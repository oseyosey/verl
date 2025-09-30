"""
LLM-as-a-Judge reward functions for VERL.

This module provides a way to compute rewards by using an LLM as a judge
to evaluate the similarity between the model response (``solution_str``) and the
reference answer (``ground_truth``). The LLM is prompted to return a score
between 0 and 1.

Highlights
----------
* **LiteLLM** integration for calling various LLM providers (Gemini, OpenAI, etc.)
* **Batched** processing for efficiency when evaluating multiple examples
* **Retry logic** with exponential backoff for handling API failures
* **Extraction logic** to parse numerical scores from LLM responses
* **Thinking mode** support for Gemini models with configurable token budgets

Usage example
~~~~~~~~~~~~~
>>> from verl.utils.reward_score.llm_judge import compute_score
>>> extra_info = {
...     "prompt_template": "Rate similarity between solutions: {REFERENCE_SOLUTION} vs {CANDIDATE_SOLUTION}",
...     "problem": "What is 2+2?"
... }
>>> compute_score(
...     data_source="llm_judge_custom",
...     solution_str="The answer is 4.",
...     ground_truth="2 + 2 = 4",
...     extra_info=extra_info
... )
0.950

Environment variables:
* Set GOOGLE_API_KEY for Gemini models
* Set OPENAI_API_KEY for OpenAI models
* etc. (see LiteLLM documentation)
"""

from __future__ import annotations

import os
import re
import time
import warnings
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

try:
    import litellm
    from litellm import batch_completion
    _HAS_LITELLM = True
except ImportError:
    _HAS_LITELLM = False
    warnings.warn(
        "LiteLLM is not installed – LLM judge reward will return 0.0 scores. "
        "Install it via `pip install litellm`.",
        RuntimeWarning,
    )

__all__ = ["compute_score", "compute_score_batched"]

# Default model and configuration
DEFAULT_MODEL = "gemini/gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.6  # Balanced temperature for scoring
DEFAULT_MAX_TOKENS = 3096  # Total tokens including thinking tokens
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0

# Thinking mode configuration
DEFAULT_THINKING_ENABLED = True  # Enable thinking mode by default for better reasoning
DEFAULT_THINKING_BUDGET = 2048   # Budget for thinking tokens (max_tokens - this = output tokens)

# Default prompt template
DEFAULT_PROMPT_TEMPLATE = """
Rate the two math problem solutions (one reference, one candidate) in terms of their similarity. Return a real value between 0-1 with 3 decimals.

INPUTS
- Problem:
{PROBLEM}

- Reference solution:
{REFERENCE_SOLUTION}

- Candidate solution:
{CANDIDATE_SOLUTION}

OUTPUT FORMAT (must follow exactly)
Output ONLY one line:
REWARD: <number between 0 and 1 with 3 decimals>
""".strip()


def _extract_reward_score(response_text: str, score_range: str = "0-1") -> Optional[float]:
    """
    Extract reward score from LLM response text.
    
    Expected format: "REWARD: X.XXX" where range depends on score_range parameter
    
    Args:
        response_text: Raw response from LLM
        score_range: Expected score range, either "0-1" or "0-100"
        
    Returns:
        Extracted score normalized to [0, 1] range, or None if extraction failed
    """
    if not response_text:
        return None
    
    # Look for "REWARD:" followed by a number
    pattern = r"REWARD:\s*([0-9]*\.?[0-9]+)"
    match = re.search(pattern, response_text.strip(), re.IGNORECASE)
    
    if match:
        try:
            score = float(match.group(1))
            
            # Normalize based on expected range
            if score_range == "0-100":
                # For 0-100 scale, normalize to 0-1
                normalized_score = score / 100.0
                return max(0.0, min(1.0, normalized_score))
            else:
                # For 0-1 scale, clamp to [0, 1] range
                return max(0.0, min(1.0, score))
        except ValueError:
            pass
    
    # Fallback: look for any number
    if score_range == "0-100":
        # For 0-100 scale, look for numbers up to 100
        pattern = r"\b(\d{1,3}(?:\.\d+)?)\b"
        matches = re.findall(pattern, response_text)
        if matches:
            try:
                score = float(matches[-1])  # Take the last match
                normalized_score = score / 100.0
                return max(0.0, min(1.0, normalized_score))
            except ValueError:
                pass
    else:
        # For 0-1 scale, look for numbers between 0 and 1
        pattern = r"\b(0\.\d{1,3}|1\.0{1,3}|0|1)\b"
        matches = re.findall(pattern, response_text)
        if matches:
            try:
                score = float(matches[-1])  # Take the last match
                return max(0.0, min(1.0, score))
            except ValueError:
                pass
    
    return None


def _detect_score_range(prompt_template: str) -> str:
    """
    Detect the expected score range from the prompt template.
    
    Args:
        prompt_template: The prompt template string
        
    Returns:
        "0-100" if template expects 0-100 scale, "0-1" otherwise
    """
    if "between 0 and 100" in prompt_template or "0-100" in prompt_template:
        return "0-100"
    return "0-1"


def _tokenize(text: str) -> List[str]:
    """Tokenise text into a list of lowercase terms.
    
    Split on whitespace and punctuation, keeping alphanumeric tokens.
    This ensures single letters like 'A', 'B', 'C' are captured correctly.
    """
    # Use regex to split on non-alphanumeric characters and extract tokens
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


def _filter_refs(refs: List[str], extra_info: dict | None) -> List[str]:
    """Return a possibly reduced list of refs according to extra_info.

    Supported options in extra_info:
    • target_gt – a string or list of strings; keep only references that exactly match any of them.
    • filter_gt_by_prompt_token (bool) and prompt – extract the
      last whitespace‐delimited token of prompt (lower‐cased) and keep only
      references that contain that token (after simple regex tokenisation).

    If filtering removes all references, the original list is returned so
    that scoring never fails with an empty pool.
    """
    if not extra_info or not isinstance(extra_info, dict):
        return refs

    # 1. Exact target string(s)
    tgt = extra_info.get("target_gt")
    if isinstance(tgt, str):
        subset = [r for r in refs if r == tgt]
        if subset:
            return subset
    elif isinstance(tgt, list):
        # Handle list of target strings - keep references that match any of them
        subset = [r for r in refs if r in tgt]
        if subset:
            return subset

    # 2. Last prompt token heuristic
    if extra_info.get("filter_gt_by_prompt_token") and "prompt" in extra_info:
        prompt_txt = str(extra_info["prompt"]).strip()
        if prompt_txt:
            last_tok = prompt_txt.split()[-1].lower()
            subset = [r for r in refs if last_tok in _tokenize(r)]
            if subset:
                return subset

    return refs


def _call_llm_single(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    thinking_enabled: bool = DEFAULT_THINKING_ENABLED,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    **model_kwargs
) -> Optional[str]:
    """
    Call LLM API for a single prompt with retry logic.
    
    Args:
        prompt: The prompt to send to the LLM
        model: LiteLLM model identifier
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate (includes thinking tokens)
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries
        retry_delay: Base delay between retries
        thinking_enabled: Whether to enable thinking mode for reasoning
        thinking_budget: Number of tokens allocated for thinking (subtracted from max_tokens)
        **model_kwargs: Additional model parameters
        
    Returns:
        LLM response text, or None if all retries failed
    """
    if not _HAS_LITELLM:
        return None
    
    retry_count = 0
    last_error = None
    null_response_count = 0
    max_null_retries = 3
    
    while retry_count < max_retries:
        try:
            # Prepare completion arguments
            completion_args = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
                **model_kwargs
            }
            
            # Add thinking mode configuration for Gemini models
            if model.startswith("gemini/") and thinking_enabled:
                completion_args["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget
                }
            elif model.startswith("gemini/") and not thinking_enabled:
                # Explicitly disable thinking mode
                completion_args["reasoning_effort"] = "disable"
            
            # Call LiteLLM
            response = litellm.completion(**completion_args)
            
            # Extract response text
            response_text = response.choices[0].message.content
            
            # Check for null/empty response
            if response_text is None or (isinstance(response_text, str) and len(response_text.strip()) == 0):
                null_response_count += 1
                if null_response_count <= max_null_retries:
                    logger.warning(f"Received null/empty response (attempt {null_response_count}/{max_null_retries}), retrying...")
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"Received null/empty response {max_null_retries} times, treating as failure")
                    return None
            
            return response_text
            
        except Exception as e:
            last_error = e
            retry_count += 1
            if retry_count < max_retries:
                logger.warning(f"Error calling LLM (attempt {retry_count}/{max_retries}): {e}")
                time.sleep(retry_delay * retry_count)  # Exponential backoff
            else:
                logger.error(f"Failed after {max_retries} retries: {e}")
    
    return None


def _call_llm_batch(
    prompts: List[str],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    thinking_enabled: bool = DEFAULT_THINKING_ENABLED,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    **model_kwargs
) -> List[Optional[str]]:
    """
    Call LLM API for multiple prompts with batch processing.
    
    Args:
        prompts: List of prompts to send to the LLM
        model: LiteLLM model identifier
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate (includes thinking tokens)
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries
        retry_delay: Base delay between retries
        thinking_enabled: Whether to enable thinking mode for reasoning
        thinking_budget: Number of tokens allocated for thinking (subtracted from max_tokens)
        **model_kwargs: Additional model parameters
        
    Returns:
        List of LLM response texts (same order as input prompts)
    """
    if not _HAS_LITELLM or not prompts:
        return [None] * len(prompts)
    
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            # Prepare batch messages
            batch_messages = []
            for prompt in prompts:
                batch_messages.append([{"role": "user", "content": prompt}])
            
            # Prepare completion arguments
            completion_args = {
                "model": model,
                "messages": batch_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
                **model_kwargs
            }
            
            # Add thinking mode configuration for Gemini models
            if model.startswith("gemini/") and thinking_enabled:
                completion_args["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget
                }
            elif model.startswith("gemini/") and not thinking_enabled:
                # Explicitly disable thinking mode
                completion_args["reasoning_effort"] = "disable"
            
            # Call LiteLLM batch completion
            responses = batch_completion(**completion_args)
            
            # Extract response texts and handle null responses
            response_texts = []
            null_indices = []
            
            for i, response in enumerate(responses):
                if response and hasattr(response, 'choices') and len(response.choices) > 0:
                    response_text = response.choices[0].message.content
                    if response_text is None or (isinstance(response_text, str) and len(response_text.strip()) == 0):
                        response_texts.append(None)
                        null_indices.append(i)
                    else:
                        response_texts.append(response_text)
                else:
                    response_texts.append(None)
                    null_indices.append(i)
            
            # If we have null responses, retry just those prompts individually
            if null_indices:
                logger.warning(f"Batch contained {len(null_indices)}/{len(prompts)} null/empty responses, retrying individually")
                
                for null_idx in null_indices:
                    try:
                        retry_response = _call_llm_single(
                            prompts[null_idx], model, temperature, max_tokens,
                            timeout, max_retries, retry_delay, thinking_enabled, thinking_budget, **model_kwargs
                        )
                        response_texts[null_idx] = retry_response
                    except Exception as e:
                        logger.error(f"Failed to retry null response for prompt {null_idx}: {e}")
                        response_texts[null_idx] = None
            
            return response_texts
            
        except Exception as e:
            last_error = e
            retry_count += 1
            if retry_count < max_retries:
                logger.warning(f"Error in batch LLM call (attempt {retry_count}/{max_retries}): {e}")
                time.sleep(retry_delay * retry_count)
            else:
                logger.error(f"Batch call failed after {max_retries} retries: {e}")
                # Fall back to individual calls
                logger.info("Falling back to individual API calls...")
                return [_call_llm_single(
                    prompt, model, temperature, max_tokens, timeout, max_retries, retry_delay, thinking_enabled, thinking_budget, **model_kwargs
                ) for prompt in prompts]
    
    return [None] * len(prompts)


def _format_prompt(
    prompt_template: str,
    problem: str,
    reference_solution: str,
    candidate_solution: str
) -> str:
    """
    Format the prompt template with the given inputs.
    
    Args:
        prompt_template: Template string with placeholders
        problem: The math problem statement
        reference_solution: The ground truth solution
        candidate_solution: The candidate solution to evaluate
        
    Returns:
        Formatted prompt string
    """
    # Check if PROBLEM placeholder exists in template
    if "{PROBLEM}" in prompt_template:
        return prompt_template.format(
            PROBLEM=problem.strip(),
            REFERENCE_SOLUTION=reference_solution.strip(),
            CANDIDATE_SOLUTION=candidate_solution.strip()
        )
    else:
        # Only format REFERENCE_SOLUTION and CANDIDATE_SOLUTION
        return prompt_template.format(
            REFERENCE_SOLUTION=reference_solution.strip(),
            CANDIDATE_SOLUTION=candidate_solution.strip()
        )


def _get_llm_config_from_extra_info(extra_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract LLM configuration parameters from extra_info."""
    if not isinstance(extra_info, dict):
        return {}
    
    config = {}
    
    # Model configuration
    if "model" in extra_info:
        config["model"] = extra_info["model"]
    if "temperature" in extra_info:
        config["temperature"] = float(extra_info["temperature"])
    if "max_tokens" in extra_info:
        config["max_tokens"] = int(extra_info["max_tokens"])
    if "timeout" in extra_info:
        config["timeout"] = int(extra_info["timeout"])
    if "max_retries" in extra_info:
        config["max_retries"] = int(extra_info["max_retries"])
    if "retry_delay" in extra_info:
        config["retry_delay"] = float(extra_info["retry_delay"])
    
    # Thinking mode parameters
    if "thinking_enabled" in extra_info:
        config["thinking_enabled"] = bool(extra_info["thinking_enabled"])
    if "thinking_budget" in extra_info:
        config["thinking_budget"] = int(extra_info["thinking_budget"])
    
    # Additional model parameters
    model_kwargs = {}
    for key in ["top_p", "top_k", "presence_penalty", "frequency_penalty"]:
        if key in extra_info:
            model_kwargs[key] = extra_info[key]
    if model_kwargs:
        config.update(model_kwargs)
    
    return config


def _single_llm_judge_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None
) -> float:
    """
    Compute LLM judge score for a single solution-ground_truth pair.
    
    Args:
        solution_str: The candidate solution to evaluate
        ground_truth: The reference solution
        extra_info: Dictionary containing prompt template and problem context
        
    Returns:
        Score between 0 and 1, or 0.0 if evaluation failed
    """
    if not extra_info or not isinstance(extra_info, dict):
        logger.warning("LLM judge requires extra_info with prompt_template and problem")
        return 0.0
    
    # Get prompt template
    prompt_template = extra_info.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)
    
    # Get problem context
    problem = extra_info.get("problem", "")
    if not problem:
        logger.warning("LLM judge requires 'problem' field in extra_info")
        return 0.0
    
    # Format the prompt
    try:
        formatted_prompt = _format_prompt(prompt_template, problem, ground_truth, solution_str)
    except KeyError as e:
        logger.error(f"Error formatting prompt template: {e}")
        return 0.0
    
    # Get LLM configuration
    llm_config = _get_llm_config_from_extra_info(extra_info)
    
    # Set defaults
    model = llm_config.get("model", DEFAULT_MODEL)
    temperature = llm_config.get("temperature", DEFAULT_TEMPERATURE)
    max_tokens = llm_config.get("max_tokens", DEFAULT_MAX_TOKENS)
    timeout = llm_config.get("timeout", DEFAULT_TIMEOUT)
    max_retries = llm_config.get("max_retries", DEFAULT_MAX_RETRIES)
    retry_delay = llm_config.get("retry_delay", DEFAULT_RETRY_DELAY)
    thinking_enabled = llm_config.get("thinking_enabled", DEFAULT_THINKING_ENABLED)
    thinking_budget = llm_config.get("thinking_budget", DEFAULT_THINKING_BUDGET)
    
    # Extract model kwargs
    model_kwargs = {k: v for k, v in llm_config.items() 
                   if k not in ["model", "temperature", "max_tokens", "timeout", "max_retries", "retry_delay", "thinking_enabled", "thinking_budget"]}
    
    # Call LLM
    response_text = _call_llm_single(
        formatted_prompt, model, temperature, max_tokens,
        timeout, max_retries, retry_delay, thinking_enabled, thinking_budget, **model_kwargs
    )
    
    if response_text is None:
        logger.warning("LLM call failed, returning 0.0")
        return 0.0
    
    # Extract score
    score_range = _detect_score_range(prompt_template)
    score = _extract_reward_score(response_text, score_range)
    if score is None:
        logger.warning(f"Failed to extract score from LLM response: {response_text[:100]}...")
        return 0.0
    
    return score




def compute_score(
    data_source: str | List[str] | None = None,
    solution_str: str | List[str] | None = None,
    ground_truth: str | List[str] | None = None,
    extra_info: dict | List[dict] | None = None,
    *,
    data_sources: List[str] | None = None,
    solution_strs: List[str] | None = None,
    ground_truths: List[str | List[str]] | None = None,
    extra_infos: List[dict | None] | None = None,
) -> float | List[float]:
    """
    LLM judge reward (single or batched).

    Behaviour mirrors verl.utils.reward_score.lexical.compute_score:
    • Single-sample mode: LLM evaluates solution_str vs ground_truth
    • Batch mode: LLM evaluates each solution against corresponding ground truth
    Returns a float or list[float] in the range [0, 1].
    
    Args:
        data_source: Data source identifier (ignored)
        solution_str: Candidate solution(s) to evaluate
        ground_truth: Reference solution(s)
        extra_info: Dictionary containing 'prompt_template' and 'problem'
        data_sources: Batch mode data sources
        solution_strs: Batch mode candidate solutions
        ground_truths: Batch mode reference solutions
        extra_infos: Batch mode extra info
        
    Returns:
        Score(s) between 0 and 1
    """
    
    # Batch mode detection
    if solution_strs is not None or ground_truths is not None:
        sols = solution_strs or []
        gts = ground_truths or []
        infos = extra_infos or [None] * len(sols)
        
        # Ensure all lists have the same length
        min_len = min(len(sols), len(gts))
        sols = sols[:min_len]
        gts = gts[:min_len]
        infos = infos[:min_len] if infos else [None] * min_len
        
        if not sols:
            return []
        
        # Check if we can use batch processing
        use_batch = True
        first_info = None
        needs_filter = False
        
        for info in infos:
            if info is None:
                use_batch = False
                break
            if first_info is None:
                first_info = info
            # For batch processing, all examples should use the same model/config
            elif (info.get("model") != first_info.get("model") or
                  info.get("temperature") != first_info.get("temperature")):
                use_batch = False
                break
            # Check if filtering is needed
            if isinstance(info, dict) and (
                "target_gt" in info or info.get("filter_gt_by_prompt_token")
            ):
                needs_filter = True
        
        if use_batch and first_info and len(sols) > 1:
            # Batch processing
            return _compute_score_batch(sols, gts, infos)
        else:
            # Individual processing
            results = []
            for sol, gt, info in zip(sols, gts, infos):
                # Handle multiple references
                refs = [gt] if isinstance(gt, str) else list(gt or [])
                refs = _filter_refs(refs, info)
                
                if not refs:
                    results.append(0.0)
                    continue
                
                # Evaluate against all references and take max
                best_score = 0.0
                for ref in refs:
                    try:
                        score = _single_llm_judge_score(str(sol), ref, info)
                        best_score = max(best_score, score)
                        if best_score == 1.0:  # Early exit if perfect score
                            break
                    except Exception as e:
                        logger.error(f"Error computing LLM judge score: {e}")
                        continue
                
                results.append(best_score)
            
            return results
    
    # Single sample mode
    if solution_str is None or ground_truth is None:
        return 0.0
    
    # Handle ground_truth as list
    refs = [ground_truth] if isinstance(ground_truth, str) else list(ground_truth or [])
    refs = _filter_refs(refs, extra_info)
    
    if not refs:
        return 0.0
    
    # Evaluate against all references and take max
    best_score = 0.0
    for ref in refs:
        try:
            score = _single_llm_judge_score(str(solution_str), ref, extra_info)
            best_score = max(best_score, score)
            if best_score == 1.0:  # Early exit if perfect score
                break
        except Exception as e:
            logger.error(f"Error computing LLM judge score for reference: {e}")
            continue
    
    return best_score


def _compute_score_batch(
    solution_strs: List[str],
    ground_truths: List[str | List[str]],
    extra_infos: List[Dict[str, Any]]
) -> List[float]:
    """
    Compute scores for a batch of examples using batch LLM API calls.
    
    Handles multiple references per solution efficiently by batching all
    (solution, reference) pairs into a single LLM call.
    
    Args:
        solution_strs: List of candidate solutions
        ground_truths: List of reference solutions (each can be str or List[str])
        extra_infos: List of extra info dictionaries
        
    Returns:
        List of scores between 0 and 1 (max score for each solution)
    """
    if not solution_strs or not extra_infos:
        return []
    
    # Get configuration from first extra_info
    first_info = extra_infos[0]
    llm_config = _get_llm_config_from_extra_info(first_info)
    
    model = llm_config.get("model", DEFAULT_MODEL)
    temperature = llm_config.get("temperature", DEFAULT_TEMPERATURE)
    max_tokens = llm_config.get("max_tokens", DEFAULT_MAX_TOKENS)
    timeout = llm_config.get("timeout", DEFAULT_TIMEOUT)
    max_retries = llm_config.get("max_retries", DEFAULT_MAX_RETRIES)
    retry_delay = llm_config.get("retry_delay", DEFAULT_RETRY_DELAY)
    thinking_enabled = llm_config.get("thinking_enabled", DEFAULT_THINKING_ENABLED)
    thinking_budget = llm_config.get("thinking_budget", DEFAULT_THINKING_BUDGET)
    
    model_kwargs = {k: v for k, v in llm_config.items() 
                   if k not in ["model", "temperature", "max_tokens", "timeout", "max_retries", "retry_delay", "thinking_enabled", "thinking_budget"]}
    
    # Build all (solution_idx, reference) pairs with filtering
    prompts = []
    prompt_mappings = []  # (solution_idx, reference_idx)
    score_ranges = []  # Store score range for each prompt
    
    for sol_idx, (sol, gt, info) in enumerate(zip(solution_strs, ground_truths, extra_infos)):
        # Handle multiple references
        refs = [gt] if isinstance(gt, str) else list(gt or [])
        refs = _filter_refs(refs, info)
        
        if not refs:
            logger.warning(f"No references after filtering for solution {sol_idx}")
            # If no references after filtering, we'll use empty ref
            refs = [""]
        
        prompt_template = info.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)
        problem = info.get("problem", "")
        score_range = _detect_score_range(prompt_template)
        
        # Create prompts for all references of this solution
        for ref_idx, ref in enumerate(refs):
            try:
                formatted_prompt = _format_prompt(prompt_template, problem, ref, str(sol))
                prompts.append(formatted_prompt)
                prompt_mappings.append((sol_idx, ref_idx))
                score_ranges.append(score_range)
            except Exception as e:
                logger.error(f"Error formatting prompt for solution {sol_idx}, ref {ref_idx}: {e}")
                prompts.append("")  # Will result in None response
                prompt_mappings.append((sol_idx, ref_idx))
                score_ranges.append(score_range)
    
    # Call LLM batch API with all prompts at once
    responses = _call_llm_batch(
        prompts, model, temperature, max_tokens,
        timeout, max_retries, retry_delay, thinking_enabled, thinking_budget, **model_kwargs
    )
    
    # Group scores by solution and take max
    solution_scores = [0.0] * len(solution_strs)
    
    for prompt_idx, response in enumerate(responses):
        sol_idx, ref_idx = prompt_mappings[prompt_idx]
        
        if response is None:
            logger.warning(f"LLM call failed for solution {sol_idx}, ref {ref_idx}")
            score = 0.0
        else:
            score_range = score_ranges[prompt_idx]
            score = _extract_reward_score(response, score_range)
            if score is None:
                logger.warning(f"Failed to extract score from response for sol {sol_idx}, ref {ref_idx}: {response[:100]}...")
                score = 0.0
        
        # Update max score for this solution
        solution_scores[sol_idx] = max(solution_scores[sol_idx], score)
    
    return solution_scores


def compute_score_batched(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str | List[str]],
    extra_infos: List[dict | None] | None = None,
) -> List[float]:
    """
    Convenience wrapper for batched LLM judge scoring.
    
    Args:
        data_sources: List of data source identifiers (ignored)
        solution_strs: List of candidate solutions
        ground_truths: List of reference solutions
        extra_infos: List of extra info dictionaries
        
    Returns:
        List of scores between 0 and 1
    """
    return compute_score(
        data_sources=data_sources,
        solution_strs=solution_strs,
        ground_truths=ground_truths,
        extra_infos=extra_infos,
    )
