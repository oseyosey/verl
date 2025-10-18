"""
Remote LLM Judge reward functions for VERL.

This module provides a remote LLM-as-a-judge functionality for computing rewards
by connecting to a vLLM server hosting models like Qwen3-32B. This design allows 
GPU-accelerated LLM inference without requiring GPU allocation on the reward worker.

The module connects to a vLLM server with OpenAI-compatible API, allowing efficient
text generation without local GPU requirements.

Key features:
- Remote vLLM server integration for GPU-accelerated text generation
- Qwen3-32B support with thinking/non-thinking modes
- Connection pooling and retry logic for reliability
- Fallback to lexical similarity when server is unavailable
- Batched processing for efficient LLM judge computation
- Compatible with llm_judge.py API for drop-in replacement

Usage example:
    from verl.utils.reward_score.llm_judge_remote import compute_score
    
    # Ensure LLM_JUDGE_SERVER_URL is set or pass server_url in extra_info
    compute_score(
        data_source="llm_judge_remote_custom",
        solution_str="The answer is 4.",
        ground_truth="2 + 2 = 4",
        extra_info={
            "problem": "What is 2+2?",
            "server_url": "http://vllm-server:8000"
        }
    )

Environment Variables:
- LLM_JUDGE_SERVER_URL: URL of the vLLM server (e.g., http://localhost:8000)
- LLM_JUDGE_SERVER_API_KEY: Optional API key for authentication
- LLM_JUDGE_SERVER_TIMEOUT: Request timeout in seconds (default: 60)

To use this module, deploy a vLLM server with Qwen3-32B:
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-32B --port 8000
"""

from __future__ import annotations

import os
import warnings
import logging
import re
from typing import List, Optional, Dict, Any
from functools import lru_cache

# Import tokenizer for lexical metrics (matching reconstruction_evaluation.py)
try:
    from transformers import AutoTokenizer
    _DEFAULT_TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
    _HAS_TOKENIZER = True
except ImportError:
    _DEFAULT_TOKENIZER = None
    _HAS_TOKENIZER = False
    warnings.warn(
        "transformers not available. Lexical metrics will use regex fallback.",
        RuntimeWarning
    )

# Try to import LLM judge client
try:
    from ...utils_rl.llm_judge_client import LLMJudgeClient, get_default_client
    _HAS_CLIENT = True
except ImportError:
    # Try alternative import paths
    try:
        from ddrl.utils_rl.llm_judge_client import LLMJudgeClient, get_default_client
        _HAS_CLIENT = True
    except ImportError:
        _HAS_CLIENT = False
        warnings.warn(
            "llm_judge_client not available. Remote LLM judge will return 0.0 when server unavailable.",
            RuntimeWarning
        )

# Note: No longer using lexical fallback - returns 0.0 when server unavailable

logger = logging.getLogger(__name__)

__all__ = ["compute_score", "compute_score_batched"]

# Default configuration (same as llm_judge.py)
DEFAULT_MODEL_NAME = "Qwen/Qwen3-32B"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_ENABLE_THINKING = False
DEFAULT_BATCH_SIZE = 128  # Optimized for 8-GPU vLLM server with batch_size_per_worker

# Import prompt templates
try:
    from .llm_judge_prompts import get_prompt_template, get_default_template
    DEFAULT_PROMPT_TEMPLATE = get_default_template()
except ImportError:
    # Fallback for standalone usage
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

# Global client instance
_GLOBAL_CLIENT: Optional[Any] = None


def _get_client(server_url: Optional[str] = None, **kwargs) -> Optional[Any]:
    """Get or create an LLM judge client."""
    global _GLOBAL_CLIENT
    
    if not _HAS_CLIENT:
        return None
    
    if server_url:
        # Create a new client for specific server
        try:
            return LLMJudgeClient(server_url=server_url, **kwargs)
        except Exception as e:
            logger.error(f"Failed to create LLM judge client for {server_url}: {e}")
            return None
    
    # Use global client
    if _GLOBAL_CLIENT is None:
        try:
            _GLOBAL_CLIENT = get_default_client()
        except Exception as e:
            logger.error(f"Failed to get default LLM judge client: {e}")
            return None
    
    return _GLOBAL_CLIENT


def _tokenize(text: str, max_tokens: Optional[int] = None) -> List[str]:
    """
    Tokenise text into a list of tokens using BERT tokenizer.
    
    This matches the tokenization in reconstruction_evaluation.py for consistency.
    Falls back to regex tokenization if BERT tokenizer is not available.
    
    Args:
        text: Text to tokenize
        max_tokens: Maximum tokens to return (for truncation)
        
    Returns:
        List of tokens
    """
    if _HAS_TOKENIZER and _DEFAULT_TOKENIZER is not None:
        # Use BERT tokenizer (same as reconstruction_evaluation.py)
        return _DEFAULT_TOKENIZER.tokenize(
            text,
            max_length=max_tokens,
            truncation=True,
        )
    else:
        # Fallback to regex tokenization
        tokens = re.findall(r'\b\w+\b', text.lower())
        if max_tokens is not None and len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return tokens


def _compute_lexical_metrics(reference: str, candidate: str) -> Dict[str, float]:
    """
    Compute lexical metrics between reference and candidate solutions.
    
    This function computes metrics required by advanced prompt templates (e.g., v4_1)
    that incorporate lexical similarity information.
    
    IMPORTANT: This implementation uses the SAME tokenizer and Jaccard calculation 
    as reconstruction_evaluation.py to ensure consistency across the codebase.
    Both use BERT's bert-base-uncased tokenizer for tokenization.
    
    Args:
        reference: Ground truth solution
        candidate: Candidate solution to evaluate
        
    Returns:
        Dict with three metrics:
        - lexical_token_overlap: Jaccard similarity (0-1), computed identically to reconstruction_evaluation.py
        - lexical_lcs_ratio: Normalized LCS ratio (0-1), normalized by ground truth length
        - length_ratio: Token length ratio (candidate/reference)
    """
    # Tokenize both texts (using BERT tokenizer to match reconstruction_evaluation.py)
    ref_tokens = _tokenize(reference)
    cand_tokens = _tokenize(candidate)
    
    # 1. Lexical token overlap (Jaccard similarity)
    # This matches the implementation in reconstruction_evaluation.py exactly
    ref_set = set(ref_tokens)
    cand_set = set(cand_tokens)
    intersection = ref_set & cand_set
    union = ref_set | cand_set
    lexical_token_overlap = len(intersection) / len(union) if union else 0.0
    
    # 2. Lexical LCS ratio (normalized by ground truth length)
    if not ref_tokens or not cand_tokens:
        lexical_lcs_ratio = 0.0
    else:
        # Compute LCS length using dynamic programming
        m, n = len(ref_tokens), len(cand_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i-1] == cand_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        # Normalize by ground truth length (reference)
        lexical_lcs_ratio = lcs_length / len(ref_tokens)
    
    # 3. Length ratio (candidate / reference)
    if not ref_tokens:
        length_ratio = 0.0 if cand_tokens else 1.0
    else:
        length_ratio = len(cand_tokens) / len(ref_tokens)
    
    return {
        "lexical_token_overlap": lexical_token_overlap,
        "lexical_lcs_ratio": lexical_lcs_ratio,
        "length_ratio": length_ratio
    }


# Removed lexical fallback - now returns 0.0 when server unavailable


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


def _format_prompt(
    prompt_template: str,
    problem: str,
    reference_solution: str,
    candidate_solution: str
) -> str:
    """
    Format the prompt template with the given inputs.
    
    Supports optional placeholders:
    - {PROBLEM}: Problem statement
    - {REFERENCE_SOLUTION}: Ground truth solution
    - {CANDIDATE_SOLUTION}: Candidate solution
    - {LEXICAL_TOKEN_OVERLAP}: Jaccard similarity metric (0-1)
    - {LEXICAL_LCS_RATIO}: Normalized LCS ratio (0-1)
    - {LENGTH_RATIO}: Length ratio (candidate/reference)
    
    Args:
        prompt_template: Template string with placeholders
        problem: The math problem statement
        reference_solution: The ground truth solution
        candidate_solution: The candidate solution to evaluate
        
    Returns:
        Formatted prompt string
    """
    # Build formatting dictionary with base values
    format_dict = {
        "REFERENCE_SOLUTION": reference_solution.strip(),
        "CANDIDATE_SOLUTION": candidate_solution.strip()
    }
    
    # Add problem if placeholder exists
    if "{PROBLEM}" in prompt_template:
        format_dict["PROBLEM"] = problem.strip()
    
    # Check if template requires lexical metrics
    needs_metrics = any(
        placeholder in prompt_template
        for placeholder in ["{LEXICAL_TOKEN_OVERLAP}", "{LEXICAL_LCS_RATIO}", "{LENGTH_RATIO}"]
    )
    
    # Compute and add lexical metrics if needed
    if needs_metrics:
        metrics = _compute_lexical_metrics(reference_solution, candidate_solution)
        format_dict["LEXICAL_TOKEN_OVERLAP"] = f"{metrics['lexical_token_overlap']:.3f}"
        format_dict["LEXICAL_LCS_RATIO"] = f"{metrics['lexical_lcs_ratio']:.3f}"
        format_dict["LENGTH_RATIO"] = f"{metrics['length_ratio']:.3f}"
    
    return prompt_template.format(**format_dict)


def _filter_refs(refs: List[str], extra_info: dict | None) -> List[str]:
    """Filter references (same as llm_judge.py)."""
    if not extra_info or not isinstance(extra_info, dict):
        return refs
    
    # 1. Exact target string(s)
    tgt = extra_info.get("target_gt")
    if isinstance(tgt, str):
        subset = [r for r in refs if r == tgt]
        if subset:
            return subset
    elif isinstance(tgt, list):
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


def _single_llm_judge_score(
    solution_str: str,
    ground_truth: str,
    client: Any,
    model_name: str = DEFAULT_MODEL_NAME,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    enable_thinking: bool = DEFAULT_ENABLE_THINKING,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    problem: str = "",
    **generation_kwargs
) -> Optional[float]:
    """Compute LLM judge score for a single solution using remote server."""
    if client is None:
        logger.error("Remote LLM judge client is not available")
        return None
    
    try:
        # Format prompt
        formatted_prompt = _format_prompt(prompt_template, problem, ground_truth, solution_str)
        
        # Generate response
        responses = client.generate_responses(
            prompts=[formatted_prompt],
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
            **generation_kwargs
        )
        
        if not responses or not responses[0]:
            logger.error("No response generated from remote server")
            return None
        
        # Extract score
        score_range = _detect_score_range(prompt_template)
        score = _extract_reward_score(responses[0], score_range)
        if score is None:
            logger.error(f"Failed to extract score from remote response: {responses[0][:100]}...")
            return None
        
        return score
        
    except Exception as e:
        logger.error(f"Error in remote LLM judge scoring: {e}")
        return None


def _batch_llm_judge_scores(
    solutions: List[str],
    references: List[str],
    problems: List[str],
    client: Any,
    model_name: str = DEFAULT_MODEL_NAME,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    enable_thinking: bool = DEFAULT_ENABLE_THINKING,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    **generation_kwargs
) -> Optional[List[float]]:
    """
    Compute LLM judge scores for multiple solution-reference pairs efficiently.
    
    This batches all prompts to the vLLM server for optimal throughput.
    
    Returns:
        List of scores if successful, None if remote server is unavailable or failed
    """
    if client is None:
        logger.error("Remote LLM judge client is not available")
        return None
    
    try:
        # Prepare all prompts
        all_prompts = []
        for solution, reference, problem in zip(solutions, references, problems):
            formatted_prompt = _format_prompt(prompt_template, problem, reference, solution)
            all_prompts.append(formatted_prompt)
        
        # Generate responses in batches
        all_responses = client.generate_responses(
            prompts=all_prompts,
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
            batch_size=batch_size,
            **generation_kwargs
        )
        
        if not all_responses:
            logger.error("No responses generated from remote server")
            return None
        
        # Extract scores
        scores = []
        failed_extractions = 0
        score_range = _detect_score_range(prompt_template)
        for i, response in enumerate(all_responses):
            if response:
                score = _extract_reward_score(response, score_range)
                if score is not None:
                    scores.append(score)
                else:
                    logger.warning(f"Failed to extract score from response {i}: {response[:100]}...")
                    scores.append(0.0)  # Use 0.0 for failed extractions
                    failed_extractions += 1
            else:
                logger.warning(f"Empty response for prompt {i}")
                scores.append(0.0)  # Use 0.0 for empty responses
                failed_extractions += 1
        
        if failed_extractions > 0:
            logger.warning(f"Failed to extract scores for {failed_extractions}/{len(all_responses)} responses")
        
        return scores
        
    except Exception as e:
        logger.error(f"Batch remote LLM judge failed: {e}")
        return None


def _best_similarity(
    sol: str, refs: List[str], client: Any,
    problem: str = "",
    model_name: str = DEFAULT_MODEL_NAME,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    enable_thinking: bool = DEFAULT_ENABLE_THINKING,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    **generation_kwargs
) -> float:
    """Find best similarity among multiple references."""
    if not refs:
        return 0.0
    
    if client is None:
        logger.warning("Remote LLM judge client unavailable, returning 0.0")
        return 0.0
    
    # For efficiency, batch all prompts
    try:
        all_prompts = []
        for ref in refs:
            formatted_prompt = _format_prompt(prompt_template, problem, ref, sol)
            all_prompts.append(formatted_prompt)
        
        responses = client.generate_responses(
            prompts=all_prompts,
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
            batch_size=batch_size,
            **generation_kwargs
        )
        
        if not responses or len(responses) != len(all_prompts):
            # Server failed
            logger.warning("Remote LLM judge server failed or returned incomplete responses, returning 0.0")
            return 0.0
        
        # Find best score
        best_score = 0.0
        score_range = _detect_score_range(prompt_template)
        for response in responses:
            if response:
                score = _extract_reward_score(response, score_range)
                if score is not None:
                    best_score = max(best_score, score)
        
        return best_score
        
    except Exception as e:
        logger.warning(f"Batch similarity computation failed: {e}, returning 0.0")
        return 0.0


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
    """Remote LLM judge similarity reward (single or batched).

    Behaviour mirrors verl.utils.reward_score.embedding_remote.compute_score:
    • Single-sample mode: solution_str vs every string in ground_truth list → best similarity.
    • Batch mode: each solution in solution_strs is compared to the references.
    Returns a float or list[float] in the range [0, 1].
    
    Remote Server Configuration (via extra_info):
    - server_url: vLLM server URL (overrides LLM_JUDGE_SERVER_URL)
    - api_key: Optional API key (overrides LLM_JUDGE_SERVER_API_KEY)
    - timeout: Request timeout in seconds
    - model_name: Model name on vLLM server (default: Qwen/Qwen3-32B)
    - temperature: Generation temperature (default: 0.7)
    - top_p: Top-p sampling parameter (default: 0.8)
    - max_new_tokens: Maximum new tokens to generate (default: 512)
    - enable_thinking: Whether to enable thinking mode (default: False)
    - problem: Problem statement for LLM judge prompt
    - prompt_template: Custom prompt template
    """

    # Extract configuration from extra_info
    config = {}
    if isinstance(extra_info, dict):
        config = extra_info.copy()
    elif extra_infos is not None and len(extra_infos) > 0:
        # In batched mode, get config from first extra_info
        first_extra_info = extra_infos[0]
        if isinstance(first_extra_info, dict):
            config = first_extra_info.copy()
    
    # Extract server configuration
    server_url = config.get("server_url")
    api_key = config.get("api_key")
    timeout = float(config.get("timeout", os.getenv("LLM_JUDGE_SERVER_TIMEOUT", "180")))
    
    # Extract LLM configuration
    model_name = config.get("model_name", config.get("model", DEFAULT_MODEL_NAME))
    
    # Handle prompt template - support both template name and direct template string
    prompt_template_config = config.get("prompt_template", "default")
    if isinstance(prompt_template_config, str) and len(prompt_template_config) < 50:
        # Assume it's a template name, try to load it
        try:
            from .llm_judge_prompts import get_prompt_template
            prompt_template = get_prompt_template(prompt_template_config)
        except (ImportError, ValueError):
            # Fallback to default if template name not found or import fails
            prompt_template = DEFAULT_PROMPT_TEMPLATE
    else:
        # Assume it's a direct template string
        prompt_template = prompt_template_config if prompt_template_config != "default" else DEFAULT_PROMPT_TEMPLATE
    
    enable_thinking = config.get("enable_thinking", DEFAULT_ENABLE_THINKING)
    temperature = float(config.get("temperature", DEFAULT_TEMPERATURE))
    top_p = float(config.get("top_p", DEFAULT_TOP_P))
    max_new_tokens = int(config.get("max_new_tokens", config.get("max_tokens", DEFAULT_MAX_NEW_TOKENS)))
    batch_size = int(config.get("batch_size", DEFAULT_BATCH_SIZE))
    
    # Get or create client
    client = _get_client(server_url=server_url, api_key=api_key, timeout=timeout)
    
    # Log if client is unavailable
    if client is None:
        logger.warning("No LLM judge server available, will return 0.0 for all scores")
    
    # Batch mode detection
    if solution_strs is not None or ground_truths is not None:
        needs_filter = False
        if extra_infos is not None:
            for ei in extra_infos:
                if isinstance(ei, dict) and (
                    "target_gt" in ei or ei.get("filter_gt_by_prompt_token")
                ):
                    needs_filter = True
                    break
        
        sols = solution_strs or []
        gts_flat: List[str] = []
        for gt in ground_truths or []:
            if isinstance(gt, list):
                gts_flat.extend(gt)
            else:
                gts_flat.append(gt)
        if not gts_flat:
            gts_flat.append("")
        
        if not needs_filter and client is not None:
            # Optimized batch processing without filtering
            all_solutions = []
            all_references = []
            all_problems = []
            pair_indices = []  # Track which solution each pair belongs to
            
            for sol_idx, sol in enumerate(sols):
                # Get problem for this solution
                problem = ""
                if extra_infos and sol_idx < len(extra_infos) and extra_infos[sol_idx]:
                    problem = extra_infos[sol_idx].get("problem", "")
                elif config:
                    problem = config.get("problem", "")
                
                # Find best reference for each solution
                if len(gts_flat) > 0:
                    # Create pairs for batch processing (always prefer batching over sequential)
                    for ref in gts_flat:
                        all_solutions.append(sol)
                        all_references.append(ref)
                        all_problems.append(problem)
                        pair_indices.append(sol_idx)
            
            if len(gts_flat) > 0 and all_solutions:
                # Batch process all pairs
                scores = _batch_llm_judge_scores(
                    all_solutions, all_references, all_problems, client,
                    model_name, prompt_template, enable_thinking,
                    temperature, top_p, max_new_tokens, batch_size
                )
                
                if scores is not None:
                    # Group scores by solution and take max
                    result_scores = []
                    for sol_idx in range(len(sols)):
                        sol_scores = [scores[i] for i, pi in enumerate(pair_indices) if pi == sol_idx]
                        best_score = max(sol_scores) if sol_scores else 0.0
                        result_scores.append(best_score)
                    return result_scores
            
            # For few references or fallback, use sequential processing
            result_scores = []
            for sol_idx, sol in enumerate(sols):
                problem = ""
                if extra_infos and sol_idx < len(extra_infos) and extra_infos[sol_idx]:
                    problem = extra_infos[sol_idx].get("problem", "")
                elif config:
                    problem = config.get("problem", "")
                
                best_score = _best_similarity(
                    sol, gts_flat, client, problem,
                    model_name, prompt_template, enable_thinking,
                    temperature, top_p, max_new_tokens, batch_size
                )
                result_scores.append(best_score)
            
            return result_scores
        
        # Filtered path with batch optimization
        if client is not None:
            # Collect all solution-reference pairs after filtering
            all_solutions = []
            all_references = []
            all_problems = []
            pair_indices = []  # Track which solution each pair belongs to
            
            defaults = [None] * len(sols) if extra_infos is None else extra_infos
            for sol_idx, (sol, gt, ei) in enumerate(zip(sols, ground_truths, defaults)):
                refs = [gt] if isinstance(gt, str) else list(gt)
                refs = _filter_refs(refs, ei)  # Apply filtering per sample
                
                # Extract per-sample problem
                sample_problem = ""
                if isinstance(ei, dict):
                    sample_problem = ei.get("problem", config.get("problem", ""))
                else:
                    sample_problem = config.get("problem", "")
                
                # Add all solution-reference pairs to the batch
                for ref in refs:
                    all_solutions.append(sol)
                    all_references.append(ref)
                    all_problems.append(sample_problem)
                    pair_indices.append(sol_idx)
            
            # Batch process all filtered pairs
            if all_solutions:
                scores = _batch_llm_judge_scores(
                    all_solutions, all_references, all_problems, client,
                    model_name, prompt_template, enable_thinking,
                    temperature, top_p, max_new_tokens, batch_size
                )
                
                if scores is not None:
                    # Group scores by solution and take max
                    result_scores = []
                    for sol_idx in range(len(sols)):
                        sol_scores = [scores[i] for i, pi in enumerate(pair_indices) if pi == sol_idx]
                        best_score = max(sol_scores) if sol_scores else 0.0
                        result_scores.append(best_score)
                    return result_scores
        
        # Fallback to sequential processing if batch failed
        res: List[float] = []
        defaults = [None] * len(sols) if extra_infos is None else extra_infos
        for sol, gt, ei in zip(sols, ground_truths, defaults):
            refs = [gt] if isinstance(gt, str) else list(gt)
            refs = _filter_refs(refs, ei)
            
            # Extract per-sample config
            sample_problem = ""
            if isinstance(ei, dict):
                sample_problem = ei.get("problem", config.get("problem", ""))
            else:
                sample_problem = config.get("problem", "")
            
            res.append(_best_similarity(
                sol, refs, client, sample_problem,
                model_name, prompt_template, enable_thinking,
                temperature, top_p, max_new_tokens, batch_size
            ))
        return res
    
    # ---------------- Single sample path ----------------
    
    if solution_str is None or ground_truth is None:
        return 0.0
    
    refs = [ground_truth] if isinstance(ground_truth, str) else list(ground_truth)
    refs = _filter_refs(refs, extra_info)
    
    problem = config.get("problem", "")
    
    return _best_similarity(
        str(solution_str), refs, client, problem,
        model_name, prompt_template, enable_thinking,
        temperature, top_p, max_new_tokens, batch_size
    )


def compute_score_batched(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str | List[str]],
    extra_infos: List[dict | None] | None = None,
):
    """Convenience wrapper for batched remote LLM judge evaluation."""
    return compute_score(
        data_sources=data_sources,
        solution_strs=solution_strs,
        ground_truths=ground_truths,
        extra_infos=extra_infos,
    )


def clear_client_cache():
    """Clear the client cache and close connections."""
    global _GLOBAL_CLIENT
    if _GLOBAL_CLIENT:
        _GLOBAL_CLIENT.close()
        _GLOBAL_CLIENT = None
    logger.info("Remote LLM judge client cache cleared")


# Test function
def _test_remote_llm_judge():
    """Simple test function to verify the remote implementation works."""
    print("[test] Testing remote LLM judge implementation...")
    
    # Test server connectivity
    server_url = os.getenv("LLM_JUDGE_SERVER_URL", "http://localhost:8000")
    print(f"\n[test] Testing connectivity to {server_url}...")
    
    client = _get_client(server_url=server_url)
    if client is None:
        print("❌ Cannot connect to remote server, testing fallback behavior...")
    else:
        print("✅ Connected to remote server")
        if client.health_check():
            print("✅ Server health check passed")
        else:
            print("⚠️ Server health check failed")
    
    # Test single scoring
    print("\n[test] Testing single scoring...")
    score = compute_score(
        data_source="llm_judge_remote_test",
        solution_str="The answer is 4.",
        ground_truth="2 + 2 = 4",
        extra_info={
            "problem": "What is 2+2?",
            "server_url": server_url,
            "max_new_tokens": 50
        }
    )
    if score is not None:
        print(f"✅ Single score: {score}")
    else:
        print("❌ Single scoring failed - returned None")
    
    # Test batch scoring
    print("\n[test] Testing batch scoring...")
    scores = compute_score(
        solution_strs=["4", "The answer is four", "2+2 equals 4"],
        ground_truths=["2 + 2 = 4", "2 + 2 = 4", "2 + 2 = 4"],
        extra_infos=[
            {"problem": "What is 2+2?", "server_url": server_url, "max_new_tokens": 50},
            {"problem": "What is 2+2?", "server_url": server_url, "max_new_tokens": 50},
            {"problem": "What is 2+2?", "server_url": server_url, "max_new_tokens": 50}
        ]
    )
    
    if scores is not None:
        print(f"✅ Batch scores: {scores}")
    else:
        print("❌ Batch scoring failed - returned None")
    
    # Clear cache
    clear_client_cache()
    print("\n[test] Test completed!")


if __name__ == "__main__":
    _test_remote_llm_judge()
