"""
Prompt templates for LLM judge evaluation.

This module contains various prompt templates for evaluating solution similarity
using LLM-as-a-judge. Templates are stored as simple string constants and can be
accessed by name or used directly.
"""


PROMPT_TEMPLATE_V0 = """
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


PROMPT_TEMPLATE_V0_1 = """TASK DESCRIPTION
Rate the two math problem solutions (one reference, one candidate) in terms of their similarity.  Return a real value between 0-1 with 3 decimals.

INPUTS
- Reference solution:
{REFERENCE_SOLUTION}

- Candidate solution:
{CANDIDATE_SOLUTION}


OUTPUT FORMAT (must follow exactly)
REWARD: <number between 0 and 1 with 3 decimals>
""".strip()


PROMPT_TEMPLATE_V1 = """
Rate the two math problem solutions (one reference, one candidate) in terms of their similarity. Return a real value between 0-1 with 3 decimals.

EVALUATION CRITERIA:
1. Mathematical correctness - Are candidate solution mathematically sound as the reference solution?
2. Solution approach - Do candidate solution use similar methods or reasoning as the reference solution? 
3. Final answer - Do candidate solution arrive at the same conclusion as the reference solution?
4. Overall clarity - Are the reasoning and solution steps correct, sonsistent, and logically sound as the reference solution?


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



PROMPT_TEMPLATE_V1_1 = """TASK DESCRIPTION
Rate the two math problem solutions (one reference, one candidate) in terms of their similarity. Return a real value between 0-1 with 3 decimals.


EVALUATION CRITERIA
1. Mathematical correctness - Are candidate solution mathematically sound as the reference solution?
2. Solution approach - Do candidate solution use similar methods or reasoning as the reference solution? 
3. Final answer - Do candidate solution arrive at the same conclusion as the reference solution?
4. Overall clarity - Are the reasoning and solution steps correct, sonsistent, and logically sound as the reference solution?


INPUTS
- Reference solution:
{REFERENCE_SOLUTION}

- Candidate solution:
{CANDIDATE_SOLUTION}

OUTPUT FORMAT (must follow exactly)
Output ONLY one line:
REWARD: <real value between 0 and 1 with 3 decimals>
""".strip()


PROMPT_TEMPLATE_V1_2 = """TASK DESCRIPTION
Rate the two math problem solutions (one reference, one candidate) in terms of their similarity. Return a real value between 0-1 with 3 decimals.


EVALUATION CRITERIA
1. Mathematical correctness - Is the candidate solution mathematically sound as the reference solution?
2. Solution approach - Do candidate solution use similar methods or reasoning as the reference solution? 
3. Final answer - Do candidate solution arrive at the same answer (enclosed in  "\\boxed{{}}") as the reference solution?
4. Overall similarity -  Do the solutions exhibit high lexical/token/semantic overlap as the reference solution?


INPUTS
- Reference solution:
{REFERENCE_SOLUTION}

- Candidate solution:
{CANDIDATE_SOLUTION}

OUTPUT FORMAT (must follow exactly)
Output ONLY one line:
REWARD: <real value between 0 and 1 with 3 decimals>
""".strip()

PROMPT_TEMPLATE_V1_3 = """TASK DESCRIPTION
Rate the two math problem solutions (one reference, one candidate) in terms of their similarity. Return a real value between 0-1 with 3 decimals.


EVALUATION CRITERIA
1. Mathematical correctness - Is the candidate solution mathematically sound as the reference solution?
2. Solution approach - Do candidate solution use similar methods or reasoning as the reference solution? 
3. Final answer - Do candidate solution arrive at the same answer (enclosed in  "\\boxed{{}}") as the reference solution?
4. Overall similarity -  Do the solutions exhibit high lexical/token/semantic overlap as the reference solution?
5. Length conformity — Is the candidate solution’s word count within ±25% of the reference solution’s word count?

INPUTS
- Reference solution:
{REFERENCE_SOLUTION}

- Candidate solution:
{CANDIDATE_SOLUTION}

OUTPUT FORMAT (must follow exactly)
Output ONLY one line:
REWARD: <real value between 0 and 1 with 3 decimals>
""".strip()

PROMPT_TEMPLATE_V2 = """TASK DESCRIPTION
Rate two solutions to the same math problem (one reference, one candidate) for similarity. The final answer in each solution is enclosed in "\\boxed{{}}". Return a real-valued score between 0 and 1 with exactly 3 decimals.

EVALUATION CRITERIA (consider all, equally)
1. Mathematical correctness — Is the candidate mathematically sound as in the reference?
2. Solution approach — Does the candidate use methods/reasoning similar to the reference?
3. Final answer — Does the candidate arrive at the same \\boxed{{}} answer as the reference?
4. Overall clarity — Are the candidate's reasoning and steps as consistent and logically sound as the reference?
5. Surface & semantic resemblance — Do the two solutions look very similar in wording, symbols, step order, and intermediate expressions (i.e., high lexical/token overlap and near-paraphrase semantics)? Penalize added filler or unnecessary rephrasing.

INPUTS
- Reference solution:
{REFERENCE_SOLUTION}

- Candidate solution:
{CANDIDATE_SOLUTION}

OUTPUT FORMAT (must follow exactly)
REWARD: <real value between 0 and 1 with 3 decimals>
""".strip()


PROMPT_TEMPLATE_V2_1 = """TASK DESCRIPTION
Rate two solutions to the same math problem (one reference, one candidate) for similarity. The final answer in each solution is enclosed in "\\boxed{{}}". Return a real-valued score between 0 and 1 with exactly 3 decimals.

EVALUATION CRITERIA (consider all, equally)
1. Mathematical correctness — Is the candidate mathematically sound as in the reference?
2. Solution approach — Does the candidate use methods/reasoning similar to the reference?
3. Final answer — Does the candidate arrive at the same \\boxed{{}} answer as the reference?
4. Overall clarity — Are the candidate's reasoning and steps as consistent and logically sound as the reference?
5. Surface & semantic resemblance — Do the two solutions look very similar in wording, symbols, step order, and intermediate expressions (i.e., high lexical/token overlap and near-paraphrase semantics)? Penalize added filler or unnecessary rephrasing.

INPUTS
- Reference solution:
{REFERENCE_SOLUTION}

- Candidate solution:
{CANDIDATE_SOLUTION}

OUTPUT FORMAT (must follow exactly)
Output ONLY one line:
REWARD: <real value between 0 and 1 with 3 decimals>
""".strip()


# Detailed prompt template - includes reasoning steps
DETAILED_PROMPT_TEMPLATE = """
You are an expert mathematics teacher evaluating the similarity between two solutions to a math problem. You need to rate how similar the candidate solution is to the reference solution.

EVALUATION CRITERIA:
1. Mathematical correctness - Are both solutions mathematically sound?
2. Solution approach - Do they use similar methods or reasoning?
3. Final answer - Do they arrive at the same conclusion?
4. Presentation clarity - Are the explanations similarly structured?

INPUTS
- Problem:
{PROBLEM}

- Reference solution:
{REFERENCE_SOLUTION}

- Candidate solution:
{CANDIDATE_SOLUTION}

INSTRUCTIONS:
1. First, analyze the mathematical correctness of both solutions
2. Compare the approaches and reasoning used
3. Evaluate how well the candidate matches the reference
4. Assign a similarity score from 0 to 1 where:
   - 0.0-0.3: Very different (wrong answer or completely different approach)
   - 0.4-0.6: Somewhat similar (some overlap but significant differences)
   - 0.7-0.8: Quite similar (similar approach and correct answer)
   - 0.9-1.0: Very similar (nearly identical reasoning and presentation)

Think through your evaluation step by step, then provide your final score.

OUTPUT FORMAT (must follow exactly)
After your reasoning, output ONLY one line:
REWARD: <number between 0 and 1 with 3 decimals>
""".strip()




# Template name mapping
PROMPT_TEMPLATES = {
    "default": PROMPT_TEMPLATE_V1,
    "detailed": DETAILED_PROMPT_TEMPLATE,
    "v0": PROMPT_TEMPLATE_V0,
    "v0_1": PROMPT_TEMPLATE_V0_1,
    "v1": PROMPT_TEMPLATE_V1,
    "v1_1": PROMPT_TEMPLATE_V1_1,
    "v1_2": PROMPT_TEMPLATE_V1_2,
    "v1_3": PROMPT_TEMPLATE_V1_3,
    "v2": PROMPT_TEMPLATE_V2,
    "v2_1": PROMPT_TEMPLATE_V2_1
}


def get_prompt_template(template_name: str) -> str:
    """
    Get a prompt template by name.
    
    Args:
        template_name: Name of the template ("default", "detailed", "v0")
        
    Returns:
        The prompt template string
        
    Raises:
        ValueError: If template name is not found
    """
    if template_name in PROMPT_TEMPLATES:
        return PROMPT_TEMPLATES[template_name]
    else:
        available_templates = list(PROMPT_TEMPLATES.keys())
        raise ValueError(
            f"Unknown prompt template: '{template_name}'. "
            f"Available templates: {available_templates}"
        )


def get_default_template() -> str:
    """Get the default prompt template."""
    return PROMPT_TEMPLATES["default"]


def list_available_templates() -> list:
    """List all available template names."""
    return list(PROMPT_TEMPLATES.keys())
