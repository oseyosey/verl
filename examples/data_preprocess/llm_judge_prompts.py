"""
LLM Judge Prompt Templates Configuration

This file contains various prompt templates for the LLM judge reward function.
You can easily modify these templates or add new ones without changing the main code.
"""

# Default prompt template for math problem similarity evaluation
DEFAULT_MATH_SIMILARITY_PROMPT = """Rate the two math problem solutions (one reference, one candidate) in terms of their similarity. Return a real value between 0-1 with 3 decimals.

INPUTS
- Problem:
{PROBLEM}

- Reference solution:
{REFERENCE_SOLUTION}

- Candidate solution:
{CANDIDATE_SOLUTION}

OUTPUT FORMAT (must follow exactly)
Output ONLY one line:
REWARD: <number between 0 and 1 with 3 decimals>"""

# Alternative prompt focusing on correctness and approach similarity
CORRECTNESS_AND_APPROACH_PROMPT = """Evaluate how similar the candidate solution is to the reference solution for the given math problem. Consider both correctness and approach. Return a score between 0-1 with 3 decimals.

PROBLEM:
{PROBLEM}

REFERENCE SOLUTION:
{REFERENCE_SOLUTION}

CANDIDATE SOLUTION:
{CANDIDATE_SOLUTION}

Scoring criteria:
- 1.0: Perfect match in correctness and approach
- 0.8-0.9: Correct answer with very similar approach
- 0.6-0.7: Correct answer with different but valid approach
- 0.4-0.5: Partially correct or minor errors
- 0.2-0.3: Major errors but some correct elements
- 0.0-0.1: Completely wrong or irrelevant

OUTPUT FORMAT (must follow exactly):
REWARD: <number between 0 and 1 with 3 decimals>"""

# Prompt focusing on mathematical reasoning quality
REASONING_QUALITY_PROMPT = """Compare the mathematical reasoning in the candidate solution against the reference solution. Focus on logical flow, mathematical accuracy, and clarity of explanation.

Given Problem:
{PROBLEM}

Reference Solution:
{REFERENCE_SOLUTION}

Candidate Solution:
{CANDIDATE_SOLUTION}

Rate the similarity in reasoning quality from 0.0 to 1.0:
- Mathematical accuracy and correctness
- Logical flow and step-by-step reasoning
- Clarity of explanation
- Use of appropriate mathematical concepts

OUTPUT FORMAT:
REWARD: <number between 0 and 1 with 3 decimals>"""

# Simplified prompt for faster evaluation
SIMPLE_SIMILARITY_PROMPT = """Compare these two math solutions and rate their similarity from 0.0 to 1.0.

Problem: {PROBLEM}
Reference: {REFERENCE_SOLUTION}
Candidate: {CANDIDATE_SOLUTION}

REWARD: <number between 0 and 1 with 3 decimals>"""

# Detailed prompt with explicit scoring rubric
DETAILED_RUBRIC_PROMPT = """Evaluate the candidate math solution against the reference solution using this rubric:

PROBLEM: {PROBLEM}

REFERENCE SOLUTION: {REFERENCE_SOLUTION}

CANDIDATE SOLUTION: {CANDIDATE_SOLUTION}

SCORING RUBRIC:
1.0: Identical or equivalent solutions with same final answer
0.9: Same answer, very similar method/approach
0.8: Same answer, similar method with minor differences
0.7: Same answer, different but equally valid method
0.6: Same answer, less elegant or more complex method
0.5: Partially correct, some right steps but errors
0.4: Some correct mathematical concepts but wrong answer
0.3: Major errors but shows some understanding
0.2: Minimal correct elements, mostly wrong
0.1: Completely different approach, wrong answer
0.0: Completely wrong or nonsensical

OUTPUT (must be exact format):
REWARD: <number between 0 and 1 with 3 decimals>"""

# Prompt template mapping for easy selection
PROMPT_TEMPLATES = {
    "default": DEFAULT_MATH_SIMILARITY_PROMPT,
    "correctness_approach": CORRECTNESS_AND_APPROACH_PROMPT,
    "reasoning_quality": REASONING_QUALITY_PROMPT,
    "simple": SIMPLE_SIMILARITY_PROMPT,
    "detailed_rubric": DETAILED_RUBRIC_PROMPT,
}

def get_prompt_template(template_name: str = "default") -> str:
    """
    Get a prompt template by name.
    
    Args:
        template_name: Name of the template to retrieve
        
    Returns:
        The prompt template string
        
    Raises:
        ValueError: If template_name is not found
    """
    if template_name not in PROMPT_TEMPLATES:
        available = ", ".join(PROMPT_TEMPLATES.keys())
        raise ValueError(f"Unknown template '{template_name}'. Available: {available}")
    
    return PROMPT_TEMPLATES[template_name]

def list_available_templates():
    """List all available prompt template names."""
    return list(PROMPT_TEMPLATES.keys())

# For backward compatibility, export the default template
DEFAULT_PROMPT_TEMPLATE = DEFAULT_MATH_SIMILARITY_PROMPT
