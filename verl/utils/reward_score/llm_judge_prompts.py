"""
Prompt templates for LLM judge evaluation.

This module contains various prompt templates for evaluating solution similarity
using LLM-as-a-judge. Templates are stored as simple string constants and can be
accessed by name or used directly.
"""

# =============================================================================
# PROMPT TEMPLATES - VERSION HISTORY
# =============================================================================


# =============================================================================
# SPECIALIZED TEMPLATES
# =============================================================================

# -----------------------------------------------------------------------------
# DETAILED - Includes reasoning steps and detailed evaluation
# -----------------------------------------------------------------------------
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



# -----------------------------------------------------------------------------
# V0 - Basic similarity rating (includes problem statement)
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# V0.1 - Basic similarity rating (no problem statement)
# -----------------------------------------------------------------------------
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

PROMPT_TEMPLATE_V0_1_QWEN = """TASK DESCRIPTION
Rate the two math problem solutions (one reference, one candidate) in terms of their similarity.  Return a real value between 0-1 with 3 decimals.

SCOPE OF COMPARISON
- Evaluate only the SOLUTION content. Ignore any text that reproduce the problem statement or outputs generic headings/boilerplate (e.g., "\\assistant", "assistant", “problem:").
- Do not give credit for tokens/phrases that appear that is not relevant to the solution (e.g. "assistant"); similarity must come from solution reasoning/derivations/explanations and final answer.


INPUTS
- Reference solution:
{REFERENCE_SOLUTION}
- Candidate solution:
{CANDIDATE_SOLUTION}
OUTPUT FORMAT (must follow exactly)
REWARD: <number between 0 and 1 with 3 decimals>
""".strip()


# -----------------------------------------------------------------------------
# V1 - V0, enhanced with evaluation criteria (includes problem statement)
# -----------------------------------------------------------------------------
PROMPT_TEMPLATE_V1 = """
Rate the two math problem solutions (one reference, one candidate) in terms of their similarity. Return a real value between 0-1 with 3 decimals.

EVALUATION CRITERIA:
1. Mathematical correctness - Are candidate solution mathematically sound as the reference solution?
2. Solution approach - Do candidate solution use similar methods or reasoning as the reference solution? 
3. Final answer - Do candidate solution arrive at the same answer (enclosed in  "\\boxed{{}}") as the reference solution?
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


# -----------------------------------------------------------------------------
# V1.1 - Enhanced with evaluation criteria (no problem statement)
# -----------------------------------------------------------------------------
PROMPT_TEMPLATE_V1_1 = """TASK DESCRIPTION
Rate the two math problem solutions (one reference, one candidate) in terms of their similarity. Return a real value between 0-1 with 3 decimals.


EVALUATION CRITERIA
1. Mathematical correctness - Are candidate solution mathematically sound as the reference solution?
2. Solution approach - Do candidate solution use similar methods or reasoning as the reference solution? 
3. Final answer - Do candidate solution arrive at the same answer (enclosed in  "\\boxed{{}}") as the reference solution?
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

# -----------------------------------------------------------------------------
# V1.2 - Enhanced criteria with boxed answer focus
# -----------------------------------------------------------------------------
PROMPT_TEMPLATE_V1_2 = """TASK DESCRIPTION
Rate the two math problem solutions (one reference, one candidate) in terms of their similarity. Return a real value between 0-1 with 3 decimals.


EVALUATION CRITERIA
1. Mathematical correctness - Is the candidate solution mathematically sound as the reference solution?
2. Solution approach - Do candidate solution use similar methods or reasoning as the reference solution? 
3. Final answer - Do candidate solution arrive at the same answer (enclosed in  "\\boxed{{}}") as the reference solution?
4. Overall clarity - Are the reasoning and solution steps correct, sonsistent, and logically sound as the reference solution?
5. Overall similarity - Do the solutions exhibit high lexical/token/semantic overlap as the reference solution?


INPUTS
- Reference solution:
{REFERENCE_SOLUTION}

- Candidate solution:
{CANDIDATE_SOLUTION}

OUTPUT FORMAT (must follow exactly)
Output ONLY one line:
REWARD: <real value between 0 and 1 with 3 decimals>
""".strip()


# -----------------------------------------------------------------------------
# V1.3 - Enhanced criteria with length conformity
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# V1.4 - Enhanced criteria with grounding on lexical metrics
# updated: changed 3. final answer from conlusion to answer (enclosed in  "\\boxed{{}}")
# -----------------------------------------------------------------------------
PROMPT_TEMPLATE_V1_4 = """Rate the two math problem solutions (one reference, one candidate) in terms of their similarity. Return a real value between 0-1 with 3 decimals.

EVALUATION CRITERIA
1. Mathematical correctness - Are the candidate solution’s steps mathematically sound as the reference solution?
2. Solution approach - Do the candidate solution use similar methods or reasoning as the reference solution?
3. Final answer - Does the candidate solution arrive at the same answer (enclosed in  "\\boxed{{}}") as the reference solution?
4. Overall clarity - Are the reasoning and solution steps correct, consistent, and logically sound as the reference solution?
5. Lexical overlap - Anchor this score to the AUXILIARY METRICS (use max(LEXICAL_TOKEN_OVERLAP, LEXICAL_LCS_RATIO) as the primary score).
6. Length proximity - Anchor this score to the AUXILIARY METRICS (use LENGTH_RATIO as the primary score).

INPUTS
- Reference solution:
{REFERENCE_SOLUTION}

- Candidate solution:
{CANDIDATE_SOLUTION}

AUXILIARY METRICS (for criteria 5–6; solution text only)
- LEXICAL_TOKEN_OVERLAP: {LEXICAL_TOKEN_OVERLAP}    # [0,1]
- LEXICAL_LCS_RATIO:     {LEXICAL_LCS_RATIO}        # [0,1] (e.g., LCS_len / |GT_tokens|)
- LENGTH_RATIO:          {LENGTH_RATIO}             # |cand_tokens| / |GT_tokens|

OUTPUT FORMAT (must follow exactly)
Output ONLY one line:
REWARD: <real value between 0 and 1 with 3 decimals>
""".strip()


# -----------------------------------------------------------------------------
# V1.5 - Enhanced criteria with grounding on lexical metrics with subscore.
# updated: changed 3. final answer from conlusion to answer (enclosed in  "\\boxed{{}}")
# -----------------------------------------------------------------------------
PROMPT_TEMPLATE_V1_5 = """Rate the two math problem solutions (one reference, one candidate) in terms of their similarity. Evaluate the similarity of the candidate solution to the reference solution on the following criteria. 
Output sub-scores for each criterion as a numerical value between 0-100. Finally, consider sub-scores and output the overall reward as a numerical value between 0-100.

EVALUATION CRITERIA
1. Mathematical correctness - Are the candidate solution’s steps mathematically sound as the reference solution?
2. Solution approach - Do the candidate solution use similar methods or reasoning as the reference solution?
3. Final answer - Does the candidate solution arrive at the same answer (enclosed in  "\\boxed{{}}") as the reference solution?
4. Overall clarity - Are the reasoning and solution steps correct, consistent, and logically sound as the reference solution?
5. Lexical overlap - Anchor this score to the AUXILIARY METRICS (use average(LEXICAL_TOKEN_OVERLAP, LEXICAL_LCS_RATIO) as the primary score).
6. Length proximity - Anchor this score to the AUXILIARY METRICS (use LENGTH_RATIO as the primary score).

INPUTS
- Reference solution:
{REFERENCE_SOLUTION}

- Candidate solution:
{CANDIDATE_SOLUTION}

AUXILIARY METRICS (for criteria 5–6; solution text only)
- LEXICAL_TOKEN_OVERLAP: {LEXICAL_TOKEN_OVERLAP}    # [0,1]
- LEXICAL_LCS_RATIO:     {LEXICAL_LCS_RATIO}        # [0,1] (e.g., LCS_len / |GT_tokens|)
- LENGTH_RATIO:          {LENGTH_RATIO}             # |cand_tokens| / |GT_tokens|

OUTPUT FORMAT (must follow exactly)
MATHEMATICAL_CORRECTNESS: <number between 0 and 100>
SOLUTION_APPROACH:  <number between 0 and 100>
FINAL_ANSWER: <number between 0 and 100>
OVERALL_CLARITY:  <number between 0 and 100>
LEXICAL_OVERLAP:  <number between 0 and 100>
LENGTH_PROXIMITY:  <number between 0 and 100>
FINAL REWARD: <number between 0 and 100>
""".strip()


# -----------------------------------------------------------------------------
# V1.6 - Enhanced criteria with grounding on lexical metrics with subscore with caps.
# -----------------------------------------------------------------------------
PROMPT_TEMPLATE_V1_6 = """Rate the two math problem solutions (one reference, one candidate) in terms of their similarity. Evaluate the similarity of the candidate solution to the reference solution on the following criteria. 
Output sub-scores for each criterion as a real value between 0-1 with 3 decimals. Finally, output the overall reward as a real value between 0-1 with 3 decimals.

EVALUATION CRITERIA
1. Mathematical correctness - Are the candidate solution’s steps mathematically sound as the reference solution?
2. Solution approach - Do the candidate solution use similar methods or reasoning as the reference solution?
3. Final answer - Does the candidate solution arrive at the same answer (enclosed in  "\\boxed{{}}") as the reference solution?
4. Overall clarity - Are the reasoning and solution steps correct, consistent, and logically sound as the reference solution?
5. Lexical overlap - Anchor this score to the AUXILIARY METRICS (LEXICAL_TOKEN_OVERLAP, LEXICAL_LCS_RATIO).
6. Length proximity - Anchor this score to the AUXILIARY METRICS (LENGTH_RATIO).

INPUTS
- Reference solution:
{REFERENCE_SOLUTION}

- Candidate solution:
{CANDIDATE_SOLUTION}

AUXILIARY METRICS (for criteria 5–6; solution text only)
- LEXICAL_TOKEN_OVERLAP: {LEXICAL_TOKEN_OVERLAP}    # [0,1]
- LEXICAL_LCS_RATIO:     {LEXICAL_LCS_RATIO}        # [0,1] (e.g., LCS_len / |GT_tokens|)
- LENGTH_RATIO:          {LENGTH_RATIO}             # |cand_tokens| / |GT_tokens|

GLOBAL CAPS (apply AFTER summing)
- If two or more of the criteria (1, 2, 3, 4) have low score, the final reward will be capped at 50. The maximum reward you can return is 50.

OUTPUT FORMAT (must follow exactly)
MATHEMATICAL_CORRECTNESS: <real value between 0 and 1 with 3 decimals>
SOLUTION_APPROACH: <real value between 0 and 1 with 3 decimals>
FINAL_ANSWER: <real value between 0 and 1 with 3 decimals>
OVERALL_CLARITY: <real value between 0 and 1 with 3 decimals>
LEXICAL_OVERLAP: <real value between 0 and 1 with 3 decimals>
LENGTH_PROXIMITY: <real value between 0 and 1 with 3 decimals>
CAP APPLIED: <yes|no>
REWARD: <real value between 0 and 1 with 3 decimals>
""".strip()





# -----------------------------------------------------------------------------
# V2 - Comprehensive evaluation with surface & semantic resemblance
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# V2.1 - Same as V2 but with explicit "Output ONLY one line" instruction
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# V3 - Scoring with hard disqualifiers and additive scoring
# -----------------------------------------------------------------------------
PROMPT_TEMPLATE_V3 = """TASK
Rate the similarity between two math problem solutions (one GROUND TRUTH, one CANDIDATE). Return a single numeric score in [0,1] with 3 decimals.

HARD DISQUALIFIER (apply BEFORE scoring; if true, output 0.000):
A) The candidate merely restates or paraphrases the problem statement, or provides no substantive solution steps (i.e., not a solution).
If A is true → REWARD: 0.000

SCOPE OF COMPARISON
- Evaluate only the SOLUTION content. Ignore any text that reproduces the problem statement or generic headings/boilerplate (e.g., “Problem:”, “### Problem”, restated prompt).
- Do not give credit for tokens/phrases that appear in the problem statement; similarity must come from solution reasoning/derivations/explanations and final answer.

SCORING (apply ONLY if no disqualifier triggered)
Compute an additive score ∈ [0, 1.0] using the five criteria below; each criterion is worth 0.20. Round the final result to 3 decimals.

1) Mathematical correctness (0.20)
   - Are the candidate’s steps mathematically valid and consistent with the GROUND TRUTH’s reasoning (beyond just the final answer)?

2) Solution approach similarity (0.20)
   - Does the candidate use comparable methods/transformations to the GROUND TRUTH (e.g., same identities, substitutions, case structure, combinatorial arguments)?

3) Lexical/token overlap of solution content (0.20)
   - Consider only solution text (exclude problem text/boilerplate). Higher score for close phrase/term/token overlap and similar ordering.

4) Length similarity (0.20)
   - Compare solution word counts (exclude problem text/boilerplate). Full credit if within ±15% of GROUND TRUTH; otherwise decrease proportionally with distance.

5) Final answer match (0.20)
   - If the candidate presents a clear final answer that exactly matches the GROUND TRUTH (e.g., same value in \\boxed{{}} when present): full credit (0.20).
   - If a clear final answer is present but does NOT match: 0.00 for this criterion.
   - If no clear final answer is presented: 0.00 for this criterion.

FINAL OUTPUT FORMAT (must follow exactly; no extra words, no reasoning)
REWARD: <number between 0 and 1 with 3 decimals>

INPUTS
- GROUND TRUTH solution:
{REFERENCE_SOLUTION}

- CANDIDATE solution:
{CANDIDATE_SOLUTION}

""".strip()


# -----------------------------------------------------------------------------
# V3.1 - Introduce Rubric for more specific scoring.
# -----------------------------------------------------------------------------
PROMPT_TEMPLATE_V3_1 = """
TASK
Rate the similarity between two math problem solutions (one GROUND TRUTH, one CANDIDATE). Return a single numeric score in [0,1] with 3 decimals.

HARD DISQUALIFIER (apply BEFORE scoring; if true, output 0.000):
A) The candidate merely restates or paraphrases the problem statement, or provides no substantive solution steps (i.e., not a solution).
If A is true → REWARD: 0.000

SCOPE OF COMPARISON
- Evaluate only the SOLUTION content (exclude PROBLEM text and generic headings/boilerplate like “Problem:”, “### Problem”).
- Do not give credit for tokens/phrases that appear in the PROBLEM; similarity must come from solution reasoning/derivations/explanations and final answer.

SCORING (use ALL five criteria; each contributes 0.20. Round final to 3 decimals.)
1) Mathematical correctness (0.20)
   0.00: no valid steps / nonsense
   0.05: some valid manipulations but major errors dominate
   0.10: partial correctness (roughly half steps valid)
   0.15: largely correct with minor slips
   0.20: fully correct and consistent

2) Solution approach similarity (0.20)
   0.00: different method entirely / irrelevant steps
   0.05: shares a small sub-step but overall different
   0.10: some overlap in method (e.g., same identity used once)
   0.15: broadly similar pipeline (same key transformations/case structure)
   0.20: closely mirrors the ground-truth approach

3) Lexical/token overlap of solution content (exclude PROBLEM) (0.20)
   Guide by qualitative token/phrase overlap and ordering (of solution text only):
   0.00: negligible (<10%) or mostly copied from PROBLEM
   0.05: low (~10–30%)
   0.10: moderate (~30–60%)
   0.15: high (~60–80%)
   0.20: very high (≈80%+), yet not copied from PROBLEM

4) Length similarity (solution text only) (0.20)
   Compare word counts after excluding PROBLEM/boilerplate:
   0.20: within ±15%
   0.15: within ±25%
   0.10: within ±40%
   0.05: within ±60%
   0.00: farther than ±60%

5) Final answer match (0.20)
   0.20: clear final answer exactly matches the ground truth (e.g., same value in \\boxed{{}} when present)
   0.05: clear final answer present but does NOT match
   0.00: no clear final answer

FINAL OUTPUT FORMAT (must follow exactly; no extra words, no reasoning)
REWARD: <number between 0 and 1 with 3 decimals>

INPUTS
- PROBLEM:
{PROBLEM}

- GROUND TRUTH solution:
{REFERENCE_SOLUTION}

- CANDIDATE solution:
{CANDIDATE_SOLUTION}
""".strip()


# -----------------------------------------------------------------------------
# V3.2 - Modified HARD DISQUALIFIER to avoid false HARD DISQUALIFIER.
# -----------------------------------------------------------------------------
PROMPT_TEMPLATE_V3_2 = """
TASK
Rate the similarity between two math problem solutions (one GROUND TRUTH, one CANDIDATE). Return a single numeric score in [0,1] with 3 decimals.

HARD DISQUALIFIER (apply BEFORE scoring; if true, output 0.000)
A) The candidate provides no substantive solution steps (i.e., only repeats the PROBLEM or only states an answer with no computation/derivation).
Notes:
- A brief restatement of the PROBLEM to set context is acceptable and should NOT trigger this disqualifier.
- Short code snippets that compute the result (e.g., using a modulus or arithmetic) count as valid solution steps.

SCOPE OF COMPARISON
- Evaluate only the SOLUTION content (exclude PROBLEM text and generic headings/boilerplate like “Problem:”, “### Problem”).
- Treat code blocks as valid reasoning for correctness/approach.

SCORING (apply ONLY if no disqualifier triggered)
Compute an additive score ∈ [0, 1.0] using the five criteria below; each criterion is worth 0.20. Round the final result to 3 decimals.

1) Mathematical correctness (0.20)
   - Are the candidate’s steps mathematically valid and consistent with the GROUND TRUTH’s reasoning (beyond just the final answer)?

2) Solution approach similarity (0.20)
   - Does the candidate use comparable methods/transformations to the GROUND TRUTH (e.g., same identities, substitutions, case structure, combinatorial arguments)?

3) Lexical/token overlap of solution content (0.20)
   - Consider only solution text (exclude problem text/boilerplate). Higher score for close phrase/term/token overlap and similar ordering.

4) Length similarity (0.20)
   - Compare solution word counts (exclude problem text/boilerplate). Full credit if within ±15% of GROUND TRUTH; otherwise decrease proportionally with distance.

5) Final answer match (0.20)
   - If the candidate presents a clear final answer that exactly matches the GROUND TRUTH (e.g., same value in \\boxed{{}} when present): full credit (0.20).
   - If a clear final answer is present but does NOT match: 0.00 for this criterion.
   - If no clear final answer is presented: 0.00 for this criterion.

FINAL OUTPUT FORMAT (must follow exactly; no extra words, no reasoning)
REWARD: <number between 0 and 1 with 3 decimals>

INPUTS
- GROUND TRUTH solution:
{REFERENCE_SOLUTION}

- CANDIDATE solution:
{CANDIDATE_SOLUTION}

""".strip()


# -----------------------------------------------------------------------------
# V3.3 - Introduce Rubric for more specific scoring.
# -----------------------------------------------------------------------------
PROMPT_TEMPLATE_V3_3 = """
TASK
Rate the similarity between two math problem solutions (one GROUND TRUTH, one CANDIDATE). Return a single numeric score in [0,1] with 3 decimals.

HARD DISQUALIFIER (apply BEFORE scoring; if true, output 0.000)
A) The candidate provides no substantive solution steps (i.e., only repeats the PROBLEM or only states an answer with no computation/derivation).
Notes:
- A brief restatement of the PROBLEM to set context is acceptable and should NOT trigger this disqualifier.
- Short code snippets that compute the result (e.g., using a modulus or arithmetic) count as valid solution steps.

SCOPE OF COMPARISON
- Evaluate only the SOLUTION content (exclude PROBLEM text and generic headings/boilerplate like “Problem:”, “### Problem”).
- Treat code blocks as valid reasoning for correctness/approach.

SCORING (use ALL four criteria; each contributes 0.25. Choose the band that best fits. Round final to 3 decimals.)

1) Solution approach similarity (0.25)
   0.00: different method entirely / irrelevant steps
   0.05: small overlap
   0.10: some overlapping techniques
   0.15: broadly similar pipeline (same key transformations/case structure)
   0.25: closely mirrors the ground-truth approach
   (Programmatic computation that mirrors the same math is acceptable.)

2) Lexical/token overlap of solution content (0.25)
   Judge by phrase/term/token overlap and ordering of the **solution text only** (discount repetition):
   0.00: negligible overlap or mostly copied from PROBLEM
   0.05: low (~10–25%)
   0.10: moderate (~25–50%)
   0.15: high (~50–75%)
   0.25: very high (≈75%+)

3) Length similarity (solution text only) (0.25)
   Compare word counts after excluding PROBLEM/boilerplate; discount repetition.
   0.25: within ±15%
   0.15: within ±25%
   0.10: within ±40%
   0.05: within ±60%
   0.00: farther than ±60%
4) Final answer match (0.25)
   0.25: clear final answer exactly matches the ground truth (e.g., same value in \\boxed{{}} when present)
   0.10: numerically equivalent but different formatting (e.g., 2/4 vs 1/2); or obviously minor format difference
   0.05: clear final answer present but does NOT match
   0.00: no clear final answer

GLOBAL CAPS (apply AFTER summing the four criteria)
- If you assigned **0.00 or 0.05** to criterion (2) Lexical/Token Overlap, cap the TOTAL at **0.50**.
- If you assigned **0.00 or 0.05** to criterion (4) Length Similarity, cap the TOTAL at **0.50**.
(If both apply, use the lower cap.)


FINAL OUTPUT FORMAT (must follow exactly; no extra words, no reasoning)
REWARD: <number between 0 and 1 with 3 decimals>

INPUTS
- PROBLEM:
{PROBLEM}

- GROUND TRUTH solution:
{REFERENCE_SOLUTION}

- CANDIDATE solution:
{CANDIDATE_SOLUTION}

""".strip()


# -----------------------------------------------------------------------------
# V3.4 - Enhanced with additional output format. Added sub-score output.
#      - Changed the scoring from [0,1] to [0,100].
#      - Added additional description in the instruction header (you are an expert mathematician)
# -----------------------------------------------------------------------------
PROMPT_TEMPLATE_V3_4 = """
TASK
You are an strict and expert mathematician evaluating the similarity between two solutions to a math problem.
Rate the similarity between two math problem solutions (one GROUND TRUTH, one CANDIDATE). Return a single numeric score in [0,100]

DISQUALIFIER (apply BEFORE scoring)
A) Candidate solution provides no solution steps or no answer (i.e., only repeats the PROBLEM or only states an answer with no computation/derivation).
Notes:
- A brief restatement of the PROBLEM to set context is acceptable and should NOT trigger this disqualifier.
- Short code snippets that compute the result (e.g., using a modulus or arithmetic) count as valid solution steps.

If candidate solution output the answer correctly, it should not trigger this disqualifier. If requirements A) is met, return 0.

SCOPE OF COMPARISON
- Evaluate only the SOLUTION content (exclude PROBLEM text and generic headings/boilerplate like “Problem:”, “### Problem”).
- Treat code blocks as valid reasoning for correctness/approach.

SCORING (use ALL four criteria; each contributes at maximum 25 points. Choose carefully the band that best fits.)

1) Solution approach similarity (25)
   0: different method entirely / irrelevant steps
   5: small overlap
   10: some overlapping techniques
   15: broadly similar pipeline (same key transformations/case structure)
   25: closely mirrors the ground-truth approach
   (Programmatic computation that mirrors the same math is acceptable.)

2) Lexical/token overlap of solution content (25)
   Judge by phrase/term/token overlap (discount repetition):
   25: very high overlap (≈75%+)
   15: high overlap  (~50–75%)
   10: moderate overlap (~25–50%)
   5: low overlap (~10–25%)
   0: negligible overlap


3) Length similarity (solution text only) (25)
   Compare word counts after excluding PROBLEM/boilerplate; discount repetition.
   25: within ±15%
   15: within ±25%
   10: within ±40%
   5: within ±60%
   0: farther than ±60%

4) Final answer match (25)
   25: clear final answer exactly matches the ground truth (e.g., same value in \\boxed{{}} when present)
   10: numerically equivalent but different formatting (e.g., 2/4 vs 1/2); or obviously minor format difference
   5: clear final answer present but does NOT match
   0: no clear final answer

GLOBAL CAPS (apply AFTER summing the four criteria).
- If you assigned **0, 5, or 10** to criterion (1), (2), or (3), the total reward will be capped at 50. The maximum reward you can return is 50.


FINAL OUTPUT FORMAT (must follow exactly.)
APPROACH: <one of 0|5|10|15|25>
LEXICAL: <one of 0|5|10|15|25>
LENGTH:  <one of 0|5|10|15|25>
ANSWER:  <one of 0|5|10|25>
CAP APPLIED: <yes|no>
REWARD: <number between 0 and 100>

INPUTS
- PROBLEM:
{PROBLEM}

- GROUND TRUTH solution:
{REFERENCE_SOLUTION}

- CANDIDATE solution:
{CANDIDATE_SOLUTION}

""".strip()

# -----------------------------------------------------------------------------
# V3.5 - Removed sub-score output for inference efficiency.
# -----------------------------------------------------------------------------
PROMPT_TEMPLATE_V3_5 = """
TASK
You are an strict and expert mathematician evaluating the similarity between two solutions to a math problem.
Rate the similarity between two math problem solutions (one GROUND TRUTH, one CANDIDATE). Return a single numeric score in [0,100]

DISQUALIFIER (apply BEFORE scoring)
A) Candidate solution provides no solution steps or no answer (i.e., only repeats the PROBLEM or only states an answer with no computation/derivation).
Notes:
- A brief restatement of the PROBLEM to set context is acceptable and should NOT trigger this disqualifier.
- Short code snippets that compute the result (e.g., using a modulus or arithmetic) count as valid solution steps.

If candidate solution output the answer correctly, it should not trigger this disqualifier. If requirements A) is met, return 0.

SCOPE OF COMPARISON
- Evaluate only the SOLUTION content (exclude PROBLEM text and generic headings/boilerplate like “Problem:”, “### Problem”).
- Treat code blocks as valid reasoning for correctness/approach.

SCORING (use ALL four criteria; each contributes at maximum 25 points. Choose carefully the band that best fits.)

1) Solution approach similarity (25)
   0: different method entirely / irrelevant steps
   5: small overlap
   10: some overlapping techniques
   15: broadly similar pipeline (same key transformations/case structure)
   25: closely mirrors the ground-truth approach
   (Programmatic computation that mirrors the same math is acceptable.)

2) Lexical/token overlap of solution content (25)
   Judge by phrase/term/token overlap (discount repetition):
   25: very high overlap (≈75%+)
   15: high overlap  (~50–75%)
   10: moderate overlap (~25–50%)
   5: low overlap (~10–25%)
   0: negligible overlap


3) Length similarity (solution text only) (25)
   Compare word counts after excluding PROBLEM/boilerplate; discount repetition.
   25: within ±15%
   15: within ±25%
   10: within ±40%
   5: within ±60%
   0: farther than ±60%

4) Final answer match (25)
   25: clear final answer exactly matches the ground truth (e.g., same value in \\boxed{{}} when present)
   10: numerically equivalent but different formatting (e.g., 2/4 vs 1/2); or obviously minor format difference
   5: clear final answer present but does NOT match
   0: no clear final answer

GLOBAL CAPS (apply AFTER summing the four criteria).
- If you assigned **0, 5, or 10** to criterion (1), (2), or (3), the total reward will be capped at 50. The maximum reward you can return is 50.


FINAL OUTPUT FORMAT (must follow exactly.)
REWARD: <number between 0 and 100>

INPUTS
- PROBLEM:
{PROBLEM}

- GROUND TRUTH solution:
{REFERENCE_SOLUTION}

- CANDIDATE solution:
{CANDIDATE_SOLUTION}

""".strip()

# -----------------------------------------------------------------------------
# V3.6 - Removed {PROBLEM} input for further inference efficiency.
# -----------------------------------------------------------------------------

PROMPT_TEMPLATE_V3_6 = """
TASK
You are an strict and expert mathematician evaluating the similarity between two solutions to a math problem.
Rate the similarity between two math problem solutions (one GROUND TRUTH, one CANDIDATE). Return a single numeric score in [0,100]

DISQUALIFIER (apply BEFORE scoring)
A) Candidate solution provides no solution steps or no answer (i.e., only repeats the PROBLEM or only states an answer with no computation/derivation).
Notes:
- A brief restatement of the PROBLEM to set context is acceptable and should NOT trigger this disqualifier.
- Short code snippets that compute the result (e.g., using a modulus or arithmetic) count as valid solution steps.

If candidate solution output the answer correctly, it should not trigger this disqualifier. If requirements A) is met, return 0.

SCOPE OF COMPARISON
- Evaluate only the SOLUTION content (exclude PROBLEM text and generic headings/boilerplate like “Problem:”, “### Problem”).
- Treat code blocks as valid reasoning for correctness/approach.

SCORING (use ALL four criteria; each contributes at maximum 25 points. Choose carefully the band that best fits.)

1) Solution approach similarity (25)
   0: different method entirely / irrelevant steps
   5: small overlap
   10: some overlapping techniques
   15: broadly similar pipeline (same key transformations/case structure)
   25: closely mirrors the ground-truth approach
   (Programmatic computation that mirrors the same math is acceptable.)

2) Lexical/token overlap of solution content (25)
   Judge by phrase/term/token overlap (discount repetition):
   25: very high overlap (≈75%+)
   15: high overlap  (~50–75%)
   10: moderate overlap (~25–50%)
   5: low overlap (~10–25%)
   0: negligible overlap


3) Length similarity (solution text only) (25)
   Compare word counts after excluding PROBLEM/boilerplate; discount repetition.
   25: within ±15%
   15: within ±25%
   10: within ±40%
   5: within ±60%
   0: farther than ±60%

4) Final answer match (25)
   25: clear final answer exactly matches the ground truth (e.g., same value in \\boxed{{}} when present)
   10: numerically equivalent but different formatting (e.g., 2/4 vs 1/2); or obviously minor format difference
   5: clear final answer present but does NOT match
   0: no clear final answer

GLOBAL CAPS (apply AFTER summing the four criteria).
- If you assigned **0, 5, or 10** to criterion (1), (2), or (3), the total reward will be capped at 50. The maximum reward you can return is 50.


FINAL OUTPUT FORMAT (must follow exactly.)
REWARD: <number between 0 and 100>

INPUTS

- GROUND TRUTH solution:
{REFERENCE_SOLUTION}

- CANDIDATE solution:
{CANDIDATE_SOLUTION}

""".strip()

PROMPT_TEMPLATE_V3_7 = """
TASK
You are an strict and expert mathematician evaluating the similarity between two solutions to a math problem.
Rate the similarity between two math problem solutions (one GROUND TRUTH, one CANDIDATE). Return a single numeric score in [0,100]

DISQUALIFIER (apply BEFORE scoring)
A) Candidate solution provides no solution steps or no answer (i.e., only repeats the PROBLEM or only states an answer with no computation/derivation).
Notes:
- A brief restatement of the PROBLEM to set context is acceptable and should NOT trigger this disqualifier.
- Short code snippets that compute the result (e.g., using a modulus or arithmetic) count as valid solution steps.

If candidate solution output the answer correctly, it should not trigger this disqualifier. If requirements A) is met, return 0.

SCOPE OF COMPARISON
- Evaluate only the SOLUTION content (exclude PROBLEM text and generic headings/boilerplate like “Problem:”, “### Problem”).
- Treat code blocks as valid reasoning for correctness/approach.

SCORING (use ALL five criteria; each contributes at maximum 20 points. Choose carefully the band that best fits.)

1) Solution approach similarity (20)
   0: different method entirely / irrelevant steps
   5: small overlap
   10: some overlapping techniques
   15: broadly similar pipeline (same key transformations/case structure)
   20: closely mirrors the ground-truth approach
   (Programmatic computation that mirrors the same math is acceptable.)

2) Lexical/token overlap of solution content (20)
   Judge by phrase/term/token overlap (discount repetition):
   20: very high overlap (≈75%+)
   15: high overlap  (~50–75%)
   10: moderate overlap (~25–50%)
   5: low overlap (~10–25%)
   0: negligible overlap

3) Length similarity (solution text only) (20)
   Compare word counts after excluding PROBLEM/boilerplate; discount repetition.
   20: within ±15%
   15: within ±25%
   10: within ±40%
   5: within ±60%
   0: farther than ±60%

4) Final answer match (20)
   20: clear final answer exactly matches the ground truth (e.g., same value in \\boxed{{}} when present)
   10: numerically equivalent but different formatting (e.g., 2/4 vs 1/2); or obviously minor format difference
   5: clear final answer present but does NOT match
   0: no clear final answer
   
5) Overall mathematical soundness (20)
   Judge internal correctness and logical consistency of the solution steps (solution text only).
   20: Fully correct and consistent; valid transformations, no contradictions/gaps.
   15: Mostly correct; minor slips/omissions that don’t affect the argument.
   10: Mixed; one–two substantive issues or gaps, but some correct progress.
   5: Largely incorrect/weak; multiple invalid steps, misused identities, contradictions.
   0: No valid reasoning; restatement/nonsense/irrelevant steps.


GLOBAL CAPS (apply AFTER summing the four criteria).
- If you assigned **0, 5, or 10** to criterion (1), (2), or (3), the total reward will be capped at 50. The maximum reward you can return is 50.
- If final answer doens't match, the maximum reward will be capped at 40. The maximum reward you can return is 40.
- If final answer doesn't match and criterion (5) is low score (betwen 0 and 10), the maximum reward will be capped at 0. The maximum reward you can return is 0.


FINAL OUTPUT FORMAT (must follow exactly.)
REWARD: <number between 0 and 100>

INPUTS

- GROUND TRUTH solution:
{REFERENCE_SOLUTION}

- CANDIDATE solution:
{CANDIDATE_SOLUTION}
""".strip()


# -----------------------------------------------------------------------------
# V4 - Data contamination detection explicitly
# -----------------------------------------------------------------------------
PROMPT_TEMPLATE_V4 = """TASK
You are an strict and math expert. Your job is to detect data contamination given two problem solution. Give a contamination rating between two math solutions (GROUND TRUTH vs CANDIDATE). Return a single numeric score in [0,100] (integer or with 1–3 decimals). 

NOTE
Focus on whether the CANDIDATE mirrors the GROUND TRUTH’s approach/reasoning/phrasing/length/answer. 
Solution correctness is only part of the criteria.


DISQUALIFIER (apply BEFORE scoring)
A) Candidate solution provides no solution steps or no answer (i.e., only repeats the PROBLEM or only states an answer with no computation/derivation).
Notes:
- A brief restatement of the PROBLEM to set context is acceptable and should NOT trigger this disqualifier.
- Short code snippets that compute the result (e.g., using a modulus or arithmetic) count as valid solution steps.

If candidate solution output the answer correctly, it should not trigger this disqualifier. If requirements A) is met, return 0.

SCOPE
- Compare SOLUTION content only. Ignore PROBLEM text and boilerplate (“Problem:”, “### Problem”, restated prompt).
- Discount repetitive padding. Do not award overlap for repeated spans.


1) Approach alignment (mirror of GT key steps/order):

2) Lexical/phrase/token overlap:

3) Length proximity:

4) Final answer exact match:

GLOBAL CAPS (apply AFTER summing)
- If Approach shows little similarity, the maximum reward will be capped at 30. The maximum reward you can return is 30.
- If Lexical shows little overlap, the maximum reward will be capped at 30. The maximum reward you can return is 30.
- If Length is not within 50\% of the GROUND TRUTH, the maximum reward will be capped at 50. The maximum reward you can return is 50.


FINAL OUTPUT (no extra words)
APPROACH: <score>
LEXICAL: <score>
LENGTH: <score>
ANSWER: <score>
CAP APPLIED: <yes|no>
REWARD: <number between 0 and 100>

INPUTS
- Problem:
{PROBLEM}

- GROUND TRUTH solution:
{REFERENCE_SOLUTION}

- CANDIDATE solution:
{CANDIDATE_SOLUTION}
""".strip()



PROMPT_TEMPLATE_V4_1 = """TASK
You are an strict and math expert. Your job is to detect data contamination given two problem solution. Give a contamination rating between two math solutions (GROUND TRUTH vs CANDIDATE). Return a final numeric score in [0,1] (number between 0 and 1 with 3 decimals). 

NOTE
Focus on whether the CANDIDATE mirrors the GROUND TRUTH’s approach/reasoning/phrasing/length/answer. 
Solution correctness is only part of the criteria.

DISQUALIFIER (apply BEFORE scoring)
A) Candidate solution provides no solution steps or no answer (i.e., only repeats the PROBLEM or only states an answer with no computation/derivation).
Notes:
- A brief restatement of the PROBLEM to set context is acceptable and should NOT trigger this disqualifier.
- Short code snippets that compute the result (e.g., using a modulus or arithmetic) count as valid solution steps.
If candidate solution output the answer correctly, it should not trigger this disqualifier. If requirements A) is met, return 0.

SCOPE
- Compare SOLUTION content only. Ignore PROBLEM text and boilerplate (“Problem:”, “### Problem”, restated prompt).
- Discount repetitive padding. Do not award overlap for repeated spans.

AUXILIARY METRICS (you MUST use these to anchor scoring for LEXICAL and LENGTH)
- LEXICAL_TOKEN_OVERLAP: {LEXICAL_TOKEN_OVERLAP}   # fraction in [0,1]
- LEXICAL_LCS_RATIO:     {LEXICAL_LCS_RATIO}       # fraction in [0,1] (normalized LCS)
- LENGTH_RATIO:          {LENGTH_RATIO}            # candidate_length / ground_truth_length (solution text only)


1) Approach alignment (mirror of GT key steps/order):
   1.00: closely mirrors the ground-truth approach
   0.75: broadly similar pipeline (same key transformations/case structure)
   0.50: some overlapping techniques
   0.25: small overlap
   0: different method entirely / irrelevant steps
   (Programmatic computation that mirrors the same math is acceptable.)

2) Lexical/phrase/token overlap:
   (Average LEXICAL_TOKEN_OVERLAP and LEXICAL_LCS_RATIO as the lexical score)

3) Length proximity:
   (Use LENGTH_RATIO as length proximity score)

4) Final answer exact match:
   1.00: clear final answer exactly matches the ground truth (e.g., same value in \\boxed{{}} when present)
   0.50: numerically equivalent but different formatting (e.g., 2/4 vs 1/2); or obviously minor format difference
   0.25: clear final answer present but does NOT match
   0.00: no clear final answer

GLOBAL CAPS (apply AFTER summing)
- If Approach shows little similarity, the maximum reward will be capped at 30. The maximum reward you can return is 30.
- If Lexical shows little overlap (e.g., max lexical metric < 0.25), the maximum reward will be capped at 30. The maximum reward you can return is 30.
- If Length is not within 50% of the GROUND TRUTH (i.e., LENGTH_RATIO < 0.5 or > 1.5), the maximum reward will be capped at 50. The maximum reward you can return is 50.

FINAL OUTPUT (no extra words)
APPROACH: <score>
LEXICAL: <score>
LENGTH: <score>
ANSWER: <score>
CAP APPLIED: <yes|no>
REWARD: <number between 0 and 1 with 3 decimals>

INPUTS
- Problem:
{PROBLEM}

- GROUND TRUTH solution:
{REFERENCE_SOLUTION}

- CANDIDATE solution:
{CANDIDATE_SOLUTION}
""".strip()


# -----------------------------------------------------------------------------
# V5 - Data contamination detection with 
# -----------------------------------------------------------------------------
PROMPT_TEMPLATE_V5 = """
TASK
Judge the likelihood of DATA CONTAMINATION between two math solutions (GROUND TRUTH vs CANDIDATE).
Return a single numeric score in [0,100] (higher = more likely contaminated/memorized).

DEFINITION (what counts as contamination)
High likelihood when the CANDIDATE closely MIRRORS or PARAPHRASES the GROUND TRUTH in: reasoning pipeline, phrasing/tokens, common sequences, length, and reaches the same final answer.
Different-but-correct methods are NOT contamination.

SCOPE
- Compare SOLUTION content only (ignore PROBLEM text and boilerplate like “Problem:”, “### Problem”).
- Discount repetitive padding; do not award overlap for repeated spans.
- Code is valid as solution steps; code-only verification that does not mirror GT reasoning counts as a different approach.

AUXILIARY METRICS (use as anchors for lexical/length)
- LEXICAL_TOKEN_OVERLAP: {LEXICAL_TOKEN_OVERLAP}   # [0,1]
- LEXICAL_LCS_RATIO:     {LEXICAL_LCS_RATIO}       # [0,1]  (e.g., LCS_len / |GT_tokens|)
- LENGTH_RATIO:          {LENGTH_RATIO}            # |cand_tokens| / |GT_tokens|

EVALUATION CRITERIA
- Style similarity: Does the CANDIDATE mirror GT’s style/framing/phrasing?
- Approach alignment: Does the CANDIDATE mirror GT’s key steps/order/transformations?
- Lexical similarity: Use average(LEXICAL_TOKEN_OVERLAP, LEXICAL_LCS_RATIO); adjust for paraphrase/ordering; discount repetition.
- Length proximity: Use LENGTH_RATIO as primary signal (closer ≈ higher).
- Final answer (enclosed in "\\boxed{{}}") : Exact match slightly increases contamination likelihood; mismatch reduces it.

FINAL OUTPUT FORMAT (must follow exactly, no extra words)
CONTAMINATION REWARD: <number between 0 and 100>

INPUTS
- PROBLEM:
{PROBLEM}

- GROUND TRUTH solution:
{REFERENCE_SOLUTION}

- CANDIDATE solution:
{CANDIDATE_SOLUTION}

""".strip()

PROMPT_TEMPLATE_V5_1 = """
TASK
Judge the likelihood of DATA CONTAMINATION between two math solutions (GROUND TRUTH vs CANDIDATE).
Return a single real-valued score in [0,1] (higher = more likely contaminated/memorized).

DEFINITION (what counts as contamination)
High likelihood when the CANDIDATE closely MIRRORS or PARAPHRASES the GROUND TRUTH in: solution steps, reasoning pipeline, phrasing/tokens, common sequences, length, and reaches the same final answer.
Different-but-correct methods are NOT contamination.

SCOPE
- Compare SOLUTION content only (ignore PROBLEM text and boilerplate like “Problem:”, “### Problem”).
- Discount repetitive padding; do not award overlap for repeated spans.
- Code is valid as solution steps; code-only verification that does not mirror GT reasoning counts as a different approach.

AUXILIARY METRICS (use as anchors for lexical/length)
- LEXICAL_TOKEN_OVERLAP: {LEXICAL_TOKEN_OVERLAP}   # [0,1]
- LEXICAL_LCS_RATIO:     {LEXICAL_LCS_RATIO}       # [0,1]  (e.g., LCS_len / |GT_tokens|)
- LENGTH_RATIO:          {LENGTH_RATIO}            # |cand_tokens| / |GT_tokens|

EVALUATION CRITERIA
- Style similarity: Does the CANDIDATE mirror GT’s style/framing/phrasing?
- Approach alignment: Does the CANDIDATE mirror GT’s key steps/order/transformations?
- Lexical similarity: Use average(LEXICAL_TOKEN_OVERLAP, LEXICAL_LCS_RATIO).
- Length proximity: Use LENGTH_RATIO as primary signal (closer ≈ higher).
- Final answer (enclosed in "\\boxed{{}}") : Exact match slightly increases contamination likelihood; mismatch reduces it.

FINAL OUTPUT FORMAT (must follow exactly, no extra words)
CONTAMINATION REWARD:  <number between 0 and 1 with 3 decimals>

INPUTS

- GROUND TRUTH solution:
{REFERENCE_SOLUTION}

- CANDIDATE solution:
{CANDIDATE_SOLUTION}

""".strip()


PROMPT_TEMPLATE_V5_2 = """
TASK
Judge the likelihood of DATA CONTAMINATION between two math solutions (GROUND TRUTH vs CANDIDATE).
Return a single real-valued score in [0,1] (higher = more likely contaminated/memorized).

DEFINITION (what counts as contamination)
High likelihood when the CANDIDATE closely MIRRORS or PARAPHRASES the GROUND TRUTH in: solution steps,reasoning pipeline, phrasing/tokens, common sequences, length, and reaches the same final answer.
Different-but-correct methods are NOT contamination.

SCOPE
- Compare SOLUTION content only (ignore PROBLEM text and boilerplate like “Problem:”, “### Problem”).
- Discount repetitive padding; do not award overlap for repeated spans.
- Code is valid as solution steps; code-only verification that does not resemble GT reasoning counts as a different approach.

AUXILIARY METRICS (use as anchors for lexical/length)
- LEXICAL_TOKEN_OVERLAP: {LEXICAL_TOKEN_OVERLAP}   # [0,1]
- LEXICAL_LCS_RATIO:     {LEXICAL_LCS_RATIO}       # [0,1]  (e.g., LCS_len / |GT_tokens|)
- LENGTH_RATIO:          {LENGTH_RATIO}            # |cand_tokens| / |GT_tokens|

EVALUATION CRITERIA
- Style similarity: Does the CANDIDATE resembles GT’s style/framing/phrasing?
- Approach alignment: Does the CANDIDATE mirror GT’s key steps/order/transformations?
- Lexical token similarity: Use LEXICAL_TOKEN_OVERLAP.
- Lexical sequence similarity: Use LEXICAL_LCS_RATIO.
- Length proximity: Use LENGTH_RATIO as primary signal (closer ≈ higher).
- Final answer (enclosed in "\\boxed{{}}") : Exact match slightly increases contamination likelihood; mismatch reduces it.

FINAL OUTPUT FORMAT (must follow exactly, no extra words)
CONTAMINATION REWARD: <number between 0 and 1 with 3 decimals>

INPUTS

- GROUND TRUTH solution:
{REFERENCE_SOLUTION}

- CANDIDATE solution:
{CANDIDATE_SOLUTION}

""".strip()


# -----------------------------------------------------------------------------
# Refine DEFINITION and EVALUATION CRITERIA from V5_2.
# -----------------------------------------------------------------------------
PROMPT_TEMPLATE_V5_3 = """
TASK
Judge the likelihood of DATA CONTAMINATION between two math solutions (GROUND TRUTH vs CANDIDATE).
Return a single real-valued score in [0,1] (higher = more likely contaminated/memorized).

DEFINITION (what counts as contamination)
High likelihood when the CANDIDATE closely MIRRORS or PARAPHRASES the GROUND TRUTH in: solution steps, reasoning structure, phrasing/tokens, common sequences, length, and reaches the same final answer.
Different-but-correct approaches are NOT contamination, pay very close attention to the reasoning structure and steps.

SCOPE
- Compare SOLUTION content only (ignore PROBLEM text and boilerplate like “Problem:”, “### Problem”).
- Discount repetitive padding; do not award overlap for repeated spans.
- Code is valid as solution steps; code-only verification that does not resemble GT reasoning counts as a different approach.

AUXILIARY METRICS (use as anchors for lexical/length)
- LEXICAL_TOKEN_OVERLAP: {LEXICAL_TOKEN_OVERLAP}   # [0,1]
- LEXICAL_LCS_RATIO:     {LEXICAL_LCS_RATIO}       # [0,1]  (e.g., LCS_len / |GT_tokens|)
- LENGTH_RATIO:          {LENGTH_RATIO}            # |cand_tokens| / |GT_tokens|

EVALUATION CRITERIA
- Style similarity: Does the CANDIDATE closely resembles GT’s solution style, framing, phrasing?
- Approach alignment: Does the CANDIDATE closely mirror GT’s solutions structure, steps, and reasoning?
- Lexical token similarity: Use LEXICAL_TOKEN_OVERLAP.
- Lexical sequence similarity: Use LEXICAL_LCS_RATIO.
- Length proximity: Use LENGTH_RATIO as primary signal (closer ≈ higher).
- Final answer (enclosed in "\\boxed{{}}") : Exact match slightly increases contamination likelihood; mismatch reduces it.


FINAL OUTPUT FORMAT (must follow exactly, no extra words)
CONTAMINATION REWARD: <number between 0 and 1 with 3 decimals>

INPUTS

- GROUND TRUTH solution:
{REFERENCE_SOLUTION}

- CANDIDATE solution:
{CANDIDATE_SOLUTION}

""".strip()


PROMPT_TEMPLATE_V5_4 = """
TASK
Judge the likelihood of DATA CONTAMINATION between two math solutions (GROUND TRUTH vs CANDIDATE).
Return a single real-valued score in [0,1] (higher = more likely contaminated/memorized).

DEFINITION (what counts as contamination)
High likelihood when the CANDIDATE closely MIRRORS or PARAPHRASES the GROUND TRUTH in: solution steps, reasoning structure, phrasing/tokens, common sequences, length, and reaches the same final answer.
Different-but-correct approaches are NOT contamination, pay very close attention to the reasoning structure and steps.

SCOPE
- Compare SOLUTION content only (ignore PROBLEM text and boilerplate like “Problem:”, “### Problem”).
- Discount repetitive padding; do not award overlap for repeated spans.
- Code is valid as solution steps; code-only verification that does not resemble GT reasoning counts as a different approach.

AUXILIARY METRICS (use as anchors for lexical/length)
- LEXICAL_TOKEN_OVERLAP: {LEXICAL_TOKEN_OVERLAP}   # [0,1]
- LEXICAL_LCS_RATIO:     {LEXICAL_LCS_RATIO}       # [0,1]  (e.g., LCS_len / |GT_tokens|)
- LENGTH_RATIO:          {LENGTH_RATIO}            # |cand_tokens| / |GT_tokens|

EVALUATION CRITERIA
- Style similarity: Does the CANDIDATE closely resembles GT’s solution style, structure, phrasing?
- Lexical token similarity: Use LEXICAL_TOKEN_OVERLAP.
- Lexical sequence similarity: Use LEXICAL_LCS_RATIO.
- Length proximity: Use LENGTH_RATIO (closer ≈ higher).

Final reward is the average of the four evaluation criteria.

FINAL OUTPUT FORMAT (must follow exactly, no extra words)
CONTAMINATION REWARD: <number between 0 and 1 with 3 decimals>

INPUTS

- GROUND TRUTH solution:
{REFERENCE_SOLUTION}

- CANDIDATE solution:
{CANDIDATE_SOLUTION}

""".strip()


PROMPT_TEMPLATE_V5_5 = """
TASK
Judge the likelihood of DATA CONTAMINATION between two math solutions (GROUND TRUTH vs CANDIDATE).
Return a single real-valued score in [0,1] (higher = more likely contaminated/memorized).

DEFINITION (what counts as contamination)
High likelihood when the CANDIDATE closely MIRRORS or PARAPHRASES the GROUND TRUTH in: solution steps, reasoning structure, phrasing/tokens, common sequences, length, and reaches the same final answer.
Different-but-correct approaches are NOT contamination, pay very close attention to the reasoning structure and steps.

SCOPE
- Compare SOLUTION content only (ignore PROBLEM text and boilerplate like “Problem:”, “### Problem”).
- Discount repetitive padding; do not award overlap for repeated spans.
- Code is valid as solution steps; code-only verification that does not resemble GT reasoning counts as a different approach.

AUXILIARY METRICS (use as anchors for lexical/length)
- LEXICAL_TOKEN_OVERLAP: {LEXICAL_TOKEN_OVERLAP}   # [0,1]
- LEXICAL_LCS_RATIO:     {LEXICAL_LCS_RATIO}       # [0,1]  (e.g., LCS_len / |GT_tokens|)
- LENGTH_RATIO:          {LENGTH_RATIO}            # |cand_tokens| / |GT_tokens|

EVALUATION CRITERIA
- Lexical token similarity: Use LEXICAL_TOKEN_OVERLAP.
- Lexical sequence similarity: Use LEXICAL_LCS_RATIO.
- Length proximity: Use LENGTH_RATIO (closer ≈ higher).

Final contamination reward SHOULD BE the average of the three evaluation criteria.

FINAL OUTPUT FORMAT (must follow exactly, no extra words)
CONTAMINATION REWARD: <number between 0 and 1 with 3 decimals>

INPUTS

- GROUND TRUTH solution:
{REFERENCE_SOLUTION}

- CANDIDATE solution:
{CANDIDATE_SOLUTION}

""".strip()

PROMPT_TEMPLATE_V5_6 = """
TASK
Judge the likelihood of DATA CONTAMINATION between two math solutions (GROUND TRUTH vs CANDIDATE).
Return a single real-valued score in [0,1] (higher = more likely contaminated/memorized).

DEFINITION (what counts as contamination)
High likelihood when the CANDIDATE closely MIRRORS or PARAPHRASES the GROUND TRUTH in: solution steps, reasoning structure, phrasing/tokens, common sequences, length, and reaches the same final answer.
Different-but-correct approaches are NOT contamination, pay very close attention to the reasoning structure and steps.

SCOPE
- Compare SOLUTION content only (ignore PROBLEM text and boilerplate like “Problem:”, “### Problem”).
- Discount repetitive padding; do not award overlap for repeated spans.
- Code is valid as solution steps; code-only verification that does not resemble GT reasoning counts as a different approach.

AUXILIARY METRICS (use as anchors for lexical/length)
- LEXICAL_TOKEN_OVERLAP: {LEXICAL_TOKEN_OVERLAP}   # [0,1]
- LEXICAL_LCS_RATIO:     {LEXICAL_LCS_RATIO}       # [0,1]  (e.g., LCS_len / |GT_tokens|)

EVALUATION CRITERIA
- Lexical token similarity: Use LEXICAL_TOKEN_OVERLAP.
- Lexical sequence similarity: Use LEXICAL_LCS_RATIO.

Final contamination reward is the average of the two evaluation criteria.

FINAL OUTPUT FORMAT (must follow exactly, no extra words)
CONTAMINATION REWARD: <number between 0 and 1 with 3 decimals>

INPUTS

- GROUND TRUTH solution:
{REFERENCE_SOLUTION}

- CANDIDATE solution:
{CANDIDATE_SOLUTION}

""".strip()


PROMPT_TEMPLATE_V5_7 = """
TASK
Judge the likelihood of DATA CONTAMINATION between two math solutions (GROUND TRUTH vs CANDIDATE).
Return a single real-valued score in [0,1] (higher = more likely contaminated/memorized).

DEFINITION (what counts as contamination)
High likelihood when the CANDIDATE closely MIRRORS or PARAPHRASES the GROUND TRUTH in: solution steps, reasoning structure, phrasing/tokens, common sequences, length, and reaches the same final answer.
Different-but-correct approaches are NOT contamination, pay very close attention to the reasoning structure and steps.

SCOPE
- Compare SOLUTION content only (ignore PROBLEM text and boilerplate like “Problem:”, “### Problem”).
- Discount repetitive padding; do not award overlap for repeated spans.
- Code is valid as solution steps; code-only verification that does not resemble GT reasoning counts as a different approach.

AUXILIARY METRICS (use as anchors for EVALUATION CRITERIA)
- LEXICAL_TOKEN_OVERLAP: {LEXICAL_TOKEN_OVERLAP}   # [0,1]

EVALUATION CRITERIA
- Lexical token similarity: Use LEXICAL_TOKEN_OVERLAP.

Final reward should directly calculated from the average of the evaluation criteria.

FINAL OUTPUT FORMAT (must follow exactly, no extra words)
CONTAMINATION REWARD: <number between 0 and 1 with 3 decimals>

INPUTS

- GROUND TRUTH solution:
{REFERENCE_SOLUTION}

- CANDIDATE solution:
{CANDIDATE_SOLUTION}

""".strip()


PROMPT_TEMPLATE_V5_8 = """
TASK
Judge the likelihood of DATA CONTAMINATION between two math solutions (GROUND TRUTH vs CANDIDATE).
Return a single real-valued score in [0,1] (higher = more likely contaminated/memorized).

AUXILIARY METRICS (use as anchors for EVALUATION CRITERIA)
- LEXICAL_TOKEN_OVERLAP: {LEXICAL_TOKEN_OVERLAP}   # [0,1]

EVALUATION CRITERIA
- Lexical token similarity: Use LEXICAL_TOKEN_OVERLAP.

Final contamination reward SHOULD BE LEXICAL_TOKEN_OVERLAP: {LEXICAL_TOKEN_OVERLAP}, 

FINAL OUTPUT FORMAT (must follow exactly, no extra words)
CONTAMINATION REWARD: <number between 0 and 1 with 3 decimals>
""".strip()




# -----------------------------------------------------------------------------
# V6 - Data similarity. This is used to judge the similarity of two math distillation solutions.
# -----------------------------------------------------------------------------

PROMPT_TEMPLATE_V6_1 = """
TASK
Judge the similarity of two math solutions (GROUND TRUTH vs CANDIDATE).
Return a single real-valued score in [0,1] (higher = more similar).

SCOPE
- Compare SOLUTION content only (ignore PROBLEM text and boilerplate like “Problem:”, “### Problem”).
- Discount repetitive padding; do not award overlap for repeated spans.
- Code is valid as solution steps; code-only verification that does not resemble GT reasoning counts as a different approach.

AUXILIARY METRICS (use as anchors for lexical/length)
- LEXICAL_TOKEN_OVERLAP: {LEXICAL_TOKEN_OVERLAP}   # [0,1]
- LEXICAL_LCS_RATIO:     {LEXICAL_LCS_RATIO}       # [0,1]  (e.g., LCS_len / |GT_tokens|)
- LENGTH_RATIO:          {LENGTH_RATIO}            # |cand_tokens| / |GT_tokens|

EVALUATION CRITERIA
- Solution approach similarity: Does the CANDIDATE closely resembles GT’s solution style, reasoning, structure, phrasing? BE VERY STRICT ON THIS.
- Lexical token similarity: Use LEXICAL_TOKEN_OVERLAP.
- Lexical sequence similarity: Use LEXICAL_LCS_RATIO.
- Length proximity: Use LENGTH_RATIO.

Final similarity reward SHOULD BE the average of the evaluation criteria.

FINAL OUTPUT FORMAT (must follow exactly, no extra words)
SIMILARITY REWARD: <number between 0 and 1 with 3 decimals>

INPUTS

- GROUND TRUTH solution:
{REFERENCE_SOLUTION}

- CANDIDATE solution:
{CANDIDATE_SOLUTION}
""".strip()



PROMPT_TEMPLATE_V6_2 = """
TASK
Judge the similarity of two math solutions (GROUND TRUTH vs CANDIDATE).
Return a single real-valued score in [0,1] (higher = more similar).

SCOPE
- Compare SOLUTION content only (ignore PROBLEM text and boilerplate like “Problem:”, “### Problem”).
- Discount repetitive padding; do not award overlap for repeated spans.
- Code is valid as solution steps; code-only verification that does not resemble GT reasoning counts as a different approach.

AUXILIARY METRICS (use as anchors for lexical/length)
- LEXICAL_TOKEN_OVERLAP: {LEXICAL_TOKEN_OVERLAP}   # [0,1]
- LEXICAL_LCS_RATIO:     {LEXICAL_LCS_RATIO}       # [0,1]  (e.g., LCS_len / |GT_tokens|)
- LENGTH_RATIO:          {LENGTH_RATIO}            # |cand_tokens| / |GT_tokens|

EVALUATION CRITERIA
- Lexical token similarity: Use LEXICAL_TOKEN_OVERLAP.
- Lexical sequence similarity: Use LEXICAL_LCS_RATIO.
- Length proximity: Use LENGTH_RATIO.

Final similarity reward SHOULD BE the average of the evaluation criteria.

FINAL OUTPUT FORMAT (must follow exactly, no extra words)
SIMILARITY REWARD: <number between 0 and 1 with 3 decimals>

INPUTS

- GROUND TRUTH solution:
{REFERENCE_SOLUTION}

- CANDIDATE solution:
{CANDIDATE_SOLUTION}
""".strip()

PROMPT_TEMPLATE_V6_3 = """
TASK
Judge the similarity of two math solutions (GROUND TRUTH vs CANDIDATE).
Return a single real-valued score in [0,1] (higher = more similar).

SCOPE
- Compare SOLUTION content only (ignore PROBLEM text and boilerplate like “Problem:”, “### Problem”).
- Discount repetitive padding; do not award overlap for repeated spans.
- Code is valid as solution steps; code-only verification that does not resemble GT reasoning counts as a different approach.

AUXILIARY METRICS (use as anchors for lexical/length)
- LEXICAL_TOKEN_OVERLAP: {LEXICAL_TOKEN_OVERLAP}   # [0,1]
- LEXICAL_LCS_RATIO:     {LEXICAL_LCS_RATIO}       # [0,1]  (e.g., LCS_len / |GT_tokens|)

EVALUATION CRITERIA
- Lexical token similarity: Use LEXICAL_TOKEN_OVERLAP.
- Lexical sequence similarity: Use LEXICAL_LCS_RATIO.

Final similarity reward SHOULD BE the average of the evaluation criteria.

FINAL OUTPUT FORMAT (must follow exactly, no extra words)
SIMILARITY REWARD: <number between 0 and 1 with 3 decimals>

INPUTS

- GROUND TRUTH solution:
{REFERENCE_SOLUTION}

- CANDIDATE solution:
{CANDIDATE_SOLUTION}
""".strip()


PROMPT_TEMPLATE_V6_4= """
TASK
Judge the similarity of two math solutions (GROUND TRUTH vs CANDIDATE).
Return a single real-valued score in [0,1] (higher = more similar).

SCOPE
- Compare SOLUTION content only (ignore PROBLEM text and boilerplate like “Problem:”, “### Problem”).
- Discount repetitive padding; do not award overlap for repeated spans.
- Code is valid as solution steps; code-only verification that does not resemble GT reasoning counts as a different approach.

AUXILIARY METRICS (use as anchors for lexical/length)
- LEXICAL_TOKEN_OVERLAP: {LEXICAL_TOKEN_OVERLAP}   # [0,1]
- LEXICAL_LCS_RATIO:     {LEXICAL_LCS_RATIO}       # [0,1]  (e.g., LCS_len / |GT_tokens|)

EVALUATION CRITERIA
- Lexical token similarity: Use LEXICAL_TOKEN_OVERLAP.
- Lexical sequence similarity: Use LEXICAL_LCS_RATIO.

BONUS CRITERIA
- If solution approach is similar, consistent, and logically sound as the GROUND TRUTH, +0.2 bonus.
- If solution answer is the same as the GROUND TRUTH (enclosed in \\boxed{{}}), +0.1 bonus.

Final similarity reward SHOULD BE the average of the evaluation criteria, plus the bonus criteria.

GLOBAL CAPS (apply AFTER summing)
- The maximum reward you can return is 1.0.

FINAL OUTPUT FORMAT (must follow exactly, no extra words)
TOKEN SIMILARITY: LEXICAL_TOKEN_OVERLAP
SEQUENCE SIMILARITY: LEXICAL_LCS_RATIO
BONUS: <real value between 0 and 0.3 with 3 decimals>
CONTAMINATION REWARD: <real value between 0 and 1 with 3 decimals>

INPUTS

- GROUND TRUTH solution:
{REFERENCE_SOLUTION}

- CANDIDATE solution:
{CANDIDATE_SOLUTION}
""".strip()

PROMPT_TEMPLATE_V6_5= """
TASK
Judge the similarity of two math solutions (GROUND TRUTH vs CANDIDATE).
Return a single real-valued score in [0,1] (higher = more similar).

SCOPE
- Compare SOLUTION content only (ignore PROBLEM text and boilerplate like “Problem:”, “### Problem”).
- Discount repetitive padding; do not award overlap for repeated spans.
- Code is valid as solution steps; code-only verification that does not resemble GT reasoning counts as a different approach.

AUXILIARY METRICS (use as anchors for lexical/length)
- LEXICAL_TOKEN_OVERLAP: {LEXICAL_TOKEN_OVERLAP}   # [0,1]
- LEXICAL_LCS_RATIO:     {LEXICAL_LCS_RATIO}       # [0,1]  (e.g., LCS_len / |GT_tokens|)

EVALUATION CRITERIA
- Lexical token similarity: Use LEXICAL_TOKEN_OVERLAP.
- Lexical sequence similarity: Use LEXICAL_LCS_RATIO.

BONUS CRITERIA
- If solution approach is similar to the GROUND TRUTH, +0.2 bonus.
- If solution answer is the same as the GROUND TRUTH (enclosed in \\boxed{{}}), +0.1 bonus.

PENALTY CRITERIA
- If solution approach is different from the GROUND TRUTH, -0.2 penalty.

Final similarity reward SHOULD BE the average of the evaluation criteria, plus the bonus criteria, minus the penalty criteria.

GLOBAL CAPS (apply AFTER summing)
- The maximum reward you can return is 1.0.
- The minimum reward you can return is 0.0.

FINAL OUTPUT FORMAT (must follow exactly, no extra words)
TOKEN SIMILARITY: LEXICAL_TOKEN_OVERLAP
SEQUENCE SIMILARITY: LEXICAL_LCS_RATIO
BONUS: <real value between 0 and 0.3 with 3 decimals>
PENALTY: <real value between 0 and 0.2 with 3 decimals>
CONTAMINATION REWARD: <real value between 0 and 1 with 3 decimals>

INPUTS

- GROUND TRUTH solution:
{REFERENCE_SOLUTION}

- CANDIDATE solution:
{CANDIDATE_SOLUTION}
""".strip()




# =============================================================================
# TEMPLATE REGISTRY AND UTILITY FUNCTIONS
# =============================================================================

# -----------------------------------------------------------------------------
# Template name mapping - All available prompt templates
# -----------------------------------------------------------------------------
PROMPT_TEMPLATES = {
    "default": PROMPT_TEMPLATE_V1,
    "detailed": DETAILED_PROMPT_TEMPLATE,
    "v0": PROMPT_TEMPLATE_V0,
    "v0_1": PROMPT_TEMPLATE_V0_1,
    "v0_1_qwen": PROMPT_TEMPLATE_V0_1_QWEN,
    "v1": PROMPT_TEMPLATE_V1,
    "v1_1": PROMPT_TEMPLATE_V1_1,
    "v1_2": PROMPT_TEMPLATE_V1_2,
    "v1_3": PROMPT_TEMPLATE_V1_3,
    "v1_4": PROMPT_TEMPLATE_V1_4,
    "v1_5": PROMPT_TEMPLATE_V1_5,
    "v2": PROMPT_TEMPLATE_V2,
    "v2_1": PROMPT_TEMPLATE_V2_1,
    "v3": PROMPT_TEMPLATE_V3,
    "v3_1": PROMPT_TEMPLATE_V3_1,
    "v3_2": PROMPT_TEMPLATE_V3_2,
    "v3_3": PROMPT_TEMPLATE_V3_3,
    "v3_4": PROMPT_TEMPLATE_V3_4,
    "v3_5": PROMPT_TEMPLATE_V3_5,
    "v3_6": PROMPT_TEMPLATE_V3_6,
    "v3_7": PROMPT_TEMPLATE_V3_7,
    "v4": PROMPT_TEMPLATE_V4,
    "v4_1": PROMPT_TEMPLATE_V4_1,
    "v5": PROMPT_TEMPLATE_V5,
    "v5_1": PROMPT_TEMPLATE_V5_1,
    "v5_2": PROMPT_TEMPLATE_V5_2,
    "v5_3": PROMPT_TEMPLATE_V5_3,
    "v5_4": PROMPT_TEMPLATE_V5_4,
    "v5_5": PROMPT_TEMPLATE_V5_5,
    "v5_6": PROMPT_TEMPLATE_V5_6,
    "v5_7": PROMPT_TEMPLATE_V5_7,
    "v5_8": PROMPT_TEMPLATE_V5_8,
    "v6_1": PROMPT_TEMPLATE_V6_1,
    "v6_2": PROMPT_TEMPLATE_V6_2,
    "v6_3": PROMPT_TEMPLATE_V6_3,
    "v6_4": PROMPT_TEMPLATE_V6_4,
    "v6_5": PROMPT_TEMPLATE_V6_5
}


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

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


# =============================================================================
# TEMPLATE EVOLUTION SUMMARY
# =============================================================================
# V0–V0.1: Basic similarity scoring (with/without problem statement)
# V1–V1.3: Add evaluation criteria; focus on boxed answers; length conformity
# V2–V2.1: Comprehensive criteria; add explicit one-line output instruction
# V3: Add hard disqualifier and additive 5-criteria rubric in [0,1]
# V3.1: Introduce banded rubric for each criterion; include PROBLEM input
# V3.2: Refine disqualifier to reduce false triggers; accept brief restatement/code
# V3.3: Switch to four-criterion banded rubric with global caps; include PROBLEM
# V3.4: Change scale to [0,100]; add sub-scores and cap reporting; expert tone
# V3.5: Keep [0,100] but remove sub-score outputs for efficiency
# V3.6: Same as V3.5 but drop PROBLEM input for further efficiency
# DETAILED: Reason-through evaluation first; then output [0,1] score
# =============================================================================
