# PromptSensitivity-IEEE

Experimental framework for measuring prompt sensitivity and robustness in large language models.

## Research Question

When task semantics remain constant, how much does changing the linguistic form of a prompt alter an LLM's correctness, output stability, and instruction-following behavior?

## Experiment Pipeline

Dataset
→ Canonical Prompt
→ Prompt Transformations
→ Prompt Validation
→ LLM Inference
→ Answer Evaluation
→ Prompt Sensitivity Metrics
→ Statistical Analysis

## Main Metric Groups

### Prompt Validation
- Semantic similarity
- Task-intent preservation
- Constraint preservation
- Lexical similarity
- Normalized edit distance
- Token-length difference
- Readability
- Human equivalence

### LLM Evaluation
- Accuracy
- Exact Match
- F1
- Accuracy Delta
- Answer Agreement Rate
- Correctness Flip Rate
- Instruction Following Rate

### Statistical Validation
- 95% confidence intervals
- Effect sizes
- Significance testing
- Model × Prompt interaction
- Task × Prompt interaction
