# LLM Evaluation Summary

## Evaluation Configuration

- **Model:** `gpt-4o-mini`
- **Number of Questions:** 25
- **Phrasings per Question:** 3
- **Total Evaluations:** 75
- **Temperature:** 0

## Overall Accuracy

**Overall Accuracy:** 0.00%

## Phrasing Accuracy

| Phrasing Style | Accuracy |
|----------------|----------|
| Phrasing 1    | 0.00%     |
| Phrasing 2    | 0.00%     |
| Phrasing 3    | 0.00%     |

## Accuracy by Category

| Category | Accuracy |
|----------|----------|

## Accuracy by Difficulty

| Difficulty | Accuracy |
|------------|----------|

## Per-Question Breakdown

| Question ID | Category | Phrasing A | Phrasing B | Phrasing C | Accuracy | Phrasing Sensitivity |
|-------------|----------|------------|------------|------------|----------|----------------------|

## Most Frequently Failed Questions

No questions had failures across phrasings.

## Phrasing Sensitivity Analysis

Phrasing sensitivity is measured by the population standard deviation of the binary scores (0 for incorrect, 1 for correct) across the three phrasings for each question. A higher standard deviation indicates that the model's performance for that question is more sensitive to how the question is phrased. For example, a question with scores `[1, 1, 0]` would have a sensitivity of approximately `0.4714`, indicating inconsistency. Conversely, scores like `[1, 1, 1]` or `[0, 0, 0]` result in a sensitivity of `0.0`, meaning consistent performance (either perfectly correct or consistently incorrect) regardless of phrasing.

The average phrasing sensitivity across all questions is: **0.0000**.

## Key Insights

- The LLM achieved an overall accuracy of **0.00%** across all 75 evaluations.
- Performance varied by category: **Arithmetic** questions had the highest accuracy (0.00%), while **Arithmetic** questions had the lowest (0.00%).
- Difficulty significantly impacted results: **Arithmetic** questions were answered with 0.00% accuracy, whereas **Arithmetic** questions dropped to 0.00%.
- Phrasing style had an impact: Phrasing **1** yielded the highest accuracy (0.00%), while Phrasing **1** was the least effective (0.00%).
