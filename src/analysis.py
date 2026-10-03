import json
import math
from collections import defaultdict
from typing import List, Dict, Any

def calculate_overall_accuracy(evaluations: List[Dict[str, Any]]) -> float:
    """Calculates the overall accuracy across all evaluations."""
    if not evaluations:
        return 0.0
    correct_responses = sum(record["score"] for record in evaluations)
    return correct_responses / len(evaluations)

def calculate_per_question_results(evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Calculates per-question accuracy and phrasing sensitivity.
    """
    question_groups = defaultdict(lambda: defaultdict(list))
    for record in evaluations:
        question_groups[record["question_id"]]["records"].append(record)
        question_groups[record["question_id"]]["scores"].append(record["score"])
        question_groups[record["question_id"]]["category"] = record["category"]
        question_groups[record["question_id"]]["difficulty"] = record["difficulty"]

    per_question_results = []
    for q_id, data in question_groups.items():
        scores = data["scores"]
        accuracy = sum(scores) / len(scores) if scores else 0.0

        # Calculate population standard deviation for phrasing sensitivity
        phrasing_sensitivity = 0.0
        if len(scores) > 1:
            mean = sum(scores) / len(scores)
            variance = sum([(x - mean) ** 2 for x in scores]) / len(scores)
            phrasing_sensitivity = math.sqrt(variance)

        per_question_results.append({
            "question_id": q_id,
            "category": data["category"],
            "difficulty": data["difficulty"],
            "phrasing_scores": scores,
            "accuracy": round(accuracy, 4),
            "phrasing_sensitivity": round(phrasing_sensitivity, 4)
        })
    return per_question_results

def calculate_grouped_accuracy(evaluations: List[Dict[str, Any]], group_key: str) -> Dict[str, float]:
    """Calculates accuracy grouped by a specified key (e.g., category, difficulty)."""
    grouped_scores = defaultdict(lambda: {"correct": 0, "total": 0})
    for record in evaluations:
        key = record[group_key]
        grouped_scores[key]["total"] += 1
        grouped_scores[key]["correct"] += record["score"]

    accuracy_by_group = {
        key: round(data["correct"] / data["total"], 4) if data["total"] > 0 else 0.0
        for key, data in grouped_scores.items()
    }
    return accuracy_by_group

def calculate_phrasing_level_accuracy(evaluations: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculates accuracy for each phrasing index (1, 2, 3)."""
    phrasing_scores = defaultdict(lambda: {"correct": 0, "total": 0})
    for record in evaluations:
        phrasing_index = str(record["phrasing_index"]) # Convert to string for dict keys
        phrasing_scores[phrasing_index]["total"] += 1
        phrasing_scores[phrasing_index]["correct"] += record["score"]

    accuracy_by_phrasing = {
        idx: round(data["correct"] / data["total"], 4) if data["total"] > 0 else 0.0
        for idx, data in phrasing_scores.items()
    }
    return accuracy_by_phrasing

def get_most_frequently_failed_questions(per_question_results: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Identifies questions with the highest number of failed phrasings.
    """
    failed_questions = []
    for q_result in per_question_results:
        failed_count = q_result["phrasing_scores"].count(0)
        if failed_count > 0:
            failed_questions.append({
                "question_id": q_result["question_id"],
                "category": q_result["category"],
                "difficulty": q_result["difficulty"],
                "failed_phrasings_count": failed_count,
                "accuracy": q_result["accuracy"]
            })
    
    failed_questions.sort(key=lambda x: x["failed_phrasings_count"], reverse=True)
    return failed_questions[:top_n]

def generate_analysis_report(
    evaluations: List[Dict[str, Any]],
    total_questions: int,
    phrasings_per_question: int,
    model_name: str
) -> Dict[str, Any]:
    """
    Generates a comprehensive analysis report from evaluation records.
    """
    overall_accuracy = calculate_overall_accuracy(evaluations)
    per_question_results = calculate_per_question_results(evaluations)
    category_accuracy = calculate_grouped_accuracy(evaluations, "category")
    difficulty_accuracy = calculate_grouped_accuracy(evaluations, "difficulty")
    phrasing_accuracy = calculate_phrasing_level_accuracy(evaluations)
    most_failed_questions = get_most_frequently_failed_questions(per_question_results)

    report = {
        "metadata": {
            "model": model_name,
            "total_questions": total_questions,
            "phrasings_per_question": phrasings_per_question,
            "total_evaluations": len(evaluations),
            "temperature": 0 # Fixed for reproducibility
        },
        "overall_accuracy": round(overall_accuracy, 4),
        "phrasing_accuracy": phrasing_accuracy,
        "category_accuracy": category_accuracy,
        "difficulty_accuracy": difficulty_accuracy,
        "question_results": per_question_results,
        "most_failed_questions": most_failed_questions,
        "evaluations": evaluations # Include raw evaluations for detailed inspection
    }
    return report

def generate_summary_markdown_report(analysis_report: Dict[str, Any]) -> str:
    """
    Generates a human-readable Markdown summary report from the analysis data.
    """
    metadata = analysis_report["metadata"]
    
    report_content = "# LLM Evaluation Summary\n\n"

    report_content += "## Evaluation Configuration\n"
    report_content += f"- **Model:** `{metadata['model']}`\n"
    report_content += f"- **Number of Questions:** {metadata['total_questions']}\n"
    report_content += f"- **Phrasings per Question:** {metadata['phrasings_per_question']}\n"
    report_content += f"- **Total Evaluations:** {metadata['total_evaluations']}\n"
    report_content += f"- **Temperature:** {metadata['temperature']}\n\n"

    report_content += "## Overall Accuracy\n"
    report_content += f"**Overall Accuracy:** {analysis_report['overall_accuracy']:.2%}\n\n"

    report_content += "## Phrasing Accuracy\n"
    report_content += "| Phrasing Style | Accuracy |\n"
    report_content += "|----------------|----------|\n"
    for idx, acc in analysis_report["phrasing_accuracy"].items():
        report_content += f"| Phrasing {idx}    | {acc:.2%}     |\n"
    report_content += "\n"

    report_content += "## Accuracy by Category\n"
    report_content += "| Category | Accuracy |\n"
    report_content += "|----------|----------|\n"
    for category, acc in analysis_report["category_accuracy"].items():
        report_content += f"| {category.title()} | {acc:.2%}     |\n"
    report_content += "\n"

    report_content += "## Accuracy by Difficulty\n"
    report_content += "| Difficulty | Accuracy |\n"
    report_content += "|------------|----------|\n"
    for difficulty, acc in analysis_report["difficulty_accuracy"].items():
        report_content += f"| {difficulty.title()} | {acc:.2%}     |\n"
    report_content += "\n"

    report_content += "## Per-Question Breakdown\n"
    report_content += "| Question ID | Category | Phrasing A | Phrasing B | Phrasing C | Accuracy | Phrasing Sensitivity |\n"
    report_content += "|-------------|----------|------------|------------|------------|----------|----------------------|\n"
    for q_result in analysis_report["question_results"]:
        scores = q_result["phrasing_scores"]
        report_content += (
            f"| {q_result['question_id']} | {q_result['category'].title()} | "
            f"{scores[0]} | {scores[1]} | {scores[2]} | "
            f"{q_result['accuracy']:.2%} | {q_result['phrasing_sensitivity']:.4f} |\n"
        )
    report_content += "\n"

    report_content += "## Most Frequently Failed Questions\n"
    if analysis_report["most_failed_questions"]:
        report_content += "| Question ID | Category | Difficulty | Failed Phrasings | Accuracy |\n"
        report_content += "|-------------|----------|------------|------------------|----------|\n"
        for q_fail in analysis_report["most_failed_questions"]:
            report_content += (
                f"| {q_fail['question_id']} | {q_fail['category'].title()} | {q_fail['difficulty'].title()} | "
                f"{q_fail['failed_phrasings_count']} | {q_fail['accuracy']:.2%} |\n"
            )
    else:
        report_content += "No questions had failures across phrasings.\n"
    report_content += "\n"

    report_content += "## Phrasing Sensitivity Analysis\n"
    report_content += (
        "Phrasing sensitivity is measured by the population standard deviation of the binary scores "
        "(0 for incorrect, 1 for correct) across the three phrasings for each question. "
        "A higher standard deviation indicates that the model's performance for that question "
        "is more sensitive to how the question is phrased. For example, a question with scores `[1, 1, 0]` "
        "would have a sensitivity of approximately `0.4714`, indicating inconsistency. "
        "Conversely, scores like `[1, 1, 1]` or `[0, 0, 0]` result in a sensitivity of `0.0`, "
        "meaning consistent performance (either perfectly correct or consistently incorrect) regardless of phrasing.\n\n"
    )
    
    avg_sensitivity = sum(q['phrasing_sensitivity'] for q in analysis_report["question_results"]) / len(analysis_report["question_results"]) if analysis_report["question_results"] else 0.0
    report_content += f"The average phrasing sensitivity across all questions is: **{avg_sensitivity:.4f}**.\n\n"


    report_content += "## Key Insights\n"
    insights = []
    
    overall_acc = analysis_report['overall_accuracy']
    insights.append(f"The LLM achieved an overall accuracy of **{overall_acc:.2%}** across all {metadata['total_evaluations']} evaluations.")

    # Category differences
    if analysis_report['category_accuracy']:
        sorted_categories = sorted(analysis_report['category_accuracy'].items(), key=lambda item: item[1])
        best_category, best_acc = sorted_categories[-1]
        worst_category, worst_acc = sorted_categories[0]
        if best_category != worst_category:
            insights.append(f"Performance varied by category: **{best_category.title()}** questions had the highest accuracy ({best_acc:.2%}), while **{worst_category.title()}** questions had the lowest ({worst_acc:.2%}).")

    # Difficulty differences
    if analysis_report['difficulty_accuracy']:
        sorted_difficulties = sorted(analysis_report['difficulty_accuracy'].items(), key=lambda item: item[1])
        best_difficulty, best_acc = sorted_difficulties[-1]
        worst_difficulty, worst_acc = sorted_difficulties[0]
        if best_difficulty != worst_difficulty:
            insights.append(f"Difficulty significantly impacted results: **{best_difficulty.title()}** questions were answered with {best_acc:.2%} accuracy, whereas **{worst_difficulty.title()}** questions dropped to {worst_acc:.2%}.")

    # Phrasing-level differences
    if analysis_report['phrasing_accuracy']:
        sorted_phrasings = sorted(analysis_report['phrasing_accuracy'].items(), key=lambda item: item[1])
        best_phrasing, best_acc = sorted_phrasings[-1]
        worst_phrasing, worst_acc = sorted_phrasings[0]
        if best_phrasing != worst_phrasing:
            insights.append(f"Phrasing style had an impact: Phrasing **{best_phrasing}** yielded the highest accuracy ({best_acc:.2%}), while Phrasing **{worst_phrasing}** was the least effective ({worst_acc:.2%}).")

    # Questions with inconsistent results (high sensitivity)
    high_sensitivity_questions = [q for q in analysis_report["question_results"] if q["phrasing_sensitivity"] > 0.0]
    if high_sensitivity_questions:
        high_sensitivity_questions.sort(key=lambda x: x["phrasing_sensitivity"], reverse=True)
        top_sensitive = high_sensitivity_questions[0]
        insights.append(f"Question **{top_sensitive['question_id']}** ({top_sensitive['category'].title()}, {top_sensitive['difficulty'].title()}) showed the highest phrasing sensitivity ({top_sensitive['phrasing_sensitivity']:.4f}), indicating inconsistent performance across its phrasings.")
    
    if analysis_report["most_failed_questions"]:
        top_failed = analysis_report["most_failed_questions"][0]
        insights.append(f"Question **{top_failed['question_id']}** ({top_failed['category'].title()}, {top_failed['difficulty'].title()}) was the most frequently failed, with {top_failed['failed_phrasings_count']} out of {metadata['phrasings_per_question']} phrasings being incorrect.")


    for i, insight in enumerate(insights):
        report_content += f"- {insight}\n"
    report_content += "\n"

    return report_content

if __name__ == "__main__":
    # This block is for testing purposes and won't run during normal script execution
    print("This module is typically run via scripts/run_evaluation.py.")
    print("To test, provide a dummy list of evaluation records.")

    # Example dummy evaluations for testing analysis functions
    dummy_evals = [
        {"question_id": "q01", "category": "math", "difficulty": "easy", "phrasing_index": 1, "score": 1},
        {"question_id": "q01", "category": "math", "difficulty": "easy", "phrasing_index": 2, "score": 1},
        {"question_id": "q01", "category": "math", "difficulty": "easy", "phrasing_index": 3, "score": 0},
        {"question_id": "q02", "category": "geo", "difficulty": "medium", "phrasing_index": 1, "score": 0},
        {"question_id": "q02", "category": "geo", "difficulty": "medium", "phrasing_index": 2, "score": 0},
        {"question_id": "q02", "category": "geo", "difficulty": "medium", "phrasing_index": 3, "score": 0},
        {"question_id": "q03", "category": "math", "difficulty": "hard", "phrasing_index": 1, "score": 1},
        {"question_id": "q03", "category": "math", "difficulty": "hard", "phrasing_index": 2, "score": 0},
        {"question_id": "q03", "category": "math", "difficulty": "hard", "phrasing_index": 3, "score": 1},
        {"question_id": "q04", "category": "cs", "difficulty": "easy", "phrasing_index": 1, "score": 1},
        {"question_id": "q04", "category": "cs", "difficulty": "easy", "phrasing_index": 2, "score": 1},
        {"question_id": "q04", "category": "cs", "difficulty": "easy", "phrasing_index": 3, "score": 1},
    ]

    total_q = 4
    phrasings_per_q = 3
    model = "test-model"

    analysis_report = generate_analysis_report(dummy_evals, total_q, phrasings_per_q, model)
    print("\n--- Generated Analysis Report (JSON) ---")
    print(json.dumps(analysis_report, indent=2))

    markdown_report = generate_summary_markdown_report(analysis_report)
    print("\n--- Generated Summary Report (Markdown) ---")
    print(markdown_report)
