import json
from typing import List, Dict, Any
from src.llm_client import LLMClient
from src.scorers import score_response

def run_evaluation_pipeline(dataset: List[Dict[str, Any]], llm_client: LLMClient) -> List[Dict[str, Any]]:
    """
    Executes the complete evaluation pipeline for a given dataset.

    Args:
        dataset (List[Dict[str, Any]]): The list of question dictionaries from the dataset.
        llm_client (LLMClient): An initialized LLMClient instance.

    Returns:
        List[Dict[str, Any]]: A list of evaluation records, each containing
                               details about a single LLM interaction and its score.
    """
    evaluation_records: List[Dict[str, Any]] = []
    total_evaluations = len(dataset) * 3 # 3 phrasings per question

    print(f"Running {total_evaluations} evaluations...")

    for q_idx, question in enumerate(dataset):
        question_id = question["id"]
        category = question["category"]
        difficulty = question["difficulty"]
        correct_answer = question["correct_answer"]
        scoring_strategy = question["scoring_strategy"]
        phrasings = question["phrasings"]

        for p_idx, phrasing in enumerate(phrasings):
            eval_num = (q_idx * 3) + p_idx + 1
            print(f"[{eval_num}/{total_evaluations}] Evaluating QID: {question_id}, Phrasing: {p_idx + 1}")

            try:
                model_response = llm_client.generate_response(phrasing)
                score = score_response(model_response, correct_answer, scoring_strategy)
            except Exception as e:
                print(f"Error during evaluation for QID {question_id}, Phrasing {p_idx + 1}: {e}")
                model_response = f"ERROR: {e}"
                score = 0 # Mark as incorrect if an error occurs

            record = {
                "question_id": question_id,
                "category": category,
                "difficulty": difficulty,
                "phrasing_index": p_idx + 1,
                "phrasing": phrasing,
                "correct_answer": correct_answer,
                "model_response": model_response,
                "score": score,
                "scoring_strategy": scoring_strategy
            }
            evaluation_records.append(record)

    return evaluation_records

if __name__ == "__main__":
    # This block is for testing purposes and won't run during normal script execution
    # as run_evaluation.py is the main entry point.
    print("This module is typically run via scripts/run_evaluation.py.")
    print("To test, ensure you have a valid .env with OPENAI_API_KEY and a dataset.")
    # Example of how it would be called:
    # from dotenv import load_dotenv
    # load_dotenv()
    #
    # # Dummy dataset for local testing
    # dummy_dataset = [
    #     {
    #         "id": "test01",
    #         "question_text": "What is 1+1?",
    #         "correct_answer": "2",
    #         "category": "arithmetic",
    #         "difficulty": "easy",
    #         "scoring_strategy": "normalized_exact_match",
    #         "phrasings": ["What is one plus one?", "Calculate 1+1.", "Sum of 1 and 1?"]
    #     }
    # ]
    #
    # try:
    #     client = LLMClient()
    #     results = run_evaluation_pipeline(dummy_dataset, client)
    #     print("\n--- Dummy Evaluation Results ---")
    #     print(json.dumps(results, indent=2))
    # except Exception as e:
    #     print(f"Error during dummy run: {e}")
