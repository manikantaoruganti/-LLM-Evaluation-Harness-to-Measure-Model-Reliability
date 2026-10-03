import os
import json
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, validator
from typing import List, Dict, Any

from src.llm_client import LLMClient
from src.runner import run_evaluation_pipeline
from src.analysis import generate_analysis_report, generate_summary_markdown_report
from src.scorers import SCORING_STRATEGIES # To validate scoring_strategy in dataset

# Define Pydantic model for dataset validation
class Question(BaseModel):
    id: str
    question_text: str
    correct_answer: str
    category: str
    difficulty: str
    scoring_strategy: str
    phrasings: List[str] = Field(min_length=3, max_length=3)

    @validator('phrasings')
    def check_phrasing_count(cls, v):
        if len(v) != 3:
            raise ValueError('phrasings must contain exactly 3 strings')
        if len(set(v)) != 3:
            raise ValueError('phrasings must be unique within a question')
        return v
    
    @validator('scoring_strategy')
    def check_scoring_strategy(cls, v):
        if v not in SCORING_STRATEGIES:
            raise ValueError(f"Unknown scoring strategy: '{v}'. Must be one of {list(SCORING_STRATEGIES.keys())}")
        return v

def load_and_validate_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """
    Loads the evaluation dataset from a JSON file and validates its structure.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        raw_dataset = json.load(f)

    if not isinstance(raw_dataset, list):
        raise ValueError("Dataset must be a JSON array of questions.")

    if len(raw_dataset) < 20:
        raise ValueError(f"Dataset must contain at least 20 unique questions, found {len(raw_dataset)}.")

    validated_dataset = []
    question_ids = set()
    for i, q_data in enumerate(raw_dataset):
        try:
            question = Question(**q_data)
            if question.id in question_ids:
                raise ValueError(f"Duplicate question ID found: '{question.id}'")
            question_ids.add(question.id)
            validated_dataset.append(question.model_dump()) # Use model_dump() for Pydantic v2
        except ValidationError as e:
            raise ValueError(f"Validation error in question {i+1} (ID: {q_data.get('id', 'N/A')}): {e}")
        except ValueError as e:
            raise ValueError(f"Data error in question {i+1} (ID: {q_data.get('id', 'N/A')}): {e}")

    print(f"Loaded {len(validated_dataset)} questions.")
    return validated_dataset

def main():
    """
    Main function to run the LLM evaluation harness.
    """
    # 1. Load environment variables
    load_dotenv()
    print("Loading environment variables...")

    # 2. Validate OPENAI_API_KEY
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key or openai_api_key == "your_openai_api_key":
        print("CRITICAL ERROR: OPENAI_API_KEY environment variable is missing or not configured.")
        print("Please copy .env.example to .env and set your OpenAI API key.")
        print("Exiting evaluation.")
        exit(1)
    
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    print(f"Using OpenAI Model: {openai_model}")

    # Define paths
    dataset_path = Path("dataset/evaluation_dataset.json")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True) # Ensure results directory exists
    evaluation_results_path = results_dir / "evaluation_results.json"
    summary_report_path = results_dir / "summary_report.md"

    try:
        # 3. Load and validate dataset
        print("Loading dataset...")
        dataset = load_and_validate_dataset(dataset_path)
        total_questions = len(dataset)
        phrasings_per_question = 3 # Fixed by dataset structure

        # Initialize LLM client
        llm_client = LLMClient()

        # 4. Execute all evaluations
        print(f"Running {total_questions * phrasings_per_question} evaluations...")
        evaluation_records = run_evaluation_pipeline(dataset, llm_client)

        # 5. Calculate metrics
        print("Calculating metrics and generating analysis report...")
        analysis_report = generate_analysis_report(
            evaluation_records,
            total_questions,
            phrasings_per_question,
            llm_client.model # Pass the actual model name used
        )

        # 6. Write results/evaluation_results.json
        print(f"Writing evaluation results to {evaluation_results_path}...")
        with open(evaluation_results_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_report, f, indent=2, ensure_ascii=False)

        # 7. Write results/summary_report.md
        print(f"Writing summary report to {summary_report_path}...")
        markdown_report_content = generate_summary_markdown_report(analysis_report)
        with open(summary_report_path, 'w', encoding='utf-8') as f:
            f.write(markdown_report_content)

        print("Evaluation complete. Results and summary report generated.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except ValueError as e:
        print(f"Dataset Validation Error: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during evaluation: {e}")
        exit(1)

if __name__ == "__main__":
    main()
