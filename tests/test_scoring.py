import pytest
import math
from src.scorers import score_response, normalized_exact_match, regex_extraction
from scripts.run_evaluation import Question, load_and_validate_dataset
from pathlib import Path
import json

# --- Test Scorers ---

def test_normalized_exact_match_correct_simple():
    assert normalized_exact_match("Canberra", "Canberra") == 1

def test_normalized_exact_match_case_insensitive():
    assert normalized_exact_match("canberra", "Canberra") == 1
    assert normalized_exact_match("CANBERRA", "canberra") == 1

def test_normalized_exact_match_whitespace_and_punctuation():
    assert normalized_exact_match(" Canberra. ", "Canberra") == 1
    assert normalized_exact_match("The answer is Canberra!", "Canberra") == 1
    assert normalized_exact_match("The answer is Canberra.", "Canberra") == 1
    assert normalized_exact_match("Canberra is the capital.", "Canberra") == 1

def test_normalized_exact_match_numeric_answers():
    assert normalized_exact_match("105", "105") == 1
    assert normalized_exact_match("The result is 105.", "105") == 1
    assert normalized_exact_match("105 miles", "105 miles") == 1
    assert normalized_exact_match("The answer is 105 miles.", "105 miles") == 1
    assert normalized_exact_match("12", "12.0") == 1 # Numeric comparison
    assert normalized_exact_match("3.14", "3.14") == 1

def test_normalized_exact_match_incorrect():
    assert normalized_exact_match("Paris", "Canberra") == 0
    assert normalized_exact_match("100", "105") == 0
    assert normalized_exact_match("The answer is 100 miles.", "105 miles") == 0

def test_regex_extraction_multiple_choice():
    assert regex_extraction("The correct answer is B.", "B") == 1
    assert regex_extraction("Option A is the best.", "A") == 1
    assert regex_extraction("The answer is C.", "C") == 1
    assert regex_extraction("The correct option is D.", "D") == 1

def test_regex_extraction_factual():
    assert regex_extraction("The capital is Canberra.", "Canberra") == 1
    assert regex_extraction("It is Canberra.", "Canberra") == 1
    assert regex_extraction("Canberra.", "Canberra") == 1 # Simple match if no prefix

def test_regex_extraction_numeric():
    assert regex_extraction("Therefore, the result is 42.", "42") == 1
    assert regex_extraction("The number is 7.", "7") == 1

def test_regex_extraction_incorrect():
    assert regex_extraction("The answer is B.", "A") == 0
    assert regex_extraction("The capital is Paris.", "Canberra") == 0

def test_score_response_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown scoring strategy: 'invalid_strategy'"):
        score_response("response", "answer", "invalid_strategy")

# --- Test Phrasing Sensitivity Calculation ---

def test_phrasing_sensitivity_all_correct():
    scores = [1, 1, 1]
    mean = sum(scores) / len(scores)
    variance = sum([(x - mean) ** 2 for x in scores]) / len(scores)
    sensitivity = math.sqrt(variance)
    assert sensitivity == 0.0

def test_phrasing_sensitivity_all_incorrect():
    scores = [0, 0, 0]
    mean = sum(scores) / len(scores)
    variance = sum([(x - mean) ** 2 for x in scores]) / len(scores)
    sensitivity = math.sqrt(variance)
    assert sensitivity == 0.0

def test_phrasing_sensitivity_mixed_scores():
    scores = [1, 1, 0]
    mean = sum(scores) / len(scores) # 2/3
    variance = sum([(x - mean) ** 2 for x in scores]) / len(scores)
    sensitivity = math.sqrt(variance)
    assert math.isclose(sensitivity, 0.4714, rel_tol=1e-4) # Approx sqrt(2/9)

def test_phrasing_sensitivity_another_mixed_scores():
    scores = [1, 0, 0]
    mean = sum(scores) / len(scores) # 1/3
    variance = sum([(x - mean) ** 2 for x in scores]) / len(scores)
    sensitivity = math.sqrt(variance)
    assert math.isclose(sensitivity, 0.4714, rel_tol=1e-4) # Approx sqrt(2/9)

# --- Test Dataset Validation (using Pydantic model) ---

def test_question_model_valid():
    valid_question_data = {
        "id": "q01",
        "question_text": "Test question",
        "correct_answer": "Test answer",
        "category": "test",
        "difficulty": "easy",
        "scoring_strategy": "normalized_exact_match",
        "phrasings": ["phrasing 1", "phrasing 2", "phrasing 3"]
    }
    question = Question(**valid_question_data)
    assert question.id == "q01"
    assert len(question.phrasings) == 3

def test_question_model_invalid_phrasing_count():
    invalid_question_data = {
        "id": "q01",
        "question_text": "Test question",
        "correct_answer": "Test answer",
        "category": "test",
        "difficulty": "easy",
        "scoring_strategy": "normalized_exact_match",
        "phrasings": ["phrasing 1", "phrasing 2"] # Only 2 phrasings
    }
    with pytest.raises(ValueError, match="phrasings must contain exactly 3 strings"):
        Question(**invalid_question_data)

def test_question_model_duplicate_phrasings():
    invalid_question_data = {
        "id": "q01",
        "question_text": "Test question",
        "correct_answer": "Test answer",
        "category": "test",
        "difficulty": "easy",
        "scoring_strategy": "normalized_exact_match",
        "phrasings": ["phrasing 1", "phrasing 1", "phrasing 3"] # Duplicate phrasing
    }
    with pytest.raises(ValueError, match="phrasings must be unique within a question"):
        Question(**invalid_question_data)

def test_question_model_missing_field():
    invalid_question_data = {
        "id": "q01",
        "question_text": "Test question",
        "correct_answer": "Test answer",
        "category": "test",
        "difficulty": "easy",
        "scoring_strategy": "normalized_exact_match",
        # "phrasings": ["phrasing 1", "phrasing 2", "phrasing 3"] # Missing phrasings
    }
    with pytest.raises(Exception): # Pydantic ValidationError
        Question(**invalid_question_data)

def test_question_model_invalid_scoring_strategy():
    invalid_question_data = {
        "id": "q01",
        "question_text": "Test question",
        "correct_answer": "Test answer",
        "category": "test",
        "difficulty": "easy",
        "scoring_strategy": "unsupported_strategy",
        "phrasings": ["phrasing 1", "phrasing 2", "phrasing 3"]
    }
    with pytest.raises(ValueError, match="Unknown scoring strategy: 'unsupported_strategy'"):
        Question(**invalid_question_data)

# --- Test load_and_validate_dataset function ---

@pytest.fixture
def temp_dataset_file(tmp_path):
    """Fixture to create a temporary dataset file for testing."""
    def _create_file(content, filename="test_dataset.json"):
        file_path = tmp_path / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(content, f)
        return file_path
    return _create_file

def test_load_and_validate_dataset_valid(temp_dataset_file):
    valid_dataset_content = [
        {
            "id": "q01", "question_text": "Q1", "correct_answer": "A1", "category": "C1", "difficulty": "easy",
            "scoring_strategy": "normalized_exact_match", "phrasings": ["P1.1", "P1.2", "P1.3"]
        },
        {
            "id": "q02", "question_text": "Q2", "correct_answer": "A2", "category": "C2", "difficulty": "medium",
            "scoring_strategy": "regex_extraction", "phrasings": ["P2.1", "P2.2", "P2.3"]
        },
        # Add more questions to meet the >=20 requirement
        *[
            {
                "id": f"q{i:02d}", "question_text": f"Q{i}", "correct_answer": f"A{i}", "category": "C_other", "difficulty": "easy",
                "scoring_strategy": "normalized_exact_match", "phrasings": [f"P{i}.1", f"P{i}.2", f"P{i}.3"]
            } for i in range(3, 26)
        ]
    ]
    file_path = temp_dataset_file(valid_dataset_content)
    dataset = load_and_validate_dataset(file_path)
    assert len(dataset) == 25
    assert dataset[0]["id"] == "q01"

def test_load_and_validate_dataset_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_and_validate_dataset(Path("non_existent_file.json"))

def test_load_and_validate_dataset_less_than_20_questions(temp_dataset_file):
    small_dataset_content = [
        {
            "id": "q01", "question_text": "Q1", "correct_answer": "A1", "category": "C1", "difficulty": "easy",
            "scoring_strategy": "normalized_exact_match", "phrasings": ["P1.1", "P1.2", "P1.3"]
        }
    ]
    file_path = temp_dataset_file(small_dataset_content)
    with pytest.raises(ValueError, match="Dataset must contain at least 20 unique questions"):
        load_and_validate_dataset(file_path)

def test_load_and_validate_dataset_duplicate_ids(temp_dataset_file):
    duplicate_id_dataset = [
        {
            "id": "q01", "question_text": "Q1", "correct_answer": "A1", "category": "C1", "difficulty": "easy",
            "scoring_strategy": "normalized_exact_match", "phrasings": ["P1.1", "P1.2", "P1.3"]
        },
        {
            "id": "q01", "question_text": "Q2", "correct_answer": "A2", "category": "C2", "difficulty": "medium",
            "scoring_strategy": "regex_extraction", "phrasings": ["P2.1", "P2.2", "P2.3"]
        },
        *[
            {
                "id": f"q{i:02d}", "question_text": f"Q{i}", "correct_answer": f"A{i}", "category": "C_other", "difficulty": "easy",
                "scoring_strategy": "normalized_exact_match", "phrasings": [f"P{i}.1", f"P{i}.2", f"P{i}.3"]
            } for i in range(3, 26)
        ]
    ]
    file_path = temp_dataset_file(duplicate_id_dataset)
    with pytest.raises(ValueError, match="Duplicate question ID found: 'q01'"):
        load_and_validate_dataset(file_path)

def test_load_and_validate_dataset_invalid_question_structure(temp_dataset_file):
    invalid_structure_dataset = [
        {
            "id": "q01", "question_text": "Q1", "correct_answer": "A1", "category": "C1", "difficulty": "easy",
            "scoring_strategy": "normalized_exact_match", "phrasings": ["P1.1", "P1.2", "P1.3"]
        },
        {
            "id": "q02", "question_text": "Q2", "correct_answer": "A2", "category": "C2", "difficulty": "medium",
            "scoring_strategy": "regex_extraction", "phrasings": ["P2.1", "P2.2"] # Missing phrasing
        },
        *[
            {
                "id": f"q{i:02d}", "question_text": f"Q{i}", "correct_answer": f"A{i}", "category": "C_other", "difficulty": "easy",
                "scoring_strategy": "normalized_exact_match", "phrasings": [f"P{i}.1", f"P{i}.2", f"P{i}.3"]
            } for i in range(3, 26)
        ]
    ]
    file_path = temp_dataset_file(invalid_structure_dataset)
    with pytest.raises(ValueError, match="Validation error in question 2"):
        load_and_validate_dataset(file_path)

def test_load_and_validate_dataset_invalid_scoring_strategy_in_dataset(temp_dataset_file):
    invalid_strategy_dataset = [
        {
            "id": "q01", "question_text": "Q1", "correct_answer": "A1", "category": "C1", "difficulty": "easy",
            "scoring_strategy": "normalized_exact_match", "phrasings": ["P1.1", "P1.2", "P1.3"]
        },
        {
            "id": "q02", "question_text": "Q2", "correct_answer": "A2", "category": "C2", "difficulty": "medium",
            "scoring_strategy": "non_existent_strategy", "phrasings": ["P2.1", "P2.2", "P2.3"]
        },
        *[
            {
                "id": f"q{i:02d}", "question_text": f"Q{i}", "correct_answer": f"A{i}", "category": "C_other", "difficulty": "easy",
                "scoring_strategy": "normalized_exact_match", "phrasings": [f"P{i}.1", f"P{i}.2", f"P{i}.3"]
            } for i in range(3, 26)
        ]
    ]
    file_path = temp_dataset_file(invalid_strategy_dataset)
    with pytest.raises(ValueError, match="Unknown scoring strategy: 'non_existent_strategy'"):
        load_and_validate_dataset(file_path)
