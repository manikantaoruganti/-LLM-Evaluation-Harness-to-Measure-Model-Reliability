import re
import string
from typing import Callable, Dict

def normalize_text(text: str) -> str:
    """
    Normalizes text by lowercasing, stripping whitespace, and removing punctuation.
    """
    text = text.lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with single
    return text

def normalized_exact_match(response: str, correct_answer: str) -> int:
    """
    Scores a response based on a normalized exact match.
    Normalizes case, whitespace, punctuation, and handles common prefixes/suffixes.
    """
    normalized_response = normalize_text(response)
    normalized_correct = normalize_text(correct_answer)

    # Handle common prefixes/suffixes in LLM responses
    # e.g., "The answer is 150 miles." -> "150 miles"
    # e.g., "150 miles." -> "150 miles"
    patterns_to_remove = [
        r"the correct answer is\s*",
        r"the answer is\s*",
        r"it is\s*",
        r"therefore the result is\s*",
        r"the result is\s*",
        r"is\s*",
        r"final answer\s*",
        r"answer\s*",
        r"option\s*[a-z]\s*",
        r"option\s*[a-z]\s*is\s*",
        r"is\s*the\s*correct\s*answer\s*",
        r"is\s*correct\s*",
        r"is\s*the\s*answer\s*"
    ]
    for pattern in patterns_to_remove:
        normalized_response = re.sub(pattern, '', normalized_response).strip()
        normalized_correct = re.sub(pattern, '', normalized_correct).strip()

    # Further normalization for numbers (e.g., "150 miles" vs "150")
    # This is a simple approach; more complex numeric parsing might be needed for specific cases.
    if normalized_correct.replace('.', '', 1).isdigit() and normalized_response.replace('.', '', 1).isdigit():
        try:
            return int(float(normalized_response) == float(normalized_correct))
        except ValueError:
            pass # Fallback to string comparison if conversion fails

    # Final comparison after extensive normalization
    return int(normalized_response == normalized_correct)

def regex_extraction(response: str, correct_answer: str) -> int:
    """
    Scores a response by extracting a specific pattern using regex and comparing it.
    Suitable for multiple-choice, numeric, or short factual answers.
    """
    normalized_correct = normalize_text(correct_answer)

    # Patterns to extract potential answers
    # Prioritize more specific patterns
    extraction_patterns = [
        r"the correct answer is\s*([A-Za-z0-9\s\.\-]+)", # "The correct answer is Canberra."
        r"the answer is\s*([A-Za-z0-9\s\.\-]+)",        # "The answer is 42."
        r"result is\s*([A-Za-z0-9\s\.\-]+)",            # "The result is 105."
        r"option\s*([A-D])",                            # "Option A"
        r"is\s*([A-Za-z0-9\s\.\-]+)",                   # "It is Canberra." (less specific, use carefully)
        r"([A-Za-z0-9\s\.\-]+)"                         # Catch-all, might need refinement
    ]

    extracted_answer = None
    for pattern in extraction_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            extracted_answer = normalize_text(match.group(1))
            break
    
    if extracted_answer is None:
        # If no specific pattern matched, try to normalize the whole response
        extracted_answer = normalize_text(response)

    return int(extracted_answer == normalized_correct)

# Dictionary to map strategy names to their respective functions
SCORING_STRATEGIES: Dict[str, Callable[[str, str], int]] = {
    "normalized_exact_match": normalized_exact_match,
    "regex_extraction": regex_extraction,
}

def score_response(response: str, correct_answer: str, strategy_name: str) -> int:
    """
    Scores an LLM response against a correct answer using a specified strategy.

    Args:
        response (str): The LLM's generated response.
        correct_answer (str): The ground truth correct answer.
        strategy_name (str): The name of the scoring strategy to use.

    Returns:
        int: 1 if the response is correct, 0 otherwise.

    Raises:
        ValueError: If an unknown scoring strategy is requested.
    """
    scorer = SCORING_STRATEGIES.get(strategy_name)
    if not scorer:
        raise ValueError(f"Unknown scoring strategy: {strategy_name}")
    return scorer(response, correct_answer)

if __name__ == "__main__":
    # Example usage and tests for scorers
    print("--- Normalized Exact Match Tests ---")
    assert score_response("The answer is 150 miles.", "150 miles", "normalized_exact_match") == 1, "Test 1 Failed"
    assert score_response("150", "150 miles", "normalized_exact_match") == 0, "Test 2 Failed (numeric mismatch)"
    assert score_response("150", "150", "normalized_exact_match") == 1, "Test 3 Failed"
    assert score_response("canberra.", "Canberra", "normalized_exact_match") == 1, "Test 4 Failed (case/punctuation)"
    assert score_response("  Canberra  ", "Canberra", "normalized_exact_match") == 1, "Test 5 Failed (whitespace)"
    assert score_response("The capital is Canberra.", "Canberra", "normalized_exact_match") == 1, "Test 6 Failed (prefix)"
    assert score_response("Paris", "London", "normalized_exact_match") == 0, "Test 7 Failed (incorrect)"
    assert score_response("The result is 42.", "42", "normalized_exact_match") == 1, "Test 8 Failed (numeric prefix)"
    assert score_response("H2O", "h2o", "normalized_exact_match") == 1, "Test 9 Failed (case insensitive)"
    assert score_response("12", "12.0", "normalized_exact_match") == 1, "Test 10 Failed (float vs int)"
    print("Normalized Exact Match Tests Passed!")

    print("\n--- Regex Extraction Tests ---")
    assert score_response("The correct answer is B.", "B", "regex_extraction") == 1, "Test 11 Failed (MCQ B)"
    assert score_response("The answer is Canberra.", "Canberra", "regex_extraction") == 1, "Test 12 Failed (Factual)"
    assert score_response("Therefore, the result is 42.", "42", "regex_extraction") == 1, "Test 13 Failed (Numeric)"
    assert score_response("Option A is correct.", "A", "regex_extraction") == 1, "Test 14 Failed (MCQ A)"
    assert score_response("The answer is Paris.", "London", "regex_extraction") == 0, "Test 15 Failed (Incorrect factual)"
    assert score_response("It is 105.", "105", "regex_extraction") == 1, "Test 16 Failed (Simple extraction)"
    assert score_response("The capital is Ottawa.", "Ottawa", "regex_extraction") == 1, "Test 17 Failed (Capital)"
    print("Regex Extraction Tests Passed!")

    try:
        score_response("response", "answer", "unknown_strategy")
        print("Test 18 Failed (Unknown strategy did not raise error)")
    except ValueError as e:
        print(f"Test 18 Passed (Unknown strategy raised error: {e})")
