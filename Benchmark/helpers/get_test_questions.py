import json
from pathlib import Path
from typing import List, Dict

def get_all_questions() -> List[Dict]:
    """
    Loads all questions from JSON files in the AtomicRedTeamTests/questions_checked/ folder.

    Returns:
        List[Dict]: A list of dictionaries, where each dictionary represents a loaded question.
    """
    # Determine the path to the repository root directory
    # __file__ is the path to the current file (Benchmark/helpers/get_test_questions.py)
    # .parent gives the parent directory
    script_dir = Path(__file__).resolve().parent  # Benchmark/helpers
    benchmark_dir = script_dir.parent             # Benchmark
    repo_root = benchmark_dir.parent              # Root directory of the project

    questions_checked_dir = repo_root / "AtomicRedTeamTests" / "questions_checked"
    
    questions: List[Dict] = []
    if not questions_checked_dir.is_dir():
        # Optional: Error handling or logging if the folder does not exist
        print(f"Warning: The folder {questions_checked_dir} was not found.")
        return questions

    for file_path in questions_checked_dir.glob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                question_data = json.load(f)
                questions.append(question_data)
        except json.JSONDecodeError:
            print(f"Error loading JSON file: {file_path}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {file_path}: {e}")
            
    return questions

if __name__ == '__main__':
    # Simple example of usage
    all_questions = get_all_questions()
    if all_questions:
        print(f"Total {len(all_questions)} questions loaded.")
        print("First question:", all_questions[0].get("technique_id"), all_questions[0].get("question_id"))
    else:
        print("No questions loaded.")
