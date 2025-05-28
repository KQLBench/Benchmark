# Standard library imports
import csv
import json
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports to work when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Third-party imports
import dotenv
import requests
import yaml
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from openai import AzureOpenAI
from pydantic import BaseModel


# Import when running from project root
from Benchmark.helpers.connector_log_analytics import LogAnalyticsConnector
from Benchmark.helpers.logging_config import get_logger, configure_logging
from Benchmark.models.benchmark_models import Question

# Get script directory and repository root for file paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

# Constants
QUESTIONS_DIR = REPO_ROOT / "AtomicRedTeamTests" / "questions_checked"
SYSTEM_PROMPT_PATH = SCRIPT_DIR / "generate_questions.txt"
TESTS_CSV_PATH = REPO_ROOT / "AtomicRedTeamTests" / "chosen_tests.csv"
OPENAI_API_VERSION = "2025-01-01-preview"
OPENAI_MODEL = "o4-mini"
OPENAI_MODEL_WEAK = "gpt-4.1"

# Initialize environment and logging
dotenv.load_dotenv()
logger = get_logger(__name__)

# Initialize services
def init_services():
    """Initialize Azure services and connectors"""
    logger.debug("Starting service initialization")
    
    # Get secrets from Key Vault
    vault_url = os.getenv("VAULT_URL")
    logger.debug(f"Using vault URL: {vault_url}")
    credential = DefaultAzureCredential()
    secret_client = SecretClient(vault_url=vault_url, credential=credential)
    logger.debug("Secret client initialized")
    
    # Initialize Log Analytics connector
    workspace_id = secret_client.get_secret("WORKSPACE-ID").value
    logger.debug(f"Retrieved workspace ID: {workspace_id[:4]}***")  # Show only first 4 chars
    connector = LogAnalyticsConnector(workspace_id)
    logger.debug("LogAnalyticsConnector initialized")
    
    # Cache table fields
    logger.debug("Retrieving all table fields from Log Analytics...")
    table_fields = connector.get_all_table_fields()
    table_count = len(table_fields) if table_fields else 0
    logger.debug(f"Retrieved {table_count} tables from Log Analytics")
    table_fields_string = json.dumps(table_fields, indent=4)
    
    # Initialize Azure OpenAI client
    logger.debug("Initializing Azure OpenAI client")
    api_key = secret_client.get_secret("AZURE-OPENAI-API-KEY").value
    azure_endpoint = secret_client.get_secret("AZURE-OPENAI-BASE-URL").value
    logger.debug(f"Using API version: {OPENAI_API_VERSION}, endpoint: {azure_endpoint}")
    client = AzureOpenAI(
        api_version=OPENAI_API_VERSION,
        api_key=api_key,
        azure_endpoint=azure_endpoint
    )
    logger.debug("Azure OpenAI client initialized")
    
    logger.debug("Service initialization complete")
    return connector, table_fields, table_fields_string, client

connector, table_fields, table_fields_string, client = init_services()


# Pydantic models
class Difficulty(str, Enum):
    """Enum for difficulty levels of detection questions"""
    EASY = "easy"
    MEDIUM = "medium"
    DIFFICULT = "difficult"

class TechnicalDetails(BaseModel):
    """Model for technical details of a detection scenario"""
    command_analysis: str
    detection_approach: str

class QuestionContent(BaseModel):
    """Model for content of a detection question"""
    context: str
    objective: str
    technical_details: TechnicalDetails
    thinking_how_to_phrase_question_and_answer: str
    prompt: str
    answer: list[str]
    difficulty: Difficulty

class ResultsVerification(BaseModel):
    """Model for verifying KQL query results against expected answers"""
    summary: str  # Summary of how the results relate to expected answers
    contains_expected_data: bool  # Whether results contain necessary data
    explanation: str  # Detailed explanation of the verification result

class KQLResponse(BaseModel):
    """Model for KQL query response"""
    KQL_query: str

def load_system_prompt() -> str:
    """Load the system prompt from file"""
    logger.debug(f"Loading system prompt from {SYSTEM_PROMPT_PATH}")
    try:
        with open(SYSTEM_PROMPT_PATH, "r") as f:
            content = f.read().strip()
            content_length = len(content)
            logger.debug(f"System prompt loaded successfully ({content_length} characters)")
            logger.debug(f"First 100 chars: {content[:100]}...")
            return content
    except Exception as e:
        logger.error(f"Error loading system prompt: {e}", exc_info=True)
        raise

def get_question_filename(test_id: str, technique: str) -> str:
    """Generate a filename for storing the question"""
    return f"{technique}_{test_id}.json"

def save_question(test_id: str, technique: str, question: Dict) -> None:
    """Save a question to the questions directory"""
    filename = get_question_filename(test_id, technique)
    question_file = QUESTIONS_DIR / filename
    
    # Ensure the directory exists
    QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(question_file, "w", encoding="utf-8") as f:
        json.dump(question, f, indent=4, ensure_ascii=False)


def get_atomic_test_details(technique: str, test_id: str) -> Dict:
    """Fetch and parse the atomic red team yaml file from GitHub to get test details"""
    import time
    start_time = time.time()
    
    url = f"https://raw.githubusercontent.com/redcanaryco/atomic-red-team/master/atomics/{technique}/{technique}.yaml"
    logger.debug(f"Fetching test details from URL: {url}")
    
    try:
        response = requests.get(url)
        logger.debug(f"HTTP response status code: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch YAML from GitHub for technique {technique}: Status {response.status_code}")
            raise ValueError(f"Failed to fetch YAML from GitHub for technique {technique}")
            
        data = yaml.safe_load(response.text)
        logger.debug(f"Successfully parsed YAML content for {technique}")
        
        for test in data["atomic_tests"]:
            if test.get("auto_generated_guid") == test_id:
                # Remove 'macos' from supported_platforms if present
                if "supported_platforms" in test:
                    original_platforms = test["supported_platforms"].copy()
                    test["supported_platforms"] = [p for p in test["supported_platforms"] if p != "macos"]
                    logger.debug(f"Platforms before filtering: {original_platforms}")
                    logger.debug(f"Platforms after filtering: {test['supported_platforms']}")
                
                duration = time.time() - start_time
                logger.debug(f"Found test {test_id} in {technique}.yaml (took {duration:.2f}s)")
                return test
                
        duration = time.time() - start_time
        logger.error(f"Test {test_id} not found in {technique}.yaml (searched for {duration:.2f}s)")
        raise ValueError(f"Test {test_id} not found in {technique}.yaml")
        
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error for {technique}: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error fetching test details for {technique}: {e}", exc_info=True)
        raise

def generate_question_content(test_id: str, technique: str) -> Dict:
    """
    Generate the content for a question using Azure OpenAI
    
    Args:
        test_id: The atomic test ID
        technique: The MITRE technique ID
    
    Returns:
        Dict containing the generated question content without KQL query
    """
    logger.info(f"Fetching test details for {technique} (Test ID: {test_id})")
    test_details = get_atomic_test_details(technique, test_id)
    test_details_string = json.dumps(test_details, indent=4)
    logger.debug(f"Test details retrieved: {test_details_string}")
    
    system_prompt = load_system_prompt()

    # Prepare user content
    user_content = f"""
        Create a detection question for this atomic red team test:
        {test_details_string}

        Available Log Analytics tables and their fields:
        {table_fields_string}

        Focus on generating a clear question with context, objective, technical details, 
        expected answer, and assign a difficulty level (must be exactly one of: 'easy', 'medium', or 'difficult') 
        based on the complexity of the detection scenario and queries needed.
        
        DO NOT generate the KQL query yet, just focus on the question content.
        """

    logger.info("Generating question content using Azure OpenAI")
    response = client.beta.chat.completions.parse(
        model=OPENAI_MODEL,
        max_completion_tokens=100_000,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        response_format=QuestionContent,
        reasoning_effort="medium"
    )

    # Extract structured content from parsed response
    content = response.choices[0].message.parsed
    content_dict = {
        "context": content.context,
        "objective": content.objective,
        "technical_details": {
            "command_analysis": content.technical_details.command_analysis,
            "detection_approach": content.technical_details.detection_approach
        },
        "thinking_how_to_phrase_question_and_answer": content.thinking_how_to_phrase_question_and_answer,
        "prompt": content.prompt,
        "answer": content.answer,
        "difficulty": content.difficulty
    }
    logger.info(f"Successfully generated question content for {technique} (Test ID: {test_id})")
    logger.debug(f"Generated content: {json.dumps(content_dict, indent=2)}")
    return content_dict

def verify_kql_results_match_answer(results: str, expected_answers: list, timespan: tuple = None) -> tuple[bool, str]:
    """
    Verify if KQL query results contain the expected answers
    
    Args:
        results: The KQL query results
        expected_answers: List of expected answers
        timespan: Optional timespan tuple
        
    Returns:
        Tuple containing (is_valid, explanation)
    """
    logger.debug(f"Verifying KQL results match expected answers: {expected_answers}")
    
    # Count rows in results
    row_count = len(results) - 1 if results and len(results) > 0 else 0
    logger.debug(f"Results contain {row_count} rows of data")
    
    # If there are no rows, return false immediately
    if row_count <= 0:
        logger.debug("No data rows in results, skipping LLM verification")
        return False, "Query returned no data rows to analyze"
    
    try:
        # Ask model to verify if results would contain the expected answers
        logger.debug("Creating verification prompt for LLM")
        verification_prompt = f"""
        Analyze these KQL query results to determine if they contain the information needed to answer the detection question.
        
        Expected answers: {json.dumps(expected_answers)} ---
        Query results: {results}
        
        Analyze the structure and content of these results. Would they provide the evidence needed to identify the activities described in the expected answers?
        """
        
        logger.debug("Calling OpenAI API to verify results")
        verification_response = client.beta.chat.completions.parse(
            model=OPENAI_MODEL_WEAK, 
            max_tokens=1_000,
            messages=[
                {"role": "system", "content": "You are a cybersecurity analyst expert in KQL and log analysis. The answer needs to be 1:1 in the results."},
                {"role": "user", "content": verification_prompt}
            ],
            response_format=ResultsVerification
        )
        
        result = verification_response.choices[0].message.parsed
        logger.debug(f"Verification result: contains_expected_data={result.contains_expected_data}")
        logger.debug(f"Verification summary: {result.summary}")
        logger.debug(f"Verification explanation: {result.explanation}")
        
        return (result.contains_expected_data, 
                result.explanation)
            
    except Exception as e:
        logger.error(f"Error during result verification: {str(e)}", exc_info=True)
        return False, f"Error verifying results: {str(e)}"

def generate_kql_query(test_id: str, technique: str, question_content: Dict,
                       retry_count: int = 0, max_retries: int = 5,
                       failed_query: str = None, error_message: str = None,
                       check_query: bool = False) -> Tuple[Optional[str], Optional[str], Optional[list]]:
    logger.debug(f"Generating KQL query for {technique} (Test ID: {test_id}, retry {retry_count}/{max_retries})")
    """
    Generate and validate a KQL query for the given question content.
    
    Args:
        test_id: The atomic test ID
        technique: The MITRE technique ID
        question_content: The previously generated question content
        retry_count: Current retry attempt (internal use)
        max_retries: Maximum number of retries for failed KQL queries
        failed_query: The KQL query that failed (for retry attempts)
        error_message: The error message from the failed KQL query (for retry attempts)
        check_query: Whether to validate the query by actually running it (default: False)
    
    Returns:
        A tuple containing the KQL query string (or None if failed), 
        a validation message (or None), and the KQL query results (or None if not applicable/failed).
    """
    if retry_count >= max_retries:
        final_error_message = f"Failed to generate valid KQL query after {max_retries} attempts. Last error context: {error_message if error_message else 'Max retries reached without specific error.'}"
        logger.error(final_error_message)
        return None, final_error_message, None
    
    # Prepare content for KQL generation
    user_content = f"""
        Generate a KQL query for this detection question:
        
        Context: {question_content['context']}
        Objective: {question_content['objective']}
        Technical details: {json.dumps(question_content['technical_details'])}
        Expected answer: {json.dumps(question_content['answer'])}

        Available Log Analytics tables and their fields:
        {table_fields_string}
        """
    
    # Add error information if this is a retry
    if retry_count > 0 and failed_query and error_message:
        user_content += f"""
        
        IMPORTANT: The previously generated KQL query failed validation. Please fix the issues in your new query.
        
        Failed Query:
        ```
        {failed_query}
        ```
        
        Error Message:
        ```
        {error_message}
        ```
        
        Please create a new KQL query that fixes these issues while still accomplishing the detection objective.
        """

    logger.info("Generating KQL query using Azure OpenAI")
    logger.debug(f"Using context:\n{user_content}")
    response = client.beta.chat.completions.parse(
        model=OPENAI_MODEL,
        max_completion_tokens=100_000,
        reasoning_effort="high",
        messages=[
            {"role": "system", "content": "You are a KQL expert. Generate a KQL query focusing on Microsoft Defender tables (DeviceEvents, DeviceProcessEvents, DeviceNetworkEvents, etc.) to detect the described security events. Don't use project-away or project-rename."},
            {"role": "user", "content": user_content}
        ],
        response_format=KQLResponse,
    )
    
    # Extract KQL query from parsed response
    kql_query = response.choices[0].message.parsed.KQL_query
    
    # Clean up the query by removing comments if needed    
    logger.info("Generated KQL query:")
    logger.info("-" * 80)
    logger.info(kql_query)
    logger.info("-" * 80)
    
    # If check_query is False, return the query with a message indicating validation was skipped
    if not check_query:
        logger.info("Skipping query validation as check_query=False")
        return kql_query, "Query validation skipped as check_query=False", None
    
    # Validate KQL syntax and content
    results = None # Initialize results to None
    try:
        results = connector.run_custom_query(kql_query)

        if results: # This implies results is not None and not empty list
            rows_count = len(results) - 1  # -1 because first row is columns
            logger.info(f"Query returned {rows_count} rows")
            logger.info("Sample of results:")
            if len(results) > 1: # Check if there are data rows beyond headers
                headers = results[0]
                logger.info(f"Columns: {', '.join(str(h) for h in headers)}")
                for row in results[1:min(len(results), 101)]:
                    logger.info(f"Row: {', '.join(str(cell) for cell in row)}")
            elif len(results) == 1: # Only headers returned
                 logger.info(f"Query returned only headers: {results[0]}")
            # else: results is empty list, handled by the outer 'if results:'
            
            is_valid, contain_message = verify_kql_results_match_answer(results, question_content['answer'])
            
            if is_valid:
                success_message = f"KQL query validation successful: {contain_message}"
                logger.info(success_message)
                return kql_query, success_message, results
            else:
                current_attempt_error = f"Query may not contain expected answers: {contain_message}"
                logger.warning(f"KQL query not matched expected answers: {contain_message}")
                if retry_count >= max_retries - 1:
                    final_fail_message = f"Failed after {max_retries} attempts - answers don't match. Last message: {contain_message}"
                    logger.error(final_fail_message)
                    return None, final_fail_message, results # Return results that caused failure
                return generate_kql_query(test_id, technique, question_content, retry_count + 1, max_retries,
                                        failed_query=kql_query, 
                                        error_message=current_attempt_error,
                                        check_query=check_query)
        else: # Query returned no results (results is None or empty list from connector)
            current_attempt_error = "Query returned no results. Try adjusting the query to match existing data."
            logger.warning(current_attempt_error)
            if retry_count >= max_retries - 1:
                final_fail_message = f"Failed after {max_retries} attempts - no results."
                logger.error(final_fail_message)
                return None, final_fail_message, results # results could be [] or None
            return generate_kql_query(test_id, technique, question_content, retry_count + 1, max_retries,
                                    failed_query=kql_query,
                                    error_message=current_attempt_error,
                                    check_query=check_query)
            
    except Exception as e: # KQL execution/syntax error
        current_attempt_error = str(e)
        logger.warning(f"KQL query validation failed: {current_attempt_error}")
        if retry_count >= max_retries - 1:
            final_fail_message = f"Failed after {max_retries} attempts - validation error: {current_attempt_error}"
            logger.error(final_fail_message)
            return None, final_fail_message, results # results might be None if query failed before execution
        logger.info("Retrying with new query generation...")
        return generate_kql_query(test_id, technique, question_content, retry_count + 1, max_retries,
                                 failed_query=kql_query, error_message=current_attempt_error,
                                 check_query=check_query)

def generate_question(test_id: str, technique: str, check_query: bool = False) -> Dict:
    """
    Generate a complete question for the benchmark LLM using Azure OpenAI
    This is a two-step process:
    1. Generate the question content (context, objective, etc.)
    2. Generate and validate the KQL query
    
    Args:
        test_id: The atomic test ID
        technique: The MITRE technique ID
        check_query: Whether to validate the query by actually running it (default: False)
        
    Returns:
        Complete question dictionary with validated KQL query or None if generation fails
    """
    import time
    start_time = time.time()
    logger.debug(f"Starting question generation for {technique} (Test ID: {test_id})")
    
    # Step 1: Generate question content
    content_start_time = time.time()
    question_content = generate_question_content(test_id, technique)
    content_duration = time.time() - content_start_time
    logger.debug(f"Content generation completed in {content_duration:.2f} seconds")
    
    # Step 2: Generate and validate KQL query
    try:
        query_start_time = time.time()
        kql_query, kql_validation_message, kql_query_results = generate_kql_query(
            test_id, technique, question_content, check_query=check_query
        )
        query_duration = time.time() - query_start_time
        logger.debug(f"KQL query generation/validation completed in {query_duration:.2f} seconds. Message: {kql_validation_message}")
        
        # Combine the content and the query
        logger.debug("Combining question content with KQL query, validation message, and results")
        complete_question = question_content.copy()
        complete_question['KQL_query'] = kql_query  # This can be None if generation failed
        complete_question['KQL_validation_message'] = kql_validation_message # Add the validation message
        complete_question['KQL_query_results'] = kql_query_results # Add KQL results (can be None)
        
        # Only save if we have a KQL query (even if validation was skipped or failed but query was generated)
        # The presence of kql_query indicates an attempt was made and a query string (even if bad) might exist.
        # If kql_query is None, it means generation/validation failed critically.
        if kql_query is not None:
            logger.debug(f"Saving complete question to file for {technique} (Test ID: {test_id})")
            save_question(test_id, technique, complete_question)
            
            total_duration = time.time() - start_time
            logger.info(f"Successfully generated and saved question for {technique} (Test ID: {test_id}) in {total_duration:.2f} seconds.")
            return complete_question
        else:
            logger.warning(f"Not saving question for {technique} (Test ID: {test_id}) - no KQL query was generated. Reason: {kql_validation_message}")
            return None # Indicates overall failure for this test
        
    except Exception as e: # Catch unexpected errors during the process
        logger.error(f"Unexpected error during question generation for {technique} (Test ID: {test_id}): {e}", exc_info=True)
        total_duration = time.time() - start_time
        logger.debug(f"Question generation process failed after {total_duration:.2f} seconds due to unexpected error.")
        return None

def generate_all_questions(check_queries: bool = False, force_regenerate: bool = False, max_workers: int = 4):
    """
    Generate questions for all enabled tests in tests.csv using multiple threads.
    By default, skips tests if a question file already exists.

    Args:
        check_queries: Whether to validate queries by actually running them (default: False)
        force_regenerate: If True, regenerate questions even if the file exists (default: False)
        max_workers: Maximum number of threads to use (default: 4)
    """
    if not TESTS_CSV_PATH.exists():
        raise FileNotFoundError(f"Tests CSV file not found at {TESTS_CSV_PATH}")

    tests_to_process = []
    with open(TESTS_CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['enabled'].lower() == 'true':
                technique = row['Technique']
                test_id = row['auto_generated_guid']
                
                filename = get_question_filename(test_id, technique)
                question_file = QUESTIONS_DIR / filename

                if question_file.exists() and not force_regenerate:
                    logger.info(f"Question for {technique} (Test ID: {test_id}) already exists, skipping. Use --force-regenerate to overwrite.")
                    continue
                elif question_file.exists() and force_regenerate:
                    logger.info(f"Question for {technique} (Test ID: {test_id}) already exists, regenerating due to --force-regenerate flag.")
                else:
                    logger.info(f"Queuing question generation for {technique} (Test ID: {test_id})")
                
                tests_to_process.append({'test_id': test_id, 'technique': technique})

    if not tests_to_process:
        logger.info("No tests to process.")
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_question, item['test_id'], item['technique'], check_queries): item for item in tests_to_process}
        
        for future in as_completed(futures):
            item = futures[future]
            technique = item['technique']
            test_id = item['test_id']
            try:
                question = future.result()
                if question:
                    logger.info(f"Successfully generated question for {technique} (Test ID: {test_id})")
                else:
                    logger.warning(f"Failed to generate question for {technique} (Test ID: {test_id})")
            except Exception as e:
                logger.error(f"Error generating question for {technique} (Test ID: {test_id}): {e}", exc_info=True)

# Function to run as main entry point
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate detection questions for Atomic Red Team tests")
    parser.add_argument("--check-queries", action="store_true", help="Validate KQL queries by actually running them")
    parser.add_argument("--force-regenerate", action="store_true", help="Regenerate questions even if they already exist")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default="INFO", help="Set the logging level (default: INFO)")
    parser.add_argument("--max-workers", type=int, default=os.cpu_count() or 4, 
                        help="Maximum number of worker threads (default: number of CPUs or 4)")
    args = parser.parse_args()

    # Configure logging with the specified level
    configure_logging(args.log_level)

    logger.info(f"Starting question generation with {args.max_workers} worker threads.")
    generate_all_questions(
        check_queries=args.check_queries, 
        force_regenerate=args.force_regenerate,
        max_workers=args.max_workers
    )
    logger.info("All question generation tasks completed.")

# This allows the script to be run both as a module and directly
if __name__ == "__main__":
    main()
