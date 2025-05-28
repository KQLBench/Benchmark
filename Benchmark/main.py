import sys
import os
import argparse
import time
from datetime import datetime, UTC
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import logging configuration first
from helpers.logging_config import configure_logging, LOG_LEVELS

from helpers.system import QuerySystem
from helpers.comparing import compare_answer, print_comparison_table
from helpers.connector_log_analytics import LogAnalyticsConnector
from Benchmark.helpers.get_test_questions import get_all_questions
from helpers.json_utils import dump_to_json

from Benchmark.configuration.models_config import MODELS_CONFIG
from Benchmark.models.benchmark_models import TestCase, TestResult, BenchmarkResult

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from dotenv import load_dotenv

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Optional

load_dotenv()

# Ensure results directory exists
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run benchmarks for KQL query generation')
    parser.add_argument('--model', type=str,
                        choices=list(MODELS_CONFIG.keys()),
                        help='Specify which model configuration to run (required unless --list-models is used)')
    parser.add_argument('--list-models', action='store_true',
                        help='List available model configurations')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for benchmark results (default: results/MODEL_KEY_TIMESTAMP.json)')
    parser.add_argument('--log-level', type=str, default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='Set the logging level (default: info)')
    parser.add_argument('--workers', type=int, default=5,
                        help='Number of parallel workers for test execution (default: 5)')
    parser.add_argument('--tries', type=int, default=5,
                        help='Number of attempts for query execution (default: 5)')
    parser.add_argument('--test-limit', type=int, default=0,
                        help='Limit the number of questions to run (0 for no limit, default: 0)')
    parser.add_argument('--query-date', type=str, default=None,
                        help='Specify the end date for KQL queries (DD.MM.YYYY). Defaults to last 24h from now.')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available model configurations (from MODELS_CONFIG):")
        for model_key, config_details in MODELS_CONFIG.items():
            print(f"- {model_key}")
            print(f"  LiteLLM Model ID: {config_details.get('model')}")
            if config_details.get('reasoning_effort'):
                 print(f"  Reasoning Effort: {config_details.get('reasoning_effort')}")
        sys.exit(0)
    
    if not args.model:
        parser.error("the following arguments are required: --model")
    
    if args.model not in MODELS_CONFIG:
        parser.error(f"Model key '{args.model}' not found in MODELS_CONFIG. Available: {list(MODELS_CONFIG.keys())}")
    
    # Validate query_date format if provided
    if args.query_date:
        try:
            datetime.strptime(args.query_date, "%d.%m.%Y")
        except ValueError:
            parser.error("Invalid date format for --query-date. Please use DD.MM.YYYY")

    return args

def run_test_case(LLM_System_model_key: str, LLM_System_max_tries: int, question_details_dict: dict, query_date_str: Optional[str] = None, querier=None):
    """Run a single test case for parallel execution using question data.
    
    Args:
        LLM_System_model_key: The model key (from MODELS_CONFIG) to configure the QuerySystem.
        LLM_System_max_tries: The number of tries for the QuerySystem.
        question_details_dict: Dictionary containing the question content.
        query_date_str: Optional query end date string (DD.MM.YYYY).
        querier: LogAnalyticsConnector instance.
        
    Returns:
        TestResult object or None if question details are missing.
    """
    if not question_details_dict:
        print(f"Warning: No question details provided. Skipping test case.")
        return None

    question_details_dict.pop('KQL_query_clean', None)
    test_case = TestCase(**question_details_dict)
        
    llm_system_instance = get_query_system_for_worker(LLM_System_model_key, LLM_System_max_tries)

    query_result = llm_system_instance.solve(test_case, query_date_str=query_date_str)
    answer_correct = compare_answer(query_result.answer, test_case.answer, test_case)
    
    test_result = TestResult(
        test_case=test_case,
        query_result=query_result,
        answer_correct=bool(answer_correct),
        cost=query_result.cost if query_result else 0.0
    )
    return test_result

# Thread-local storage for QuerySystem instances
thread_local_storage = threading.local()

def get_query_system_for_worker(model_key_from_args: str, max_tries_from_args: int):
    """Gets or creates a QuerySystem instance for the current thread, configured with provided model key and tries."""
    if not hasattr(thread_local_storage, 'query_system'):
        # Initialize QuerySystem with the model key from args and max_tries from args
        # The __init__ of QuerySystem will call configure with these.
        thread_local_storage.query_system = QuerySystem(initial_model_name=model_key_from_args, default_max_tries=max_tries_from_args)
    return thread_local_storage.query_system

def main():
    args = parse_args()
    
    # Configure logging with specified level
    configure_logging(log_level=LOG_LEVELS[args.log_level])

    # Load all available questions from JSON files
    all_questions_from_json = get_all_questions()
    if not all_questions_from_json:
        print("No questions found. Exiting.")
        sys.exit(1)

    # Apply test limit if specified
    if args.test_limit > 0:
        all_questions_from_json = all_questions_from_json[:args.test_limit]

    # Get workspace ID from Key Vault
    vault_url = os.getenv("VAULT_URL")
    credential = DefaultAzureCredential()
    secret_client = SecretClient(vault_url=vault_url, credential=credential)
    workspace_id = secret_client.get_secret("WORKSPACE-ID").value

    querier = LogAnalyticsConnector(workspace_id)

    model_key_to_run = args.model
    max_tries_for_run = args.tries

    # Prepare configuration details for BenchmarkResult
    selected_model_full_config = MODELS_CONFIG[model_key_to_run]
    run_configuration_dict = {
        "model_name": model_key_to_run,
        "litellm_model_id": selected_model_full_config.get("model"),
        "reasoning_effort_used": selected_model_full_config.get("reasoning_effort"),
        "configured_max_tries": max_tries_for_run
    }
    
    benchmark_result = BenchmarkResult(configuration=run_configuration_dict)
    model_start_time = time.time()
            
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_question_details = {}
        for question_dict in all_questions_from_json:
            if not question_dict: continue
            future = executor.submit(
                run_test_case, 
                model_key_to_run,
                max_tries_for_run,
                question_dict,
                args.query_date,
                querier
            )
            future_to_question_details[future] = question_dict
        
        test_results = []
        for future in as_completed(future_to_question_details):
            question_info_for_error_reporting = future_to_question_details[future]
            try:
                test_result = future.result()
                if test_result: test_results.append(test_result)
            except Exception as e:
                guid_for_error = question_info_for_error_reporting.get('question_id', 'UNKNOWN_GUID')
                print(f"Error processing test (Question ID: {guid_for_error}): {str(e)}")
    
    benchmark_result.test_results = test_results
    benchmark_result.total_cost = sum(tr.cost for tr in test_results if tr and hasattr(tr, 'cost'))
    
    # MODIFIED: Calculate total LLM formulate KQL errors
    total_llm_formulate_kql_errors = sum(tr.query_result.llm_formulate_kql_errors for tr in test_results if tr and tr.query_result)
    benchmark_result.llm_formulate_kql_errors_total = total_llm_formulate_kql_errors
    benchmark_result.average_llm_formulate_kql_errors_per_test = total_llm_formulate_kql_errors / len(test_results) if test_results else 0.0

    benchmark_result.total_benchmark_time = time.time() - model_start_time
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_model_name_for_file = model_key_to_run.replace("/", "_")
        
    if args.output:
        if "{model_name}" in args.output.lower() or "{model_key}" in args.output.lower():
            output_file = args.output.format(model_name=current_model_name_for_file, model_key=current_model_name_for_file, timestamp=timestamp)
        elif args.output.endswith(".json"):
            output_file = args.output
        else:
            output_file = os.path.join(args.output, f"{timestamp}_{current_model_name_for_file}.json")
    else:
        output_file = os.path.join(RESULTS_DIR, f"{timestamp}_{current_model_name_for_file}.json")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f_obj:
        dump_to_json(benchmark_result, f_obj, indent=4)
    print(f"\nBenchmark results for model '{model_key_to_run}' saved to: {output_file}")

    print(f"\n{'-' * 30} Model: {model_key_to_run} {'-' * 30}")
    print_comparison_table(benchmark_result)
    
    print("\n--- Benchmark Summary ---")
    summary_stats = benchmark_result.statistics
    print(f"LiteLLM Model ID: {benchmark_result.configuration['litellm_model_id']}")
    if benchmark_result.configuration['reasoning_effort_used'] is not None:
        print(f"Reasoning Effort: {benchmark_result.configuration['reasoning_effort_used']}")
    print(f"Configured Max Tries Per Test: {benchmark_result.configuration['configured_max_tries']}")

    print(f"Total Tests: {summary_stats['total_tests']}")
    print(f"Successful Tests: {summary_stats['successful_tests']}")
    print(f"Success Rate: {summary_stats['success_rate']:.2f}%")
    # MODIFIED: Print new consolidated error statistic
    print(f"Total LLM Formulated KQL Errors: {summary_stats['llm_formulate_kql_errors_total']}")
    print(f"Average LLM Formulated KQL Errors Per Test: {summary_stats['average_llm_formulate_kql_errors_per_test']:.2f}")
    print(f"Total Cost: ${summary_stats['total_cost']:.6f}")
    print(f"Average Cost Per Test: ${summary_stats['average_cost_per_test']:.6f}")
    print(f"Total Benchmark Time: {summary_stats['total_benchmark_time']:.2f}s")
    print(f"Average Execution Time Per Test: {summary_stats['avg_execution_time']:.2f}s")
    print(f"Average Attempts Per Test: {summary_stats['average_attempts']:.2f}")
        
if __name__ == "__main__":
    parse_args()
    main()
