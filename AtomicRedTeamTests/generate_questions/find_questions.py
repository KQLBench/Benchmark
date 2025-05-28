# Standard library imports
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
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
from Benchmark.helpers.logging_config import get_logger, configure_logging

# Get script directory and repository root for file paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

# Constants
CHOSEN_TESTS_CSV_PATH = REPO_ROOT / "AtomicRedTeamTests" / "chosen_tests.csv"
SYSTEM_PROMPT_PATH = SCRIPT_DIR / "evaluate_tests.txt"
OPENAI_API_VERSION = "2025-01-01-preview"
OPENAI_MODEL = "o3-mini"

# URLs for atomic red team indexes
LINUX_INDEX_URL = "https://raw.githubusercontent.com/redcanaryco/atomic-red-team/refs/heads/master/atomics/Indexes/linux-index.yaml"
WINDOWS_INDEX_URL = "https://raw.githubusercontent.com/redcanaryco/atomic-red-team/refs/heads/master/atomics/Indexes/windows-index.yaml"

# Initialize environment and logging
dotenv.load_dotenv()
configure_logging()
logger = get_logger(__name__)

@dataclass
class TestGrade:
    """Evaluation result for an atomic red team test"""
    needs_dependencies_not_installed_with_prereq: bool = False
    needs_parameters_to_be_changed_to_work: bool = False
    requires_active_directory: bool = False
    requires_multiple_machines: bool = False
    likely_to_fail: bool = False
    not_findable_by_defender_for_endpoint_logs: bool = False
    additional_notes: str = ""
    
    def is_suitable(self) -> bool:
        """Check if the test is suitable for a simple environment"""
        # If any evaluation criteria is True, the test is not suitable
        return not any([
            self.needs_dependencies_not_installed_with_prereq,
            self.needs_parameters_to_be_changed_to_work,
            self.requires_active_directory,
            self.requires_multiple_machines,
            self.likely_to_fail,
            self.not_findable_by_defender_for_endpoint_logs
        ])

class TestEvaluationResponse(BaseModel):
    """Model for the evaluation response from GPT"""
    needs_dependencies_not_installed_with_prereq: bool
    needs_parameters_to_be_changed_to_work: bool
    requires_active_directory: bool
    requires_multiple_machines: bool
    likely_to_fail: bool
    not_findable_by_defender_for_endpoint_logs: bool
    explanation: str

def init_services():
    """Initialize Azure services and OpenAI client"""
    # Get secrets from Key Vault
    vault_url = os.getenv("VAULT_URL")
    credential = DefaultAzureCredential()
    secret_client = SecretClient(vault_url=vault_url, credential=credential)
    
    # Initialize Azure OpenAI client
    client = AzureOpenAI(
        api_version=OPENAI_API_VERSION,
        api_key=secret_client.get_secret("AZURE-OPENAI-API-KEY").value,
        azure_endpoint=secret_client.get_secret("AZURE-OPENAI-BASE-URL").value
    )
    
    return client

def create_system_prompt():
    """Create system prompt for test evaluation"""
    with open(SYSTEM_PROMPT_PATH, "r") as f:
        return f.read().strip()

def fetch_index(url: str) -> List[str]:
    """Fetch and parse atomic red team index from GitHub to extract technique IDs"""
    logger.info(f"Fetching index from {url}")
    response = requests.get(url)
    if response.status_code != 200:
        logger.error(f"Failed to fetch index from {url}, status code: {response.status_code}")
        return []
    
    try:
        data = yaml.safe_load(response.text)
        technique_ids = []
        
        # Handle the nested dictionary structure
        if isinstance(data, dict):
            # Iterate through tactics
            for tactic, techniques in data.items():
                if isinstance(techniques, dict):
                    # Add all technique IDs to the list
                    technique_ids.extend(techniques.keys())
            
            logger.info(f"Found {len(technique_ids)} techniques in {url}")
            return technique_ids
        else:
            logger.error(f"Unexpected data format from {url} - not a dictionary")
            return []
    except Exception as e:
        logger.error(f"Error parsing index from {url}: {str(e)}")
        return []

def get_atomic_test_details(technique: str) -> Dict:
    """Fetch and parse the atomic red team yaml file from GitHub to get test details"""
    url = f"https://raw.githubusercontent.com/redcanaryco/atomic-red-team/master/atomics/{technique}/{technique}.yaml"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch YAML from GitHub for technique {technique}")
    
    return yaml.safe_load(response.text)

def evaluate_test(client: AzureOpenAI, technique_id: str, test: Dict) -> Optional[TestGrade]:
    """
    Evaluate a single atomic red team test using Azure OpenAI
    
    Args:
        client: Azure OpenAI client
        technique_id: The MITRE technique ID (e.g., T1059.001)
        test: The test details dictionary
    
    Returns:
        TestGrade object if evaluation succeeded, None otherwise
    """
    try:
        # Format test details for the model
        test_details_string = json.dumps(test, indent=2)
        
        # Prepare prompt for evaluation
        system_prompt = create_system_prompt()
        user_prompt = f"""
        Please evaluate this Atomic Red Team test for compatibility with our simple test environment:
        
        Technique ID: {technique_id}
        Test Name: {test.get('name', 'Unknown')}
        Test ID: {test.get('auto_generated_guid', 'Unknown')}
        
        Test Details:
        {test_details_string}
        
        Evaluate strictly and return structured assessment. Focus on:
        1. Does it need dependencies not installed with prereq?
        2. Does it need parameters configured to work?
        3. Does it require Active Directory?
        4. Does it require multiple machines?
        5. Is it likely to fail or be unreliable?
        6. Is it NOT findable by Microsoft Defender for Endpoint logs? (Mark true if this activity would likely NOT be captured in Defender logs)
        
        BE STRICT in your evaluation - only approve tests that are VERY likely to work without issues.
        """
        
        logger.info(f"Evaluating {technique_id} - {test.get('name', 'Unknown')} ({test.get('auto_generated_guid', 'Unknown')})")
        
        response = client.beta.chat.completions.parse(
            model=OPENAI_MODEL,
            max_completion_tokens=10000,
            reasoning_effort="high",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=TestEvaluationResponse,
        )
        
        evaluation = response.choices[0].message.parsed
        
        # Create TestGrade from evaluation
        grade = TestGrade(
            needs_dependencies_not_installed_with_prereq=evaluation.needs_dependencies_not_installed_with_prereq,
            needs_parameters_to_be_changed_to_work=evaluation.needs_parameters_to_be_changed_to_work,
            requires_active_directory=evaluation.requires_active_directory,
            requires_multiple_machines=evaluation.requires_multiple_machines,
            likely_to_fail=evaluation.likely_to_fail,
            not_findable_by_defender_for_endpoint_logs=evaluation.not_findable_by_defender_for_endpoint_logs,
            additional_notes=evaluation.explanation
        )
        
        suitability = "SUITABLE" if grade.is_suitable() else "NOT SUITABLE"
        logger.info(f"Evaluation result for {technique_id}: {suitability}")
        
        return grade
    
    except Exception as e:
        logger.error(f"Error evaluating test {technique_id} - {test.get('name', 'Unknown')}: {str(e)}")
        return None

def process_technique(client: AzureOpenAI, technique_id: str, suitable_tests: List[Dict], output_path: Path, thread_pool: ThreadPoolExecutor):
    """
    Process all tests for a given technique using thread pool
    
    Args:
        client: Azure OpenAI client
        technique_id: The MITRE technique ID (e.g., T1059.001)
        suitable_tests: List to append suitable tests to
        output_path: Path to save CSV file when new tests are found
        thread_pool: ThreadPoolExecutor for parallel test evaluation
    """
    try:
        import random
        
        # Get all tests for this technique
        technique_data = get_atomic_test_details(technique_id)
        atomic_tests = technique_data.get('atomic_tests', [])
        
        logger.info(f"Processing {len(atomic_tests)} tests for technique {technique_id}")
        
        # Prepare valid tests (those that support Windows or Linux)
        valid_tests = []
        for test in atomic_tests:
            platforms = test.get('supported_platforms', [])
            if any(platform in platforms for platform in ['windows', 'linux']):
                valid_tests.append(test)
            else:
                test_id = test.get('auto_generated_guid')
                test_name = test.get('name', 'Unknown')
                logger.info(f"Skipping {test_id} - {test_name} (unsupported platforms: {platforms})")
        
        # Randomize order of valid tests
        random.shuffle(valid_tests)
        logger.info(f"Shuffled {len(valid_tests)} valid tests for technique {technique_id}")
        
        # Submit all tests to thread pool and collect futures
        futures = []
        for test in valid_tests:
            test_id = test.get('auto_generated_guid')
            test_name = test.get('name', 'Unknown')
            platforms = test.get('supported_platforms', [])
            
            # Submit test for evaluation
            future = thread_pool.submit(evaluate_test, client, technique_id, test)
            futures.append((future, test, test_id, test_name, platforms))
        
        # Process results as they complete
        for future, test, test_id, test_name, platforms in futures:
            try:
                # Get the evaluation result
                grade = future.result()
                
                if grade and grade.is_suitable():
                    # Test is suitable, add to our list
                    suitable_test = {
                        'Technique': technique_id,
                        'TestName': test_name,
                        'auto_generated_guid': test_id,
                        'supported_platforms': ','.join(platforms),
                        'TimeoutSeconds': '120',  # Default timeout
                        'InputArgs': '',  # Empty by default
                        'enabled': 'TRUE',
                        'notes': grade.additional_notes[:100] if grade.additional_notes else ''  # Truncate notes
                    }
                    
                    suitable_tests.append(suitable_test)
                    logger.info(f"Added suitable test: {technique_id} - {test_name} ({test_id})")
                    
                    # Save to CSV immediately after finding a suitable test
                    save_to_csv([suitable_test], output_path, append=True)
                else:
                    if grade:
                        reasons = [field for field in asdict(grade).keys() 
                                  if field != 'additional_notes' and getattr(grade, field)]
                        logger.info(f"Test not suitable: {test_id} - {test_name}, reasons: {reasons}")
                    else:
                        logger.info(f"Test evaluation failed: {test_id} - {test_name}")
            except Exception as e:
                logger.error(f"Error processing evaluation for {test_id} - {test_name}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error processing technique {technique_id}: {str(e)}")

def save_to_csv(suitable_tests: List[Dict], output_path: Path, append: bool = False):
    """
    Save suitable tests to CSV file
    
    Args:
        suitable_tests: List of dictionaries containing test data
        output_path: Path to save the CSV file
        append: If True, append to existing file; if False, overwrite existing file
    """
    logger.info(f"Saving {len(suitable_tests)} suitable tests to {output_path}")
    
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fieldnames = ['Technique', 'TestName', 'auto_generated_guid', 'supported_platforms', 
                     'TimeoutSeconds', 'InputArgs', 'enabled', 'notes']
        
        # Sanitize text fields to handle Unicode characters
        sanitized_tests = []
        for test in suitable_tests:
            sanitized_test = {}
            for key, value in test.items():
                if isinstance(value, str):
                    # Replace problematic Unicode characters
                    sanitized_test[key] = value.replace('\u2010', '-').replace('\u2013', '-')
                else:
                    sanitized_test[key] = value
            sanitized_tests.append(sanitized_test)
        
        # Check if file exists and we're in append mode
        file_exists = output_path.exists() and append
        
        mode = 'a' if file_exists else 'w'
        with open(output_path, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(sanitized_tests)
            
        logger.info(f"Successfully saved suitable tests to {output_path}")
    
    except Exception as e:
        logger.error(f"Error saving suitable tests to CSV: {str(e)}")

def main():
    """Main function to find suitable atomic red team tests"""
    import argparse
    import random
    
    parser = argparse.ArgumentParser(description="Find suitable Atomic Red Team tests for a simple environment")
    parser.add_argument("--limit", type=int, default=None, 
                        help="Limit the number of techniques to evaluate (useful for testing)")
    parser.add_argument("--output", type=str, default=None,
                        help=f"Output CSV file path (default: {CHOSEN_TESTS_CSV_PATH})")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for shuffling (default: None, uses system time)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of worker threads to use for evaluation (default: 4)")
    
    args = parser.parse_args()
    
    # Set random seed if specified
    if args.seed is not None:
        random.seed(args.seed)
        logger.info(f"Using random seed: {args.seed}")
    
    # Determine output path
    output_path = Path(args.output) if args.output else CHOSEN_TESTS_CSV_PATH
    
    # Initialize CSV file with headers (will be overwritten if already exists)
    if output_path.exists():
        logger.info(f"Output file {output_path} already exists. Will append new tests.")
    else:
        # Create empty file with just headers
        save_to_csv([], output_path, append=False)
        logger.info(f"Created new output file {output_path}")
    
    # Initialize OpenAI client
    client = init_services()
    
    # Fetch Linux and Windows indexes
    linux_index = fetch_index(LINUX_INDEX_URL)
    windows_index = fetch_index(WINDOWS_INDEX_URL)
    
    # Combine all unique techniques from both sources
    all_techniques = set(linux_index).union(set(windows_index))
    all_techniques_list = list(all_techniques)
    
    # Shuffle the combined list
    random.shuffle(all_techniques_list)
    logger.info(f"Created and shuffled a list of {len(all_techniques_list)} unique techniques from both Linux and Windows")
    
    # Use the shuffled list as our unique techniques
    unique_techniques = all_techniques_list
    
    # Apply limit if specified
    if args.limit and args.limit > 0:
        unique_techniques = unique_techniques[:args.limit]
        logger.info(f"Limited evaluation to {args.limit} techniques")
    
    # List to store suitable tests (for tracking only)
    suitable_tests = []
    
    # Initialize thread pool
    with ThreadPoolExecutor(max_workers=args.workers) as thread_pool:
        logger.info(f"Using thread pool with {args.workers} workers")
        
        # Process each technique
        for technique_id in unique_techniques:
            process_technique(client, technique_id, suitable_tests, output_path, thread_pool)
    
    logger.info(f"Evaluation complete. Found {len(suitable_tests)} suitable tests.")

if __name__ == "__main__":
    main()
