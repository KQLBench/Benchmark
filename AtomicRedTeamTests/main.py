import csv
import paramiko
import logging
import time
import threading
from queue import Queue
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import io
import argparse
import re # Added for platform string splitting
from pathlib import Path # Added for path handling
import os # Keep os for os.getenv and os.makedirs (though mkdir can be done by pathlib)

from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# Path definitions using pathlib
SCRIPT_DIR = Path(__file__).resolve().parent
LOGS_DIR = SCRIPT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
log_filename = LOGS_DIR / f"atomic_tests_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Creating the constant for workers outside since it's used in function signatures
WORKERS_COUNT = int(os.getenv("WORKERS_COUNT", 4))

# Variables for credentials that will be loaded only when needed
WINDOWS_HOST = None
WINDOWS_USERNAME = None
WINDOWS_PASSWORD = None
LINUX_HOST = None
LINUX_USERNAME = None
LINUX_PRIVATE_KEY = None

def load_credentials():
    """Load credentials from Azure Key Vault"""
    global WINDOWS_HOST, WINDOWS_USERNAME, WINDOWS_PASSWORD, LINUX_HOST, LINUX_USERNAME, LINUX_PRIVATE_KEY
    
    # Load secrets from Azure Key Vault
    load_dotenv()
    vault_url = os.getenv("VAULT_URL")
    credential = DefaultAzureCredential()
    secret_client = SecretClient(vault_url=vault_url, credential=credential)

    # Get secrets directly from Key Vault
    WINDOWS_HOST = secret_client.get_secret("WINDOWS-HOST").value
    WINDOWS_USERNAME = secret_client.get_secret("WINDOWS-USERNAME").value
    WINDOWS_PASSWORD = secret_client.get_secret("WINDOWS-PASSWORD").value
    LINUX_HOST = secret_client.get_secret("LINUX-HOST").value
    LINUX_USERNAME = secret_client.get_secret("LINUX-USERNAME").value
    LINUX_PRIVATE_KEY = secret_client.get_secret("LINUX-PRIVATE-KEY").value

# Create reports directory if it doesn't exist
REPORTS_DIR = SCRIPT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Generate timestamp for report filename
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

# CSV-Datei mit den Tests
TESTS_CSV = SCRIPT_DIR / "chosen_tests.csv"
# Report-Datei
REPORT_CSV = REPORTS_DIR / f"atomic_tests_report-{timestamp}.csv"

# Constants for dictionary keys, OS types, actions, statuses, etc.
# CSV Header and Test Dictionary Keys
KEY_TECHNIQUE = "Technique"
KEY_AUTO_GENERATED_GUID = "auto_generated_guid"
KEY_TIMEOUT_SECONDS = "TimeoutSeconds"
KEY_INPUT_ARGS = "InputArgs"
KEY_TEST_NAME = "TestName"
KEY_ENABLED = "enabled"
KEY_SUPPORTED_PLATFORMS = "supported_platforms"

# OS Types
OS_WINDOWS = "windows"
OS_LINUX = "linux"
OS_ALL = "all"

# Test Actions
ACTION_GET_PREREQS = "GetPrereqs"
ACTION_TEST = "Test"
ACTION_CLEANUP = "Cleanup"

# Statuses
STATUS_SUCCESS = "Success"
STATUS_FAILURE = "Failure"
STATUS_ERROR = "Error"

# SSH
SSH_SESSION_NOT_ACTIVE = "SSH session not active"

# Default values
DEFAULT_TIMEOUT = "120"

# Report CSV Fieldnames
FIELD_TIMESTAMP = "timestamp"
FIELD_HOST = "host"
FIELD_OS = "os"
FIELD_TECHNIQUE = "technique"
FIELD_TEST_NAME = "test_name"
FIELD_GUID = "guid"
FIELD_ACTION = "action"
FIELD_COMMAND = "command"
FIELD_STATUS = "status"
FIELD_OUTPUT = "output"

# Funktionen zum Ausführen von Befehlen

def _execute_ssh_command(os_type: str, command_to_run: str, max_retries: int = 2, retry_delay: int = 5) -> tuple[str, str]:
    """
    Helper function to execute a command over SSH with retry logic.
    Returns (status, output).
    """
    for attempt in range(max_retries + 1):
        try:
            ssh_client = get_ssh_connection(os_type)
            if not ssh_client:
                return STATUS_ERROR, f"Failed to get SSH connection to {os_type.capitalize()}"
            
            stdin, stdout, stderr = ssh_client.exec_command(command_to_run)
            output = stdout.read().decode(errors='replace') + stderr.read().decode(errors='replace')
            exit_status = stdout.channel.recv_exit_status()
            status = STATUS_SUCCESS if exit_status == 0 else STATUS_FAILURE
            
            if len(output) > 10000: # Truncate long outputs
                output = output[:10000] + "... (output truncated)"
            return status, output.strip()
        
        except Exception as e:
            if SSH_SESSION_NOT_ACTIVE in str(e) and attempt < max_retries:
                logger.warning(f"SSH session not active for {os_type}, retrying in {retry_delay} seconds (attempt {attempt+1}/{max_retries+1})")
                time.sleep(retry_delay)
                thread_id = threading.get_ident()
                connection_pool = windows_ssh_connections if os_type == OS_WINDOWS else linux_ssh_connections
                if thread_id in connection_pool:
                    try:
                        connection_pool[thread_id].close()
                    except:
                        pass # Ignore errors during close
                    del connection_pool[thread_id]
                continue # Retry the command execution
            
            logger.error(f"Error executing {os_type} test command ({command_to_run}): {e}")
            return STATUS_ERROR, str(e)
    # Should not be reached if max_retries >= 0, but as a fallback:
    return STATUS_ERROR, f"Failed after {max_retries + 1} attempts for {os_type} executing: {command_to_run}"

def run_windows_test(technique, guid, timeout, input_args, extra_flag=""):
    """
    Führt einen Windows-Test über SSH aus.
    extra_flag kann z.B. "-GetPrereqs" oder "-Cleanup" sein.
    Returns (command_executed, status, output)
    """
    ps_cmd_template = f"Invoke-AtomicTest {technique} -TestGuids {guid} {extra_flag} -TimeoutSeconds {timeout} {input_args}"
    # Use PowerShell to execute the command
    full_command_to_run = f"powershell -Command \"{ps_cmd_template}\""
    
    status, output = _execute_ssh_command(OS_WINDOWS, full_command_to_run)
    
    return ps_cmd_template, status, output

def run_linux_test(technique, guid, timeout, input_args, extra_flag=""):
    """
    Führt einen Linux-Test über SSH aus.
    Hier wird angenommen, dass PowerShell Core (pwsh) installiert ist.
    Returns (command_executed, status, output)
    """
    cmd_inner_template = f"Invoke-AtomicTest {technique} -TestGuids {guid} {extra_flag} -TimeoutSeconds {timeout} {input_args}"
    # Befehl in PowerShell ausführen
    full_command_to_run = f"sudo pwsh -Command \"{cmd_inner_template}\""

    status, output = _execute_ssh_command(OS_LINUX, full_command_to_run)
    
    return cmd_inner_template, status, output # Return the core Invoke-AtomicTest command for logging consistency

# Connection pools
linux_ssh_connections: Dict[int, paramiko.SSHClient] = {}
windows_ssh_connections: Dict[int, paramiko.SSHClient] = {}

def get_ssh_connection(os_type=OS_LINUX) -> paramiko.SSHClient:
    """Get or create an SSH connection for the current thread"""
    thread_id = threading.get_ident()
    connection_pool = linux_ssh_connections if os_type == OS_LINUX else windows_ssh_connections
    host = LINUX_HOST if os_type == OS_LINUX else WINDOWS_HOST
    username = LINUX_USERNAME if os_type == OS_LINUX else WINDOWS_USERNAME
    
    # If connection exists, check if it's active
    if thread_id in connection_pool:
        ssh = connection_pool[thread_id]
        try:
            # Check if transport is active
            if ssh.get_transport() and ssh.get_transport().is_active():
                return ssh
            else:
                logger.warning(f"SSH connection for thread {thread_id} to {os_type} is inactive, reconnecting...")
                # Close the inactive connection before recreating
                try:
                    ssh.close()
                except:
                    pass
                # Remove from pool so we'll create a new one
                del connection_pool[thread_id]
        except Exception as e:
            logger.warning(f"Error checking SSH connection for thread {thread_id}: {e}, reconnecting...")
            # Remove from pool so we'll create a new one
            del connection_pool[thread_id]
    
    # Create a new connection
    if thread_id not in connection_pool:
        logger.info(f"Creating new SSH connection for thread {thread_id} to {os_type}")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            if os_type == OS_LINUX:
                # Linux uses private key authentication
                key = paramiko.RSAKey.from_private_key(io.StringIO(LINUX_PRIVATE_KEY))
                ssh.connect(host, username=username, pkey=key)
            else: # OS_WINDOWS
                # Windows uses password authentication
                ssh.connect(host, username=username, password=WINDOWS_PASSWORD)
            
            # Enable keepalive packets to prevent connection from timing out
            transport = ssh.get_transport()
            transport.set_keepalive(30)  # Send keepalive every 30 seconds
            
            connection_pool[thread_id] = ssh
            logger.info(f"SSH connection established for thread {thread_id} to {os_type}")
        except Exception as e:
            logger.error(f"Failed to establish SSH connection for thread {thread_id} to {os_type}: {e}")
            return None
    return connection_pool[thread_id]

# Lese Tests aus der CSV
tests = []
# This initial read is later overridden in main(), but if it were to be used, it should use constants.
# For now, the main() function's read is the effective one.
with TESTS_CSV.open(newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row.get(KEY_ENABLED, "").strip().upper() == "TRUE":
            tests.append(row)

# Queue for results
results_queue = Queue()

# Open the CSV file at the beginning
csv_file = None
csv_writer = None

def initialize_csv_writer():
    """Initialize the CSV writer at the beginning"""
    global csv_file, csv_writer
    csv_file = REPORT_CSV.open('w', newline='', encoding='utf-8')
    fieldnames = [
        FIELD_TIMESTAMP, FIELD_HOST, FIELD_OS, FIELD_TECHNIQUE, 
        FIELD_TEST_NAME, FIELD_GUID, FIELD_ACTION, FIELD_COMMAND, 
        FIELD_STATUS, FIELD_OUTPUT
    ]
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

def write_result_to_csv(result):
    """Write a single result to the CSV file"""
    if csv_writer:
        csv_writer.writerow(result)
        csv_file.flush()  # Ensure it's written to disk immediately

def run_single_test(test, os_type):
    """Execute a single test for the specified OS"""
    technique = test[KEY_TECHNIQUE].strip()
    guid = test[KEY_AUTO_GENERATED_GUID].strip()
    timeout = test.get(KEY_TIMEOUT_SECONDS, DEFAULT_TIMEOUT).strip() or DEFAULT_TIMEOUT
    input_args = test.get(KEY_INPUT_ARGS, "").strip()
    test_name = test.get(KEY_TEST_NAME, "").strip()
    
    logger.info(f"Processing {os_type} test: {test_name} ({technique})")
    
    host_ip = WINDOWS_HOST if os_type == OS_WINDOWS else LINUX_HOST
    run_test_func = run_windows_test if os_type == OS_WINDOWS else run_linux_test

    # GetPrereqs
    cmd, status, output = run_test_func(technique, guid, timeout, input_args, extra_flag=f"-{ACTION_GET_PREREQS}")
    result = {
        FIELD_TIMESTAMP: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        FIELD_HOST: host_ip,
        FIELD_OS: os_type,
        FIELD_TECHNIQUE: technique,
        FIELD_TEST_NAME: test_name,
        FIELD_GUID: guid,
        FIELD_COMMAND: cmd,
        FIELD_ACTION: ACTION_GET_PREREQS,
        FIELD_STATUS: status,
        FIELD_OUTPUT: output
    }
    results_queue.put(result)
    write_result_to_csv(result)

    # Test ausführen
    cmd, status, output = run_test_func(technique, guid, timeout, input_args)
    result = {
        FIELD_TIMESTAMP: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        FIELD_HOST: host_ip,
        FIELD_OS: os_type,
        FIELD_TECHNIQUE: technique,
        FIELD_TEST_NAME: test_name,
        FIELD_GUID: guid,
        FIELD_COMMAND: cmd,
        FIELD_ACTION: ACTION_TEST,
        FIELD_STATUS: status,
        FIELD_OUTPUT: output
    }
    results_queue.put(result)
    write_result_to_csv(result)
    
    # Run cleanup after test
    logger.info(f"Running cleanup for {os_type} test: {test_name}")
    cmd, status, output = run_test_func(technique, guid, timeout, input_args, extra_flag=f"-{ACTION_CLEANUP}")
    result = {
        FIELD_TIMESTAMP: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        FIELD_HOST: host_ip,
        FIELD_OS: os_type,
        FIELD_TECHNIQUE: technique,
        FIELD_TEST_NAME: test_name,
        FIELD_GUID: guid,
        FIELD_COMMAND: cmd,
        FIELD_ACTION: ACTION_CLEANUP,
        FIELD_STATUS: status,
        FIELD_OUTPUT: output
    }
    results_queue.put(result)
    write_result_to_csv(result)

# Helper function to create result dictionary
def _create_result_dict(host_val: str, os_val: str, technique_val: str, test_name_val: str, 
                        guid_val: str, command_val: str, action_val: str, status_val: str, 
                        output_val: str) -> Dict:
    return {
        FIELD_TIMESTAMP: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        FIELD_HOST: host_val,
        FIELD_OS: os_val,
        FIELD_TECHNIQUE: technique_val,
        FIELD_TEST_NAME: test_name_val,
        FIELD_GUID: guid_val,
        FIELD_COMMAND: command_val,
        FIELD_ACTION: action_val,
        FIELD_STATUS: status_val,
        FIELD_OUTPUT: output_val
    }

def run_single_test(test, os_type):
    """Execute a single test for the specified OS"""
    technique = test[KEY_TECHNIQUE].strip()
    guid = test[KEY_AUTO_GENERATED_GUID].strip()
    timeout = test.get(KEY_TIMEOUT_SECONDS, DEFAULT_TIMEOUT).strip() or DEFAULT_TIMEOUT
    input_args = test.get(KEY_INPUT_ARGS, "").strip()
    test_name = test.get(KEY_TEST_NAME, "").strip()
    
    logger.info(f"Processing {os_type} test: {test_name} ({technique})")
    
    host_ip = WINDOWS_HOST if os_type == OS_WINDOWS else LINUX_HOST
    run_test_func = run_windows_test if os_type == OS_WINDOWS else run_linux_test

    # GetPrereqs
    cmd, status, output = run_test_func(technique, guid, timeout, input_args, extra_flag=f"-{ACTION_GET_PREREQS}")
    result = _create_result_dict(host_ip, os_type, technique, test_name, guid, cmd, ACTION_GET_PREREQS, status, output)
    results_queue.put(result)
    write_result_to_csv(result)

    # Test ausführen
    cmd, status, output = run_test_func(technique, guid, timeout, input_args) # No extra_flag for main test
    result = _create_result_dict(host_ip, os_type, technique, test_name, guid, cmd, ACTION_TEST, status, output)
    results_queue.put(result)
    write_result_to_csv(result)
    
    # Run cleanup after test
    logger.info(f"Running cleanup for {os_type} test: {test_name}")
    cmd, status, output = run_test_func(technique, guid, timeout, input_args, extra_flag=f"-{ACTION_CLEANUP}")
    result = _create_result_dict(host_ip, os_type, technique, test_name, guid, cmd, ACTION_CLEANUP, status, output)
    results_queue.put(result)
    write_result_to_csv(result)

def parse_args():
    parser = argparse.ArgumentParser(description='Run Atomic Red Team Tests')
    parser.add_argument('--technique', '-t', help='Run tests for specific technique (e.g., T1003)')
    parser.add_argument('--guid', '-g', help='Run test with specific GUID')
    parser.add_argument('--platform', '-p', choices=[OS_WINDOWS, OS_LINUX, OS_ALL], 
                       default=OS_ALL, help='Run tests for specific platform')
    parser.add_argument('--list', '-l', action='store_true', 
                       help='List all available tests')
    return parser.parse_args()

def filter_tests(tests: List[Dict], technique: Optional[str] = None, 
                guid: Optional[str] = None, platform: str = OS_ALL) -> List[Dict]:
    filtered_tests = []
    for test in tests:
        if not test.get(KEY_ENABLED, "").strip().upper() == "TRUE":
            continue

        # Check if test matches filters
        matches = True
        if technique and test[KEY_TECHNIQUE].strip() != technique:
            matches = False
        if guid and test[KEY_AUTO_GENERATED_GUID].strip() != guid:
            matches = False
        if platform != OS_ALL:
            platforms_raw = test.get(KEY_SUPPORTED_PLATFORMS, "")
            # Split by comma or pipe, filter out empty strings after split, then strip and lower
            current_test_platforms = [p.strip().lower() for p in re.split(r'[|,]', platforms_raw) if p.strip()]
            if platform not in current_test_platforms:
                matches = False

        if matches:
            filtered_tests.append(test)

    return filtered_tests

def list_available_tests(tests: List[Dict]):
    print("\nAvailable Tests:")
    print("{:<15} {:<40} {:<36} {:<20}".format(
        KEY_TECHNIQUE.capitalize(), "Test Name", "GUID", "Platforms")) # Test Name, GUID, Platforms are for display
    print("-" * 111)
    
    for test in tests:
        if test.get(KEY_ENABLED, "").strip().upper() == "TRUE":
            test_name_val = test.get(KEY_TEST_NAME, "")
            print("{:<15} {:<40} {:<36} {:<20}".format(
                test.get(KEY_TECHNIQUE, ""),
                test_name_val[:37] + "..." if len(test_name_val) > 40 else test_name_val,
                test.get(KEY_AUTO_GENERATED_GUID, ""),
                test.get(KEY_SUPPORTED_PLATFORMS, "")))

def main():
    args = parse_args()

    # Read all tests
    all_tests_from_csv = [] # Renamed to avoid confusion with the global 'tests' variable
    with TESTS_CSV.open(newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            all_tests_from_csv.append(row)

    # If --list argument is provided, show available tests and exit
    if args.list:
        # Use all_tests_from_csv for listing, ensuring it reflects the current file content
        list_available_tests(all_tests_from_csv)
        return

    # Load credentials only when we need to run tests
    logger.info("Loading credentials from Azure Key Vault...")
    load_credentials()
    
    # Initialize the CSV writer
    initialize_csv_writer()

    # Filter tests based on command line arguments using all tests read from CSV
    filtered_tests = filter_tests(all_tests_from_csv, args.technique, args.guid, args.platform)

    if not filtered_tests:
        print("No matching tests found with the specified criteria.")
        return

    # Prepare test tasks
    test_tasks = []
    logger.info("Preparing test tasks")
    for test_item in filtered_tests: # Renamed 'test' to 'test_item' to avoid conflict with module
        platforms_raw = test_item.get(KEY_SUPPORTED_PLATFORMS, "")
        # Split by comma or pipe, filter out empty strings after split, then strip and lower
        current_test_platforms = [p.strip().lower() for p in re.split(r'[|,]', platforms_raw) if p.strip()]
        
        if args.platform == OS_ALL or args.platform in current_test_platforms:
            if OS_WINDOWS in current_test_platforms and (args.platform == OS_ALL or args.platform == OS_WINDOWS):
                test_tasks.append((test_item, OS_WINDOWS))
            if OS_LINUX in current_test_platforms and (args.platform == OS_ALL or args.platform == OS_LINUX):
                test_tasks.append((test_item, OS_LINUX))

    # Execute tests in parallel using ThreadPoolExecutor
    logger.info(f"Starting execution of {len(test_tasks)} test(s) with {WORKERS_COUNT} worker threads")
    with ThreadPoolExecutor(max_workers=WORKERS_COUNT) as executor:
        futures = [executor.submit(run_single_test, test, os_type) 
                  for test, os_type in test_tasks]
        for future in futures:
            future.result()

    # Collect all results from the queue
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())

    # Close all connections
    logger.info("Closing all connections")
    for ssh in linux_ssh_connections.values():
        try:
            ssh.close()
        except Exception as e:
            logger.error(f"Error closing Linux SSH connection: {e}")
            
    for ssh in windows_ssh_connections.values():
        try:
            ssh.close()
        except Exception as e:
            logger.error(f"Error closing Windows SSH connection: {e}")

    # Close the CSV file
    if csv_file:
        csv_file.close()

    # Calculate statistics (only counting actual test executions, not prereq checks or cleanup)
    test_results = [r for r in results if r[FIELD_ACTION] == ACTION_TEST]
    total_tests = len(test_results)
    successful_tests = len([r for r in test_results if r[FIELD_STATUS] == STATUS_SUCCESS])
    failed_tests = total_tests - successful_tests
    windows_tests = len([r for r in test_results if r[FIELD_OS] == OS_WINDOWS])
    linux_tests = len([r for r in test_results if r[FIELD_OS] == OS_LINUX])

    logger.info("\nTest Statistics:")
    logger.info("---------------")
    logger.info(f"Total tests executed: {total_tests}")
    logger.info(f"Successful tests: {successful_tests}")
    logger.info(f"Failed tests: {failed_tests}")
    logger.info(f"Windows tests: {windows_tests}")
    logger.info(f"Linux tests: {linux_tests}")
    logger.info(f"\nResults saved to {REPORT_CSV}")
    logger.info(f"Log file saved to {log_filename}")

if __name__ == "__main__":
    main()
