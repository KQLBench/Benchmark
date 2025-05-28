# Benchmark Component

This document outlines the usage of the Benchmark component, which is responsible for evaluating the performance of Large Language Models (LLMs) in generating Kusto Query Language (KQL) queries. It uses test scenarios derived from Atomic Red Team tests and measures the effectiveness and accuracy of the generated queries against expected outcomes.

## Directory Structure

```
Benchmark/
├── __init__.py
├── main.py               # Main script to run benchmarks
├── README.md             # This file
├── configuration/
│   └── models_config.py  # Model configurations, LLM parameters, and benchmark settings
├── helpers/              # Utility scripts for the benchmark framework
│   ├── __init__.py
│   ├── get_test_questions.py # Script to load test questions
│   └── ...               # Other helper modules
├── logs/                 # Directory for storing benchmark execution logs
├── models/               # Pydantic models for benchmark data structures
│   ├── __init__.py
│   └── benchmark_models.py # Defines TestCase, TestResult, BenchmarkResult etc.
└── results/              # Default directory for storing benchmark result JSON 
```

## Running the Benchmark

The core functionality of this component is executed via the `main.py` script. This script runs the benchmark tests against a specified LLM, evaluates the generated KQL queries, and outputs the results.

### Prerequisites

*   Ensure that the Azure environment (Log Analytics workspace, permissions) is set up as described in the main KQLBench documentation and the `Ingest` component README.
*   A Python environment with all necessary dependencies installed. Refer to the project's global requirements file or setup guide.
*   The `Benchmark/configuration/models_config.py` file must be correctly configured with the desired LLM endpoints and parameters.
*   Test questions should be accessible by the `get_all_questions` helper, typically sourced from the `AtomicRedTeamTests/questions_checked/` directory.

### Command-Line Arguments

The `main.py` script accepts the following command-line arguments:

| Argument         | Description                                                                      | Choices / Default                     |
|------------------|----------------------------------------------------------------------------------|---------------------------------------|
| `--model`        | Specify which model configuration from `models_config.py` to run. Required unless `--list-models` is used. | (List from `MODELS_CONFIG.keys()`)    |
| `--list-models`  | List available model configurations.                                             | `False`                               |
| `--output`       | Output file for benchmark results.                                               | `results/MODEL_KEY_TIMESTAMP.json`    |
| `--log-level`    | Set the logging level.                                                           | `debug`, `info`, `warning`, `error`, `critical` (Default: `info`) |
| `--workers`      | Number of parallel workers for test execution.                                   | `5`                                   |
| `--tries`        | Number of attempts for query execution per test case.                            | `5`                                   |
| `--test-limit`   | Limit the number of questions to run (0 for no limit).                           | `0`                                   |
| `--query-date`   | Specify the end date for KQL queries (DD.MM.YYYY). Defaults to the last 24 hours from the current time. | `None`                                |

### Examples

**1. List available models:**
```powershell
python main.py --list-models
```

**2. Run benchmark for a specific model:**
This command runs the benchmark for the `gpt-4o` model, using default settings for other parameters. Results will be saved in the `results/` directory.
```powershell
python main.py --model gpt-4o
```

**3. Run benchmark with custom parameters:**
This command runs the benchmark for the `gpt-4.1-finetuned` model, limits execution to 10 test questions, sets the log level to debug, specifies a custom output file, uses 10 workers, and allows 3 tries per query.
```powershell
python main.py --model gpt-4.1-finetuned --test-limit 10 --log-level debug --output results/my_custom_run.json --workers 10 --tries 3
```

## Configuration

Model configurations, including API keys (if managed directly, though Key Vault is recommended), endpoints, and other LLM-specific parameters, are defined in `Benchmark/configuration/models_config.py`. Ensure this file is up-to-date with the models you intend to benchmark.

## Output

The benchmark script generates a JSON file in the `results/` directory (or the path specified by `--output`). This file contains an array of `TestResult` objects, where each object includes:
-   `test_case`: Details of the question, including the original query, expected answer, and metadata.
-   `query_result`: Information about the LLM's attempt, including the generated KQL query, execution status, and any errors.
-   `answer_correct`: A boolean indicating whether the LLM's generated query produced the correct result when compared against the test case's expected answer.
-   `cost`: The estimated cost for the LLM interaction for that specific test case, if applicable and calculable.
