<!-- filepath: d:\Development\HSLU\WIPRO\Benchmark_Temp\Dataset\README.md -->
# Dataset Component

This document describes the datasets used within the KQLBench project, primarily focusing on data for finetuning Large Language Models (LLMs) to generate Kusto Query Language (KQL) queries.

## Directory Structure

```
Dataset/
├── gpt-4.1_finetuning.csv  # Finetuning data for specific GPT models
├── gpt-4o_finetuning.csv   # Finetuning data for specific GPT models
├── README.md               # This file
└── finetuning/             # (Optional) Supporting scripts, logs, or detailed information related to finetuning processes
```

## Finetuning Datasets

The primary purpose of the datasets in this component is to provide high-quality data for finetuning LLMs. This process aims to enhance their ability to accurately and efficiently generate KQL queries based on natural language prompts, particularly for security-related scenarios.

### Files

-   `gpt-4.1_finetuning.csv`: This CSV file contains prompt-completion pairs specifically curated for finetuning models like GPT-4.1.
-   `gpt-4o_finetuning.csv`: This CSV file contains prompt-completion pairs specifically curated for finetuning models like GPT-4o.

### Data Format

The CSV files typically follow a structure containing pairs of prompts and their corresponding ideal KQL query completions. For example, a common structure might include columns such as:

-   `prompt`: A natural language question or a description of a scenario requiring a KQL query.
-   `completion`: The expected KQL query that addresses the prompt.

**Example Row (Illustrative):**
```csv
prompt,completion
"Show all successful user authentications in the last hour.","SigninLogs | where Status.errorCode == 0 and TimeGenerated > ago(1h) | project UserPrincipalName, AppDisplayName, IPAddress, Location"
```
*Note: The actual column names and structure should be confirmed by inspecting the CSV files.*

Alternatively, for models that use a structured message format (like OpenAI's chat models), the CSV might contain JSONL strings, where each line is a JSON object:
**Example JSONL line within a CSV cell (Illustrative):**
`"{"messages": [{"role": "system", "content": "You are a KQL expert."}, {"role": "user", "content": "User prompt..."}, {"role": "assistant", "content": "KQL query..."}]}"`

It is recommended to inspect the header row of the CSV files to determine the exact format.

### Data Source

The data for these finetuning datasets is primarily derived from security scenarios, potentially based on or inspired by Atomic Red Team tests. This ensures that the LLMs are finetuned on relevant and practical examples for generating KQL queries in a security context. The questions and KQL queries are carefully curated to provide effective training material.

### Usage

These CSV files can be used with various LLM finetuning frameworks and APIs, such as:
-   OpenAI's fine-tuning API
-   Hugging Face's training libraries
-   Other custom training scripts

Ensure the data is preprocessed into the specific format required by the chosen finetuning platform. For example, OpenAI expects data in a JSONL format.

## `finetuning/` Subdirectory

The `finetuning/` subdirectory is intended to store supplementary materials related to the finetuning process. This may include:
-   Scripts for data preprocessing or conversion (e.g., CSV to JSONL).
-   Detailed logs from finetuning runs.
-   Configuration files for finetuning jobs.
-   Evaluation scripts or results for finetuned models.

Currently, this directory might be a placeholder or contain specific utility scripts. Refer to its contents for more details.
