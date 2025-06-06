# KQLBench: Benchmarking KQL Generation for Security

KQLBench is a benchmark designed to test how well Large Language Models (LLMs) can write Kusto Query Language (KQL) queries for cybersecurity use cases. The tests are based on real-world scenarios provided by Atomic Red Team, ensuring practical and relevant results.

Each Atomic Red Team test is run in a controlled lab environment to generate realistic security logs. These logs are collected, prepared, and reloaded into a workspace to ensure consistency and simplicity during testing.

For each scenario, KQLBench sets clear expected results. Instead of just checking whether a query is correct, it verifies if the query actually produces the expected outcomes.

LLMs using KQLBench can directly execute their generated queries in the provided log workspace. This setup enables immediate feedback and clear insight into the practical usefulness of the queries.

KQLBench helps security analysts and researchers easily evaluate how effective LLMs are at creating useful KQL queries for threat detection and analysis.

## Components

KQLBench is comprised of four key components, each playing a vital role in the benchmarking process:

1.  **[Atomic Red Team Tests](AtomicRedTeamTests)**:
    Manages the execution of Atomic Red Team tests to generate realistic security log data. This component includes scripts for setting up virtual machine environments and automating the detonation of specified tests, forming the foundation for the benchmark's scenarios.

2.  **[Benchmark](Benchmark)**:
    Houses the core benchmarking framework. This includes the logic for evaluating LLM-generated KQL queries against the collected security logs, measuring their effectiveness and accuracy based on predefined expected outcomes for each test scenario.

3.  **[Dataset](Dataset)**:
    Contains resources for fine-tuning Large Language Models, including datasets and test runs (e.g., `gpt-4.1_finetuning.csv`, `gpt-4o_finetuning.csv`).

4.  **[Ingest](Ingest)**:
    Handles the data pipeline for the benchmark. This component includes scripts to extract table schemas from the target environment, process raw logs generated by the Atomic Red Team tests, set up the necessary data structures (like schemas and Data Collection Rules), and ingest the prepared logs into the analysis workspace with realistic timestamps.