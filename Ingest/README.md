# Ingest Component

This document outlines the procedures for setting up the necessary Azure environment and ingesting security logs for the KQLBench benchmark. It includes steps for configuring Azure resources, managing permissions, and processing log data.

## Directory Structure

```
Ingest/
├── export_logs.py
├── export_schemas.py
├── ingest_logs.py
├── logs/                 # Default directory for storing *.json log files to be ingested
├── README.md             # This file
├── setup.py
└── tables/               # Directory for storing table schema definitions (e.g., tables.json)
```

## Setup Benchmark Environment

Follow these steps to prepare your Azure environment for the benchmark. This involves creating necessary tables, Data Collection Rules (DCRs), and assigning permissions.

### 1. Setup Environment (Tables and DCR)

The `setup.py` script is used to create the required table structures and Data Collection Rule (DCR) in your Azure Log Analytics workspace. This DCR will be used for ingesting logs.

**Important:** The location specified must be the same as your Log Analytics workspace.

**Command:**
```powershell
python setup.py \\
  --subscription-id <SUBSCRIPTION_ID> \\
  --resource-group <RESOURCE_GROUP> \\
  --workspace-name <WORKSPACE_NAME> \\
  --location <LOCATION> \\
  --dcr-name <DCR_NAME> \\
  --tables-file <PATH_TO_TABLES_FILE>
```

**Example:**
```powershell
python setup.py --subscription-id aaaaaaa-aaaa-aaaa-bbbb-bbbbbbbbbbbb --resource-group benchmark_resource --workspace-name benchmark_workspace --location swedencentral --dcr-name MyLogsDCR --tables-file tables.json
```
This command sets up the environment using `tables.json` for schema definitions in the `swedencentral` region, within the `benchmark_resource` group and `benchmark_workspace`.

### 2. Configure Lab Permissions

Proper permissions are required for the scripts to interact with your Azure resources. Assign the following roles to the service principal or user account that will be running the scripts.

#### a. Workspace Permissions (Table Operations)

This permission allows creating and managing tables in the Log Analytics workspace.

**Command:**
```powershell
az role assignment create \\
  --assignee <ASSIGNEE_OBJECT_ID> \\
  --role "Log Analytics Contributor" \\
  --scope /subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<RESOURCE_GROUP>/providers/Microsoft.OperationalInsights/workspaces/<WORKSPACE_NAME>
```

**Example:**
```powershell
az role assignment create --assignee 9a0959fb-b383-4c25-9728-10d9e31b63bc --role "Log Analytics Contributor" --scope /subscriptions/aaaaaaa-aaaa-aaaa-bbbb-bbbbbbbbbbbb/resourceGroups/wipro/providers/Microsoft.OperationalInsights/workspaces/benchmark_workspace
```

#### b. Data Collection Rule (DCR) Permissions

This permission allows managing and using the DCR for log ingestion.

**Command:**
```powershell
az role assignment create \\
  --assignee <ASSIGNEE_OBJECT_ID> \\
  --role "Monitoring Contributor" \\
  --scope /subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<RESOURCE_GROUP>/providers/Microsoft.Insights/dataCollectionRules/<DCR_NAME>
```

**Example:**
```powershell
az role assignment create --assignee 9a0959fb-b383-4c25-9728-10d9e31b63bc --role "Monitoring Contributor" --scope /subscriptions/aaaaaaa-aaaa-aaaa-bbbb-bbbbbbbbbbbb/resourceGroups/wipro/providers/Microsoft.Insights/dataCollectionRules/MyLogsDCR
```

Additionally, assign the "Monitoring Metrics Publisher" role to allow publishing metrics for the DCR.
**Command:**
```powershell
az role assignment create \\
  --assignee <ASSIGNEE_OBJECT_ID> \\
  --role "Monitoring Metrics Publisher" \\
  --scope /subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<RESOURCE_GROUP>/providers/Microsoft.Insights/dataCollectionRules/<DCR_NAME>
```
**Example:**
```powershell
az role assignment create --assignee 8adab05f-ce2b-4753-954f-f7a18c767b65 --role "Monitoring Metrics Publisher" --scope /subscriptions/aaaaaaa-aaaa-aaaa-bbbb-bbbbbbbbbbbb/resourceGroups/wipro/providers/Microsoft.Insights/dataCollectionRules/MyLogsDCR
```

### 3. Ingest Logs

Once the tables and DCR are configured and permissions are set, you can ingest your logs using the `ingest_logs.py` script.

This script processes all `*.json` files located in the specified logs directory (defaults to `./logs/`). It intelligently adjusts record timestamps to ensure they appear recent relative to the current time. To comply with Azure API limit, the script automatically splits large log files into smaller chunks before uploading.

**Command:**
```powershell
python ingest_logs.py \\
  --subscription-id <SUBSCRIPTION_ID> \\
  --resource-group <RESOURCE_GROUP> \\
  --dcr-name <DCR_NAME> \\
  --logs-dir [PATH_TO_LOGS_DIRECTORY]
```

**Example (using default `./logs/` directory):**
```powershell
python ingest_logs.py \\
  --subscription-id aaaaaaa-aaaa-aaaa-bbbb-bbbbbbbbbbbb \\
  --resource-group benchmark_resource \\
  --dcr-name MyLogsDCR
```

**Example (specifying a custom logs directory):**
```powershell
python ingest_logs.py \\
  --subscription-id aaaaaaa-aaaa-aaaa-bbbb-bbbbbbbbbbbb \\
  --resource-group benchmark_resource \\
  --dcr-name MyLogsDCR \\
  --logs-dir /path/to/my/exported_logs
```

## Extract Logs (Optional)

These steps are for users who wish to use different table structures or extract their own logs from an existing Log Analytics workspace.

### 1. Export Table Schemas

Use the `export_schemas.py` script to export the schemas of the tables you intend to use. The output is a JSON file that will be used by the `setup.py` script.

**Command:**
```powershell
python export_schemas.py \\
  --subscription-id <SUBSCRIPTION_ID> \\
  --resource-group <RESOURCE_GROUP> \\
  --workspace-name <WORKSPACE_NAME> \\
  --tables "<TABLE_NAME_1>,<TABLE_NAME_2>,..." \\
  --output-file <OUTPUT_FILENAME>.json
```

**Example:**
```powershell
python export_schemas.py --subscription-id aaaaaaa-aaaa-aaaa-bbbb-bbbbbbbbbbbb --resource-group benchmark_resource --workspace-name benchmark_workspace --tables "DeviceEvents,DeviceFileCertificateInfo,DeviceFileEvents,DeviceImageLoadEvents,DeviceInfo,DeviceLogonEvents,DeviceNetworkEvents,DeviceNetworkInfo,DeviceProcessEvents,DeviceRegistryEvents" --output-file tables.json
```
This command exports schemas for the specified tables from `benchmark_workspace` into `tables.json`.

### 2. Export Logs

After exporting schemas, use the `export_logs.py` script to export log data from a specific table within a defined time range.

**Command:**
```powershell
python export_logs.py \\
  --subscription-id <SUBSCRIPTION_ID> \\
  --resource-group <RESOURCE_GROUP> \\
  --workspace-name <WORKSPACE_NAME> \\
  --table <TABLE_NAME> \\
  --start-time <YYYY-MM-DDTHH:MM:SSZ> \\
  --end-time <YYYY-MM-DDTHH:MM:SSZ> \\
  --output-file <OUTPUT_FILENAME>.json
```

**Example:**
```powershell
python export_logs.py --subscription-id aaaaaaa-aaaa-aaaa-bbbb-bbbbbbbbbbbb --resource-group benchmark_resource --workspace-name benchmark_workspace --table DeviceEvents --start-time 2025-04-01T06:00:00Z --end-time 2025-04-01T06:05:00Z --output-file DeviceEvents.json
```
This command exports `DeviceEvents` logs from `benchmark_workspace` between the specified timestamps into `DeviceEvents.json`.
