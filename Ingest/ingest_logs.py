#!/usr/bin/env python3
"""
ingest_logs.py – Adjusts timestamps and ingests JSON records into a DCR.

Timestamps in the input file are adjusted relative to the latest record,
setting its primary timestamp to the current time while preserving deltas,
before ingesting via an Azure Data Collection Rule (DCR).
This script processes all *.json files found in the specified logs directory.

Usage
-----
python ingest_logs.py \\
    --subscription-id  <SUB> \\
    --resource-group   <RG> \\
    --dcr-name         <DCR_NAME> \\
    --logs-dir         [PATH_TO_LOGS_DIRECTORY] (default: ./logs)

Each JSON file must be an array of objects, each containing a "Type" field
matching the target table (e.g. "DeviceFileEvents") and timestamp fields
in the format "YYYY-MM-DDTHH:MM:SS.ffffffZ".

Notes
-----
* Requires an Azure identity resolvable by DefaultAzureCredential.
* Console output is plain ASCII – no emojis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timezone, timedelta # Restored timedelta, kept timezone

import requests
from azure.identity import DefaultAzureCredential
import gzip

# ── Constants ────────────────────────────────────────────────────────────────

ARM_SCOPE           = "https://management.azure.com"
INGEST_SCOPE        = "https://monitor.azure.com"
INGEST_API_VERSION  = "2023-01-01"
TIMESTAMP_FORMAT    = "%Y-%m-%dT%H:%M:%S.%fZ" # Format for parsing/formatting timestamps
MAX_COMPRESSED_PAYLOAD_BYTES = 900 * 1024 # 900 KB to be safe, well under 1MB API limit

# ── Timestamp Handling Helpers ───────────────────────────────────────────────

def parse_timestamp_robust(ts_str: Any) -> Any | datetime:
    """
    Attempts to parse a string into a datetime object using the specific format.
    Handles fractional seconds with more or less than 6 digits.
    Returns the original value if not a string or if parsing fails/format incorrect.
    """
    if not isinstance(ts_str, str):
        return ts_str

    try:
        if 'T' not in ts_str or not ts_str.endswith('Z'):
            return ts_str

        dot_index = ts_str.rfind('.')
        z_index = ts_str.rfind('Z')

        if dot_index != -1 and z_index != -1 and z_index > dot_index:
            fractional_part = ts_str[dot_index + 1:z_index]
            if len(fractional_part) > 6:
                # Truncate to 6 digits for strptime
                parseable_str = ts_str[:dot_index + 7] + 'Z'
            elif len(fractional_part) < 6:
                 # Pad with zeros to ensure 6 digits for %f
                 parseable_str = ts_str[:z_index] + '0'*(6-len(fractional_part)) + 'Z'
            else: # Exactly 6 digits
                 parseable_str = ts_str

            dt_object = datetime.strptime(parseable_str, TIMESTAMP_FORMAT)
            return dt_object.replace(tzinfo=timezone.utc) # Make it timezone-aware (UTC)
        else:
            return ts_str # Doesn't have the expected '.<digits>Z' structure

    except (ValueError, TypeError):
        return ts_str # Parsing failed

def format_timestamp(ts: Any) -> Any | str:
    """Formats a datetime object back into the ISO 8601 string format with 6 microsecond digits and Z."""
    if isinstance(ts, datetime):
        # Format to 6 microseconds and add the 'Z'
        return ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond:06d}Z"
    return ts # Return original if not a datetime object

def process_timestamps_recursive(data: Any, operation_func) -> Any:
    """
    Recursively traverses data, applying 'operation_func' (parse or format)
    to relevant values (strings for parse, datetimes for format).
    """
    if isinstance(data, dict):
        return {key: process_timestamps_recursive(value, operation_func) for key, value in data.items()}
    elif isinstance(data, list):
        return [process_timestamps_recursive(item, operation_func) for item in data]
    # Apply parsing only to strings
    elif isinstance(data, str) and operation_func == parse_timestamp_robust:
        return operation_func(data)
    # Apply formatting only to datetimes
    elif isinstance(data, datetime) and operation_func == format_timestamp:
         return operation_func(data)
    # Return other types or types not matching the operation unchanged
    else:
        return data

# ── Azure Interaction Helpers ────────────────────────────────────────────────

def get_bearer(scope: str) -> str:
    """Return an AAD bearer token for *scope*."""
    try:
        # Use exclude_interactive unless explicitly needed for user login flow
        token = (
            DefaultAzureCredential(exclude_interactive_browser_credential=True)
            .get_token(f"{scope}/.default")
        )
        return token.token
    except Exception as e:
        print(f"Failed to get Azure token for scope {scope}: {e}", file=sys.stderr)
        print("Ensure you are logged in via Azure CLI, VS Code, or have environment variables set.", file=sys.stderr)
        sys.exit(1)


def get_dcr_info(subscription_id: str,
                 resource_group: str,
                 dcr_name: str) -> tuple[str, str]:
    """
    Return the immutableId and logs ingestion endpoint for the given DCR.
    """
    url = (
        f"{ARM_SCOPE}/subscriptions/{subscription_id}"
        f"/resourceGroups/{resource_group}"
        f"/providers/Microsoft.Insights/dataCollectionRules/{dcr_name}"
        f"?api-version=2023-03-11" # Use a reasonably recent API version
    )
    headers = {"Authorization": f"Bearer {get_bearer(ARM_SCOPE)}"}
    print(f"Fetching DCR info for '{dcr_name}'...")
    try:
        resp = requests.get(url, headers=headers, timeout=30) # Added timeout
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        print("Request timed out while fetching DCR info.", file=sys.stderr)
        sys.exit(1)
    except requests.HTTPError:
        print(f"Failed to fetch DCR info – HTTP {resp.status_code}", file=sys.stderr)
        print(f"URL: {url}", file=sys.stderr)
        try:
            print(f"Response: {resp.text}", file=sys.stderr)
        except Exception:
             print("Response body could not be decoded.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred fetching DCR info: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        dcr_data = resp.json()
        props = dcr_data.get("properties", {})
        imm   = props.get("immutableId")
        # Handle potential variations in endpoint property names
        ep    = (props.get("logsIngestion", {}).get("endpoint")
                or props.get("endpoints", {}).get("logsIngestion"))
        if not imm or not ep:
            print("DCR immutableId or logs ingestion endpoint missing in response.", file=sys.stderr)
            print(f"Response properties: {props}", file=sys.stderr)
            sys.exit(1)
        print("DCR info retrieved successfully.")
        return imm, ep
    except json.JSONDecodeError:
        print("Failed to decode JSON response while fetching DCR info.", file=sys.stderr)
        print(f"Response text: {resp.text}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error processing DCR info response: {e}", file=sys.stderr)
        sys.exit(1)


def post_records(immutable: str,
                 endpoint: str,
                 table_name: str,
                 records: List[Dict]) -> None:
    """
    POST a batch of records to the DCR stream for *table_name*.
    """
    if not records:
        print(f"No records to ingest for {table_name}.")
        return

    stream = f"Custom-{table_name}" # Assuming custom tables
    # strip https:// if present
    host = endpoint.removeprefix("https://")
    url  = (
        f"https://{host}/dataCollectionRules/{immutable}"
        f"/streams/{stream}?api-version={INGEST_API_VERSION}"
    )
    headers = {
        "Authorization": f"Bearer {get_bearer(INGEST_SCOPE)}",
        "Content-Type": "application/json",
        "Content-Encoding": "gzip" # Added Gzip compression
    }

    try:
        payload_bytes = json.dumps(records).encode('utf-8')
        compressed_payload = gzip.compress(payload_bytes)
    except Exception as e:
        print(f"Failed to serialize or compress payload for {table_name}: {e}", file=sys.stderr)
        return # Don't attempt post if payload fails

    print(f"Ingesting {len(records)} records to {table_name} stream...")
    try:
        resp = requests.post(url, headers=headers, data=compressed_payload, timeout=60) # Increased timeout
        status = "succeeded" if resp.status_code in (200, 204) else f"failed ({resp.status_code})"
        print(f"Ingest {table_name}: {status}")
        if resp.status_code not in (200, 204):
            try:
                print(f"Response: {resp.text}", file=sys.stderr)
            except Exception:
                print("Response body could not be decoded.", file=sys.stderr)
    except requests.exceptions.Timeout:
        print(f"Request timed out while ingesting data for {table_name}.", file=sys.stderr)
    except requests.exceptions.RequestException as e:
         print(f"Network error during ingestion for {table_name}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred during ingestion for {table_name}: {e}", file=sys.stderr)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Adjust timestamps and ingest JSON logs into a DCR.",
        formatter_class=argparse.RawDescriptionHelpFormatter # Preserve formatting
        )
    ap.add_argument("--subscription-id", required=True, help="Azure Subscription ID")
    ap.add_argument("--resource-group",  required=True, help="Azure Resource Group name")
    ap.add_argument("--dcr-name",        required=True, help="Name of the existing Data Collection Rule")
    ap.add_argument("--logs-dir",        default="Ingest/logs", help="Directory containing JSON log files to ingest (default: Ingest/logs)")
    args = ap.parse_args()

    # --- Retrieve DCR Details (once) ---
    immutable, endpoint = get_dcr_info(
        args.subscription_id, args.resource_group, args.dcr_name
    )

    logs_directory = Path(args.logs_dir)
    if not logs_directory.is_dir():
        print(f"Error: Logs directory not found at {logs_directory}", file=sys.stderr)
        sys.exit(1)

    json_files_to_process = list(logs_directory.glob('*.json'))
    if not json_files_to_process:
        print(f"No *.json files found in directory: {logs_directory}")
        sys.exit(0)
    
    print(f"Found {len(json_files_to_process)} JSON files to process in {logs_directory}.")

    for current_file_path in json_files_to_process:
        print(f"\n--- Processing file: {current_file_path} ---")

        # --- 1. Load Raw Records ---
        print(f"Reading log entries from: {current_file_path}")
        if not current_file_path.is_file(): # Should not happen with glob but good practice
            print(f"Error: Input file not found at {current_file_path}", file=sys.stderr)
            continue # Skip to next file
        try:
            all_recs_raw = json.loads(current_file_path.read_text(encoding='utf-8')) # Specify encoding
            if not isinstance(all_recs_raw, list):
                print(f"Error: Input file content must be a JSON array in {current_file_path}. Skipping.", file=sys.stderr)
                continue # Skip to next file
        except json.JSONDecodeError as exc:
            print(f"Error parsing JSON from input file {current_file_path}: {exc}. Skipping.", file=sys.stderr)
            continue # Skip to next file
        except Exception as exc:
            print(f"Error reading input file {current_file_path}: {exc}. Skipping.", file=sys.stderr)
            continue # Skip to next file
        
        print(f"Read {len(all_recs_raw)} records from {current_file_path}.")
        if not all_recs_raw:
            print(f"Input file {current_file_path} is empty. Nothing to ingest.")
            continue # Skip to next file

        # --- 2. Parse Timestamps ---
        print("Parsing timestamps...")
        parsed_recs = process_timestamps_recursive(all_recs_raw, parse_timestamp_robust)
        print("Parsing complete.")

        # --- 3. Sort by Timestamp ---
        print("Sorting records by timestamp...")
        parsed_recs.sort(
            key=lambda x: x.get("Timestamp") if isinstance(x.get("Timestamp"), datetime) else datetime.min,
            reverse=True # Latest first
        )
        print("Sorting complete.")

        # --- 4. Calculate Time Shift ---
        anchor_entry = parsed_recs[0]
        original_anchor_dt = anchor_entry.get("Timestamp")

        if not isinstance(original_anchor_dt, datetime):
            print("Error: Could not find a valid 'Timestamp' field in the latest record.", file=sys.stderr)
            print("Cannot perform timestamp adjustment. Check input data format.", file=sys.stderr)
            # Decide whether to proceed without adjustment or exit
            # For safety, let's exit if adjustment is expected but can't be performed.
            sys.exit(1)

        # Use UTC time for the new anchor, consistent with 'Z' notation
        # new_anchor_dt = datetime.utcnow()
        new_anchor_dt = datetime.now(timezone.utc) # Use timezone-aware UTC datetime
        global_shift_delta = new_anchor_dt - original_anchor_dt

        print(f"Original Anchor Timestamp: {format_timestamp(original_anchor_dt)}")
        print(f"New Anchor Timestamp:      {format_timestamp(new_anchor_dt)}")
        print(f"Calculated Shift (Delta):  {global_shift_delta}")

        # --- 5. Adjust Timestamps ---
        print("Adjusting timestamps...")
        # 5a. Adjust anchor entry (preserving internal deltas)
        for key, value in anchor_entry.items():
            if isinstance(value, datetime):
                if key == "Timestamp":
                    anchor_entry[key] = new_anchor_dt
                else:
                    try:
                        original_internal_delta = value - original_anchor_dt
                        new_internal_dt = new_anchor_dt + original_internal_delta
                        anchor_entry[key] = new_internal_dt
                    except TypeError:
                        # Should not happen if parsing worked, but safety check
                        print(f"Warning: Could not calculate delta for key '{key}' in anchor. Skipping adjustment.", file=sys.stderr)


        # 5b. Adjust other entries (applying global shift)
        for i, entry in enumerate(parsed_recs):
            if i == 0: continue # Skip anchor
            for key, value in entry.items():
                if isinstance(value, datetime):
                    try:
                        entry[key] = value + global_shift_delta
                    except TypeError:
                        print(f"Warning: Could not apply global shift for key '{key}' in record {i+1}. Skipping adjustment.", file=sys.stderr)
        print("Timestamp adjustment complete.")

        # --- 6. Format Timestamps Back to Strings ---
        print("Formatting adjusted timestamps back to strings...")
        adjusted_recs_serializable = process_timestamps_recursive(parsed_recs, format_timestamp)
        print("Formatting complete.")

        # --- 7. Group Records by Type ---
        print("Grouping records by 'Type'...")
        groups: Dict[str, List[Dict]] = {}
        skipped_count = 0
        for rec in adjusted_recs_serializable: # Use the adjusted records
            tbl = rec.get("Type")
            if not tbl or not isinstance(tbl, str):
                print("Record missing 'Type' field or Type is not a string; skipping.", file=sys.stderr)
                skipped_count += 1
                continue
            groups.setdefault(tbl, []).append(rec)

        if skipped_count > 0:
            print(f"Skipped {skipped_count} records due to missing/invalid 'Type'.")

        if not groups:
            print(f"No valid records remaining after grouping in {current_file_path}. Nothing to ingest for this file.")
            continue # Skip to next file
        print(f"Grouped records from {current_file_path} into {len(groups)} types: {list(groups.keys())}")

        # --- 9. Post Each Batch ---
        for table_name, recs in groups.items():
            current_chunk: List[Dict] = []
            current_chunk_size_compressed = 0 # Approximate size of current chunk (compressed)
            records_processed_for_table = 0

            for i, record in enumerate(recs):
                # Estimate size of adding this record
                # This is an approximation. Actual compression depends on entire payload.
                # For simplicity, we'll check size after adding to chunk and serializing/compressing.
                
                temp_chunk = current_chunk + [record]
                try:
                    payload_bytes = json.dumps(temp_chunk).encode('utf-8')
                    compressed_payload = gzip.compress(payload_bytes)
                    estimated_next_size = len(compressed_payload)
                except Exception as e:
                    print(f"Warning: Could not estimate size for record, skipping record: {e}", file=sys.stderr)
                    continue # Skip this record if it causes issues

                if current_chunk and estimated_next_size > MAX_COMPRESSED_PAYLOAD_BYTES:
                    # Current chunk is full (or adding next record makes it too full)
                    # Post the current_chunk
                    print(f"Chunk for {table_name} reached size {current_chunk_size_compressed} bytes with {len(current_chunk)} records. Posting.")
                    post_records(immutable, endpoint, table_name, current_chunk)
                    records_processed_for_table += len(current_chunk)
                    current_chunk = [record] # Start new chunk with current record
                    try:
                        payload_bytes = json.dumps(current_chunk).encode('utf-8')
                        current_chunk_size_compressed = len(gzip.compress(payload_bytes))
                    except Exception as e: # Should not happen with a single record, but safety
                        print(f"Warning: Could not estimate size for initial record of new chunk: {e}", file=sys.stderr)
                        current_chunk_size_compressed = MAX_COMPRESSED_PAYLOAD_BYTES +1 # Force new chunk if problematic
                else:
                    # Add to current chunk
                    current_chunk = temp_chunk
                    current_chunk_size_compressed = estimated_next_size
                
                # Post the last chunk if it's the end of records for this table
                if i == len(recs) - 1 and current_chunk:
                    print(f"Posting final chunk for {table_name} with {len(current_chunk)} records, size {current_chunk_size_compressed} bytes.")
                    post_records(immutable, endpoint, table_name, current_chunk)
                    records_processed_for_table += len(current_chunk)
            
            print(f"Finished processing {records_processed_for_table}/{len(recs)} records for table {table_name} from file {current_file_path}.")


    print("\n--- All files processed. Ingestion process finished. ---")

if __name__ == "__main__":
    main()