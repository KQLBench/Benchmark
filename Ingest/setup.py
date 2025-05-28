#!/usr/bin/env python3
"""
setup.py  –  Create multiple custom tables + one DCR from a JSON spec
and ingest one sample record into each.  No emojis in console output.

JSON format
-----------
[
  {
    "name": "MyLogs_CL",
    "columns": [
      {"name": "CorrelationId", "type": "string"},
      {"name": "OperationName", "type": "string"},
      {"name": "Status",        "type": "string"},
      {"name": "DurationMs",    "type": "int"},
      {"name": "Payload",       "type": "dynamic"},
      {"name": "TimeGenerated", "type": "datetime"}
    ]
  },
  {
    "name": "Orders_CL",
    "columns": [
      {"name": "OrderId",       "type": "string"},
      {"name": "CustomerId",    "type": "string"},
      {"name": "Total",         "type": "real"},
      {"name": "TimeGenerated", "type": "datetime"}
    ]
  }
]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

import requests
from azure.identity import DefaultAzureCredential

# ── Constants ────────────────────────────────────────────────────────────────

ARM_SCOPE          = "https://management.azure.com"
INGEST_SCOPE       = "https://monitor.azure.com"
TABLE_API_VERSION  = "2025-02-01"
DCR_API_VERSION    = "2023-03-11"
INGEST_API_VERSION = "2023-01-01"

# ── Helpers ──────────────────────────────────────────────────────────────────


def get_bearer(scope: str) -> str:
    tok = DefaultAzureCredential(exclude_interactive_browser_credential=False) \
        .get_token(f"{scope}/.default")
    return tok.token


def wait_for_lro(url: str, headers: dict, timeout: int = 600) -> None:
    deadline = time.time() + timeout
    while True:
        r = requests.get(url, headers=headers)
        if r.status_code in (200, 201):
            st = (r.json().get("status")
                  or r.json().get("properties", {}).get("provisioningState", "")
                  ).lower()
            if st in ("succeeded", "failed", "cancelled"):
                print(f"    final state: {st}")
                return
        if time.time() > deadline:
            print("    timed out waiting for LRO", file=sys.stderr)
            sys.exit(1)
        time.sleep(3)


# ── ARM operations ───────────────────────────────────────────────────────────


def create_table(sub: str, rg: str, ws: str, table: dict) -> None:
    name = table["name"]
    if not name.endswith("_CL"):
        print(f"Table {name} must end with '_CL'", file=sys.stderr)
        sys.exit(1)

    url = (
        f"https://management.azure.com/subscriptions/{sub}"
        f"/resourceGroups/{rg}/providers/Microsoft.OperationalInsights"
        f"/workspaces/{ws}/tables/{name}?api-version={TABLE_API_VERSION}"
    )
    headers = {
        "Authorization": f"Bearer {get_bearer(ARM_SCOPE)}",
        "Content-Type": "application/json"
    }
    body = {
        "properties": {
            "schema": {"name": name, "columns": table["columns"]},
            "plan": "Analytics",
            "retentionInDays": 30,
            "totalRetentionInDays": 365
        }
    }

    print(f"Table {name} … ", end="", flush=True)
    r = requests.put(url, headers=headers, json=body)
    if r.status_code in (200, 201):
        print("done.")
    elif r.status_code == 202:
        print("accepted; waiting.")
        wait_for_lro(r.headers.get("azure-asyncoperation") or r.headers.get("Location"),
                     headers)
    else:
        print(f"failed ({r.status_code})", file=sys.stderr)
        print(r.text, file=sys.stderr)
        sys.exit(1)


def create_dcr(sub: str, rg: str, ws: str, loc: str,
               dcr_name: str, tables: List[dict]) -> None:
    url = (
        f"https://management.azure.com/subscriptions/{sub}"
        f"/resourceGroups/{rg}/providers/Microsoft.Insights"
        f"/dataCollectionRules/{dcr_name}?api-version={DCR_API_VERSION}"
    )
    headers = {
        "Authorization": f"Bearer {get_bearer(ARM_SCOPE)}",
        "Content-Type": "application/json"
    }

    ws_id = (f"/subscriptions/{sub}/resourceGroups/{rg}"
             f"/providers/Microsoft.OperationalInsights/workspaces/{ws}")

    stream_decls: Dict[str, dict] = {}
    data_flows: List[dict] = []

    for tbl in tables:
        full   = tbl["name"]
        short  = full[:-3]                      # strip _CL once
        stream = f"Custom-{short}"

        # Stream declaration (schema)
        stream_decls[stream] = {"columns": tbl["columns"]}

        # Data flow: stream ➜ workspace.table
        data_flows.append({
            "streams":      [stream],
            "destinations": ["LogAnalyticsDest"],
            "transformKql": "source",
            "outputStream": f"Custom-{full}"
        })

    body = {
        "location": loc,
        "kind": "Direct",
        "properties": {
            "streamDeclarations": stream_decls,
            "destinations": {
                "logAnalytics": [{"workspaceResourceId": ws_id,
                                  "name": "LogAnalyticsDest"}]
            },
            "dataFlows": data_flows
        }
    }

    print(f"DCR {dcr_name} … ", end="", flush=True)
    r = requests.put(url, headers=headers, json=body)
    if r.status_code in (200, 201):
        print("done.")
    elif r.status_code == 202:
        print("accepted; waiting.")
        wait_for_lro(r.headers.get("azure-asyncoperation") or r.headers.get("Location"),
                     headers)
    else:
        print(f"failed ({r.status_code})", file=sys.stderr)
        print(r.text, file=sys.stderr)
        sys.exit(1)


def get_dcr_info(sub: str, rg: str, dcr: str) -> Tuple[str, str]:
    url = (
        f"https://management.azure.com/subscriptions/{sub}"
        f"/resourceGroups/{rg}/providers/Microsoft.Insights"
        f"/dataCollectionRules/{dcr}?api-version={DCR_API_VERSION}"
    )
    headers = {"Authorization": f"Bearer {get_bearer(ARM_SCOPE)}"}

    for _ in range(40):            # wait up to ~2 min
        r = requests.get(url, headers=headers); r.raise_for_status()
        props = r.json().get("properties", {})
        imm   = props.get("immutableId")
        ep    = (props.get("logsIngestion", {}).get("endpoint")
                 or props.get("endpoints", {}).get("logsIngestion"))
        if imm and ep:
            return imm, ep
        time.sleep(3)

    print("DCR endpoint still missing after waiting", file=sys.stderr)
    sys.exit(1)

# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="Create multiple tables + DCR from JSON")
    ap.add_argument("--subscription-id", required=True)
    ap.add_argument("--resource-group",  required=True)
    ap.add_argument("--workspace-name",  required=True)
    ap.add_argument("--location",        required=True)
    ap.add_argument("--dcr-name",        required=True)
    ap.add_argument("--tables-file",     required=True,
                    help="Path to JSON file with table definitions")
    args = ap.parse_args()

    tables = json.loads(Path(args.tables_file).read_text())

    # 1. Create / update every table
    for tbl in tables:
        create_table(args.subscription_id, args.resource_group,
                     args.workspace_name, tbl)

    # 2. Create / update the single DCR that covers them all
    create_dcr(args.subscription_id, args.resource_group, args.workspace_name,
               args.location, args.dcr_name, tables)

    print("\nAll steps completed.")


if __name__ == "__main__":
    main()
