#!/usr/bin/env python3
"""
export_logs.py – Export Log Analytics records from one table and time window.

Usage
-----
python export_logs.py \
    --subscription-id  <SUB> \
    --resource-group   <RG> \
    --workspace-name   <WS> \
    --table            <TABLE> \
    --start-time       2025-04-01T00:00:00Z \
    --end-time         2025-04-01T06:00:00Z \
    [--output-file    out.json]

All output is JSON. Requires an Azure identity resolvable by DefaultAzureCredential.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import requests
from azure.identity import DefaultAzureCredential

# ── Constants ────────────────────────────────────────────────────────────────

ARM_SCOPE               = "https://management.azure.com"
API_VERSION             = "2025-02-01"  # for workspace lookup
LOG_API_SCOPE           = "https://api.loganalytics.io"
LOG_QUERY_URL_TEMPLATE  = f"{LOG_API_SCOPE}/v1/workspaces/{{workspace_id}}/query"

# ── Helpers ──────────────────────────────────────────────────────────────────

def get_bearer(scope: str) -> str:
    """Return an AAD bearer token for *scope*."""
    token = (
        DefaultAzureCredential(exclude_interactive_browser_credential=False)
        .get_token(f"{scope}/.default")
    )
    return token.token

def get_workspace_id(subscription_id: str,
                     resource_group: str,
                     workspace_name: str) -> str:
    """
    Return the workspace's customerId (GUID) from the ARM workspace resource.
    """
    url = (
        f"{ARM_SCOPE}/subscriptions/{subscription_id}"
        f"/resourceGroups/{resource_group}"
        f"/providers/Microsoft.OperationalInsights"
        f"/workspaces/{workspace_name}"
        f"?api-version={API_VERSION}"
    )
    headers = {"Authorization": f"Bearer {get_bearer(ARM_SCOPE)}"}
    resp = requests.get(url, headers=headers)
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        print("Failed to get workspace ID.", file=sys.stderr)
        print(resp.text, file=sys.stderr)
        sys.exit(1)

    props = resp.json().get("properties", {})
    cid = props.get("customerId")
    if not cid:
        print("Workspace customerId not found.", file=sys.stderr)
        sys.exit(1)
    return cid

def fetch_logs(workspace_id: str,
               table: str,
               start: str,
               end: str) -> List[Dict[str, Any]]:
    """
    Query the given table for records between start and end,
    return a list of dicts (one per record).
    """
    kql = (
        f"{table}"
        f"| where TimeGenerated >= datetime({start})"
        f" and TimeGenerated <= datetime({end})"
    )
    url = LOG_QUERY_URL_TEMPLATE.format(workspace_id=workspace_id)
    headers = {
        "Authorization": f"Bearer {get_bearer(LOG_API_SCOPE)}",
        "Content-Type": "application/json"
    }
    body = {"query": kql}
    resp = requests.post(url, headers=headers, json=body)
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        print(f"Failed to query logs – HTTP {resp.status_code}", file=sys.stderr)
        print(resp.text, file=sys.stderr)
        sys.exit(1)

    data = resp.json().get("tables", [])
    if not data:
        return []

    tbl = data[0]
    cols = [c["name"] for c in tbl.get("columns", [])]
    rows = tbl.get("rows", [])
    return [dict(zip(cols, row)) for row in rows]

# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Export logs from a Log Analytics table")
    ap.add_argument("--subscription-id", required=True)
    ap.add_argument("--resource-group",  required=True)
    ap.add_argument("--workspace-name",  required=True)
    ap.add_argument("--table",           required=True,
                    help="Analytics table name, e.g. DeviceEvents")
    ap.add_argument("--start-time",      required=True,
                    help="ISO8601, e.g. 2025-04-01T00:00:00Z")
    ap.add_argument("--end-time",        required=True,
                    help="ISO8601, e.g. 2025-04-01T06:00:00Z")
    ap.add_argument("--output-file",
                    help="Path to write JSON output (default stdout)")
    args = ap.parse_args()

    # Validate timestamps
    for ts in (args.start_time, args.end_time):
        try:
            datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            print(f"Invalid ISO8601 timestamp: {ts}", file=sys.stderr)
            sys.exit(1)

    ws_id   = get_workspace_id(
        args.subscription_id, args.resource_group, args.workspace_name
    )
    records = fetch_logs(ws_id, args.table, args.start_time, args.end_time)

    if not records:
        print("No records returned.", file=sys.stderr)

    output = json.dumps(records, indent=2, default=str)
    if args.output_file:
        Path(args.output_file).write_text(output)
    else:
        print(output)

if __name__ == "__main__":
    main()
