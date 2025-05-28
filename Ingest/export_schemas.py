#!/usr/bin/env python3
"""
get_builtin_table_defs.py – Fetch column schemas for built-in Log Analytics
tables and write them to JSON.

Usage
-----
python get_builtin_table_defs.py \
    --subscription-id <SUB> \
    --resource-group  <RG> \
    --workspace-name  <WS> \
    [--tables Heartbeat AzureActivity] \
    [--output-file builtins.json]

If --tables is omitted, all built-in tables in the workspace are queried.

Notes
-----
* Auth and ARM scope identical to the custom-table helper.
* Requires an Azure identity resolvable by DefaultAzureCredential.
* Console output is plain ASCII – no emojis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import requests
from azure.identity import DefaultAzureCredential

# ── Constants ────────────────────────────────────────────────────────────────

ARM_SCOPE   = "https://management.azure.com"
API_VERSION = "2025-02-01"


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_bearer(scope: str) -> str:
    """Return an AAD bearer token for *scope*."""
    token = (
        DefaultAzureCredential(exclude_interactive_browser_credential=False)
        .get_token(f"{scope}/.default")
    )
    return token.token


def list_tables(subscription_id: str, resource_group: str,
                workspace_name: str) -> List[str]:
    """
    Return every table name in the workspace, regardless of type.

    Built-ins are later distinguished by absence of the _CL suffix.
    """
    url = (
        f"https://management.azure.com/subscriptions/{subscription_id}"
        f"/resourceGroups/{resource_group}/providers/Microsoft.OperationalInsights"
        f"/workspaces/{workspace_name}/tables"
        f"?api-version={API_VERSION}"
    )
    headers = {"Authorization": f"Bearer {get_bearer(ARM_SCOPE)}"}
    resp = requests.get(url, headers=headers)
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        print("Failed to list tables.", file=sys.stderr)
        print(resp.text, file=sys.stderr)
        sys.exit(1)

    items = resp.json().get("value", [])
    return [item.get("name") for item in items if "name" in item]


def fetch_table(subscription_id: str, resource_group: str, workspace_name: str,
                table_name: str) -> Dict[str, Any]:
    """Return the column schema of *table_name* or an empty dict on error."""
    url = (
        f"https://management.azure.com/subscriptions/{subscription_id}"
        f"/resourceGroups/{resource_group}/providers/Microsoft.OperationalInsights"
        f"/workspaces/{workspace_name}/tables/{table_name}"
        f"?api-version={API_VERSION}"
    )
    headers = {"Authorization": f"Bearer {get_bearer(ARM_SCOPE)}"}
    resp = requests.get(url, headers=headers)
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        print(f"Failed to fetch {table_name} – HTTP {resp.status_code}",
              file=sys.stderr)
        print(resp.text, file=sys.stderr)
        return {}

    props   = resp.json().get("properties", {})
    
    schema = props.get("schema", {})
    
    return {"name": table_name, "columns": schema.get("standardColumns", []) + schema.get("columns", [])}


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch built-in Log Analytics table schemas")
    ap.add_argument("--subscription-id", required=True)
    ap.add_argument("--resource-group",  required=True)
    ap.add_argument("--workspace-name",  required=True)
    ap.add_argument("--tables", nargs="+", metavar="TABLE",
                    help="Specific built-in tables to query (default: all)")
    ap.add_argument("--output-file",
                    help="Path to write JSON output (default stdout)")
    args = ap.parse_args()

    # Assemble the list of target tables
    if args.tables:
        targets = args.tables
    else:
        all_tables = list_tables(args.subscription_id, args.resource_group,
                                 args.workspace_name)
        # Keep only tables that don't look like custom logs
        targets = [t for t in all_tables if not t.endswith("_CL")]

    if not targets:
        print("No built-in tables found to query.", file=sys.stderr)
        sys.exit(1)

    definitions: List[Dict[str, Any]] = []
    for tbl in targets:
        if tbl.endswith("_CL"):
            print(f"Skipping custom table {tbl}.", file=sys.stderr)
            continue
        schema = fetch_table(args.subscription_id, args.resource_group,
                             args.workspace_name, tbl)
        if schema:
            definitions.append(schema)

    if not definitions:
        print("No table definitions retrieved.", file=sys.stderr)
        sys.exit(1)

    json_text = json.dumps(definitions, indent=2)
    if args.output_file:
        Path(args.output_file).write_text(json_text)
    else:
        print(json_text)


if __name__ == "__main__":
    main()
