from datetime import datetime, timedelta, UTC
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from azure.identity import DefaultAzureCredential
from azure.monitor.query import LogsQueryClient, LogsQueryStatus
from Benchmark.helpers.logging_config import get_logger
import os
from dotenv import load_dotenv

# Get module-specific logger
logger = get_logger(__name__)


tables_to_exclude = [
    "MyLogs_CL",
    "Orders_CL",
    "Usage"
]

class LogAnalyticsConnector:
    def __init__(self, workspace_id: str):
        self.workspace_id = workspace_id
        self.credential = DefaultAzureCredential()
        self.client = LogsQueryClient(self.credential)

    def get_table_fields(self, table_name: str) -> List[str]:
        """
        Returns all fields of a table as a list of formatted strings.
        """
        query = f"{table_name} | limit 1"
        
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(days=30)
        timespan = (start_time, end_time)
        
        logger.debug(f"Getting fields for table: {table_name}")
        try:
            result = self.client.query_workspace(self.workspace_id, query, timespan=timespan)
            
            if result.status == LogsQueryStatus.SUCCESS and result.tables:
                table = result.tables[0]
                if not table.columns:
                    logger.info(f"Table '{table_name}' has no columns.")
                    return []
                field_strings = [f"{col}: {col_type}" for col, col_type in 
                               sorted(zip(table.columns, table.columns_types), key=lambda x: x[0])]
                return field_strings
            elif result.status == LogsQueryStatus.PARTIAL:
                 logger.warning(f"Partially successful in getting fields for table: {table_name}. Error: {result.error if result.error else 'N/A'}")
                 return []
            else:
                logger.warning(f"No fields found or query failed for table: {table_name}. Status: {result.status}, Error: {result.error if result.error else 'N/A'}")
                return []
        except Exception as e:
            logger.error(f"Exception in get_table_fields for table '{table_name}': {str(e)}")
            return []


    def get_available_tables(self) -> List[str]:
        """
        Ruft alle verfügbaren Tabellen in einem Azure Log Analytics Workspace ab und gibt sie als Liste zurück.
        """
        query = "search * | summarize by $table"
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(days=30)
        timespan = (start_time, end_time)
        
        logger.debug("Getting available tables in workspace")
        result = self.client.query_workspace(self.workspace_id, query, timespan=timespan)
        if result.status == LogsQueryStatus.SUCCESS and result.tables:
            return result
        logger.warning("No tables found in workspace")
        return []

    def get_all_table_fields(self) -> dict:
        """
        Ruft alle verfügbaren Tabellen ab und gibt deren Feldinformationen parallel aus.
        Gibt ein dict zurück, bei dem die Werte Listen von Feldstrings sind.
        """
        result = self.get_available_tables()

        
        table_fields = {}
        
        table_names_to_query = []
        if hasattr(result, 'status') and result.status == LogsQueryStatus.SUCCESS and result.tables:
            table = result.tables[0]
            table_names_to_query = [row[0] for row in table.rows if row[0] not in tables_to_exclude]
        elif isinstance(result, list) and not result:
             logger.info("No tables available to get fields for.")
        elif not hasattr(result, 'status'):
            table_names_to_query = [name for name in result if name not in tables_to_exclude]

        if table_names_to_query:
            # Use ThreadPoolExecutor for parallel execution
            with ThreadPoolExecutor(max_workers=10) as executor:
                # Create a dictionary of future: table_name
                future_to_table = {
                    executor.submit(self.get_table_fields, name): name 
                    for name in table_names_to_query
                }
                
                # Process completed futures as they finish
                for future in as_completed(future_to_table):
                    table_name = future_to_table[future]
                    try:
                        table_fields[table_name] = future.result()
                    except Exception as e:
                        logger.error(f"Error processing future for table {table_name}: {str(e)}")
                        table_fields[table_name] = []
                        
        return table_fields

    def run_custom_query(self, query: str, timespan: tuple[datetime, datetime] = None) -> List[List]:
        """
        Führt eine benutzerdefinierte KQL-Abfrage aus.
        
        Args:
            query: Die KQL-Abfrage
            timespan: Optional tuple of (start_time, end_time) as datetime objects
            row_limit: Maximale Anzahl der zurückzugebenden Zeilen
        """
        
        logger.info(f"Executing KQL query from {timespan[0]} to {timespan[1]}")
        logger.debug(f"Full query:\n{query}")
        result = self.client.query_workspace(self.workspace_id, query, timespan=timespan)
        if result.status == LogsQueryStatus.SUCCESS and result.tables:
            output = []
            for table in result.tables:
                # Convert columns to list of plain strings
                columns = [str(col) for col in table.columns]
                output.append(columns)
                
                # Convert each row to a list of native Python types
                for row in table.rows:
                    # Convert each cell to a native Python type
                    python_row = []
                    for cell in row:
                        if isinstance(cell, (str, int, float, bool, type(None))):
                            python_row.append(cell)
                        else:
                            # Convert any complex types to strings
                            python_row.append(str(cell))
                    output.append(python_row)
                    
            return output
        else:
            logger.error(f"Query failed: {result.error}")
            raise Exception(f"Query failed: {result.error}")


if __name__ == "__main__":
    load_dotenv()  # Ladt Variable us .env Datei
    # Workspace ID us de Umgäbigsvariable lade
    workspace_id = os.getenv("WORKSPACE_ID")
    
    if not workspace_id:
        print("Fehler: WORKSPACE_ID isch nid i de Umgäbigsvariable (.env) gsetzt.")
        print("Bitte füeg WORKSPACE_ID zu diner .env Datei hinzu.")
    else:
        print(f"Benutze Workspace ID: {workspace_id}")
        querier = LogAnalyticsConnector(workspace_id)
        
        # Focus on Defender tables - use their actual names with _CL suffix for this direct test
    defender_tables_actual_names = [
        "DeviceEvents_CL",
        "DeviceProcessEvents_CL",
        "DeviceNetworkEvents_CL",
        "DeviceFileEvents_CL",
        "DeviceRegistryEvents_CL",
        "DeviceImageLoadEvents_CL",
        "DeviceLogonEvents_CL"
        # Add other _CL tables if needed for this test
    ]
    
    all_fields_results = querier.get_all_table_fields() # This returns actual names as keys, values are List[str]

    print("Fields for specific Defender tables (actual names with _CL):")
    for actual_table_name in defender_tables_actual_names:
        if actual_table_name in all_fields_results:
            print(f"\n{actual_table_name}:")
            fields_list = all_fields_results[actual_table_name]
            if fields_list:
                print("\n".join(fields_list))
            else:
                print("  No fields found or error retrieving fields.")
        else:
            print(f"\n{actual_table_name}: Table not found in results or no fields retrieved.")
