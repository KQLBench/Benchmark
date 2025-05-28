import os
import re # Import re for regex operations
import time
import threading
import json # Import json for parsing function arguments
from datetime import datetime, timedelta, UTC
from typing import List, Dict, Any, Optional, Union

# LiteLLM import
import litellm 
# from openai import AzureOpenAI # Removed AzureOpenAI

from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

from Benchmark.helpers.connector_log_analytics import LogAnalyticsConnector
# from Benchmark.configuration.model_prices import model_prices # Removed old model_prices
from Benchmark.configuration.models_config import MODELS_CONFIG # New consolidated config
from Benchmark.helpers.litellm_client_setup import setup_litellm_client # New client setup
from Benchmark.helpers.logging_config import get_logger
from dotenv import load_dotenv

from Benchmark.models.benchmark_models import TestCase, QueryResult # ModelConfig removed

# Get module-specific logger
logger = get_logger(__name__)
load_dotenv()

DEFENDER_TABLES_ORIGINAL = [
    "DeviceEvents",
    "DeviceProcessEvents",
    "DeviceNetworkEvents",
    "DeviceFileEvents",
    "DeviceRegistryEvents",
    "DeviceImageLoadEvents",
    "DeviceLogonEvents",
    "DeviceFileCertificateInfo",
    "DeviceNetworkInfo",
    "DeviceInfo"
]
TABLE_SUFFIX = "_CL"

class QuerySystem():
    def __init__(self, initial_model_name: Optional[str] = None, default_max_tries: int = 1) -> None:
        vault_url = os.getenv("VAULT_URL")
        credential = DefaultAzureCredential()
        self.secret_client = SecretClient(vault_url=vault_url, credential=credential)
        
        self.workspace_id = self.secret_client.get_secret("WORKSPACE-ID").value
        self.querier = LogAnalyticsConnector(self.workspace_id)
        # self.client = AzureOpenAI(...) # Client is now handled by litellm globally
        
        # Set defaults first
        self.model_name_key: Optional[str] = None       # User-facing model key from MODELS_CONFIG (e.g., "gpt-4o")
        self.litellm_model_id: Optional[str] = None   # Actual model ID for litellm (e.g., "azure/gpt-4o")
        self.current_provider: Optional[str] = None
        self.max_tries = 1
        self.reasoning_effort = None
        self.current_model_config: Optional[Dict[str, Any]] = None # Stores the full config dict for the current model

        if initial_model_name:
            # When called from __init__, pass the default_max_tries
            self.configure(initial_model_name, default_max_tries)
        else:
            if MODELS_CONFIG:
                default_model_key = list(MODELS_CONFIG.keys())[0]
                logger.info(f"No initial model provided, configuring with default: {default_model_key}")
                self.configure(default_model_key, default_max_tries)
            else:
                logger.error("MODELS_CONFIG is empty. Cannot configure a default model.")

        self.cost = 0.0
        self.current_solve_cost = 0.0
        self._cost_lock = threading.RLock()
        
        logger.info(f"QuerySystem initializing. Effective LiteLLM model: {self.litellm_model_id}, Max Tries: {self.max_tries}, Reasoning Effort: {self.reasoning_effort}")
        print(f"--- QuerySystem Initialization --- EMM {self.litellm_model_id} ----")

        logger.info("Initializing available tables and fields...")
        self.table_fields_cache = {}
        actual_all_table_fields = self.querier.get_all_table_fields()
        self.display_all_table_fields = {}
        for actual_name, fields in actual_all_table_fields.items():
            display_name = self._map_actual_to_display(actual_name)
            self.display_all_table_fields[display_name] = fields
        self.display_available_tables = sorted(list(self.display_all_table_fields.keys()))
        
        print("\nAvailable Tables:")
        if not self.display_available_tables:
            print("  No tables available.")
        else:
            for table_name in self.display_available_tables:
                num_fields = len(self.display_all_table_fields[table_name])
                print(f"  Table: {table_name}, Fields: {num_fields}")
        print("--- End QuerySystem Initialization ---\n")
        logger.info(f"Initialized fields for {len(self.display_available_tables)} display tables (actual: {len(actual_all_table_fields)})")

    def _map_display_to_actual(self, display_name: str) -> str:
        """Maps a display table name (e.g., DeviceEvents) to its actual name (e.g., DeviceEvents_CL)."""
        if display_name in DEFENDER_TABLES_ORIGINAL:
            return f"{display_name}{TABLE_SUFFIX}"
        return display_name

    def _map_actual_to_display(self, actual_name: str) -> str:
        """Maps an actual table name (e.g., DeviceEvents_CL) to its display name (e.g., DeviceEvents)."""
        if actual_name.endswith(TABLE_SUFFIX):
            original_name = actual_name[:-len(TABLE_SUFFIX)]
            if original_name in DEFENDER_TABLES_ORIGINAL:
                return original_name
        return actual_name

    def _translate_kql_query_to_actual_tables(self, kql_query: str) -> str:
        """Translates table names in a KQL query from display names to actual names."""
        translated_query = kql_query
        for display_name in DEFENDER_TABLES_ORIGINAL:
            actual_name = self._map_display_to_actual(display_name)
            # Use regex word boundaries to replace whole table names only
            translated_query = re.sub(r'\b' + re.escape(display_name) + r'\b', actual_name, translated_query)
        if kql_query != translated_query:
            logger.debug(f"KQL query translated: Original='{kql_query}', Translated='{translated_query}'")
        return translated_query
        
    def _update_cost(self, response) -> None:
        """Calculate and update the cost based on LiteLLM response (thread-safe).
        Relies on response.response_cost if provided by LiteLLM.
        If not, falls back to token usage if available, but without explicit pricing from config.
        """
        cost_delta = 0.0
        input_tokens = 0
        output_tokens = 0

        try:
            # Attempt 1: Directly from response_obj.response_cost
            if hasattr(response, 'response_cost') and response.response_cost is not None:
                cost_delta = response.response_cost
                logger.debug(f"Cost provided by LiteLLM directly via response.response_cost: ${cost_delta:.6f}")
            # Attempt 2: From response_obj._hidden_params["response_cost"] (as per user example)
            elif hasattr(response, '_hidden_params') and isinstance(response._hidden_params, dict) and response._hidden_params.get("response_cost") is not None:
                cost_delta = response._hidden_params["response_cost"]
                logger.debug(f"Cost provided by LiteLLM via response._hidden_params['response_cost']: ${cost_delta:.6f}")
            else:
                # Attempt 3: Calculate using litellm.completion_cost() with extracted values
                logger.debug("response.response_cost and response._hidden_params['response_cost'] not found or None. Attempting litellm.completion_cost().")
                calculated_cost = None
                # Prioritize the key from MODELS_CONFIG as it might have custom pricing.
                # Fallback to response.model or self.litellm_model_id only if the primary key is not set.
                model_name_for_costing = self.model_name_key 
                if not model_name_for_costing:
                    if hasattr(response, 'model') and response.model:
                        model_name_for_costing = response.model
                    elif self.litellm_model_id: # Fallback to the model ID used in the call
                        model_name_for_costing = self.litellm_model_id
                    else:
                        logger.warning("Cannot determine model name for cost calculation.")
                
                if hasattr(response, 'usage'):
                    current_input_tokens = getattr(response.usage, 'prompt_tokens', 0)
                    current_output_tokens = getattr(response.usage, 'completion_tokens', 0)

                if model_name_for_costing and isinstance(model_name_for_costing, str) and current_input_tokens is not None and current_output_tokens is not None:
                    try:
                        # Removed logic for preparing custom cost_kwargs
                        # Relying on litellm.model_cost being populated by setup_litellm_client

                        calculated_cost = litellm.completion_cost(
                            model=model_name_for_costing,
                            completion_response=response # Pass the entire response object
                            # Removed **cost_kwargs
                        )
                        if calculated_cost is not None:
                            cost_delta = calculated_cost
                            # Updated debug log to reflect reliance on internal/registered pricing
                            logger.debug(f"Cost calculated by litellm.completion_cost() using internal/registered pricing: ${cost_delta:.6f} for model '{model_name_for_costing}'")
                        else:
                            logger.warning(
                                f"litellm.completion_cost() returned None for model '{model_name_for_costing}'. "
                                f"Token usage: Input={current_input_tokens}, Output={current_output_tokens}. Cost will be $0.00."
                            )
                    except Exception as cc_exc:
                        logger.error(f"Error calling litellm.completion_cost() for model '{model_name_for_costing}': {cc_exc}")
                else:
                    logger.warning("Could not attempt litellm.completion_cost(). Missing model name, model name not a string, or token usage.")
            
            # Update token counts for logging if not already done (primarily for the direct cost cases)
            if input_tokens == 0 and output_tokens == 0 and hasattr(response, 'usage'):
                input_tokens = getattr(response.usage, 'prompt_tokens', 0)
                output_tokens = getattr(response.usage, 'completion_tokens', 0)

            # Final logging if cost is still 0 despite token usage
            if cost_delta == 0.0 and (input_tokens > 0 or output_tokens > 0) and calculated_cost is None:
                logger.warning(
                    f"Cost is $0.00 for model '{model_name_for_costing or self.litellm_model_id}'. Token usage: Input={input_tokens}, Output={output_tokens}. All cost retrieval methods failed or returned zero."
                )
            elif cost_delta == 0.0 and input_tokens == 0 and output_tokens == 0:
                logger.warning(f"Cost is $0.00 for model '{model_name_for_costing or self.litellm_model_id}'. No token usage reported. All cost retrieval methods failed or returned zero.")

        except Exception as e:
            logger.error(f"Error during cost calculation: {e}. Cost may be inaccurate.")
            # Ensure cost_delta remains 0 or its last known value if error occurs mid-calculation.

        with self._cost_lock:
            self.cost += cost_delta
        self.current_solve_cost += cost_delta
        
        logger.info(f"Cost updated: +${cost_delta:.6f} (current solve: ${self.current_solve_cost:.6f}, instance total: ${self.cost:.6f})")
        if input_tokens > 0 or output_tokens > 0:
             logger.debug(f"Token usage for last call (from response.usage if available): Input: {input_tokens}, Output: {output_tokens}")

        
    def configure(self, model_name_key: str, max_tries_override: int) -> None:
        """Configure the QuerySystem to use a specific model from MODELS_CONFIG.
        
        Args:
            model_name_key: The user-facing model name (key in MODELS_CONFIG).
            max_tries_override: The number of tries for query generation for this configuration.
        """

        model_to_configure = model_name_key

        if model_to_configure not in MODELS_CONFIG:
            logger.error(f"Model key '{model_to_configure}' not found in MODELS_CONFIG.")
            raise ValueError(f"Model key '{model_to_configure}' not found in MODELS_CONFIG.")

        self.model_name_key = model_to_configure
        self.current_model_config = MODELS_CONFIG[self.model_name_key]
        
        self.litellm_model_id = self.current_model_config.get("model")
        # self.current_provider = self.current_model_config.get("provider") # Removed
        
        # Set max_tries from the explicit parameter
        self.max_tries = max_tries_override
        
        # Reasoning effort is now top-level in the model's config, can be None
        self.reasoning_effort = self.current_model_config.get("reasoning_effort") 

        if not self.litellm_model_id:
            logger.error(f"Model '{self.model_name_key}' is missing 'model' in MODELS_CONFIG.")
            raise ValueError(f"Incomplete configuration for model '{self.model_name_key}'.")

        try:
            setup_litellm_client(self.secret_client, self.model_name_key)
        except Exception as e:
            logger.error(f"Failed to setup LiteLLM client for {self.model_name_key}: {e}")
            raise
        
        logger.info(f"QuerySystem (re)configured. Name: {self.model_name_key}, LiteLLM Model: {self.litellm_model_id}, Max Tries: {self.max_tries}, Reasoning Effort: {self.reasoning_effort}")

    def _call_llm(self, # Renamed from _call_azure_openai
                    messages: List[Dict[str, str]],
                    tool_schema: Optional[Dict[str, Any]] = None,
                    max_tokens_for_completion: int = 9000) -> Union[Any, Dict[str, Any]]:
        """Centralized function to call LLM via LiteLLM with consistent parameters
        
        Args:
            messages: List of message dictionaries to send to the API
            tool_schema: Optional OpenAI tool schema for structured output (triggers function calling)
            max_tokens_for_completion: Maximum tokens for regular completion (default: 9000 for non-tool calls)
            
        Returns:
            If tool_schema is provided, returns a dictionary of arguments from the LLM's function call.
            Otherwise, returns the raw LiteLLM response object.
        """
        if not self.litellm_model_id:
            logger.error("LiteLLM model not configured. Call configure() first.")
            raise ValueError("LiteLLM model not configured.")

        kwargs = {
            "model": self.litellm_model_id,
            "messages": messages,
            "temperature": 1, # Set temperature to 0
        }

        # Add provider-specific or reasoning-effort specific parameters if needed
        # LiteLLM passes through additional kwargs to the provider
        if self.reasoning_effort and not tool_schema: # Reasoning effort usually for non-tool calls
            # The way to pass "reasoning_effort" depends on the provider.
            # For Azure OpenAI, it was `extra_body={"reasoning_effort": self.reasoning_effort}`
            # For other providers, it might be different or not supported.
            # LiteLLM might handle some common ones, or we might need custom logic per provider.
            if self.litellm_model_id and self.litellm_model_id.startswith("azure/"):
                 # For Azure, it's passed in `extra_body`
                kwargs["extra_body"] = kwargs.get("extra_body", {}) # Ensure extra_body exists
                kwargs["extra_body"].update({"reasoning_effort": self.reasoning_effort})
                kwargs["max_tokens"] = 100_000 # As per original code for reasoning effort
                logger.debug(f"Applying reasoning_effort: {self.reasoning_effort} for Azure model {self.litellm_model_id}")
            else:
                # For generic OpenAI or other providers, 'reasoning_effort' is not a standard param.
                # Some custom models or endpoints might support it via metadata or other means.
                # For now, log if it's set for a non-Azure provider and not using tools.
                current_provider_inferred = self.litellm_model_id.split('/')[0] if self.litellm_model_id and '/' in self.litellm_model_id else "unknown"
                logger.warning(f"Reasoning effort '{self.reasoning_effort}' set for non-Azure provider '{current_provider_inferred}'. It might not be applied.")
                # Still set max_tokens if reasoning_effort was specified, as per original logic
                kwargs["max_tokens"] = kwargs.get("max_tokens", max_tokens_for_completion) 

        if tool_schema:
            kwargs["tools"] = [tool_schema]
            # LiteLLM uses "tool_choice": {"type": "function", "function": {"name": "my_function"}} format for specific function
            # or "tool_choice": "auto" / "required"
            kwargs["tool_choice"] = {"type": "function", "function": {"name": tool_schema["function"]["name"]}}
            # max_tokens for tool calls is usually small or managed by the API
            # kwargs.pop("max_tokens", None) # Remove if reasoning_effort set it
            # kwargs.pop("max_completion_tokens", None) # Ensure this isn't passed
        else:
            if "max_tokens" not in kwargs: # If not already set by reasoning_effort logic
                 kwargs["max_tokens"] = max_tokens_for_completion

        try:
            logger.debug(f"Calling LiteLLM with kwargs: { {k:v for k,v in kwargs.items() if k != 'messages'} }") # Avoid logging full messages list here
            # logger.debug(f"Full messages for LiteLLM call: {json.dumps(messages)}") # Log messages separately if needed
            raw_response = litellm.completion(**kwargs)
            self._update_cost(raw_response) 

            if tool_schema:
                # LiteLLM response structure for tool calls:
                # response.choices[0].message.tool_calls[0].function (name, arguments)
                message = raw_response.choices[0].message
                if message.tool_calls:
                    tool_call = message.tool_calls[0] # Assuming one tool call as before
                    if tool_call.function:
                        function_name = tool_call.function.name
                        expected_function_name = tool_schema["function"]["name"]
                        if function_name == expected_function_name:
                            try:
                                arguments_str = tool_call.function.arguments
                                logger.debug(f"Function call arguments for {function_name} from LiteLLM: {arguments_str}")
                                arguments = json.loads(arguments_str)
                                return arguments
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse JSON arguments from LiteLLM for {function_name}: {arguments_str}. Error: {e}")
                                raise ValueError(f"Invalid JSON arguments from LLM for {function_name}: {e}") from e
                        else:
                            logger.error(f"LiteLLM called unexpected function: {function_name}. Expected: {expected_function_name}")
                            raise ValueError(f"LiteLLM called unexpected function: {function_name}")
                
                logger.error(f"LiteLLM did not make the expected function call to {tool_schema['function']['name']}. Response: {raw_response}")
                raise ValueError(f"LiteLLM did not make the expected function call to {tool_schema['function']['name']}")
            else: # No tool_schema, regular completion
                return raw_response # Return the full LiteLLM response object

        except litellm.exceptions.APIError as e:
            logger.error(f"LiteLLM APIError: {e}")
            self._update_cost(e) # some errors might contain usage/cost info
            raise
        except Exception as e:
            logger.error(f"Error during LiteLLM call: {e}")
            # Potentially log kwargs for debugging if safe
            raise

    def generate_query(self, current_test_question: str, query_date_str: Optional[str] = None) -> Optional[QueryResult]:
        """Generate and improve KQL query with multiple attempts using structured output"""
        
        kql_tool_schema = {
            "type": "function",
            "function": {
                "name": "KQLQuery",
                "description": "Generates a KQL query and an explanation for it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "explanation": {
                            "type": "string",
                            "description": "An explanation of the KQL query."
                        },
                        "query": {
                            "type": "string",
                            "description": "The KQL query."
                        }
                    },
                    "required": ["explanation", "query"]
                }
            }
        }

        failed_queries = []
        all_attempts = []  # Track all attempts, including successful ones
        execution_start_time = time.time()
        
        for attempt_idx in range(self.max_tries): # Renamed attempt to attempt_idx to avoid confusion with attempt dictionary
            try:
                # Use display_all_table_fields for LLM prompt
                table_fields_for_llm_prompt = self.display_all_table_fields
                
                # Build message history with previous failed attempts
                messages = [
                    {"role": "system", "content": "You are a Kusto Query Language (KQL) query expert. Generate or improve the KQL query. Timerange will be set in function call."}
                ]
                
                # Always include table information
                # Format table fields as "table: fields" pairs using display names
                tables_with_fields_str_list = []
                # Add fields information for up to 10 display tables
                tables_to_show_fields_for = self.display_available_tables[:10]
                
                for display_table_name in tables_to_show_fields_for:
                    if display_table_name in table_fields_for_llm_prompt: # table_fields_for_llm_prompt is self.display_all_table_fields
                        tables_with_fields_str_list.append(f"{display_table_name}: {table_fields_for_llm_prompt[display_table_name]}")
                
                fields_info = "\n\nTable fields details:\n" + "\n\n".join(tables_with_fields_str_list)
                
                messages.append({"role": "user", "content": f"Generate a KQL query to answer this question: {current_test_question}------{fields_info}"})
                
                if failed_queries:
                    context = "Previous attempts that returned no results or had errors:\n" # MODIFIED context message
                    for i, q_info in enumerate(failed_queries, 1): # MODIFIED to iterate q_info
                        context += f"Attempt {i}: Query: {q_info['query']}\nError: {q_info['error']}\n"
                    # Removed the 'details' list generation as it was redundant with the improved context
                    messages.append({"role": "user", "content": context})
                
                # response_obj is now a dictionary from the parsed tool call
                response_obj = self._call_llm(messages, tool_schema=kql_tool_schema)
                kql_query_from_llm = response_obj.get("query", "")
                explanation = response_obj.get("explanation", "")

                # Translate KQL query to use actual table names before execution
                kql_query_to_execute = self._translate_kql_query_to_actual_tables(kql_query_from_llm)
                
                # Create attempt record
                current_attempt_dict = {
                    "attempt_number": attempt_idx + 1,
                    "query_llm": kql_query_from_llm, # Store the query as generated by LLM
                    "query_executed": kql_query_to_execute, # Store the translated query
                    "explanation": explanation,
                    "timestamp": datetime.now().isoformat(),
                }
                
                # Execute the translated query
                try:
                    # Query for the last 24 hours or specified date
                    if query_date_str:
                        try:
                            end_time_dt = datetime.strptime(query_date_str, "%d.%m.%Y")
                            # Ensure the time is set to the end of the day, and it's timezone-aware (UTC)
                            end_time_24h = datetime(end_time_dt.year, end_time_dt.month, end_time_dt.day, 23, 59, 59, 999999, tzinfo=UTC)
                            start_time_24h = end_time_24h - timedelta(days=1) # Query for the full specified day
                        except ValueError:
                            logger.error(f"Invalid date format provided for query_date: {query_date_str}. Defaulting to last 1 day from now.")
                            end_time_24h = datetime.now(UTC)
                            start_time_24h = end_time_24h - timedelta(days=1) 
                    else:
                        end_time_24h = datetime.now(UTC)
                        start_time_24h = end_time_24h - timedelta(days=1)
                    
                    query_timerange_24h = (start_time_24h, end_time_24h)
                    logger.debug(f"Querying KQL from {start_time_24h.isoformat()} to {end_time_24h.isoformat()}")
                    
                    results = self.querier.run_custom_query(kql_query_to_execute, timespan=query_timerange_24h)
                    current_attempt_dict["status"] = "executed"
                    current_attempt_dict["results_count"] = len(results) - 1 if results else 0  # Subtract header row
                except Exception as e:
                    error_msg = str(e)
                    # REMOVED specific syntax/semantic error counting here
                    logger.error(f"Error in query execution attempt {attempt_idx + 1}: {error_msg}")
                    current_attempt_dict["status"] = "error" # This is the key status
                    current_attempt_dict["error"] = error_msg
                    failed_queries.append({
                        "query": kql_query_from_llm, # Log the LLM's query
                        "error": error_msg
                    })
                    all_attempts.append(current_attempt_dict)
                    continue
                
                # Process the results
                if not results or len(results) <= 1:  # Only headers, no data
                    logger.info(f"Attempt {attempt_idx + 1} returned no results")
                    current_attempt_dict["status"] = "no_results"
                    failed_queries.append({
                        "query": kql_query_from_llm, # Log the LLM's query
                        "error": "Query returned no results"
                    })
                    all_attempts.append(current_attempt_dict)
                    continue
                elif len(results) > 100: # This condition might need review based on typical result sizes
                    logger.info(f"Attempt {attempt_idx + 1} returned too many results ({len(results) -1})")
                    current_attempt_dict["status"] = "too_many_results"
                    current_attempt_dict["results_count"] = len(results) - 1
                    failed_queries.append({
                        "query": kql_query_from_llm, # Log the LLM's query
                        "error": "Query returned too many results"
                    })
                    all_attempts.append(current_attempt_dict)
                    continue
                
                # Check if results contain the answer
                try:
                    contains_answer, answer, result_summary = self.check_if_result_contains_answer(results, current_test_question)
                    current_attempt_dict["result_summary"] = result_summary
                    
                    if contains_answer:
                        logger.info(f"Attempt {attempt_idx + 1} returned results that contain the answer")
                        logger.info(f"Found answer: {answer}")
                        current_attempt_dict["status"] = "success"
                        current_attempt_dict["answer"] = answer
                        current_attempt_dict["contains_answer"] = True
                        all_attempts.append(current_attempt_dict)
                        
                        execution_time = time.time() - execution_start_time
                        # Calculate llm_formulate_kql_errors before returning
                        llm_formulate_kql_errors_count = sum(1 for att in all_attempts if att.get("status") == "error")
                        return QueryResult(
                            query=kql_query_from_llm, # Return the LLM's original query
                            raw_results=results,
                            answer=answer,
                            attempts=attempt_idx + 1,
                            execution_time=execution_time,
                            all_attempts=all_attempts,
                            llm_formulate_kql_errors=llm_formulate_kql_errors_count # MODIFIED
                        )
                    else:
                        logger.info(f"Attempt {attempt_idx + 1} returned results that do not contain the answer to the question")
                        current_attempt_dict["status"] = "no_answer"
                        current_attempt_dict["contains_answer"] = False
                        failed_queries.append({
                            "query": kql_query_from_llm, # Log the LLM's query
                            "results": f"Summary of the results: {result_summary}",
                            "error": "Query returned results that do not contain the answer"
                        })
                        all_attempts.append(current_attempt_dict)
                        continue  # Try next attempt since this one didn't contain the answer
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error checking results for answer: {error_msg}")
                    current_attempt_dict["status"] = "analysis_error" # This status is for answer checking, not KQL formulation
                    current_attempt_dict["error"] = error_msg
                    failed_queries.append({
                        "query": kql_query_from_llm, # Log the LLM's query
                        "error": f"Error analyzing results: {error_msg}"
                    })
                    all_attempts.append(current_attempt_dict)
                    continue  # Try next attempt on error
                
            except Exception as e: # This outer exception is for errors in the attempt loop itself (e.g. LLM call)
                logger.error(f"Attempt {attempt_idx + 1} loop failed critically: {str(e)}")
                # Consider if this should count as an llm_formulate_kql_error
                # For now, it's a loop failure, not necessarily a KQL execution error status.
                # To be safe, add a placeholder to all_attempts if it's not already handled.
                # This situation should be rare.
                # Ensure all_attempts has an entry for this failed attempt_idx if not already added.
                if not any(att.get("attempt_number") == attempt_idx + 1 for att in all_attempts):
                    all_attempts.append({
                        "attempt_number": attempt_idx + 1,
                        "status": "loop_error", # Custom status for this case
                        "error": f"Critical failure in attempt loop: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    })
                continue
        
        # If all attempts fail, return empty values in QueryResult but include all attempts
        logger.warning("All query attempts failed to find an answer or were exhausted.")
        llm_formulate_kql_errors_count = sum(1 for att in all_attempts if att.get("status") == "error")
        return QueryResult(
            query="",
            raw_results=[],
            answer="",
            attempts=self.max_tries,
            all_attempts=all_attempts,
            llm_formulate_kql_errors=llm_formulate_kql_errors_count # MODIFIED
        )

    def check_if_result_contains_answer(self, KQL_result, current_test_question: str) -> tuple[bool, str, str]:
        """Check if the KQL results contain the answer to the question
        
        Args:
            KQL_result: The results from the KQL query
            current_test_question: The question being asked
            
        Returns:
            tuple: (contains_answer, answer, result_summary)
        """
        try:
            contains_answer_tool_schema = {
                "type": "function",
                "function": {
                    "name": "ContainsAnswer",
                    "description": "Analyzes KQL query results to determine if they contain the answer to a given question. The answer should be a specific word or phrase, not just 'yes' or 'no'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "thinking": {
                                "type": "string",
                                "description": "The thought process followed to analyze the results and arrive at the conclusion."
                            },
                            "result_summary": {
                                "type": "string",
                                "description": "A concise summary of the KQL query results."
                            },
                            "contains": {
                                "type": "boolean",
                                "description": "True if the results contain the answer to the question, False otherwise."
                            },
                            "answer": {
                                "type": "string",
                                "description": "The specific answer to the question, extracted from the results. This should be the actual information, not just 'yes' or 'no'."
                            }
                        },
                        "required": ["thinking", "result_summary", "contains", "answer"]
                    }
                }
            }
            
            # Print question for debugging before checking results
            logger.debug(f"Checking results for question: {current_test_question}")
            messages = [
                {"role": "system", "content": "Analyze the KQL query results to check if they contain the answer to the question. Answer should be only 1 word to answer the question. Not just yes or no for answer. Specify the actuall answer needed for the question. Check if the results actually are related to the question"},
                {"role": "user", "content": f"Question: {current_test_question} ------Results: {KQL_result}"}
            ]
            
            # response_obj is now a dictionary from the parsed tool call
            response_obj = self._call_llm(messages, tool_schema=contains_answer_tool_schema)
            return (
                response_obj.get("contains", False),
                response_obj.get("answer", ""),
                response_obj.get("result_summary", "")
            )
        except Exception as e:
            logger.error(f"Error checking if results contain answer: {e}")
            return False, "", f"Error analyzing results: {e}"

    def solve(self, question_data: Union[str, TestCase], query_date_str: Optional[str] = None) -> QueryResult:
        """Solve the query using configured settings
        
        Args:
            question_data: Either a simple question string or a TestCase object.
            query_date_str: Optional query end date string (DD.MM.YYYY).
            
        Returns:
            QueryResult object containing query, results, answer and attempts
        """
        # Reset cost for this specific solve operation
        self.current_solve_cost = 0.0
        
        # Handle either string or TestCase object (which has a .prompt attribute)
        if isinstance(question_data, str):
            current_test_question = question_data 
        elif hasattr(question_data, 'prompt') and isinstance(question_data.prompt, str): # Check if it's like TestCase
            current_test_question = question_data.prompt
        elif isinstance(question_data, TestCase): # Explicitly check for TestCase type
            current_test_question = question_data.prompt
        else:
            logger.error(f"Invalid question_data type: {type(question_data)}. Expected str or TestCase-like object with a .prompt string attribute.")
            # Return an empty or error QueryResult
            return QueryResult(
                query="", raw_results=[], answer="", attempts=0, cost=0.0, 
                error_message=f"Invalid input type: {type(question_data)}"
            )
        
        query_result_obj = self.generate_query(current_test_question, query_date_str=query_date_str) 
        
        # Assign the accumulated cost for this solve operation to the QueryResult
        if query_result_obj:
            query_result_obj.cost = self.current_solve_cost
        
        # Log success if query is not empty
        if query_result_obj and query_result_obj.query:
            logger.info(f"Found solution after {query_result_obj.attempts} attempt(s) with cost ${query_result_obj.cost:.6f}")
        elif query_result_obj: # Case where all attempts failed but we still have a QueryResult
             query_result_obj.cost = self.current_solve_cost # Ensure cost is set even on full failure
             logger.warning(f"All query attempts failed for '{current_test_question}'. Cost for attempts: ${query_result_obj.cost:.6f}") # Used local variable

        return query_result_obj

