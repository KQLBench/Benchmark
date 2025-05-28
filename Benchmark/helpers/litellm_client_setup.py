import os
import litellm
from azure.keyvault.secrets import SecretClient
from Benchmark.configuration.models_config import MODELS_CONFIG
from Benchmark.helpers.logging_config import get_logger
import logging

logger = get_logger(__name__)

# Store fetched secrets to avoid refetching for the same provider in a session
_fetched_secrets_cache = {}
_litellm_callbacks_registered = False

def setup_litellm_client(secret_client: SecretClient, model_name_key: str):
    """
    Configures LiteLLM environment variables for the specified model's provider.
    API keys are fetched from Azure Key Vault.
    Also registers global LiteLLM callbacks if not already done.

    Args:
        secret_client: Azure Key Vault SecretClient instance.
        model_name_key: The user-facing model name (key in MODELS_CONFIG).
    """
    global _litellm_callbacks_registered

    if not MODELS_CONFIG.get(model_name_key):
        logger.error(f"Model '{model_name_key}' not found in MODELS_CONFIG.")
        raise ValueError(f"Model '{model_name_key}' not found in MODELS_CONFIG.")

    config = MODELS_CONFIG[model_name_key]
    litellm_model_id = config.get("model") # provider is part of this string now

    if not litellm_model_id:
        logger.error(f"LiteLLM model ID (model) not specified for '{model_name_key}' in MODELS_CONFIG.")
        raise ValueError(f"LiteLLM model ID not specified for '{model_name_key}'.")

    # Register custom pricing if available in the config for this model_name_key
    input_cost_pm = config.get("input_cost_per_million_tokens")
    output_cost_pm = config.get("output_cost_per_million_tokens")

    if input_cost_pm is not None and output_cost_pm is not None:
        # Ensure litellm.model_cost dictionary exists
        if not hasattr(litellm, 'model_cost') or litellm.model_cost is None:
            litellm.model_cost = {}
        
        # Convert per-million to per-token
        input_cost_pt = input_cost_pm / 1_000_000
        output_cost_pt = output_cost_pm / 1_000_000

        litellm.model_cost[model_name_key] = {
            "input_cost_per_token": input_cost_pt,
            "output_cost_per_token": output_cost_pt
        }
        logger.info(f"Registered custom pricing for model key '{model_name_key}' with LiteLLM: Input ${input_cost_pt:.8f}/tok, Output ${output_cost_pt:.8f}/tok")
    # Fallback for old per-token keys, if they exist and new ones don't
    elif "input_cost_per_token" in config and "output_cost_per_token" in config:
        if not hasattr(litellm, 'model_cost') or litellm.model_cost is None:
            litellm.model_cost = {}
        litellm.model_cost[model_name_key] = {
            "input_cost_per_token": config["input_cost_per_token"],
            "output_cost_per_token": config["output_cost_per_token"]
        }
        logger.info(f"Registered custom (per-token) pricing for model key '{model_name_key}' with LiteLLM: Input ${config['input_cost_per_token']}/tok, Output ${config['output_cost_per_token']}/tok")
    elif model_name_key not in litellm.model_cost: # Log if no custom pricing and not already known by litellm
        logger.debug(f"No custom pricing found in MODELS_CONFIG for '{model_name_key}'. Relying on LiteLLM's internal pricing if available.")

    provider = None
    if "/" in litellm_model_id:
        provider = litellm_model_id.split('/', 1)[0]
        logger.debug(f"Provider explicitly found in litellm_model_id: {provider}")
    else:
        # Only attempt inference based on the key if no provider prefix was found in the model ID.
        logger.debug(f"No provider prefix in litellm_model_id ('{litellm_model_id}'). Attempting inference based on model key '{model_name_key}'.")
        # Default to openai if no prefix, or handle as an error/warning if specific prefixes are expected
        # For this case, let's assume if no prefix, it might be an OpenAI model or a direct model ID LiteLLM recognizes.
        # However, the request implies a prefix is expected for provider-specific keys.
        # If a model is like "o1" and we need "OPENAI_API_KEY", we must infer 'openai' or make it explicit.
        # Let's try to infer common ones or default to a generic approach if the key is just the model name.
        if model_name_key.startswith("o1") or model_name_key.startswith("o3") or model_name_key.startswith("o4") or model_name_key == "o3":
            provider = "openai" # Inferring based on common model key patterns
        elif "sonnet" in litellm_model_id:
            provider = "anthropic"
        elif "gemini" in litellm_model_id:
            provider = "gemini"
        # Add more specific inferences if needed

    if not provider:
        logger.warning(f"Could not reliably infer provider from model ID '{litellm_model_id}' for model key '{model_name_key}'. API key setup might be incorrect for some providers.")
        # Fallback or error if provider is essential for key fetching logic below and cannot be inferred.
        # For now, we will proceed, and the key fetching logic might try a generic key based on the model_name_key or fail if specific provider logic is hit.

    logger.info(f"Setting up LiteLLM client for model key: {model_name_key}, inferred provider: {provider}, litellm_id: {litellm_model_id}")

    # Cache fetched secrets to avoid redundant Key Vault calls for the same inferred provider
    if provider and provider not in _fetched_secrets_cache:
        _fetched_secrets_cache[provider] = True # Mark provider as processed

        if provider == "azure":
            try:
                os.environ["AZURE_API_KEY"] = secret_client.get_secret("AZURE-OPENAI-API-KEY").value
                os.environ["AZURE_API_BASE"] = secret_client.get_secret("AZURE-OPENAI-BASE-URL").value
                azure_api_version = config.get("api_version", "2025-01-01-preview")
                os.environ["AZURE_API_VERSION"] = azure_api_version
                logger.info(f"Set Azure OpenAI credentials for LiteLLM (Version: {azure_api_version}).")
            except Exception as e:
                logger.error(f"Failed to get Azure OpenAI secrets: {e}. Ensure AZURE-OPENAI-API-KEY and AZURE-OPENAI-BASE-URL are in Key Vault.")
                raise
        elif provider == "azure_ai": # New provider type for Azure AI
            try:
                os.environ["AZURE_AI_API_KEY"] = secret_client.get_secret("AZURE-AI-API-KEY").value
                os.environ["AZURE_AI_API_BASE"] = secret_client.get_secret("AZURE-AI-BASE-URL").value
                # api_version is not typically set for generic Azure AI endpoints unless specified by the model/LiteLLM documentation
                logger.info(f"Set Azure AI (azure_ai) credentials for LiteLLM. Key: AZURE_AI_API_KEY, Base: AZURE_AI_API_BASE")
            except Exception as e:
                logger.error(f"Failed to get Azure AI secrets: {e}. Ensure AZURE-AI-API-KEY and AZURE-AI-BASE-URL are in Key Vault.")
                raise
        else: # Handles openai, anthropic, gemini, etc.
            secret_name = f"{provider.upper()}-API-KEY"
            env_var_name = secret_name
            
            try:
                api_key_value = secret_client.get_secret(secret_name).value
                env_var_name_os_env = env_var_name.replace("-", "_")
                os.environ[env_var_name_os_env] = api_key_value
                logger.info(f"Set {env_var_name_os_env} for {provider} provider from Key Vault secret '{secret_name}'.")
            except Exception as e:
                logger.error(f"Failed to get API key '{secret_name}' for provider '{provider}' from Key Vault: {e}")
                # If the provider was inferred, and this key is not found, it might be an issue.
                # If no provider was inferred, this block might not be hit correctly.
                raise
    elif provider:
        logger.debug(f"Credentials for inferred provider '{provider}' already processed/set in this session.")
    else:
        logger.warning(f"No provider could be inferred for {litellm_model_id}. Specific API key environment variables may not be set.")

    # Register LiteLLM callbacks if not already done in this session.
    if not _litellm_callbacks_registered:
        # Attempt to reduce LiteLLM's own logger verbosity if app log level is INFO or higher.
        # This is a general setting; specific control over LiteLLM's logger might require deeper integration if available.
        # --- START: COMMENT OUT THIS BLOCK ---
        # if logger.getEffectiveLevel() >= logging.INFO: # logging needs to be imported
        #     litellm.set_verbose = False
        #     logger.debug("Attempted to set litellm.set_verbose = False.")
        #     try:
        #         litellm_logger = logging.getLogger("litellm")
        #         if litellm_logger:
        #             litellm_logger.setLevel(logging.WARNING) # Set litellm logger to WARNING
        #             # Also, ensure its handlers do not bypass this level
        #             for handler in litellm_logger.handlers:
        #                 handler.setLevel(logging.WARNING)
        #             # If litellm adds handlers to root logger, this won't stop them.
        #             # We could also consider litellm_logger.propagate = False, but that might be too aggressive
        #             # if other tools expect to capture litellm's warnings/errors via the root logger.
        #             logger.debug("Set 'litellm' logger level and its handlers to WARNING.")
        #     except Exception as e:
        #         logger.warning(f"Could not get or set 'litellm' logger level/handlers: {e}")
        # else: # Application is in DEBUG mode
        #     litellm.set_verbose = True 
        #     logger.debug("litellm.set_verbose = True (app is in DEBUG mode).")
        #     try:
        #         litellm_logger = logging.getLogger("litellm")
        #         if litellm_logger:
        #             litellm_logger.setLevel(logging.DEBUG) # Match app's debug level
        #             for handler in litellm_logger.handlers:
        #                 handler.setLevel(logging.DEBUG) # Ensure handlers also respect DEBUG
        #             logger.debug("Set 'litellm' logger level and its handlers to DEBUG.")
        #     except Exception as e:
        #         logger.warning(f"Could not get or set 'litellm' logger level/handlers for DEBUG: {e}")
        # --- END: COMMENT OUT THIS BLOCK ---
        _litellm_callbacks_registered = True
    
    # Cost registration with litellm.model_cost is removed as litellm handles this.
    # Global LiteLLM settings (verbose, callbacks, etc.) can be set here if needed once.
    # Example: litellm.set_verbose = True (do this guardedly, perhaps based on log level)
    # litellm.success_callback = [your_success_callback_function]
    # litellm.failure_callback = [your_failure_callback_function]

# To ensure this setup runs when the module is imported, you could call it,
# but it's better to call it explicitly from QuerySystem.__init__ or configure.
# For example, in QuerySystem:
# from .litellm_client_setup import setup_litellm_client
# setup_litellm_client(self.secret_client, self.model_name_key) 