# Benchmark/configuration/models_config.py

# This file consolidates model definitions, litellm parameters,
# and benchmark-specific settings.
# Pricing information has been removed as LiteLLM is expected to provide it.

MODELS_CONFIG = {
    "gpt-4.1": {
        "model": "azure/gpt-4.1"
    },    
    "gpt-4.1-finetuned": {
        "model": "azure/gpt-41-kql", 
        "input_cost_per_million_tokens": 2,
        "output_cost_per_million_tokens": 8
    },
    "gpt-4o": {
        "model": "azure/gpt-4o", 
    },
    "gpt-4o-kql-improved": {
        "model": "azure/gpt-4o-2024-08-06-kql-improved", 
    },
    "gpt-4o-mini": {
        "model": "azure/gpt-4o-mini", 
    },
    "gpt-4.1-kql-improved": {
        "model": "azure/gpt-41-kql", 
    },
    "gpt-4.1-mini": {
        "model": "azure/gpt-4.1-mini", 
    },
    "gpt-4.5-preview": {
        "model": "azure/gpt-4.5-preview", 
    },
    "o1-high": {
        "model": "azure/o1",
        "reasoning_effort": "high"
    },
    "o1-medium": {
        "model": "azure/o1", 
        "reasoning_effort": "medium"
    },
    "o1-low": {
        "model": "azure/o1", 
        "reasoning_effort": "low"
    },
    "o3-mini-high": {
        "model": "azure/o3-mini", 
        "reasoning_effort": "high"
    },
    "o3-mini-medium": {
        "model": "azure/o3-mini", 
        "reasoning_effort": "medium"
    },
    "o3-mini-low": {
        "model": "azure/o3-mini", 
        "reasoning_effort": "low"
    },
     "o3": {
        "model": "azure/o3", # Replace "o3"
    },
    "o4-mini-high": {
        "model": "azure/o4-mini", # Replace "o4-mini"
        "reasoning_effort": "high"
    },
    "o4-mini-medium": {
        "model": "azure/o4-mini", # Replace "o4-mini"
        "reasoning_effort": "medium"
    },
    "o4-mini-low": {
        "model": "azure/o4-mini", # Replace "o4-mini"
        "reasoning_effort": "low"
    },
    "sonnet-3.5": {
        "model": "anthropic/claude-3-5-sonnet-20240620",
    },
    "sonnet-3.7": { # This model ID might be hypothetical or an internal name.
        "model": "anthropic/claude-3-7-sonnet-placeholder", # Placeholder, update with actual if available
    },
    "gemini-2.5-pro-preview-05-06": { 
        "model": "gemini/gemini-2.5-pro-preview-05-06", 
    },
    "grok-3-mini-beta": { 
        "model": "xai/grok-3-mini-beta", 
    },
    "grok-3-beta": { 
        "model": "xai/grok-3-beta", 
    },
    "DeepSeek-V3-0324": { # Does not support Tool Calls
        "model": "azure_ai/DeepSeek-V3-0324", 
    },
    "DeepSeek-R1": { # Does not support Tool Calls
        "model": "azure_ai/DeepSeek-R1", 
    },
    "Phi-4-reasoning": { # Does not support Tool Calls
        "model": "azure_ai/Phi-4-reasoning", 
    },
    "Llama-3.3-70B-Instruct": { # Does not support Tool Calls
        "model": "azure_ai/Llama-3.3-70B-Instruct", 
    }
}

