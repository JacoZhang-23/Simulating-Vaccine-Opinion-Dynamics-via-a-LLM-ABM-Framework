# config.py - HKUST API Version

"""
Model configuration file for the HKUST API setup.
Contains all tunable parameters for experiments and calibration.
"""
import os

# ===============================================================================
# Simulation runtime parameters
# ===============================================================================
SIMULATION_PARAMS = {
    'default_steps': 72,           # Default number of simulation steps.
    'sample_proportion': 1,        # Default sampling ratio (1.0 = 100%).
    'log_level': 'INFO',           # Log level: DEBUG, INFO, WARNING, ERROR.
}

# ===============================================================================
# API parameters - Local API
# ===============================================================================
# Local API Keys (dummy keys for local API)
API_KEYS_STR = "abc123" 
API_KEYS_LIST = [key.strip() for key in API_KEYS_STR.split(',') if key.strip()]

API_PARAMS = {
    # Local API key list (dummy values are fine for a local API).
    'api_keys': API_KEYS_LIST if API_KEYS_LIST else ["abc123"],
    
    # Local API Base URL
    'api_base_url': "http://localhost:7890/v1",
    
    # Local API Model Name
    'llm_model_name': 'Qwen/Qwen3-8B',
    
    # Maximum concurrency per API key.
    'max_concurrency_per_key': 160,  # Local API uses high concurrency.
    
    # Batch-dialogue parameters.
    'batch_size': 28,  # Kept conservative to avoid token limits.
    # concurrent_batches = number of API keys × concurrency per key.
    'concurrent_batches': len(API_KEYS_LIST) * 100 if API_KEYS_LIST else 100,
}

# ===============================================================================
# Model-level parameters
# ===============================================================================
MODEL_PARAMS = {
    # Belief-distance threshold for interaction.
    'belief_threshold': 2.0,
    
    # Alpha openness is sampled from U(0, 1), so no extra parameter is needed.
}

# ===============================================================================
# Agent-level parameters
# ===============================================================================
AGENT_PARAMS = {
    # Semantic resonance weight w.
    'resonance_weight': 0.3,
    
    # Belief threshold for vaccination decisions.
    'vaccination_belief_threshold': 0
}

# ===============================================================================
# Data collection parameters
# ===============================================================================
DATA_COLLECTION_PARAMS = {}


# ===============================================================================
# Main helper: return all model parameters
# ===============================================================================
def get_model_params():
    """
    Merge all parameter groups into one dictionary for model initialization.
    """
    params = {}
    params.update(API_PARAMS)
    params.update(MODEL_PARAMS)
    params.update(AGENT_PARAMS)
    params.update(DATA_COLLECTION_PARAMS)
    
    return params
