"""
Centralized configuration for neuralmail.
"""

import os
import json
import logging

logger = logging.getLogger(__name__)


def get_config_path():
    """Get the path to the config file in the current working directory."""
    return os.path.join(os.getcwd(), "config.json")


def load_config():
    """Load the configuration from the config file."""
    config_path = get_config_path()
    if not os.path.exists(config_path):
        logger.info(
            f"Config file not found at {config_path}. Creating a default config."
        )
        return create_default_config()

    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading config file: {e}. A default config will be used.")
        return create_default_config()


def save_config(config):
    """Save the configuration to the config file."""
    config_path = get_config_path()
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        logger.info(f"Config saved to {config_path}")
    except IOError as e:
        logger.error(f"Error saving config file: {e}")


def create_default_config():
    """Create a default configuration."""
    config = {
        "llm_provider": "openrouter",
        "llm_model": "google/gemini-2.5-flash",
        "llm_api_key": "YOUR_OPENAI_API_KEY",
        "llm_base_url": "https://openrouter.ai/api/v1",
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "embedding_api_key": "YOUR_OPENAI_API_KEY",
        "embedding_base_url": None,
        "max_tokens": 8000,
        "llm_max_context": 256000,
        "accounts": [
            {
                "account_name": "Default Account",
                "imap_host": "",
                "imap_user": "",
                "imap_pass": "",
                "imap_port": 993,
                "use_ssl": True,
                "db_path": "emails.db",
                "selected_folders": [],
            }
        ],
    }
    save_config(config)
    return config


# Load the configuration at startup
config = load_config()
