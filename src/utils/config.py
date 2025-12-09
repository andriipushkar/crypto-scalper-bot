"""
Configuration loading and validation.

Handles loading YAML configs and validating required fields.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional

import yaml
from loguru import logger


def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    logger.debug(f"Loaded config from {config_path}")
    return config


def load_risk_config(config_path: str = "config/risk.yaml") -> Dict[str, Any]:
    """
    Load risk management configuration.

    Args:
        config_path: Path to risk config file

    Returns:
        Risk configuration dictionary (empty dict if file not found)
    """
    path = Path(config_path)

    if not path.exists():
        logger.warning(f"Risk config not found: {config_path}, using defaults")
        return {}

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    logger.debug(f"Loaded risk config from {config_path}")
    return config


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Required sections
    required_sections = ["exchange", "trading"]
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")

    if errors:
        return errors

    # Exchange validation
    exchange = config.get("exchange", {})

    if "testnet" not in exchange:
        errors.append("exchange.testnet must be specified")

    # Trading validation
    trading = config.get("trading", {})

    symbols = trading.get("symbols", [])
    if not symbols:
        errors.append("trading.symbols must have at least one symbol")

    leverage = trading.get("leverage", 1)
    if not 1 <= leverage <= 125:
        errors.append(f"trading.leverage must be 1-125, got {leverage}")

    margin_type = trading.get("margin_type", "CROSSED")
    if margin_type not in ("CROSSED", "ISOLATED"):
        errors.append(f"trading.margin_type must be CROSSED or ISOLATED")

    # Strategies validation
    strategies = config.get("strategies", {})

    any_enabled = False
    for name, strat_config in strategies.items():
        if strat_config.get("enabled", False):
            any_enabled = True

            # Validate strategy-specific configs
            if name == "orderbook_imbalance":
                threshold = strat_config.get("imbalance_threshold", 1.5)
                if not 1.0 < threshold < 10.0:
                    errors.append(f"strategies.{name}.imbalance_threshold should be 1.0-10.0")

            if name == "volume_spike":
                multiplier = strat_config.get("volume_multiplier", 3.0)
                if not 1.5 <= multiplier <= 20.0:
                    errors.append(f"strategies.{name}.volume_multiplier should be 1.5-20.0")

    if not any_enabled:
        errors.append("At least one strategy must be enabled")

    # Data validation
    data = config.get("data", {})
    storage = data.get("storage", {})

    db_path = storage.get("database_path", "")
    if db_path:
        # Check if parent directory exists or can be created
        db_dir = Path(db_path).parent
        if not db_dir.exists():
            try:
                db_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created database directory: {db_dir}")
            except Exception as e:
                errors.append(f"Cannot create database directory {db_dir}: {e}")

    return errors


def get_api_credentials(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Get API credentials from environment variables.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with api_key and api_secret
    """
    exchange = config.get("exchange", {})
    testnet = exchange.get("testnet", True)

    if testnet:
        api_key_env = exchange.get("api_key_env", "BINANCE_TESTNET_API_KEY")
        api_secret_env = exchange.get("api_secret_env", "BINANCE_TESTNET_API_SECRET")
    else:
        api_key_env = "BINANCE_API_KEY"
        api_secret_env = "BINANCE_API_SECRET"

    return {
        "api_key": os.getenv(api_key_env, ""),
        "api_secret": os.getenv(api_secret_env, ""),
    }


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge multiple configuration dictionaries.

    Later configs override earlier ones.

    Args:
        *configs: Configuration dictionaries to merge

    Returns:
        Merged configuration
    """
    result = {}

    for config in configs:
        result = _deep_merge(result, config)

    return result


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print a summary of the configuration.

    Args:
        config: Configuration dictionary
    """
    exchange = config.get("exchange", {})
    trading = config.get("trading", {})
    strategies = config.get("strategies", {})

    logger.info("-" * 40)
    logger.info("Configuration Summary")
    logger.info("-" * 40)

    # Exchange
    env = "TESTNET" if exchange.get("testnet", True) else "MAINNET"
    logger.info(f"Environment: {env}")

    # Trading
    logger.info(f"Symbols: {trading.get('symbols', [])}")
    logger.info(f"Leverage: {trading.get('leverage', 1)}x")
    logger.info(f"Margin: {trading.get('margin_type', 'CROSSED')}")

    # Strategies
    enabled_strategies = [
        name for name, cfg in strategies.items()
        if cfg.get("enabled", False)
    ]
    logger.info(f"Strategies: {enabled_strategies}")

    logger.info("-" * 40)
