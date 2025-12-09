"""
Logging configuration for the trading bot.

Uses loguru for structured, async-friendly logging.
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logger(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "1 day",
    retention: str = "30 days",
    format_string: Optional[str] = None,
) -> None:
    """
    Configure the logger for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (None for console only)
        rotation: When to rotate log files
        retention: How long to keep old log files
        format_string: Custom format string
    """
    # Remove default handler
    logger.remove()
    
    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Console handler
    logger.add(
        sys.stdout,
        format=format_string,
        level=level,
        colorize=True,
    )
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format=format_string.replace("<green>", "").replace("</green>", "")
                   .replace("<level>", "").replace("</level>", "")
                   .replace("<cyan>", "").replace("</cyan>", ""),
            level=level,
            rotation=rotation,
            retention=retention,
            compression="gz",
        )
    
    logger.info(f"Logger initialized with level={level}")


def get_logger(name: str):
    """
    Get a logger instance with a specific name.
    
    Args:
        name: Logger name (usually module name)
    
    Returns:
        Logger instance bound with the name
    """
    return logger.bind(name=name)


# Trade-specific logging
class TradeLogger:
    """
    Specialized logger for trade events.
    Logs trades in a structured format for later analysis.
    """
    
    def __init__(self, log_dir: str = "data/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate file for trades
        self.trade_file = self.log_dir / "trades.jsonl"
        
        # Add trade-specific handler
        logger.add(
            str(self.trade_file),
            format="{message}",
            filter=lambda record: record["extra"].get("trade_log", False),
            rotation="1 day",
            retention="90 days",
        )
    
    def log_signal(
        self,
        strategy: str,
        signal_type: str,
        strength: float,
        symbol: str,
        metadata: dict,
    ) -> None:
        """Log a trading signal."""
        import json
        from datetime import datetime
        
        record = {
            "type": "signal",
            "timestamp": datetime.utcnow().isoformat(),
            "strategy": strategy,
            "signal_type": signal_type,
            "strength": strength,
            "symbol": symbol,
            "metadata": metadata,
        }
        logger.bind(trade_log=True).info(json.dumps(record))
    
    def log_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float],
        status: str,
    ) -> None:
        """Log an order event."""
        import json
        from datetime import datetime
        
        record = {
            "type": "order",
            "timestamp": datetime.utcnow().isoformat(),
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "quantity": quantity,
            "price": price,
            "status": status,
        }
        logger.bind(trade_log=True).info(json.dumps(record))
    
    def log_fill(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        commission: float,
        realized_pnl: Optional[float] = None,
    ) -> None:
        """Log an order fill."""
        import json
        from datetime import datetime
        
        record = {
            "type": "fill",
            "timestamp": datetime.utcnow().isoformat(),
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "commission": commission,
            "realized_pnl": realized_pnl,
        }
        logger.bind(trade_log=True).info(json.dumps(record))
    
    def log_position(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        unrealized_pnl: float,
        action: str,  # "open", "close", "update"
    ) -> None:
        """Log a position change."""
        import json
        from datetime import datetime
        
        record = {
            "type": "position",
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "unrealized_pnl": unrealized_pnl,
            "action": action,
        }
        logger.bind(trade_log=True).info(json.dumps(record))


# Module-level trade logger instance
trade_logger: Optional[TradeLogger] = None


def get_trade_logger() -> TradeLogger:
    """Get or create the trade logger instance."""
    global trade_logger
    if trade_logger is None:
        trade_logger = TradeLogger()
    return trade_logger
