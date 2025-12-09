"""
Alerts module.

Provides advanced alerting capabilities for crypto trading.
"""

from .alert_system import (
    AdvancedAlertSystem,
    AlertSystemConfig,
    Alert,
    AlertCondition,
    TriggeredAlert,
    AlertType,
    AlertPriority,
    AlertStatus,
    NotificationChannel,
    NotificationSender,
    PriceAlertChecker,
    VolumeAlertChecker,
    WhaleAlertChecker,
    OnChainMetricsChecker,
    TechnicalIndicatorChecker,
    create_alert_system,
)

__all__ = [
    "AdvancedAlertSystem",
    "AlertSystemConfig",
    "Alert",
    "AlertCondition",
    "TriggeredAlert",
    "AlertType",
    "AlertPriority",
    "AlertStatus",
    "NotificationChannel",
    "NotificationSender",
    "PriceAlertChecker",
    "VolumeAlertChecker",
    "WhaleAlertChecker",
    "OnChainMetricsChecker",
    "TechnicalIndicatorChecker",
    "create_alert_system",
]
