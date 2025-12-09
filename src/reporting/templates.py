"""
Report Templates

Provides templates for various report types.
"""
from typing import Any, Dict, List, Optional
from datetime import datetime
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum


class ReportTemplate(Enum):
    """Available report templates."""
    DAILY_SUMMARY = "daily_summary"
    WEEKLY_SUMMARY = "weekly_summary"
    MONTHLY_SUMMARY = "monthly_summary"
    TRADE_LOG = "trade_log"
    STRATEGY_ANALYSIS = "strategy_analysis"
    RISK_REPORT = "risk_report"
    TAX_REPORT = "tax_report"


# HTML Templates for email reports
DAILY_SUMMARY_HTML = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #1a1a2e; color: white; padding: 20px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f5f5f5; }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
        .metric-label {{ font-size: 12px; color: #666; }}
        .positive {{ color: #4caf50; }}
        .negative {{ color: #f44336; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f5f5f5; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Daily Trading Report</h1>
        <p>{date}</p>
    </div>

    <div class="metrics">
        <div class="metric">
            <div class="metric-value {pnl_class}">${total_pnl:,.2f}</div>
            <div class="metric-label">Daily P&L</div>
        </div>
        <div class="metric">
            <div class="metric-value">{total_trades}</div>
            <div class="metric-label">Trades</div>
        </div>
        <div class="metric">
            <div class="metric-value">{win_rate:.1f}%</div>
            <div class="metric-label">Win Rate</div>
        </div>
        <div class="metric">
            <div class="metric-value">${ending_balance:,.2f}</div>
            <div class="metric-label">Balance</div>
        </div>
    </div>

    <h2>Today's Trades</h2>
    <table>
        <tr>
            <th>Time</th>
            <th>Symbol</th>
            <th>Side</th>
            <th>Entry</th>
            <th>Exit</th>
            <th>P&L</th>
        </tr>
        {trade_rows}
    </table>

    <p style="color: #666; font-size: 12px; margin-top: 30px;">
        Generated at {generated_at}
    </p>
</body>
</html>
"""

TRADE_ROW_HTML = """
<tr>
    <td>{time}</td>
    <td>{symbol}</td>
    <td>{side}</td>
    <td>${entry_price:,.2f}</td>
    <td>${exit_price:,.2f}</td>
    <td class="{pnl_class}">${pnl:,.2f}</td>
</tr>
"""

ALERT_EMAIL_HTML = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .alert {{ padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .alert-critical {{ background: #ffebee; border-left: 4px solid #f44336; }}
        .alert-warning {{ background: #fff3e0; border-left: 4px solid #ff9800; }}
        .alert-info {{ background: #e3f2fd; border-left: 4px solid #2196f3; }}
        .alert-title {{ font-weight: bold; font-size: 18px; }}
        .alert-message {{ margin-top: 10px; }}
        .details {{ background: #f5f5f5; padding: 10px; margin-top: 10px; font-family: monospace; }}
    </style>
</head>
<body>
    <div class="alert alert-{severity}">
        <div class="alert-title">{title}</div>
        <div class="alert-message">{message}</div>
        {details_html}
    </div>
    <p style="color: #666; font-size: 12px;">
        Trading Bot Alert - {timestamp}
    </p>
</body>
</html>
"""


def format_daily_report_html(
    date: datetime,
    total_pnl: Decimal,
    total_trades: int,
    win_rate: Decimal,
    ending_balance: Decimal,
    trades: List[Dict[str, Any]],
) -> str:
    """Format daily report as HTML."""
    trade_rows = ""
    for trade in trades:
        pnl = trade.get('pnl', 0)
        trade_rows += TRADE_ROW_HTML.format(
            time=trade.get('exit_time', '')[:8] if trade.get('exit_time') else '',
            symbol=trade.get('symbol', ''),
            side=trade.get('side', ''),
            entry_price=trade.get('entry_price', 0),
            exit_price=trade.get('exit_price', 0),
            pnl=pnl,
            pnl_class='positive' if pnl >= 0 else 'negative',
        )

    return DAILY_SUMMARY_HTML.format(
        date=date.strftime('%B %d, %Y'),
        total_pnl=float(total_pnl),
        pnl_class='positive' if total_pnl >= 0 else 'negative',
        total_trades=total_trades,
        win_rate=float(win_rate),
        ending_balance=float(ending_balance),
        trade_rows=trade_rows,
        generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
    )


def format_alert_html(
    severity: str,
    title: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> str:
    """Format alert as HTML email."""
    details_html = ""
    if details:
        details_str = "\n".join(f"{k}: {v}" for k, v in details.items())
        details_html = f'<div class="details">{details_str}</div>'

    return ALERT_EMAIL_HTML.format(
        severity=severity,
        title=title,
        message=message,
        details_html=details_html,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
    )


# Markdown templates for Telegram
DAILY_SUMMARY_MARKDOWN = """
*Daily Trading Report*
_{date}_

*Summary*
├ P&L: `{pnl_emoji} ${total_pnl:,.2f}`
├ Trades: `{total_trades}`
├ Win Rate: `{win_rate:.1f}%`
└ Balance: `${ending_balance:,.2f}`

*Performance*
├ Best Trade: `${best_trade:,.2f}`
├ Worst Trade: `${worst_trade:,.2f}`
└ Avg Trade: `${avg_trade:,.2f}`

{trade_summary}
"""


def format_daily_report_markdown(
    date: datetime,
    total_pnl: Decimal,
    total_trades: int,
    win_rate: Decimal,
    ending_balance: Decimal,
    trades: List[Dict[str, Any]],
) -> str:
    """Format daily report as Markdown for Telegram."""
    pnl_emoji = "📈" if total_pnl >= 0 else "📉"

    pnls = [t.get('pnl', 0) for t in trades]
    best_trade = max(pnls) if pnls else 0
    worst_trade = min(pnls) if pnls else 0
    avg_trade = sum(pnls) / len(pnls) if pnls else 0

    trade_summary = ""
    if trades:
        trade_summary = "*Recent Trades*\n"
        for trade in trades[-5:]:
            emoji = "✅" if trade.get('pnl', 0) >= 0 else "❌"
            trade_summary += f"{emoji} {trade.get('symbol')}: ${trade.get('pnl', 0):,.2f}\n"

    return DAILY_SUMMARY_MARKDOWN.format(
        date=date.strftime('%B %d, %Y'),
        pnl_emoji=pnl_emoji,
        total_pnl=float(total_pnl),
        total_trades=total_trades,
        win_rate=float(win_rate),
        ending_balance=float(ending_balance),
        best_trade=float(best_trade),
        worst_trade=float(worst_trade),
        avg_trade=float(avg_trade),
        trade_summary=trade_summary,
    )
