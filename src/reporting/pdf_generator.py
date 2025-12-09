"""
PDF Report Generator for Trading Bot

Generates professional PDF reports with charts, tables, and analytics.
"""
import io
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from dataclasses import dataclass

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image, PageBreak, HRFlowable
    )
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.widgets.markers import makeMarker
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from loguru import logger


@dataclass
class ReportData:
    """Data structure for report generation."""
    # Period info
    start_date: datetime
    end_date: datetime
    report_type: str  # daily, weekly, monthly, custom

    # Performance metrics
    total_pnl: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal
    profit_factor: Decimal
    sharpe_ratio: Decimal
    max_drawdown: Decimal
    average_trade: Decimal

    # Equity curve
    equity_curve: List[Dict[str, Any]]

    # Trade list
    trades: List[Dict[str, Any]]

    # Strategy breakdown
    strategy_performance: Dict[str, Dict[str, Any]]

    # Symbol breakdown
    symbol_performance: Dict[str, Dict[str, Any]]

    # Account info
    starting_balance: Decimal
    ending_balance: Decimal


class PDFReportGenerator:
    """Generates PDF reports for trading performance."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if REPORTLAB_AVAILABLE:
            self.styles = getSampleStyleSheet()
            self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='Title',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1a1a2e'),
        ))

        self.styles.add(ParagraphStyle(
            name='Subtitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.HexColor('#4a4a6a'),
        ))

        self.styles.add(ParagraphStyle(
            name='MetricLabel',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.gray,
        ))

        self.styles.add(ParagraphStyle(
            name='MetricValue',
            parent=self.styles['Normal'],
            fontSize=16,
            fontName='Helvetica-Bold',
        ))

    def generate_report(
        self,
        data: ReportData,
        filename: Optional[str] = None,
    ) -> str:
        """
        Generate a PDF trading report.

        Args:
            data: Report data
            filename: Optional output filename

        Returns:
            Path to generated PDF
        """
        if not REPORTLAB_AVAILABLE:
            logger.warning("reportlab not installed, generating text report")
            return self._generate_text_report(data, filename)

        if filename is None:
            filename = f"trading_report_{data.start_date.strftime('%Y%m%d')}_{data.end_date.strftime('%Y%m%d')}.pdf"

        filepath = self.output_dir / filename

        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
        )

        story = []

        # Title
        story.append(Paragraph("Trading Performance Report", self.styles['Title']))
        story.append(Paragraph(
            f"{data.start_date.strftime('%B %d, %Y')} - {data.end_date.strftime('%B %d, %Y')}",
            self.styles['Subtitle']
        ))
        story.append(Spacer(1, 20))

        # Summary section
        story.extend(self._build_summary_section(data))
        story.append(Spacer(1, 20))

        # Equity curve chart
        story.extend(self._build_equity_chart(data))
        story.append(Spacer(1, 20))

        # Performance metrics
        story.extend(self._build_metrics_table(data))
        story.append(PageBreak())

        # Strategy breakdown
        story.extend(self._build_strategy_breakdown(data))
        story.append(Spacer(1, 20))

        # Symbol breakdown
        story.extend(self._build_symbol_breakdown(data))
        story.append(PageBreak())

        # Trade list
        story.extend(self._build_trade_list(data))

        # Build PDF
        doc.build(story)
        logger.info(f"Generated PDF report: {filepath}")
        return str(filepath)

    def _build_summary_section(self, data: ReportData) -> List:
        """Build summary section with key metrics."""
        elements = []

        elements.append(Paragraph("Summary", self.styles['Heading2']))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
        elements.append(Spacer(1, 10))

        # Summary table
        summary_data = [
            ["Starting Balance", f"${data.starting_balance:,.2f}",
             "Ending Balance", f"${data.ending_balance:,.2f}"],
            ["Total P&L", f"${data.total_pnl:,.2f}",
             "Return", f"{((data.ending_balance / data.starting_balance - 1) * 100):.2f}%"],
            ["Total Trades", str(data.total_trades),
             "Win Rate", f"{data.win_rate:.1f}%"],
            ["Winning Trades", str(data.winning_trades),
             "Losing Trades", str(data.losing_trades)],
        ]

        table = Table(summary_data, colWidths=[2*inch, 1.5*inch, 2*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
            ('FONTNAME', (3, 0), (3, -1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (1, 0), (1, 0), colors.green if data.total_pnl >= 0 else colors.red),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))

        elements.append(table)
        return elements

    def _build_equity_chart(self, data: ReportData) -> List:
        """Build equity curve chart."""
        elements = []

        elements.append(Paragraph("Equity Curve", self.styles['Heading2']))
        elements.append(Spacer(1, 10))

        if not data.equity_curve:
            elements.append(Paragraph("No equity data available", self.styles['Normal']))
            return elements

        # Create matplotlib chart
        fig, ax = plt.subplots(figsize=(7, 3))

        dates = [point['timestamp'] for point in data.equity_curve]
        equity = [float(point['equity']) for point in data.equity_curve]

        ax.plot(dates, equity, color='#2196F3', linewidth=1.5)
        ax.fill_between(dates, equity, alpha=0.3, color='#2196F3')

        ax.set_xlabel('')
        ax.set_ylabel('Equity ($)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.grid(True, alpha=0.3)

        # Save to buffer
        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format='png', dpi=150)
        plt.close(fig)
        buf.seek(0)

        # Add to PDF
        img = Image(buf, width=6.5*inch, height=2.5*inch)
        elements.append(img)

        return elements

    def _build_metrics_table(self, data: ReportData) -> List:
        """Build performance metrics table."""
        elements = []

        elements.append(Paragraph("Performance Metrics", self.styles['Heading2']))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
        elements.append(Spacer(1, 10))

        metrics_data = [
            ["Metric", "Value", "Metric", "Value"],
            ["Profit Factor", f"{data.profit_factor:.2f}",
             "Sharpe Ratio", f"{data.sharpe_ratio:.2f}"],
            ["Max Drawdown", f"{data.max_drawdown:.2f}%",
             "Average Trade", f"${data.average_trade:,.2f}"],
            ["Best Trade", f"${max((t.get('pnl', 0) for t in data.trades), default=0):,.2f}",
             "Worst Trade", f"${min((t.get('pnl', 0) for t in data.trades), default=0):,.2f}"],
        ]

        table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 2*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f5f5f5')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))

        elements.append(table)
        return elements

    def _build_strategy_breakdown(self, data: ReportData) -> List:
        """Build strategy performance breakdown."""
        elements = []

        elements.append(Paragraph("Strategy Performance", self.styles['Heading2']))
        elements.append(Spacer(1, 10))

        if not data.strategy_performance:
            elements.append(Paragraph("No strategy data available", self.styles['Normal']))
            return elements

        table_data = [["Strategy", "Trades", "Win Rate", "P&L", "Profit Factor"]]

        for strategy, perf in data.strategy_performance.items():
            table_data.append([
                strategy,
                str(perf.get('trades', 0)),
                f"{perf.get('win_rate', 0):.1f}%",
                f"${perf.get('pnl', 0):,.2f}",
                f"{perf.get('profit_factor', 0):.2f}",
            ])

        table = Table(table_data, colWidths=[2*inch, 1*inch, 1.2*inch, 1.5*inch, 1.3*inch])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e3f2fd')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
        ]))

        elements.append(table)
        return elements

    def _build_symbol_breakdown(self, data: ReportData) -> List:
        """Build symbol performance breakdown."""
        elements = []

        elements.append(Paragraph("Symbol Performance", self.styles['Heading2']))
        elements.append(Spacer(1, 10))

        if not data.symbol_performance:
            elements.append(Paragraph("No symbol data available", self.styles['Normal']))
            return elements

        table_data = [["Symbol", "Trades", "Win Rate", "P&L"]]

        for symbol, perf in data.symbol_performance.items():
            table_data.append([
                symbol,
                str(perf.get('trades', 0)),
                f"{perf.get('win_rate', 0):.1f}%",
                f"${perf.get('pnl', 0):,.2f}",
            ])

        table = Table(table_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 2*inch])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8f5e9')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
        ]))

        elements.append(table)
        return elements

    def _build_trade_list(self, data: ReportData) -> List:
        """Build trade list table."""
        elements = []

        elements.append(Paragraph("Trade History", self.styles['Heading2']))
        elements.append(Spacer(1, 10))

        if not data.trades:
            elements.append(Paragraph("No trades in this period", self.styles['Normal']))
            return elements

        table_data = [["Date", "Symbol", "Side", "Entry", "Exit", "P&L"]]

        for trade in data.trades[-50:]:  # Last 50 trades
            pnl = trade.get('pnl', 0)
            table_data.append([
                trade.get('exit_time', '')[:10] if trade.get('exit_time') else '',
                trade.get('symbol', ''),
                trade.get('side', ''),
                f"${trade.get('entry_price', 0):,.2f}",
                f"${trade.get('exit_price', 0):,.2f}",
                f"${pnl:,.2f}",
            ])

        table = Table(table_data, colWidths=[1.2*inch, 1.2*inch, 0.8*inch, 1.2*inch, 1.2*inch, 1.2*inch])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f5f5f5')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('PADDING', (0, 0), (-1, -1), 4),
            ('ALIGN', (3, 0), (-1, -1), 'RIGHT'),
        ]))

        elements.append(table)
        return elements

    def _generate_text_report(self, data: ReportData, filename: Optional[str]) -> str:
        """Generate text report as fallback."""
        if filename is None:
            filename = f"trading_report_{data.start_date.strftime('%Y%m%d')}.txt"

        filepath = self.output_dir / filename

        report = f"""
TRADING PERFORMANCE REPORT
{data.start_date.strftime('%B %d, %Y')} - {data.end_date.strftime('%B %d, %Y')}
{'=' * 60}

SUMMARY
-------
Starting Balance: ${data.starting_balance:,.2f}
Ending Balance:   ${data.ending_balance:,.2f}
Total P&L:        ${data.total_pnl:,.2f}
Return:           {((data.ending_balance / data.starting_balance - 1) * 100):.2f}%

PERFORMANCE METRICS
-------------------
Total Trades:    {data.total_trades}
Winning Trades:  {data.winning_trades}
Losing Trades:   {data.losing_trades}
Win Rate:        {data.win_rate:.1f}%
Profit Factor:   {data.profit_factor:.2f}
Sharpe Ratio:    {data.sharpe_ratio:.2f}
Max Drawdown:    {data.max_drawdown:.2f}%

STRATEGY BREAKDOWN
------------------
"""
        for strategy, perf in data.strategy_performance.items():
            report += f"{strategy}: {perf.get('trades', 0)} trades, ${perf.get('pnl', 0):,.2f}\n"

        filepath.write_text(report)
        return str(filepath)


def generate_daily_report(trades: List, equity_curve: List, metrics: Dict) -> str:
    """Convenience function to generate daily report."""
    generator = PDFReportGenerator()

    data = ReportData(
        start_date=datetime.now().replace(hour=0, minute=0, second=0),
        end_date=datetime.now(),
        report_type="daily",
        total_pnl=Decimal(str(metrics.get('total_pnl', 0))),
        realized_pnl=Decimal(str(metrics.get('realized_pnl', 0))),
        unrealized_pnl=Decimal(str(metrics.get('unrealized_pnl', 0))),
        total_trades=metrics.get('total_trades', 0),
        winning_trades=metrics.get('winning_trades', 0),
        losing_trades=metrics.get('losing_trades', 0),
        win_rate=Decimal(str(metrics.get('win_rate', 0))),
        profit_factor=Decimal(str(metrics.get('profit_factor', 0))),
        sharpe_ratio=Decimal(str(metrics.get('sharpe_ratio', 0))),
        max_drawdown=Decimal(str(metrics.get('max_drawdown', 0))),
        average_trade=Decimal(str(metrics.get('average_trade', 0))),
        equity_curve=equity_curve,
        trades=trades,
        strategy_performance=metrics.get('strategy_performance', {}),
        symbol_performance=metrics.get('symbol_performance', {}),
        starting_balance=Decimal(str(metrics.get('starting_balance', 10000))),
        ending_balance=Decimal(str(metrics.get('ending_balance', 10000))),
    )

    return generator.generate_report(data)
