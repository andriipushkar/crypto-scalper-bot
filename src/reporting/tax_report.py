"""
Tax Report Generator for Trading Bot

Generates tax reports for crypto trading in various formats.
"""
import csv
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
import io

from loguru import logger


class TaxMethod(Enum):
    """Tax calculation methods."""
    FIFO = "fifo"       # First In, First Out
    LIFO = "lifo"       # Last In, First Out
    HIFO = "hifo"       # Highest In, First Out
    AVERAGE = "average"  # Average Cost Basis


class TaxJurisdiction(Enum):
    """Supported tax jurisdictions."""
    US = "us"
    UK = "uk"
    EU = "eu"
    CANADA = "canada"
    AUSTRALIA = "australia"
    UKRAINE = "ukraine"


@dataclass
class TaxLot:
    """Represents a tax lot for cost basis tracking."""
    lot_id: str
    symbol: str
    quantity: Decimal
    cost_basis: Decimal  # Per unit
    acquisition_date: datetime
    source: str  # trade, transfer, airdrop


@dataclass
class TaxableEvent:
    """Represents a taxable event."""
    event_id: str
    event_type: str  # sale, trade, transfer
    timestamp: datetime
    symbol: str
    quantity: Decimal
    proceeds: Decimal
    cost_basis: Decimal
    gain_loss: Decimal
    holding_period: str  # short-term, long-term
    matched_lots: List[str] = field(default_factory=list)


@dataclass
class TaxSummary:
    """Tax summary for a period."""
    year: int
    jurisdiction: TaxJurisdiction

    # Gains/Losses
    short_term_gains: Decimal = Decimal("0")
    short_term_losses: Decimal = Decimal("0")
    long_term_gains: Decimal = Decimal("0")
    long_term_losses: Decimal = Decimal("0")
    net_gain_loss: Decimal = Decimal("0")

    # Totals
    total_proceeds: Decimal = Decimal("0")
    total_cost_basis: Decimal = Decimal("0")
    total_trades: int = 0

    # Fees
    total_fees: Decimal = Decimal("0")

    # Events
    events: List[TaxableEvent] = field(default_factory=list)


class TaxReportGenerator:
    """Generates tax reports for crypto trading."""

    # Short-term holding period by jurisdiction (days)
    SHORT_TERM_PERIODS = {
        TaxJurisdiction.US: 365,
        TaxJurisdiction.UK: 0,  # No distinction in UK
        TaxJurisdiction.CANADA: 0,  # All capital gains
        TaxJurisdiction.AUSTRALIA: 365,
        TaxJurisdiction.UKRAINE: 365,
    }

    def __init__(
        self,
        jurisdiction: TaxJurisdiction = TaxJurisdiction.US,
        method: TaxMethod = TaxMethod.FIFO,
        output_dir: str = "reports/tax",
    ):
        self.jurisdiction = jurisdiction
        self.method = method
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tax_lots: Dict[str, List[TaxLot]] = {}  # symbol -> lots

    def add_acquisition(
        self,
        symbol: str,
        quantity: Decimal,
        cost_basis: Decimal,
        timestamp: datetime,
        source: str = "trade",
    ) -> TaxLot:
        """Add a new tax lot from acquisition."""
        lot = TaxLot(
            lot_id=f"{symbol}-{timestamp.timestamp()}-{quantity}",
            symbol=symbol,
            quantity=quantity,
            cost_basis=cost_basis,
            acquisition_date=timestamp,
            source=source,
        )

        if symbol not in self.tax_lots:
            self.tax_lots[symbol] = []

        self.tax_lots[symbol].append(lot)
        return lot

    def process_disposal(
        self,
        symbol: str,
        quantity: Decimal,
        proceeds: Decimal,
        timestamp: datetime,
    ) -> TaxableEvent:
        """Process a disposal and calculate gain/loss."""
        if symbol not in self.tax_lots:
            raise ValueError(f"No tax lots found for {symbol}")

        # Select lots based on method
        lots = self._select_lots(symbol, quantity)

        total_cost_basis = Decimal("0")
        matched_lot_ids = []
        remaining_qty = quantity

        for lot in lots:
            if remaining_qty <= 0:
                break

            lot_qty = min(lot.quantity, remaining_qty)
            total_cost_basis += lot_qty * lot.cost_basis
            matched_lot_ids.append(lot.lot_id)

            # Reduce lot quantity
            lot.quantity -= lot_qty
            remaining_qty -= lot_qty

        # Remove empty lots
        self.tax_lots[symbol] = [l for l in self.tax_lots[symbol] if l.quantity > 0]

        # Calculate gain/loss
        gain_loss = proceeds - total_cost_basis

        # Determine holding period
        earliest_lot = min(lots, key=lambda l: l.acquisition_date)
        holding_days = (timestamp - earliest_lot.acquisition_date).days
        short_term_threshold = self.SHORT_TERM_PERIODS.get(self.jurisdiction, 365)
        holding_period = "short-term" if holding_days <= short_term_threshold else "long-term"

        return TaxableEvent(
            event_id=f"disposal-{timestamp.timestamp()}",
            event_type="sale",
            timestamp=timestamp,
            symbol=symbol,
            quantity=quantity,
            proceeds=proceeds,
            cost_basis=total_cost_basis,
            gain_loss=gain_loss,
            holding_period=holding_period,
            matched_lots=matched_lot_ids,
        )

    def _select_lots(self, symbol: str, quantity: Decimal) -> List[TaxLot]:
        """Select tax lots based on accounting method."""
        lots = self.tax_lots.get(symbol, [])

        if self.method == TaxMethod.FIFO:
            # Sort by acquisition date (oldest first)
            return sorted(lots, key=lambda l: l.acquisition_date)

        elif self.method == TaxMethod.LIFO:
            # Sort by acquisition date (newest first)
            return sorted(lots, key=lambda l: l.acquisition_date, reverse=True)

        elif self.method == TaxMethod.HIFO:
            # Sort by cost basis (highest first)
            return sorted(lots, key=lambda l: l.cost_basis, reverse=True)

        elif self.method == TaxMethod.AVERAGE:
            # Calculate average cost and treat as single lot
            total_qty = sum(l.quantity for l in lots)
            if total_qty == 0:
                return []
            avg_cost = sum(l.quantity * l.cost_basis for l in lots) / total_qty
            return [TaxLot(
                lot_id="average",
                symbol=symbol,
                quantity=total_qty,
                cost_basis=avg_cost,
                acquisition_date=min(l.acquisition_date for l in lots),
                source="average",
            )]

        return lots

    def generate_annual_report(
        self,
        year: int,
        trades: List[Dict[str, Any]],
    ) -> TaxSummary:
        """Generate annual tax report from trades."""
        summary = TaxSummary(
            year=year,
            jurisdiction=self.jurisdiction,
        )

        # Filter trades for the year
        year_start = datetime(year, 1, 1)
        year_end = datetime(year, 12, 31, 23, 59, 59)

        for trade in trades:
            trade_time = trade.get('exit_time') or trade.get('timestamp')
            if isinstance(trade_time, str):
                trade_time = datetime.fromisoformat(trade_time.replace('Z', '+00:00'))

            if not (year_start <= trade_time <= year_end):
                continue

            # Process acquisition (entry)
            entry_time = trade.get('entry_time')
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))

            symbol = trade.get('symbol', 'UNKNOWN')
            quantity = Decimal(str(trade.get('quantity', 0)))
            entry_price = Decimal(str(trade.get('entry_price', 0)))
            exit_price = Decimal(str(trade.get('exit_price', 0)))

            # Add acquisition
            self.add_acquisition(
                symbol=symbol,
                quantity=quantity,
                cost_basis=entry_price,
                timestamp=entry_time,
            )

            # Process disposal
            proceeds = quantity * exit_price
            event = self.process_disposal(
                symbol=symbol,
                quantity=quantity,
                proceeds=proceeds,
                timestamp=trade_time,
            )

            summary.events.append(event)
            summary.total_trades += 1
            summary.total_proceeds += proceeds
            summary.total_cost_basis += event.cost_basis

            # Categorize gain/loss
            if event.holding_period == "short-term":
                if event.gain_loss >= 0:
                    summary.short_term_gains += event.gain_loss
                else:
                    summary.short_term_losses += abs(event.gain_loss)
            else:
                if event.gain_loss >= 0:
                    summary.long_term_gains += event.gain_loss
                else:
                    summary.long_term_losses += abs(event.gain_loss)

            # Add fees
            fees = Decimal(str(trade.get('commission', 0) or trade.get('fees', 0)))
            summary.total_fees += fees

        # Calculate net gain/loss
        summary.net_gain_loss = (
            summary.short_term_gains - summary.short_term_losses +
            summary.long_term_gains - summary.long_term_losses
        )

        return summary

    def export_csv(self, summary: TaxSummary, filename: Optional[str] = None) -> str:
        """Export tax report to CSV (IRS Form 8949 format)."""
        if filename is None:
            filename = f"tax_report_{summary.year}_{self.jurisdiction.value}.csv"

        filepath = self.output_dir / filename

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header (Form 8949 format)
            writer.writerow([
                "Description",
                "Date Acquired",
                "Date Sold",
                "Proceeds",
                "Cost Basis",
                "Adjustment Code",
                "Adjustment Amount",
                "Gain or Loss",
            ])

            for event in summary.events:
                # Get acquisition date from matched lots
                acq_date = event.timestamp - timedelta(days=30)  # Simplified

                writer.writerow([
                    f"{event.quantity} {event.symbol}",
                    acq_date.strftime("%m/%d/%Y"),
                    event.timestamp.strftime("%m/%d/%Y"),
                    f"{event.proceeds:.2f}",
                    f"{event.cost_basis:.2f}",
                    "",
                    "",
                    f"{event.gain_loss:.2f}",
                ])

            # Summary rows
            writer.writerow([])
            writer.writerow(["SUMMARY"])
            writer.writerow(["Short-term Gains", f"${summary.short_term_gains:.2f}"])
            writer.writerow(["Short-term Losses", f"${summary.short_term_losses:.2f}"])
            writer.writerow(["Long-term Gains", f"${summary.long_term_gains:.2f}"])
            writer.writerow(["Long-term Losses", f"${summary.long_term_losses:.2f}"])
            writer.writerow(["Net Gain/Loss", f"${summary.net_gain_loss:.2f}"])
            writer.writerow(["Total Fees", f"${summary.total_fees:.2f}"])

        logger.info(f"Generated tax CSV: {filepath}")
        return str(filepath)

    def export_json(self, summary: TaxSummary, filename: Optional[str] = None) -> str:
        """Export tax report to JSON."""
        if filename is None:
            filename = f"tax_report_{summary.year}_{self.jurisdiction.value}.json"

        filepath = self.output_dir / filename

        data = {
            "year": summary.year,
            "jurisdiction": summary.jurisdiction.value,
            "method": self.method.value,
            "summary": {
                "short_term_gains": str(summary.short_term_gains),
                "short_term_losses": str(summary.short_term_losses),
                "long_term_gains": str(summary.long_term_gains),
                "long_term_losses": str(summary.long_term_losses),
                "net_gain_loss": str(summary.net_gain_loss),
                "total_proceeds": str(summary.total_proceeds),
                "total_cost_basis": str(summary.total_cost_basis),
                "total_trades": summary.total_trades,
                "total_fees": str(summary.total_fees),
            },
            "events": [
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type,
                    "timestamp": e.timestamp.isoformat(),
                    "symbol": e.symbol,
                    "quantity": str(e.quantity),
                    "proceeds": str(e.proceeds),
                    "cost_basis": str(e.cost_basis),
                    "gain_loss": str(e.gain_loss),
                    "holding_period": e.holding_period,
                }
                for e in summary.events
            ],
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Generated tax JSON: {filepath}")
        return str(filepath)

    def generate_summary_text(self, summary: TaxSummary) -> str:
        """Generate human-readable tax summary."""
        return f"""
TAX SUMMARY - {summary.year}
Jurisdiction: {summary.jurisdiction.value.upper()}
Method: {self.method.value.upper()}
{'=' * 50}

CAPITAL GAINS/LOSSES
--------------------
Short-term Gains:   ${summary.short_term_gains:>12,.2f}
Short-term Losses:  ${summary.short_term_losses:>12,.2f}
Long-term Gains:    ${summary.long_term_gains:>12,.2f}
Long-term Losses:   ${summary.long_term_losses:>12,.2f}

NET GAIN/LOSS:      ${summary.net_gain_loss:>12,.2f}

TOTALS
------
Total Proceeds:     ${summary.total_proceeds:>12,.2f}
Total Cost Basis:   ${summary.total_cost_basis:>12,.2f}
Total Trades:       {summary.total_trades:>12}
Total Fees:         ${summary.total_fees:>12,.2f}

Note: This is for informational purposes only.
Consult a tax professional for official tax filing.
"""
