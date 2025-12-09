"""
Multi-Account Management

Manage multiple trading accounts across different exchanges.
"""
import asyncio
from typing import Any, Callable, Dict, List, Optional, Set
from datetime import datetime
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import uuid

from loguru import logger


class AccountType(Enum):
    """Account types."""
    SPOT = "spot"
    MARGIN = "margin"
    FUTURES = "futures"
    OPTIONS = "options"


class AccountStatus(Enum):
    """Account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    ERROR = "error"


@dataclass
class Account:
    """Trading account."""
    id: str
    name: str
    exchange: str
    account_type: AccountType
    api_key: str
    api_secret: str
    passphrase: Optional[str] = None  # For some exchanges
    testnet: bool = False
    status: AccountStatus = AccountStatus.INACTIVE
    balance: Dict[str, Decimal] = field(default_factory=dict)
    positions: Dict[str, Any] = field(default_factory=dict)
    last_sync: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "name": self.name,
            "exchange": self.exchange,
            "account_type": self.account_type.value,
            "testnet": self.testnet,
            "status": self.status.value,
            "balance": {k: str(v) for k, v in self.balance.items()},
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "created_at": self.created_at.isoformat(),
        }

        if include_secrets:
            data["api_key"] = self.api_key
            data["api_secret"] = self.api_secret[:8] + "..."

        return data


@dataclass
class AccountGroup:
    """Group of accounts for aggregate operations."""
    id: str
    name: str
    account_ids: List[str] = field(default_factory=list)
    allocation: Dict[str, Decimal] = field(default_factory=dict)  # account_id -> percentage
    created_at: datetime = field(default_factory=datetime.now)


class MultiAccountManager:
    """Manage multiple trading accounts."""

    def __init__(self):
        self.accounts: Dict[str, Account] = {}
        self.groups: Dict[str, AccountGroup] = {}
        self.exchange_clients: Dict[str, Any] = {}  # account_id -> exchange client

        # Callbacks
        self.on_balance_update: Optional[Callable] = None
        self.on_position_update: Optional[Callable] = None
        self.on_account_error: Optional[Callable] = None

        # Sync state
        self._sync_tasks: Dict[str, asyncio.Task] = {}

    def add_account(
        self,
        name: str,
        exchange: str,
        api_key: str,
        api_secret: str,
        account_type: AccountType = AccountType.FUTURES,
        passphrase: Optional[str] = None,
        testnet: bool = False,
        metadata: Optional[Dict] = None,
    ) -> Account:
        """Add a new trading account."""
        account = Account(
            id=str(uuid.uuid4())[:8],
            name=name,
            exchange=exchange.lower(),
            account_type=account_type,
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
            testnet=testnet,
            metadata=metadata or {},
        )

        self.accounts[account.id] = account
        logger.info(f"Added account: {name} ({exchange})")

        return account

    def remove_account(self, account_id: str) -> bool:
        """Remove an account."""
        if account_id not in self.accounts:
            return False

        # Stop sync task if running
        if account_id in self._sync_tasks:
            self._sync_tasks[account_id].cancel()
            del self._sync_tasks[account_id]

        # Remove from groups
        for group in self.groups.values():
            if account_id in group.account_ids:
                group.account_ids.remove(account_id)
            if account_id in group.allocation:
                del group.allocation[account_id]

        del self.accounts[account_id]
        logger.info(f"Removed account: {account_id}")

        return True

    def get_account(self, account_id: str) -> Optional[Account]:
        """Get account by ID."""
        return self.accounts.get(account_id)

    def get_accounts_by_exchange(self, exchange: str) -> List[Account]:
        """Get all accounts for an exchange."""
        return [a for a in self.accounts.values() if a.exchange == exchange.lower()]

    def get_active_accounts(self) -> List[Account]:
        """Get all active accounts."""
        return [a for a in self.accounts.values() if a.status == AccountStatus.ACTIVE]

    def create_group(
        self,
        name: str,
        account_ids: List[str],
        allocation: Optional[Dict[str, Decimal]] = None,
    ) -> AccountGroup:
        """Create an account group."""
        # Verify accounts exist
        for acc_id in account_ids:
            if acc_id not in self.accounts:
                raise ValueError(f"Account {acc_id} not found")

        # Default equal allocation
        if not allocation:
            pct = Decimal("100") / len(account_ids)
            allocation = {acc_id: pct for acc_id in account_ids}

        group = AccountGroup(
            id=str(uuid.uuid4())[:8],
            name=name,
            account_ids=account_ids,
            allocation=allocation,
        )

        self.groups[group.id] = group
        logger.info(f"Created group: {name} with {len(account_ids)} accounts")

        return group

    async def connect_account(self, account_id: str) -> bool:
        """Connect and initialize an account."""
        account = self.accounts.get(account_id)
        if not account:
            return False

        try:
            # Create exchange client based on exchange type
            client = await self._create_exchange_client(account)
            self.exchange_clients[account_id] = client

            # Sync initial state
            await self._sync_account(account)

            account.status = AccountStatus.ACTIVE
            logger.info(f"Connected account: {account.name}")

            return True

        except Exception as e:
            account.status = AccountStatus.ERROR
            logger.error(f"Failed to connect account {account.name}: {e}")

            if self.on_account_error:
                self.on_account_error(account, str(e))

            return False

    async def disconnect_account(self, account_id: str) -> bool:
        """Disconnect an account."""
        account = self.accounts.get(account_id)
        if not account:
            return False

        # Stop sync task
        if account_id in self._sync_tasks:
            self._sync_tasks[account_id].cancel()
            del self._sync_tasks[account_id]

        # Close client
        if account_id in self.exchange_clients:
            client = self.exchange_clients[account_id]
            if hasattr(client, 'close'):
                await client.close()
            del self.exchange_clients[account_id]

        account.status = AccountStatus.INACTIVE
        logger.info(f"Disconnected account: {account.name}")

        return True

    async def connect_all(self) -> Dict[str, bool]:
        """Connect all accounts."""
        results = {}
        for account_id in self.accounts:
            results[account_id] = await self.connect_account(account_id)
        return results

    async def disconnect_all(self) -> None:
        """Disconnect all accounts."""
        for account_id in list(self.exchange_clients.keys()):
            await self.disconnect_account(account_id)

    async def _create_exchange_client(self, account: Account) -> Any:
        """Create exchange client for account."""
        # This would integrate with your exchange clients
        # For now, return a mock client
        return MockExchangeClient(account)

    async def _sync_account(self, account: Account) -> None:
        """Sync account state from exchange."""
        client = self.exchange_clients.get(account.id)
        if not client:
            return

        try:
            # Sync balance
            balance = await client.get_balance()
            account.balance = {k: Decimal(str(v)) for k, v in balance.items()}

            # Sync positions
            positions = await client.get_positions()
            account.positions = positions

            account.last_sync = datetime.now()

            if self.on_balance_update:
                self.on_balance_update(account)

        except Exception as e:
            logger.error(f"Sync failed for {account.name}: {e}")

    def start_sync_loop(self, account_id: str, interval_seconds: int = 30):
        """Start background sync loop for account."""
        if account_id in self._sync_tasks:
            return

        async def sync_loop():
            while True:
                account = self.accounts.get(account_id)
                if not account or account.status != AccountStatus.ACTIVE:
                    break
                await self._sync_account(account)
                await asyncio.sleep(interval_seconds)

        task = asyncio.create_task(sync_loop())
        self._sync_tasks[account_id] = task

    def stop_sync_loop(self, account_id: str):
        """Stop background sync loop."""
        if account_id in self._sync_tasks:
            self._sync_tasks[account_id].cancel()
            del self._sync_tasks[account_id]

    async def place_order(
        self,
        account_id: str,
        symbol: str,
        side: str,
        quantity: Decimal,
        order_type: str = "market",
        price: Optional[Decimal] = None,
    ) -> Optional[Dict[str, Any]]:
        """Place order on specific account."""
        client = self.exchange_clients.get(account_id)
        if not client:
            return None

        try:
            result = await client.place_order(
                symbol=symbol,
                side=side,
                quantity=float(quantity),
                order_type=order_type,
                price=float(price) if price else None,
            )
            logger.info(f"Order placed on {account_id}: {result}")
            return result

        except Exception as e:
            logger.error(f"Order failed on {account_id}: {e}")
            return None

    async def place_group_order(
        self,
        group_id: str,
        symbol: str,
        side: str,
        total_quantity: Decimal,
        order_type: str = "market",
        price: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """Place order across account group with allocation."""
        group = self.groups.get(group_id)
        if not group:
            raise ValueError(f"Group {group_id} not found")

        results = {}
        tasks = []

        for account_id in group.account_ids:
            allocation_pct = group.allocation.get(account_id, Decimal("0"))
            quantity = total_quantity * allocation_pct / 100

            if quantity > 0:
                task = self.place_order(
                    account_id=account_id,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type=order_type,
                    price=price,
                )
                tasks.append((account_id, task))

        for account_id, task in tasks:
            results[account_id] = await task

        return results

    async def close_position(self, account_id: str, symbol: str) -> bool:
        """Close position on specific account."""
        client = self.exchange_clients.get(account_id)
        if not client:
            return False

        try:
            await client.close_position(symbol)
            return True
        except Exception as e:
            logger.error(f"Close position failed on {account_id}: {e}")
            return False

    async def close_all_positions(self, account_id: Optional[str] = None) -> Dict[str, bool]:
        """Close all positions on account(s)."""
        results = {}

        if account_id:
            account_ids = [account_id]
        else:
            account_ids = list(self.exchange_clients.keys())

        for acc_id in account_ids:
            account = self.accounts.get(acc_id)
            if account and account.positions:
                for symbol in list(account.positions.keys()):
                    results[f"{acc_id}:{symbol}"] = await self.close_position(acc_id, symbol)

        return results

    def get_total_balance(self, currency: str = "USDT") -> Decimal:
        """Get total balance across all accounts."""
        total = Decimal("0")
        for account in self.accounts.values():
            total += account.balance.get(currency, Decimal("0"))
        return total

    def get_total_exposure(self) -> Dict[str, Decimal]:
        """Get total exposure by symbol across accounts."""
        exposure: Dict[str, Decimal] = {}

        for account in self.accounts.values():
            for symbol, position in account.positions.items():
                qty = Decimal(str(position.get("quantity", 0)))
                if position.get("side") == "short":
                    qty = -qty
                exposure[symbol] = exposure.get(symbol, Decimal("0")) + qty

        return exposure

    def get_summary(self) -> Dict[str, Any]:
        """Get multi-account summary."""
        return {
            "total_accounts": len(self.accounts),
            "active_accounts": len(self.get_active_accounts()),
            "total_groups": len(self.groups),
            "total_balance_usdt": float(self.get_total_balance("USDT")),
            "total_exposure": {k: float(v) for k, v in self.get_total_exposure().items()},
            "accounts": [
                a.to_dict()
                for a in self.accounts.values()
            ],
        }


class MockExchangeClient:
    """Mock exchange client for testing."""

    def __init__(self, account: Account):
        self.account = account
        self._balance = {"USDT": 10000.0}
        self._positions = {}

    async def get_balance(self) -> Dict[str, float]:
        return self._balance.copy()

    async def get_positions(self) -> Dict[str, Any]:
        return self._positions.copy()

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        return {
            "order_id": str(uuid.uuid4())[:8],
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "status": "filled",
        }

    async def close_position(self, symbol: str) -> bool:
        if symbol in self._positions:
            del self._positions[symbol]
        return True

    async def close(self):
        pass
