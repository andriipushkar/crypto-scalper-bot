"""
Tests for Multi-Account Management

Comprehensive tests for multi-account trading.
"""
import pytest
import asyncio
from datetime import datetime
from decimal import Decimal

from src.trading.multi_account import (
    MultiAccountManager, Account, AccountGroup,
    AccountType, AccountStatus, MockExchangeClient
)


class TestAccount:
    """Tests for Account dataclass."""

    def test_account_creation(self):
        """Test account creation."""
        account = Account(
            id="acc_001",
            name="Main Account",
            exchange="binance",
            account_type=AccountType.FUTURES,
            api_key="test_key",
            api_secret="test_secret",
        )

        assert account.id == "acc_001"
        assert account.name == "Main Account"
        assert account.status == AccountStatus.INACTIVE

    def test_account_to_dict(self):
        """Test account serialization."""
        account = Account(
            id="acc_001",
            name="Test",
            exchange="binance",
            account_type=AccountType.FUTURES,
            api_key="key",
            api_secret="secret",
        )

        data = account.to_dict(include_secrets=False)

        assert "id" in data
        assert "api_key" not in data
        assert "api_secret" not in data

        data_with_secrets = account.to_dict(include_secrets=True)
        assert "api_key" in data_with_secrets

    def test_account_default_values(self):
        """Test account default values."""
        account = Account(
            id="acc_001",
            name="Test",
            exchange="binance",
            account_type=AccountType.SPOT,
            api_key="key",
            api_secret="secret",
        )

        assert account.testnet is False
        assert len(account.balance) == 0
        assert len(account.positions) == 0


class TestAccountGroup:
    """Tests for AccountGroup."""

    def test_group_creation(self):
        """Test group creation."""
        group = AccountGroup(
            id="grp_001",
            name="Main Group",
            account_ids=["acc_001", "acc_002"],
            allocation={"acc_001": Decimal("60"), "acc_002": Decimal("40")},
        )

        assert group.name == "Main Group"
        assert len(group.account_ids) == 2
        assert group.allocation["acc_001"] == Decimal("60")


class TestMultiAccountManager:
    """Tests for MultiAccountManager."""

    @pytest.fixture
    def manager(self):
        """Create account manager."""
        return MultiAccountManager()

    def test_add_account(self, manager):
        """Test adding account."""
        account = manager.add_account(
            name="Test Account",
            exchange="binance",
            api_key="test_key",
            api_secret="test_secret",
            account_type=AccountType.FUTURES,
        )

        assert account.id in manager.accounts
        assert account.name == "Test Account"
        assert len(manager.accounts) == 1

    def test_remove_account(self, manager):
        """Test removing account."""
        account = manager.add_account(
            name="Test",
            exchange="binance",
            api_key="key",
            api_secret="secret",
        )

        result = manager.remove_account(account.id)

        assert result is True
        assert account.id not in manager.accounts

    def test_remove_nonexistent_account(self, manager):
        """Test removing non-existent account."""
        result = manager.remove_account("nonexistent")
        assert result is False

    def test_get_account(self, manager):
        """Test getting account."""
        account = manager.add_account(
            name="Test",
            exchange="binance",
            api_key="key",
            api_secret="secret",
        )

        retrieved = manager.get_account(account.id)
        assert retrieved is account

        assert manager.get_account("nonexistent") is None

    def test_get_accounts_by_exchange(self, manager):
        """Test filtering accounts by exchange."""
        manager.add_account("Binance 1", "binance", "k1", "s1")
        manager.add_account("Binance 2", "binance", "k2", "s2")
        manager.add_account("Bybit 1", "bybit", "k3", "s3")

        binance_accounts = manager.get_accounts_by_exchange("binance")

        assert len(binance_accounts) == 2

    def test_get_active_accounts(self, manager):
        """Test getting active accounts."""
        acc1 = manager.add_account("Test 1", "binance", "k1", "s1")
        acc2 = manager.add_account("Test 2", "binance", "k2", "s2")

        acc1.status = AccountStatus.ACTIVE

        active = manager.get_active_accounts()

        assert len(active) == 1
        assert active[0] is acc1

    def test_create_group(self, manager):
        """Test creating account group."""
        acc1 = manager.add_account("Test 1", "binance", "k1", "s1")
        acc2 = manager.add_account("Test 2", "binance", "k2", "s2")

        group = manager.create_group(
            name="Main Group",
            account_ids=[acc1.id, acc2.id],
        )

        assert group.id in manager.groups
        assert len(group.account_ids) == 2
        # Default equal allocation
        assert group.allocation[acc1.id] == Decimal("50")

    def test_create_group_with_allocation(self, manager):
        """Test creating group with custom allocation."""
        acc1 = manager.add_account("Test 1", "binance", "k1", "s1")
        acc2 = manager.add_account("Test 2", "binance", "k2", "s2")

        group = manager.create_group(
            name="Custom Group",
            account_ids=[acc1.id, acc2.id],
            allocation={acc1.id: Decimal("70"), acc2.id: Decimal("30")},
        )

        assert group.allocation[acc1.id] == Decimal("70")
        assert group.allocation[acc2.id] == Decimal("30")

    def test_create_group_invalid_account(self, manager):
        """Test creating group with invalid account."""
        with pytest.raises(ValueError):
            manager.create_group(
                name="Invalid Group",
                account_ids=["nonexistent"],
            )

    @pytest.mark.asyncio
    async def test_connect_account(self, manager):
        """Test connecting account."""
        account = manager.add_account(
            name="Test",
            exchange="binance",
            api_key="key",
            api_secret="secret",
        )

        result = await manager.connect_account(account.id)

        assert result is True
        assert account.status == AccountStatus.ACTIVE
        assert account.id in manager.exchange_clients

    @pytest.mark.asyncio
    async def test_disconnect_account(self, manager):
        """Test disconnecting account."""
        account = manager.add_account("Test", "binance", "k", "s")
        await manager.connect_account(account.id)

        result = await manager.disconnect_account(account.id)

        assert result is True
        assert account.status == AccountStatus.INACTIVE
        assert account.id not in manager.exchange_clients

    @pytest.mark.asyncio
    async def test_connect_all(self, manager):
        """Test connecting all accounts."""
        manager.add_account("Test 1", "binance", "k1", "s1")
        manager.add_account("Test 2", "binance", "k2", "s2")

        results = await manager.connect_all()

        assert len(results) == 2
        assert all(results.values())

    @pytest.mark.asyncio
    async def test_place_order(self, manager):
        """Test placing order on account."""
        account = manager.add_account("Test", "binance", "k", "s")
        await manager.connect_account(account.id)

        result = await manager.place_order(
            account_id=account.id,
            symbol="BTCUSDT",
            side="buy",
            quantity=Decimal("0.1"),
        )

        assert result is not None
        assert result["status"] == "filled"

    @pytest.mark.asyncio
    async def test_place_group_order(self, manager):
        """Test placing order across group."""
        acc1 = manager.add_account("Test 1", "binance", "k1", "s1")
        acc2 = manager.add_account("Test 2", "binance", "k2", "s2")

        await manager.connect_all()

        group = manager.create_group(
            name="Group",
            account_ids=[acc1.id, acc2.id],
            allocation={acc1.id: Decimal("60"), acc2.id: Decimal("40")},
        )

        results = await manager.place_group_order(
            group_id=group.id,
            symbol="BTCUSDT",
            side="buy",
            total_quantity=Decimal("1"),
        )

        assert len(results) == 2
        assert all(r is not None for r in results.values())

    def test_get_total_balance(self, manager):
        """Test getting total balance across accounts."""
        acc1 = manager.add_account("Test 1", "binance", "k1", "s1")
        acc2 = manager.add_account("Test 2", "binance", "k2", "s2")

        acc1.balance["USDT"] = Decimal("5000")
        acc2.balance["USDT"] = Decimal("3000")

        total = manager.get_total_balance("USDT")

        assert total == Decimal("8000")

    def test_get_total_exposure(self, manager):
        """Test getting total exposure."""
        acc1 = manager.add_account("Test 1", "binance", "k1", "s1")
        acc2 = manager.add_account("Test 2", "binance", "k2", "s2")

        acc1.positions["BTCUSDT"] = {"quantity": 1, "side": "long"}
        acc2.positions["BTCUSDT"] = {"quantity": 0.5, "side": "long"}
        acc2.positions["ETHUSDT"] = {"quantity": 2, "side": "short"}

        exposure = manager.get_total_exposure()

        assert exposure["BTCUSDT"] == Decimal("1.5")
        assert exposure["ETHUSDT"] == Decimal("-2")

    def test_get_summary(self, manager):
        """Test getting summary."""
        manager.add_account("Test 1", "binance", "k1", "s1")
        manager.add_account("Test 2", "binance", "k2", "s2")

        summary = manager.get_summary()

        assert summary["total_accounts"] == 2
        assert "accounts" in summary


class TestMockExchangeClient:
    """Tests for mock exchange client."""

    @pytest.fixture
    def client(self):
        """Create mock client."""
        account = Account(
            id="test",
            name="Test",
            exchange="binance",
            account_type=AccountType.FUTURES,
            api_key="key",
            api_secret="secret",
        )
        return MockExchangeClient(account)

    @pytest.mark.asyncio
    async def test_get_balance(self, client):
        """Test getting balance."""
        balance = await client.get_balance()

        assert "USDT" in balance
        assert balance["USDT"] == 10000.0

    @pytest.mark.asyncio
    async def test_get_positions(self, client):
        """Test getting positions."""
        positions = await client.get_positions()

        assert isinstance(positions, dict)

    @pytest.mark.asyncio
    async def test_place_order(self, client):
        """Test placing order."""
        result = await client.place_order(
            symbol="BTCUSDT",
            side="buy",
            quantity=0.1,
            order_type="market",
        )

        assert "order_id" in result
        assert result["status"] == "filled"

    @pytest.mark.asyncio
    async def test_close_position(self, client):
        """Test closing position."""
        result = await client.close_position("BTCUSDT")
        assert result is True
