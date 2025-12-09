"""
Kafka Client for Event-Driven Architecture

Provides async Kafka producer and consumer for trading events.
"""
import asyncio
import json
from typing import Any, Callable, Dict, List, Optional, Set
from datetime import datetime
from decimal import Decimal
from dataclasses import dataclass, asdict
import logging

try:
    from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
    from aiokafka.errors import KafkaError, KafkaConnectionError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

from loguru import logger


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types."""

    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


@dataclass
class KafkaConfig:
    """Kafka configuration."""
    bootstrap_servers: str = "localhost:9092"
    client_id: str = "trading-bot"
    group_id: str = "trading-bot-group"

    # Producer settings
    acks: str = "all"
    compression_type: str = "gzip"
    max_batch_size: int = 16384
    linger_ms: int = 5

    # Consumer settings
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = True
    auto_commit_interval_ms: int = 1000
    max_poll_records: int = 500

    # Security
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None

    # SSL
    ssl_cafile: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None


class KafkaProducer:
    """Async Kafka producer for publishing trading events."""

    # Topic definitions
    TOPICS = {
        "market_data": "trading.market-data",
        "signals": "trading.signals",
        "orders": "trading.orders",
        "positions": "trading.positions",
        "trades": "trading.trades",
        "alerts": "trading.alerts",
        "metrics": "trading.metrics",
        "audit": "trading.audit",
    }

    def __init__(self, config: KafkaConfig):
        self.config = config
        self._producer: Optional[AIOKafkaProducer] = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to Kafka cluster."""
        if not KAFKA_AVAILABLE:
            logger.warning("aiokafka not installed, using mock producer")
            self._connected = True
            return

        try:
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self.config.bootstrap_servers,
                client_id=self.config.client_id,
                acks=self.config.acks,
                compression_type=self.config.compression_type,
                max_batch_size=self.config.max_batch_size,
                linger_ms=self.config.linger_ms,
                value_serializer=lambda v: json.dumps(v, cls=DecimalEncoder).encode(),
                key_serializer=lambda k: k.encode() if k else None,
            )
            await self._producer.start()
            self._connected = True
            logger.info(f"Connected to Kafka: {self.config.bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Kafka cluster."""
        if self._producer:
            await self._producer.stop()
            self._connected = False
            logger.info("Disconnected from Kafka")

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def publish(
        self,
        topic: str,
        message: Dict[str, Any],
        key: Optional[str] = None,
        partition: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> bool:
        """
        Publish message to Kafka topic.

        Args:
            topic: Topic name or key from TOPICS
            message: Message payload
            key: Optional message key for partitioning
            partition: Optional specific partition
            headers: Optional message headers

        Returns:
            True if published successfully
        """
        # Resolve topic name
        topic_name = self.TOPICS.get(topic, topic)

        # Add metadata
        message["_metadata"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "producer": self.config.client_id,
        }

        if not KAFKA_AVAILABLE or not self._producer:
            logger.debug(f"Mock publish to {topic_name}: {message}")
            return True

        try:
            # Convert headers
            kafka_headers = None
            if headers:
                kafka_headers = [(k, v.encode()) for k, v in headers.items()]

            await self._producer.send_and_wait(
                topic=topic_name,
                value=message,
                key=key,
                partition=partition,
                headers=kafka_headers,
            )
            return True

        except Exception as e:
            logger.error(f"Failed to publish to {topic_name}: {e}")
            return False

    async def publish_market_data(
        self,
        symbol: str,
        data_type: str,
        data: Dict[str, Any],
    ) -> bool:
        """Publish market data event."""
        return await self.publish(
            topic="market_data",
            message={
                "symbol": symbol,
                "type": data_type,
                "data": data,
            },
            key=symbol,
        )

    async def publish_signal(
        self,
        symbol: str,
        strategy: str,
        signal_type: str,
        strength: float,
        metadata: Dict[str, Any],
    ) -> bool:
        """Publish trading signal event."""
        return await self.publish(
            topic="signals",
            message={
                "symbol": symbol,
                "strategy": strategy,
                "signal_type": signal_type,
                "strength": strength,
                "metadata": metadata,
            },
            key=f"{symbol}:{strategy}",
        )

    async def publish_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        status: str,
        details: Dict[str, Any],
    ) -> bool:
        """Publish order event."""
        return await self.publish(
            topic="orders",
            message={
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "status": status,
                "details": details,
            },
            key=order_id,
        )

    async def publish_position(
        self,
        symbol: str,
        action: str,
        position: Dict[str, Any],
    ) -> bool:
        """Publish position event."""
        return await self.publish(
            topic="positions",
            message={
                "symbol": symbol,
                "action": action,
                "position": position,
            },
            key=symbol,
        )

    async def publish_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Publish alert event."""
        return await self.publish(
            topic="alerts",
            message={
                "alert_type": alert_type,
                "severity": severity,
                "message": message,
                "details": details or {},
            },
            key=alert_type,
        )


class KafkaConsumer:
    """Async Kafka consumer for processing trading events."""

    def __init__(
        self,
        config: KafkaConfig,
        topics: List[str],
        handler: Callable[[str, Dict[str, Any]], None],
    ):
        self.config = config
        self.topics = [KafkaProducer.TOPICS.get(t, t) for t in topics]
        self.handler = handler
        self._consumer: Optional[AIOKafkaConsumer] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start consuming messages."""
        if not KAFKA_AVAILABLE:
            logger.warning("aiokafka not installed, consumer not started")
            return

        try:
            self._consumer = AIOKafkaConsumer(
                *self.topics,
                bootstrap_servers=self.config.bootstrap_servers,
                client_id=f"{self.config.client_id}-consumer",
                group_id=self.config.group_id,
                auto_offset_reset=self.config.auto_offset_reset,
                enable_auto_commit=self.config.enable_auto_commit,
                auto_commit_interval_ms=self.config.auto_commit_interval_ms,
                max_poll_records=self.config.max_poll_records,
                value_deserializer=lambda v: json.loads(v.decode()),
            )
            await self._consumer.start()
            self._running = True
            self._task = asyncio.create_task(self._consume_loop())
            logger.info(f"Started consuming from: {self.topics}")
        except Exception as e:
            logger.error(f"Failed to start consumer: {e}")
            raise

    async def stop(self) -> None:
        """Stop consuming messages."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._consumer:
            await self._consumer.stop()
            logger.info("Consumer stopped")

    async def _consume_loop(self) -> None:
        """Main consume loop."""
        while self._running:
            try:
                async for message in self._consumer:
                    if not self._running:
                        break

                    try:
                        await self._process_message(message)
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consumer error: {e}")
                await asyncio.sleep(1)

    async def _process_message(self, message) -> None:
        """Process a single message."""
        topic = message.topic
        value = message.value

        logger.debug(f"Received from {topic}: {value}")

        # Call handler
        if asyncio.iscoroutinefunction(self.handler):
            await self.handler(topic, value)
        else:
            self.handler(topic, value)


class KafkaClient:
    """
    High-level Kafka client combining producer and consumer.

    Usage:
        client = KafkaClient(config)
        await client.connect()

        # Publish events
        await client.publish("signals", {...})

        # Subscribe to events
        await client.subscribe(["signals", "orders"], handler)

        await client.disconnect()
    """

    def __init__(self, config: Optional[KafkaConfig] = None):
        self.config = config or KafkaConfig()
        self.producer = KafkaProducer(self.config)
        self._consumers: List[KafkaConsumer] = []

    async def connect(self) -> None:
        """Connect to Kafka."""
        await self.producer.connect()

    async def disconnect(self) -> None:
        """Disconnect from Kafka."""
        for consumer in self._consumers:
            await consumer.stop()
        await self.producer.disconnect()

    async def publish(
        self,
        topic: str,
        message: Dict[str, Any],
        key: Optional[str] = None,
    ) -> bool:
        """Publish message to topic."""
        return await self.producer.publish(topic, message, key)

    async def subscribe(
        self,
        topics: List[str],
        handler: Callable[[str, Dict[str, Any]], None],
    ) -> KafkaConsumer:
        """Subscribe to topics with handler."""
        consumer = KafkaConsumer(self.config, topics, handler)
        await consumer.start()
        self._consumers.append(consumer)
        return consumer

    @property
    def is_connected(self) -> bool:
        return self.producer.is_connected
