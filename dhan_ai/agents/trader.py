"""
trader.py — Trader Sub-Agent

Executes trades based on trading plans from the Manager agent.
Handles order construction, validation, execution, and
confirmation for the Indian stock market (NSE/BSE).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from dhan_ai.agents.base_agent import AgentRole, BaseAgent

logger = logging.getLogger(__name__)


class OrderType(str, Enum):
    """Supported order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LOSS_LIMIT = "stop_loss_limit"


class OrderSide(str, Enum):
    """Buy or sell."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """Lifecycle status of an order."""

    PENDING = "pending"
    VALIDATED = "validated"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class Exchange(str, Enum):
    """Indian stock exchanges."""

    NSE = "NSE"
    BSE = "BSE"


@dataclass
class Order:
    """Represents a single trade order."""

    order_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    symbol: str = ""
    exchange: Exchange = Exchange.NSE
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: int = 0
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    rationale: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)


class TraderAgent(BaseAgent):
    """Converts trading plans into executable orders.

    Responsibilities:
      - Build orders from Manager's trading plans
      - Validate orders against exchange rules and risk limits
      - Execute orders (via broker API — placeholder)
      - Track order status and confirmations
      - Maintain an order ledger
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(role=AgentRole.TRADER, config=config)
        self.default_exchange: Exchange = Exchange(
            self.config.get("default_exchange", "NSE")
        )
        self.dry_run: bool = self.config.get("dry_run", True)
        self.min_order_value: float = self.config.get("min_order_value", 500.0)
        self.max_order_value: float = self.config.get("max_order_value", 500_000.0)
        self.order_ledger: List[Order] = []

    async def _execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trading plans.

        Expected context keys:
          - ``trading_plans``: list of plan dicts from the Manager
          - ``market_data``: dict of symbol -> price/volume data
          - ``capital``: available trading capital
          - ``risk_assessment``: dict from the Risk Manager
          - ``stop_losses``: suggested stop-loss prices
        """
        plans = context.get("trading_plans", [])
        market_data = context.get("market_data", {})
        capital: float = context.get("capital", 0.0)
        stop_losses: Dict[str, Optional[float]] = context.get("stop_losses", {})

        orders = self._build_orders(plans, market_data, capital, stop_losses)
        validated = self._validate_orders(orders)
        executed = await self._execute_orders(validated)

        self.order_ledger.extend(executed)

        return {
            "orders": [self._order_to_dict(o) for o in executed],
            "total_orders": len(executed),
            "filled": sum(1 for o in executed if o.status == OrderStatus.FILLED),
            "rejected": sum(1 for o in executed if o.status == OrderStatus.REJECTED),
            "dry_run": self.dry_run,
        }

    # ------------------------------------------------------------------
    # Order building
    # ------------------------------------------------------------------

    def _build_orders(
        self,
        plans: List[Dict[str, Any]],
        market_data: Dict[str, Any],
        capital: float,
        stop_losses: Dict[str, Optional[float]],
    ) -> List[Order]:
        """Convert trading plans into Order objects."""
        orders: List[Order] = []

        for plan in plans:
            symbol = plan.get("symbol", "")
            action = plan.get("action", "")
            alloc_pct = plan.get("allocation_pct", 0.0)

            sym_data = market_data.get(symbol, {})
            prices = sym_data.get("close", [])
            if not prices:
                self._logger.warning("No price data for %s; skipping order", symbol)
                continue

            current_price = prices[-1]
            if current_price <= 0:
                continue

            order_value = capital * alloc_pct
            quantity = max(1, int(order_value / current_price))

            side = self._action_to_side(action)
            if side is None:
                continue

            order = Order(
                symbol=symbol,
                exchange=self.default_exchange,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity,
                price=current_price,
                stop_loss=stop_losses.get(symbol),
                target_price=plan.get("target_price"),
                rationale=plan.get("rationale", ""),
            )
            orders.append(order)

        return orders

    def _validate_orders(self, orders: List[Order]) -> List[Order]:
        """Validate orders against exchange and risk constraints."""
        validated: List[Order] = []

        for order in orders:
            issues = self._check_order(order)
            if issues:
                order.status = OrderStatus.REJECTED
                order.metadata["rejection_reasons"] = issues
                self._logger.warning(
                    "Order %s rejected: %s", order.order_id, "; ".join(issues)
                )
            else:
                order.status = OrderStatus.VALIDATED
            validated.append(order)

        return validated

    def _check_order(self, order: Order) -> List[str]:
        """Return a list of validation issues (empty if valid)."""
        issues: List[str] = []

        if order.quantity <= 0:
            issues.append("quantity must be positive")

        if order.price is not None and order.price <= 0:
            issues.append("price must be positive")

        order_value = (order.price or 0) * order.quantity
        if order_value < self.min_order_value:
            issues.append(
                f"order value {order_value:.2f} below minimum {self.min_order_value}"
            )
        if order_value > self.max_order_value:
            issues.append(
                f"order value {order_value:.2f} exceeds maximum {self.max_order_value}"
            )

        return issues

    async def _execute_orders(self, orders: List[Order]) -> List[Order]:
        """Submit validated orders to the broker.

        In dry-run mode, orders are marked as FILLED without actual
        execution.  Real broker integration will be added once the
        Dhan API credentials are configured.
        """
        for order in orders:
            if order.status == OrderStatus.REJECTED:
                continue

            if self.dry_run:
                order.status = OrderStatus.FILLED
                order.metadata["execution_mode"] = "dry_run"
                self._logger.info(
                    "[DRY RUN] %s %d x %s @ %s",
                    order.side.value.upper(),
                    order.quantity,
                    order.symbol,
                    order.price,
                )
            else:
                order.status = OrderStatus.SUBMITTED
                order.metadata["execution_mode"] = "live"
                # Placeholder for actual broker API call
                self._logger.info(
                    "[LIVE] Submitting %s %d x %s @ %s",
                    order.side.value.upper(),
                    order.quantity,
                    order.symbol,
                    order.price,
                )

        return orders

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _action_to_side(action: str) -> Optional[OrderSide]:
        buy_actions = {"open_long", "scale_in"}
        sell_actions = {"open_short", "close", "scale_out"}
        if action in buy_actions:
            return OrderSide.BUY
        if action in sell_actions:
            return OrderSide.SELL
        return None

    @staticmethod
    def _order_to_dict(order: Order) -> Dict[str, Any]:
        return {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "exchange": order.exchange.value,
            "side": order.side.value,
            "order_type": order.order_type.value,
            "quantity": order.quantity,
            "price": order.price,
            "trigger_price": order.trigger_price,
            "stop_loss": order.stop_loss,
            "target_price": order.target_price,
            "status": order.status.value,
            "rationale": order.rationale,
            "created_at": order.created_at,
            "metadata": order.metadata,
        }
