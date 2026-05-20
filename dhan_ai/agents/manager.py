"""
manager.py — Manager Sub-Agent

Handles portfolio management, position sizing, strategy selection,
and coordination of insights from other agents into a cohesive
trading plan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from dhan_ai.agents.base_agent import AgentRole, BaseAgent

logger = logging.getLogger(__name__)


class StrategyType(str, Enum):
    """Pre-defined trading strategy archetypes."""

    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    SWING = "swing"


class PositionAction(str, Enum):
    """Action recommended for a position."""

    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    CLOSE = "close"
    HOLD = "hold"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"


@dataclass
class PositionPlan:
    """A single position recommendation produced by the manager."""

    symbol: str
    action: PositionAction
    strategy: StrategyType
    allocation_pct: float
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ManagerAgent(BaseAgent):
    """Coordinates analysis, risk, and research into executable trading plans.

    Responsibilities:
      - Aggregate signals from Analyst and Researcher
      - Select appropriate strategy per symbol
      - Size positions according to portfolio rules
      - Generate consolidated trading plans for the Trader agent
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(role=AgentRole.MANAGER, config=config)
        self.max_positions: int = self.config.get("max_positions", 10)
        self.max_single_allocation: float = self.config.get("max_single_allocation", 0.10)
        self.default_strategy: StrategyType = StrategyType(
            self.config.get("default_strategy", "momentum")
        )
        self.portfolio: Dict[str, Dict[str, Any]] = {}

    async def _execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build a trading plan from combined sub-agent outputs.

        Expected context keys:
          - ``analyst_signals``: list of signal dicts from the Analyst
          - ``risk_assessment``: dict from the Risk Manager
          - ``research_insights``: dict from the Researcher
          - ``portfolio_state``: current portfolio holdings
          - ``capital``: available trading capital
        """
        signals = context.get("analyst_signals", [])
        risk = context.get("risk_assessment", {})
        research = context.get("research_insights", {})
        portfolio_state = context.get("portfolio_state", {})
        capital: float = context.get("capital", 0.0)

        self.portfolio = portfolio_state

        plans = self._generate_plans(signals, risk, research, capital)
        plans = self._apply_position_limits(plans)

        return {
            "trading_plans": [self._plan_to_dict(p) for p in plans],
            "active_positions": len(self.portfolio),
            "capital_allocated": sum(p.allocation_pct for p in plans),
            "strategies_used": list({p.strategy.value for p in plans}),
        }

    # ------------------------------------------------------------------
    # Plan generation
    # ------------------------------------------------------------------

    def _generate_plans(
        self,
        signals: List[Dict[str, Any]],
        risk: Dict[str, Any],
        research: Dict[str, Any],
        capital: float,
    ) -> List[PositionPlan]:
        """Convert analyst signals into position plans."""
        risk_level = risk.get("overall_risk", "medium")
        allocation_budget = self._risk_adjusted_budget(risk_level)

        plans: List[PositionPlan] = []
        research_sentiments: Dict[str, str] = research.get("sentiments", {})

        for signal in signals:
            symbol = signal.get("symbol", "")
            direction = signal.get("direction", "")
            strength = signal.get("strength", "weak")

            if strength == "weak" and risk_level == "high":
                continue

            sentiment = research_sentiments.get(symbol, "neutral")
            if self._sentiment_conflicts(direction, sentiment):
                continue

            strategy = self._select_strategy(signal, research)
            alloc = self._size_position(strength, allocation_budget, capital)

            action = PositionAction.OPEN_LONG if direction == "bullish" else PositionAction.OPEN_SHORT
            if symbol in self.portfolio:
                action = PositionAction.SCALE_IN if direction == "bullish" else PositionAction.SCALE_OUT

            plans.append(
                PositionPlan(
                    symbol=symbol,
                    action=action,
                    strategy=strategy,
                    allocation_pct=alloc,
                    rationale=signal.get("description", ""),
                )
            )

        return plans

    def _apply_position_limits(self, plans: List[PositionPlan]) -> List[PositionPlan]:
        """Trim plans to respect maximum position count."""
        open_count = len(self.portfolio)
        available_slots = max(0, self.max_positions - open_count)

        new_opens = [
            p for p in plans if p.action in (PositionAction.OPEN_LONG, PositionAction.OPEN_SHORT)
        ]
        others = [
            p for p in plans if p.action not in (PositionAction.OPEN_LONG, PositionAction.OPEN_SHORT)
        ]

        return others + new_opens[:available_slots]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _risk_adjusted_budget(self, risk_level: str) -> float:
        budgets = {"low": 1.0, "medium": 0.7, "high": 0.4, "extreme": 0.1}
        return budgets.get(risk_level, 0.5)

    @staticmethod
    def _sentiment_conflicts(direction: str, sentiment: str) -> bool:
        if direction == "bullish" and sentiment == "negative":
            return True
        if direction == "bearish" and sentiment == "positive":
            return True
        return False

    def _select_strategy(
        self,
        signal: Dict[str, Any],
        research: Dict[str, Any],
    ) -> StrategyType:
        indicator = signal.get("indicator", "")
        if indicator in ("rsi", "bollinger"):
            return StrategyType.MEAN_REVERSION
        if indicator in ("macd", "ema"):
            return StrategyType.MOMENTUM
        return self.default_strategy

    def _size_position(
        self,
        strength: str,
        budget_factor: float,
        capital: float,
    ) -> float:
        base = {"strong": 0.08, "moderate": 0.05, "weak": 0.02}
        alloc = base.get(strength, 0.02) * budget_factor
        return min(alloc, self.max_single_allocation)

    @staticmethod
    def _plan_to_dict(plan: PositionPlan) -> Dict[str, Any]:
        return {
            "symbol": plan.symbol,
            "action": plan.action.value,
            "strategy": plan.strategy.value,
            "allocation_pct": plan.allocation_pct,
            "entry_price": plan.entry_price,
            "target_price": plan.target_price,
            "stop_loss": plan.stop_loss,
            "rationale": plan.rationale,
            "metadata": plan.metadata,
        }
