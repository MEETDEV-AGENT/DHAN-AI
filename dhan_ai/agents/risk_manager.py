"""
risk_manager.py — Risk Management Sub-Agent

Evaluates and manages trading risk across positions,
enforcing portfolio-level and per-trade risk limits
for the Indian stock market.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from dhan_ai.agents.base_agent import AgentRole, BaseAgent

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Overall portfolio risk classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class RiskFlag(str, Enum):
    """Specific risk condition flags."""

    CONCENTRATION = "concentration"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    LIQUIDITY = "liquidity"
    MARGIN = "margin"
    NEWS_RISK = "news_risk"


@dataclass
class RiskAlert:
    """An alert raised when a risk threshold is breached."""

    flag: RiskFlag
    severity: RiskLevel
    symbol: Optional[str]
    description: str
    current_value: float
    threshold: float
    recommendation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class RiskManagerAgent(BaseAgent):
    """Assesses and enforces risk constraints across the portfolio.

    Responsibilities:
      - Portfolio-level risk scoring
      - Per-position exposure checks
      - Drawdown monitoring
      - Volatility-based position sizing recommendations
      - Stop-loss / trailing-stop suggestions
      - Alert generation when risk limits are breached
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(role=AgentRole.RISK_MANAGER, config=config)
        self.max_portfolio_risk: float = self.config.get("max_portfolio_risk", 0.02)
        self.max_single_risk: float = self.config.get("max_single_risk", 0.01)
        self.max_drawdown_pct: float = self.config.get("max_drawdown_pct", 0.05)
        self.max_concentration_pct: float = self.config.get("max_concentration_pct", 0.20)
        self.volatility_lookback: int = self.config.get("volatility_lookback", 20)

    async def _execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full risk assessment pipeline.

        Expected context keys:
          - ``portfolio_state``: dict of symbol -> position info
          - ``market_data``: dict of symbol -> price/volume data
          - ``capital``: total available capital
          - ``analyst_signals``: optional signals from the analyst
        """
        portfolio = context.get("portfolio_state", {})
        market_data = context.get("market_data", {})
        capital: float = context.get("capital", 0.0)

        alerts: List[RiskAlert] = []
        alerts.extend(self._check_concentration(portfolio, capital))
        alerts.extend(self._check_drawdown(portfolio, market_data))
        alerts.extend(self._check_volatility(market_data))

        overall_risk = self._compute_overall_risk(alerts)
        position_limits = self._compute_position_limits(overall_risk, capital)
        stop_losses = self._suggest_stop_losses(portfolio, market_data)

        return {
            "overall_risk": overall_risk.value,
            "alerts": [self._alert_to_dict(a) for a in alerts],
            "position_limits": position_limits,
            "stop_losses": stop_losses,
            "risk_score": self._risk_score(alerts),
        }

    # ------------------------------------------------------------------
    # Risk checks
    # ------------------------------------------------------------------

    def _check_concentration(
        self,
        portfolio: Dict[str, Dict[str, Any]],
        capital: float,
    ) -> List[RiskAlert]:
        """Flag positions that exceed concentration limits."""
        alerts: List[RiskAlert] = []
        if capital <= 0:
            return alerts

        for symbol, pos in portfolio.items():
            value = pos.get("market_value", 0.0)
            pct = value / capital
            if pct > self.max_concentration_pct:
                alerts.append(
                    RiskAlert(
                        flag=RiskFlag.CONCENTRATION,
                        severity=RiskLevel.HIGH if pct > self.max_concentration_pct * 1.5 else RiskLevel.MEDIUM,
                        symbol=symbol,
                        description=f"{symbol} concentration at {pct:.1%} of portfolio",
                        current_value=pct,
                        threshold=self.max_concentration_pct,
                        recommendation=f"Reduce {symbol} to below {self.max_concentration_pct:.0%}",
                    )
                )
        return alerts

    def _check_drawdown(
        self,
        portfolio: Dict[str, Dict[str, Any]],
        market_data: Dict[str, Any],
    ) -> List[RiskAlert]:
        """Check each position for drawdown from entry price."""
        alerts: List[RiskAlert] = []

        for symbol, pos in portfolio.items():
            entry = pos.get("entry_price", 0)
            sym_data = market_data.get(symbol, {})
            prices = sym_data.get("close", [])
            if not prices or entry <= 0:
                continue

            current = prices[-1]
            drawdown = (entry - current) / entry

            if drawdown > self.max_drawdown_pct:
                severity = RiskLevel.EXTREME if drawdown > self.max_drawdown_pct * 2 else RiskLevel.HIGH
                alerts.append(
                    RiskAlert(
                        flag=RiskFlag.DRAWDOWN,
                        severity=severity,
                        symbol=symbol,
                        description=f"{symbol} drawdown at {drawdown:.1%}",
                        current_value=drawdown,
                        threshold=self.max_drawdown_pct,
                        recommendation=f"Consider stop-loss for {symbol}",
                    )
                )
        return alerts

    def _check_volatility(
        self,
        market_data: Dict[str, Any],
    ) -> List[RiskAlert]:
        """Flag symbols with unusually high recent volatility."""
        alerts: List[RiskAlert] = []

        for symbol, data in market_data.items():
            prices = data.get("close", [])
            if len(prices) < self.volatility_lookback:
                continue

            recent = prices[-self.volatility_lookback :]
            mean = sum(recent) / len(recent)
            if mean == 0:
                continue
            variance = sum((p - mean) ** 2 for p in recent) / len(recent)
            vol = (variance ** 0.5) / mean

            if vol > 0.05:
                alerts.append(
                    RiskAlert(
                        flag=RiskFlag.VOLATILITY,
                        severity=RiskLevel.HIGH if vol > 0.10 else RiskLevel.MEDIUM,
                        symbol=symbol,
                        description=f"{symbol} volatility at {vol:.2%}",
                        current_value=vol,
                        threshold=0.05,
                        recommendation=f"Reduce position size for {symbol}",
                    )
                )

        return alerts

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_overall_risk(alerts: List[RiskAlert]) -> RiskLevel:
        if any(a.severity == RiskLevel.EXTREME for a in alerts):
            return RiskLevel.EXTREME
        high_count = sum(1 for a in alerts if a.severity == RiskLevel.HIGH)
        if high_count >= 3:
            return RiskLevel.EXTREME
        if high_count >= 1:
            return RiskLevel.HIGH
        if alerts:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _compute_position_limits(
        self,
        risk_level: RiskLevel,
        capital: float,
    ) -> Dict[str, Any]:
        """Adjust position limits based on current risk level."""
        multipliers = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 0.7,
            RiskLevel.HIGH: 0.4,
            RiskLevel.EXTREME: 0.1,
        }
        mult = multipliers[risk_level]
        return {
            "max_position_value": capital * self.max_single_risk * mult * 100,
            "max_portfolio_risk_pct": self.max_portfolio_risk * mult,
            "risk_multiplier": mult,
        }

    @staticmethod
    def _suggest_stop_losses(
        portfolio: Dict[str, Dict[str, Any]],
        market_data: Dict[str, Any],
    ) -> Dict[str, Optional[float]]:
        """Suggest stop-loss prices (2% below current price, as a simple default)."""
        stops: Dict[str, Optional[float]] = {}
        for symbol in portfolio:
            data = market_data.get(symbol, {})
            prices = data.get("close", [])
            if prices:
                stops[symbol] = round(prices[-1] * 0.98, 2)
            else:
                stops[symbol] = None
        return stops

    @staticmethod
    def _risk_score(alerts: List[RiskAlert]) -> float:
        """Numeric 0-1 risk score from alerts."""
        if not alerts:
            return 0.0
        weights = {
            RiskLevel.LOW: 0.1,
            RiskLevel.MEDIUM: 0.3,
            RiskLevel.HIGH: 0.6,
            RiskLevel.EXTREME: 1.0,
        }
        total = sum(weights[a.severity] for a in alerts)
        return min(1.0, total / max(1, len(alerts)))

    @staticmethod
    def _alert_to_dict(alert: RiskAlert) -> Dict[str, Any]:
        return {
            "flag": alert.flag.value,
            "severity": alert.severity.value,
            "symbol": alert.symbol,
            "description": alert.description,
            "current_value": alert.current_value,
            "threshold": alert.threshold,
            "recommendation": alert.recommendation,
            "metadata": alert.metadata,
        }
