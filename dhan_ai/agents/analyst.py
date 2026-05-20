"""
analyst.py — Analyst Sub-Agent

Performs market data analysis including trend detection,
pattern recognition, and anomaly identification across
Indian stock market instruments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from dhan_ai.agents.base_agent import AgentRole, BaseAgent

logger = logging.getLogger(__name__)


class SignalStrength(str, Enum):
    """Strength of an analytical signal."""

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


class TrendDirection(str, Enum):
    """Direction of a detected trend."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"


@dataclass
class MarketSignal:
    """A single analytical signal produced by the analyst."""

    symbol: str
    direction: TrendDirection
    strength: SignalStrength
    indicator: str
    value: float
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class AnalystAgent(BaseAgent):
    """Analyses market data to produce actionable trading signals.

    Responsibilities:
      - Technical indicator computation (RSI, MACD, Bollinger Bands, etc.)
      - Trend and pattern detection
      - Anomaly / volume spike identification
      - Signal generation for the orchestrator
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(role=AgentRole.ANALYST, config=config)
        self.indicators: List[str] = self.config.get(
            "indicators", ["rsi", "macd", "bollinger", "vwap", "ema"]
        )
        self.lookback_periods: int = self.config.get("lookback_periods", 20)
        self.signal_threshold: float = self.config.get("signal_threshold", 0.6)

    async def _execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run analysis pipeline on the provided market data.

        Expected context keys:
          - ``market_data``: dict of symbol -> price/volume data
          - ``watchlist``: optional list of symbols to focus on
        """
        market_data: Dict[str, Any] = context.get("market_data", {})
        watchlist: List[str] = context.get("watchlist", list(market_data.keys()))

        if not market_data:
            self._logger.warning("No market data provided; skipping analysis")
            return {"signals": [], "summary": "No data to analyse."}

        signals = await self._analyse_symbols(market_data, watchlist)
        summary = self._build_summary(signals)

        return {
            "signals": [self._signal_to_dict(s) for s in signals],
            "summary": summary,
            "symbols_analysed": len(watchlist),
            "indicators_used": self.indicators,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _analyse_symbols(
        self,
        market_data: Dict[str, Any],
        watchlist: List[str],
    ) -> List[MarketSignal]:
        """Run indicator analysis for each symbol on the watchlist."""
        signals: List[MarketSignal] = []

        for symbol in watchlist:
            symbol_data = market_data.get(symbol)
            if symbol_data is None:
                self._logger.debug("No data for %s; skipping", symbol)
                continue

            for indicator in self.indicators:
                signal = self._compute_indicator(symbol, symbol_data, indicator)
                if signal is not None:
                    signals.append(signal)

        self._logger.info("Generated %d signals across %d symbols", len(signals), len(watchlist))
        return signals

    def _compute_indicator(
        self,
        symbol: str,
        data: Dict[str, Any],
        indicator: str,
    ) -> Optional[MarketSignal]:
        """Compute a single indicator for a symbol.

        Placeholder implementation — concrete indicator math will be
        added once the data layer is wired up.
        """
        prices: List[float] = data.get("close", [])
        if len(prices) < self.lookback_periods:
            return None

        recent = prices[-self.lookback_periods :]

        if indicator == "rsi":
            return self._compute_rsi(symbol, recent)
        if indicator == "macd":
            return self._compute_macd(symbol, recent)
        if indicator == "bollinger":
            return self._compute_bollinger(symbol, recent)
        if indicator == "vwap":
            volumes: List[float] = data.get("volume", [])
            return self._compute_vwap(symbol, recent, volumes[-self.lookback_periods :])
        if indicator == "ema":
            return self._compute_ema(symbol, recent)

        return None

    # ------------------------------------------------------------------
    # Indicator stubs (to be replaced with real math)
    # ------------------------------------------------------------------

    def _compute_rsi(self, symbol: str, prices: List[float]) -> Optional[MarketSignal]:
        """Relative Strength Index stub."""
        if len(prices) < 2:
            return None

        gains = [max(0, prices[i] - prices[i - 1]) for i in range(1, len(prices))]
        losses = [max(0, prices[i - 1] - prices[i]) for i in range(1, len(prices))]

        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 1

        rs = avg_gain / avg_loss if avg_loss != 0 else 100
        rsi = 100 - (100 / (1 + rs))

        if rsi > 70:
            return MarketSignal(
                symbol=symbol,
                direction=TrendDirection.BEARISH,
                strength=SignalStrength.MODERATE,
                indicator="rsi",
                value=rsi,
                description=f"RSI overbought at {rsi:.1f}",
            )
        if rsi < 30:
            return MarketSignal(
                symbol=symbol,
                direction=TrendDirection.BULLISH,
                strength=SignalStrength.MODERATE,
                indicator="rsi",
                value=rsi,
                description=f"RSI oversold at {rsi:.1f}",
            )
        return None

    def _compute_macd(self, symbol: str, prices: List[float]) -> Optional[MarketSignal]:
        """MACD stub — fast/slow EMA crossover detection."""
        if len(prices) < 12:
            return None

        fast_ema = self._ema(prices, 12)
        slow_ema = self._ema(prices, 26) if len(prices) >= 26 else self._ema(prices, len(prices))
        macd_val = fast_ema - slow_ema

        direction = TrendDirection.BULLISH if macd_val > 0 else TrendDirection.BEARISH
        return MarketSignal(
            symbol=symbol,
            direction=direction,
            strength=SignalStrength.WEAK,
            indicator="macd",
            value=macd_val,
            description=f"MACD {'above' if macd_val > 0 else 'below'} signal line",
        )

    def _compute_bollinger(self, symbol: str, prices: List[float]) -> Optional[MarketSignal]:
        """Bollinger Bands stub."""
        mean = sum(prices) / len(prices)
        variance = sum((p - mean) ** 2 for p in prices) / len(prices)
        std_dev = variance ** 0.5
        upper = mean + 2 * std_dev
        lower = mean - 2 * std_dev
        current = prices[-1]

        if current > upper:
            return MarketSignal(
                symbol=symbol,
                direction=TrendDirection.BEARISH,
                strength=SignalStrength.MODERATE,
                indicator="bollinger",
                value=current,
                description=f"Price above upper Bollinger Band ({upper:.2f})",
            )
        if current < lower:
            return MarketSignal(
                symbol=symbol,
                direction=TrendDirection.BULLISH,
                strength=SignalStrength.MODERATE,
                indicator="bollinger",
                value=current,
                description=f"Price below lower Bollinger Band ({lower:.2f})",
            )
        return None

    def _compute_vwap(
        self,
        symbol: str,
        prices: List[float],
        volumes: List[float],
    ) -> Optional[MarketSignal]:
        """VWAP stub."""
        if not volumes or len(volumes) != len(prices):
            return None
        total_vol = sum(volumes)
        if total_vol == 0:
            return None
        vwap = sum(p * v for p, v in zip(prices, volumes)) / total_vol
        current = prices[-1]

        if current > vwap * 1.02:
            return MarketSignal(
                symbol=symbol,
                direction=TrendDirection.BULLISH,
                strength=SignalStrength.WEAK,
                indicator="vwap",
                value=vwap,
                description=f"Price above VWAP ({vwap:.2f})",
            )
        if current < vwap * 0.98:
            return MarketSignal(
                symbol=symbol,
                direction=TrendDirection.BEARISH,
                strength=SignalStrength.WEAK,
                indicator="vwap",
                value=vwap,
                description=f"Price below VWAP ({vwap:.2f})",
            )
        return None

    def _compute_ema(self, symbol: str, prices: List[float]) -> Optional[MarketSignal]:
        """EMA crossover stub (9/21)."""
        if len(prices) < 21:
            return None
        short = self._ema(prices, 9)
        long = self._ema(prices, 21)
        diff = short - long

        if abs(diff) / long < 0.001:
            return None

        direction = TrendDirection.BULLISH if diff > 0 else TrendDirection.BEARISH
        return MarketSignal(
            symbol=symbol,
            direction=direction,
            strength=SignalStrength.WEAK,
            indicator="ema",
            value=diff,
            description=f"EMA-9 {'above' if diff > 0 else 'below'} EMA-21",
        )

    @staticmethod
    def _ema(prices: List[float], period: int) -> float:
        """Simple EMA calculation."""
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
        return ema

    @staticmethod
    def _signal_to_dict(signal: MarketSignal) -> Dict[str, Any]:
        return {
            "symbol": signal.symbol,
            "direction": signal.direction.value,
            "strength": signal.strength.value,
            "indicator": signal.indicator,
            "value": signal.value,
            "description": signal.description,
            "metadata": signal.metadata,
        }

    def _build_summary(self, signals: List[MarketSignal]) -> str:
        bullish = sum(1 for s in signals if s.direction == TrendDirection.BULLISH)
        bearish = sum(1 for s in signals if s.direction == TrendDirection.BEARISH)
        return (
            f"Analysis complete: {len(signals)} signals "
            f"({bullish} bullish, {bearish} bearish, "
            f"{len(signals) - bullish - bearish} sideways)"
        )
