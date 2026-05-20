"""
orchestrator.py — Agent Orchestrator

Central coordinator that manages the lifecycle and data flow
between all five sub-agents:

  1. Analyst   — market data analysis and signal generation
  2. Researcher — web-based research via Pinchtab
  3. Risk Manager — risk assessment and limit enforcement
  4. Manager   — strategy selection and position planning
  5. Trader    — order construction and execution

Execution order:
  Analyst + Researcher (parallel)
  → Risk Manager
  → Manager (receives signals, research, and risk)
  → Trader (executes the Manager's plans)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dhan_ai.agents.analyst import AnalystAgent
from dhan_ai.agents.base_agent import (
    AgentMessage,
    AgentResult,
    AgentRole,
    AgentStatus,
    BaseAgent,
)
from dhan_ai.agents.manager import ManagerAgent
from dhan_ai.agents.researcher import ResearcherAgent
from dhan_ai.agents.risk_manager import RiskManagerAgent
from dhan_ai.agents.trader import TraderAgent

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""

    analyst: Dict[str, Any] = field(default_factory=dict)
    manager: Dict[str, Any] = field(default_factory=dict)
    researcher: Dict[str, Any] = field(default_factory=dict)
    risk_manager: Dict[str, Any] = field(default_factory=dict)
    trader: Dict[str, Any] = field(default_factory=dict)
    dry_run: bool = True


@dataclass
class PipelineResult:
    """Aggregated result from one full orchestration cycle."""

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    analyst: Optional[Dict[str, Any]] = None
    researcher: Optional[Dict[str, Any]] = None
    risk_manager: Optional[Dict[str, Any]] = None
    manager: Optional[Dict[str, Any]] = None
    trader: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    success: bool = True


class AgentOrchestrator:
    """Coordinates the five sub-agents in a structured pipeline.

    Usage::

        config = OrchestratorConfig(dry_run=True)
        orchestrator = AgentOrchestrator(config)

        context = {
            "market_data": {...},
            "watchlist": ["RELIANCE", "TCS", "INFY"],
            "capital": 1_000_000,
            "portfolio_state": {},
        }
        result = await orchestrator.run(context)
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        self._agents: Dict[AgentRole, BaseAgent] = {}
        self._initialise_agents()
        self._logger = logging.getLogger("dhan_ai.agents.orchestrator")

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialise_agents(self) -> None:
        """Instantiate all five sub-agents."""
        self._agents[AgentRole.ANALYST] = AnalystAgent(self.config.analyst)
        self._agents[AgentRole.MANAGER] = ManagerAgent(self.config.manager)
        self._agents[AgentRole.RESEARCHER] = ResearcherAgent(self.config.researcher)
        self._agents[AgentRole.RISK_MANAGER] = RiskManagerAgent(self.config.risk_manager)

        trader_config = {**self.config.trader, "dry_run": self.config.dry_run}
        self._agents[AgentRole.TRADER] = TraderAgent(trader_config)

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    async def run(self, context: Dict[str, Any]) -> PipelineResult:
        """Execute the full orchestration pipeline.

        Parameters
        ----------
        context:
            Shared context dict containing at minimum:
              - ``market_data``: market price/volume data
              - ``watchlist``: list of symbols to process
              - ``capital``: available trading capital
              - ``portfolio_state``: current holdings

        Returns
        -------
        PipelineResult:
            Aggregated results from all agents.
        """
        result = PipelineResult()
        self._logger.info("Pipeline started")

        # Phase 1 — Analyst + Researcher run in parallel
        analyst_result, researcher_result = await self._phase_analysis(context)
        result.analyst = analyst_result.data if analyst_result else None
        result.researcher = researcher_result.data if researcher_result else None

        if analyst_result and analyst_result.status == AgentStatus.FAILED:
            result.errors.extend(analyst_result.errors)
        if researcher_result and researcher_result.status == AgentStatus.FAILED:
            result.errors.extend(researcher_result.errors)

        # Phase 2 — Risk Manager
        risk_result = await self._phase_risk(context)
        result.risk_manager = risk_result.data if risk_result else None
        if risk_result and risk_result.status == AgentStatus.FAILED:
            result.errors.extend(risk_result.errors)

        # Phase 3 — Manager (needs analyst + researcher + risk outputs)
        manager_context = self._build_manager_context(
            context, analyst_result, researcher_result, risk_result
        )
        manager_result = await self._phase_management(manager_context)
        result.manager = manager_result.data if manager_result else None
        if manager_result and manager_result.status == AgentStatus.FAILED:
            result.errors.extend(manager_result.errors)

        # Phase 4 — Trader (needs manager plans + risk limits)
        trader_context = self._build_trader_context(
            context, manager_result, risk_result
        )
        trader_result = await self._phase_trading(trader_context)
        result.trader = trader_result.data if trader_result else None
        if trader_result and trader_result.status == AgentStatus.FAILED:
            result.errors.extend(trader_result.errors)

        result.success = len(result.errors) == 0
        self._logger.info(
            "Pipeline finished — success=%s errors=%d",
            result.success,
            len(result.errors),
        )
        return result

    # ------------------------------------------------------------------
    # Pipeline phases
    # ------------------------------------------------------------------

    async def _phase_analysis(
        self, context: Dict[str, Any]
    ) -> tuple[Optional[AgentResult], Optional[AgentResult]]:
        """Phase 1: Run Analyst and Researcher concurrently."""
        self._logger.info("Phase 1: Analysis + Research")

        analyst = self._agents[AgentRole.ANALYST]
        researcher = self._agents[AgentRole.RESEARCHER]

        analyst_task = asyncio.create_task(analyst.run(context))
        researcher_task = asyncio.create_task(researcher.run(context))

        analyst_result, researcher_result = await asyncio.gather(
            analyst_task, researcher_task, return_exceptions=True
        )

        ar = analyst_result if isinstance(analyst_result, AgentResult) else None
        rr = researcher_result if isinstance(researcher_result, AgentResult) else None

        if isinstance(analyst_result, Exception):
            self._logger.error("Analyst raised: %s", analyst_result)
        if isinstance(researcher_result, Exception):
            self._logger.error("Researcher raised: %s", researcher_result)

        return ar, rr

    async def _phase_risk(self, context: Dict[str, Any]) -> Optional[AgentResult]:
        """Phase 2: Run Risk Manager."""
        self._logger.info("Phase 2: Risk Assessment")
        risk_agent = self._agents[AgentRole.RISK_MANAGER]
        return await risk_agent.run(context)

    async def _phase_management(
        self, context: Dict[str, Any]
    ) -> Optional[AgentResult]:
        """Phase 3: Run Manager with aggregated inputs."""
        self._logger.info("Phase 3: Strategy & Planning")
        manager = self._agents[AgentRole.MANAGER]
        return await manager.run(context)

    async def _phase_trading(self, context: Dict[str, Any]) -> Optional[AgentResult]:
        """Phase 4: Execute trades."""
        self._logger.info("Phase 4: Trade Execution")
        trader = self._agents[AgentRole.TRADER]
        return await trader.run(context)

    # ------------------------------------------------------------------
    # Context builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_manager_context(
        base_context: Dict[str, Any],
        analyst_result: Optional[AgentResult],
        researcher_result: Optional[AgentResult],
        risk_result: Optional[AgentResult],
    ) -> Dict[str, Any]:
        ctx = dict(base_context)
        if analyst_result and analyst_result.data:
            ctx["analyst_signals"] = analyst_result.data.get("signals", [])
        if researcher_result and researcher_result.data:
            ctx["research_insights"] = researcher_result.data
        if risk_result and risk_result.data:
            ctx["risk_assessment"] = risk_result.data
        return ctx

    @staticmethod
    def _build_trader_context(
        base_context: Dict[str, Any],
        manager_result: Optional[AgentResult],
        risk_result: Optional[AgentResult],
    ) -> Dict[str, Any]:
        ctx = dict(base_context)
        if manager_result and manager_result.data:
            ctx["trading_plans"] = manager_result.data.get("trading_plans", [])
        if risk_result and risk_result.data:
            ctx["stop_losses"] = risk_result.data.get("stop_losses", {})
            ctx["risk_assessment"] = risk_result.data
        return ctx

    # ------------------------------------------------------------------
    # Inter-agent messaging
    # ------------------------------------------------------------------

    def send_message(
        self,
        sender: AgentRole,
        recipient: AgentRole,
        payload: Dict[str, Any],
    ) -> None:
        """Send a message from one agent to another."""
        msg = AgentMessage(sender=sender, recipient=recipient, payload=payload)
        agent = self._agents.get(recipient)
        if agent is None:
            self._logger.error("Unknown recipient: %s", recipient)
            return
        agent.receive_message(msg)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_agent(self, role: AgentRole) -> BaseAgent:
        """Return a sub-agent by role."""
        return self._agents[role]

    def agent_statuses(self) -> Dict[str, str]:
        """Return the current status of every sub-agent."""
        return {role.value: agent.status.value for role, agent in self._agents.items()}

    def __repr__(self) -> str:
        statuses = ", ".join(
            f"{r.value}={a.status.value}" for r, a in self._agents.items()
        )
        return f"<AgentOrchestrator [{statuses}]>"
