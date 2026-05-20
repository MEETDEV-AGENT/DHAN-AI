"""
base_agent.py — Base Agent Interface

Defines the abstract base class and shared types for all
Dhan AI sub-agents.
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    """Roles available in the orchestration system."""

    ANALYST = "analyst"
    MANAGER = "manager"
    RESEARCHER = "researcher"
    RISK_MANAGER = "risk_manager"
    TRADER = "trader"


class AgentStatus(str, Enum):
    """Lifecycle status of an agent."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentMessage:
    """Message exchanged between agents via the orchestrator."""

    sender: AgentRole
    recipient: AgentRole
    payload: Dict[str, Any]
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class AgentResult:
    """Standardised result returned by every agent after execution."""

    agent_role: AgentRole
    status: AgentStatus
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class BaseAgent(ABC):
    """Abstract base class that every sub-agent must implement."""

    def __init__(self, role: AgentRole, config: Optional[Dict[str, Any]] = None):
        self.role = role
        self.config = config or {}
        self.status = AgentStatus.IDLE
        self._inbox: List[AgentMessage] = []
        self._logger = logging.getLogger(f"dhan_ai.agents.{role.value}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self, context: Dict[str, Any]) -> AgentResult:
        """Execute the agent's main logic with the given context.

        Sets status to RUNNING, delegates to the subclass ``_execute``
        implementation, and wraps the result in an ``AgentResult``.
        """
        self.status = AgentStatus.RUNNING
        self._logger.info("Agent %s started", self.role.value)
        try:
            data = await self._execute(context)
            self.status = AgentStatus.COMPLETED
            self._logger.info("Agent %s completed", self.role.value)
            return AgentResult(agent_role=self.role, status=self.status, data=data)
        except Exception as exc:
            self.status = AgentStatus.FAILED
            self._logger.error("Agent %s failed: %s", self.role.value, exc)
            return AgentResult(
                agent_role=self.role,
                status=self.status,
                errors=[str(exc)],
            )

    def receive_message(self, message: AgentMessage) -> None:
        """Enqueue an inter-agent message for processing."""
        self._inbox.append(message)
        self._logger.debug(
            "Agent %s received message %s from %s",
            self.role.value,
            message.message_id,
            message.sender.value,
        )

    def get_messages(self) -> List[AgentMessage]:
        """Return and clear the inbox."""
        messages = list(self._inbox)
        self._inbox.clear()
        return messages

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    async def _execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Core logic implemented by each sub-agent.

        Parameters
        ----------
        context:
            Shared execution context provided by the orchestrator.

        Returns
        -------
        dict:
            Agent-specific output data.
        """

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} role={self.role.value} status={self.status.value}>"
