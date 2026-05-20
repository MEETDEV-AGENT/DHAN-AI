"""
dhan_ai.agents — Multi-Agent Orchestration System

Sub-agents:
  - Analyst: Market data analysis (trends, patterns, anomalies)
  - Manager: Portfolio management and strategy coordination
  - Researcher: Web-based market research via Pinchtab
  - RiskManager: Risk assessment and mitigation
  - Trader: Trade execution and order management
"""

from dhan_ai.agents.base_agent import AgentRole, AgentStatus, BaseAgent
from dhan_ai.agents.orchestrator import AgentOrchestrator

__all__ = [
    "AgentRole",
    "AgentStatus",
    "BaseAgent",
    "AgentOrchestrator",
]
