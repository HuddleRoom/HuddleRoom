from __future__ import annotations

# pylint: disable=not-callable

from dataclasses import dataclass, field
import re
from typing import Any
import uuid

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.agent import Agent
from huddleroom.models.memory_item import MemoryItem
from huddleroom.models.session import Session
from huddleroom.models.task import Task


ACTIVE_TASK_STATUSES = ("backlog", "ready", "in_progress", "blocked")
ACTIVE_SESSION_STATUSES = ("pending", "running")
DEFAULT_CONTEXT_WORK_FUNCTIONS = (
    "planning",
    "investigation",
    "implementation",
    "review",
    "validation",
    "summarization",
)
WEAK_FIT_THRESHOLD = 45
OUTCOME_HINT_ROWS_PER_AGENT = 25
OUTCOME_HINT_TERMS = ("success", "successful", "successfully", "accepted", "passed", "completed")


@dataclass(frozen=True)
class WorkFunctionProfile:
    name: str
    role_terms: tuple[str, ...] = ()
    capability_terms: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role_terms": list(self.role_terms),
            "capability_terms": list(self.capability_terms),
            "keywords": list(self.keywords),
        }


@dataclass(frozen=True)
class RosterLoad:
    active_tasks: int = 0
    active_sessions: int = 0
    outcome_hint_count: int = 0

    @property
    def penalty(self) -> int:
        return (self.active_tasks * 7) + (self.active_sessions * 4)

    def to_dict(self) -> dict[str, int]:
        return {
            "active_tasks": self.active_tasks,
            "active_sessions": self.active_sessions,
            "outcome_hint_count": self.outcome_hint_count,
            "penalty": self.penalty,
        }


@dataclass(frozen=True)
# pylint: disable=too-many-instance-attributes
class RosterFit:
    agent_id: uuid.UUID
    name: str
    role: str
    capabilities: list[str]
    provider: str
    model: str
    adapter_type: str
    work_function: str
    score: int
    weak: bool
    matched_signals: list[str] = field(default_factory=list)
    load: RosterLoad = field(default_factory=RosterLoad)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": str(self.agent_id),
            "name": self.name,
            "role": self.role,
            "capabilities": list(self.capabilities),
            "provider": self.provider,
            "model": self.model,
            "adapter_type": self.adapter_type,
            "work_function": self.work_function,
            "score": self.score,
            "weak": self.weak,
            "matched_signals": list(self.matched_signals),
            "load": self.load.to_dict(),
        }


PROFILES: dict[str, WorkFunctionProfile] = {
    "planning": WorkFunctionProfile(
        name="planning",
        role_terms=("planner", "product", "strategist", "pm"),
        capability_terms=("planning", "requirements", "roadmap", "strategy"),
        keywords=("plan", "planning", "requirements", "roadmap", "strategy", "product"),
    ),
    "investigation": WorkFunctionProfile(
        name="investigation",
        role_terms=("investigator", "researcher", "analyst"),
        capability_terms=("investigation", "research", "analysis", "diagnostics"),
        keywords=("investigate", "research", "analyze", "diagnose", "root cause"),
    ),
    "implementation": WorkFunctionProfile(
        name="implementation",
        role_terms=("developer", "engineer", "code-writer", "code writer", "coder"),
        capability_terms=("implementation", "coding", "development", "engineering"),
        keywords=("implement", "code", "build", "fix", "develop"),
    ),
    "review": WorkFunctionProfile(
        name="review",
        role_terms=("reviewer", "code reviewer", "qa"),
        capability_terms=("review", "code_review", "quality"),
        keywords=("review", "critique", "inspect", "quality"),
    ),
    "validation": WorkFunctionProfile(
        name="validation",
        role_terms=("validator", "tester", "qa"),
        capability_terms=("validation", "testing", "verification"),
        keywords=("validate", "test", "verify", "passed"),
    ),
    "summarization": WorkFunctionProfile(
        name="summarization",
        role_terms=("summarizer", "writer", "technical writer"),
        capability_terms=("summarization", "summary", "documentation"),
        keywords=("summarize", "summary", "document", "brief"),
    ),
    "management": WorkFunctionProfile(
        name="management",
        # Spec 8.3 step 2: management, ownership, planning, product,
        # coordination, or team-lead responsibility.
        role_terms=("manager", "team_lead", "lead", "owner", "product", "coordinator", "pm"),
        capability_terms=("management", "coordination", "planning", "leadership", "ownership"),
        keywords=("management", "coordination", "ownership", "planning", "product", "lead"),
    ),
}


class OrchestrationRosterMapper:
    def score_agent_definition(self, agent: Agent, work_function: str) -> RosterFit:
        """Score definition fit only, excluding load and outcome history."""
        return self._fit_agent(
            agent,
            self._profile_for(work_function),
            required_capabilities=(),
            domain_hints=(),
            load=RosterLoad(),
        )

    async def rank_agents(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        work_function: str,
        required_capabilities: list[str] | None = None,
        domain_hints: list[str] | None = None,
    ) -> list[RosterFit]:
        agents = await self._active_agents(db)
        loads = await self.loads_by_agent(db, project_id)
        return self._rank_agent_list(
            agents,
            loads,
            work_function,
            required_capabilities=required_capabilities,
            domain_hints=domain_hints,
        )

    def _rank_agent_list(
        self,
        agents: list[Agent],
        loads: dict[uuid.UUID, RosterLoad],
        work_function: str,
        required_capabilities: list[str] | None = None,
        domain_hints: list[str] | None = None,
    ) -> list[RosterFit]:
        profile = self._profile_for(work_function)
        required = tuple(
            sorted(
                {
                    self._normalize_token(capability)
                    for capability in (required_capabilities or [])
                    if self._normalize_token(capability)
                }
            )
        )
        domain = tuple(self._normalize_token(hint) for hint in (domain_hints or []))

        fits = [
            self._fit_agent(agent, profile, required, domain, loads.get(agent.id, RosterLoad()))
            for agent in agents
        ]
        return sorted(fits, key=lambda fit: (-fit.score, fit.load.penalty, fit.name.lower(), str(fit.agent_id)))

    async def context_snapshot(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        work_functions: tuple[str, ...] = DEFAULT_CONTEXT_WORK_FUNCTIONS,
        limit_per_function: int = 3,
    ) -> dict[str, Any]:
        limit = max(1, limit_per_function)
        agents = await self._active_agents(db)
        snapshot: dict[str, Any] = {
            "active_agent_count": len(agents),
            "work_functions": {},
        }
        loads = await self.loads_by_agent(db, project_id)
        for work_function in work_functions:
            fits = self._rank_agent_list(agents, loads, work_function)
            snapshot["work_functions"][work_function] = [fit.to_dict() for fit in fits[:limit]]
        return snapshot

    async def _active_agents(self, db: AsyncSession) -> list[Agent]:
        result = await db.execute(
            select(Agent)
            .where(Agent.is_active.is_(True))
            .order_by(Agent.name.asc(), Agent.id.asc())
        )
        return list(result.scalars().all())

    async def loads_by_agent(self, db: AsyncSession, project_id: uuid.UUID) -> dict[uuid.UUID, RosterLoad]:
        return await self._loads_by_agent(db, project_id)

    async def _loads_by_agent(self, db: AsyncSession, project_id: uuid.UUID) -> dict[uuid.UUID, RosterLoad]:
        active_tasks = await self._active_task_counts(db, project_id)
        active_sessions = await self._active_session_counts(db, project_id)
        outcome_hints = await self._outcome_hint_counts(db, project_id)
        agent_ids = {
            agent_id
            for agent_id in (set(active_tasks) | set(active_sessions) | set(outcome_hints))
            if agent_id is not None
        }
        return {
            agent_id: RosterLoad(
                active_tasks=active_tasks.get(agent_id, 0),
                active_sessions=active_sessions.get(agent_id, 0),
                outcome_hint_count=outcome_hints.get(agent_id, 0),
            )
            for agent_id in agent_ids
        }

    async def _active_task_counts(self, db: AsyncSession, project_id: uuid.UUID) -> dict[uuid.UUID, int]:
        result = await db.execute(
            select(Task.assigned_to, func.count(Task.id))
            .where(
                Task.project_id == project_id,
                Task.assigned_to.is_not(None),
                Task.status.in_(ACTIVE_TASK_STATUSES),
            )
            .group_by(Task.assigned_to)
        )
        return {agent_id: count for agent_id, count in result.all() if agent_id is not None}

    async def _active_session_counts(self, db: AsyncSession, project_id: uuid.UUID) -> dict[uuid.UUID, int]:
        result = await db.execute(
            select(Session.agent_id, func.count(Session.id))
            .where(
                Session.project_id == project_id,
                Session.status.in_(ACTIVE_SESSION_STATUSES),
            )
            .group_by(Session.agent_id)
        )
        return {agent_id: count for agent_id, count in result.all() if agent_id is not None}

    async def _outcome_hint_counts(self, db: AsyncSession, project_id: uuid.UUID) -> dict[uuid.UUID, int]:
        keywords = tuple(f"%{term}%" for term in OUTCOME_HINT_TERMS)
        candidates = (
            select(
                MemoryItem.agent_id.label("agent_id"),
                MemoryItem.content.label("content"),
                func.row_number()
                .over(
                    partition_by=MemoryItem.agent_id,
                    order_by=MemoryItem.created_at.desc(),
                )
                .label("row_number"),
            )
            .where(
                MemoryItem.shared.is_(True),
                or_(
                    and_(MemoryItem.scope == "project", MemoryItem.project_id == project_id),
                    MemoryItem.scope == "global",
                ),
                or_(*(MemoryItem.content.ilike(keyword) for keyword in keywords)),
            )
            .subquery()
        )
        result = await db.execute(
            select(candidates.c.agent_id, candidates.c.content)
            .where(candidates.c.row_number <= OUTCOME_HINT_ROWS_PER_AGENT)
        )
        counts: dict[uuid.UUID, int] = {}
        for agent_id, content in result.all():
            if agent_id is not None and self._tokens(content).intersection(OUTCOME_HINT_TERMS):
                counts[agent_id] = counts.get(agent_id, 0) + 1
        return counts

    def _fit_agent(
        self,
        agent: Agent,
        profile: WorkFunctionProfile,
        required_capabilities: tuple[str, ...],
        domain_hints: tuple[str, ...],
        load: RosterLoad,
    ) -> RosterFit:
        score = 20
        signals: list[str] = []
        role_tokens = self._tokens(agent.role)
        raw_capabilities = tuple(agent.capabilities or ())
        capabilities = {self._normalize_token(capability) for capability in raw_capabilities}
        capability_tokens = {token for capability in raw_capabilities for token in self._tokens(capability)}
        searchable_tokens = role_tokens | capabilities | capability_tokens | set(domain_hints)

        for term in profile.role_terms:
            normalized = self._normalize_token(term)
            if self._matches_tokens(normalized, role_tokens):
                score += 35
                signals.append(f"role:{normalized}")

        for capability in profile.capability_terms:
            normalized = self._normalize_token(capability)
            if normalized in capabilities:
                score += 30
                signals.append(f"capability:{normalized}")

        for capability in required_capabilities:
            if capability in capabilities:
                score += 40
                signals.append(f"required_capability:{capability}")

        for keyword in profile.keywords:
            normalized = self._normalize_token(keyword)
            if self._matches_tokens(normalized, searchable_tokens):
                score += 5
                signals.append(f"keyword:{normalized}")

        if agent.adapter_type == "api":
            score += 3
            signals.append("adapter:api")

        if isinstance(agent.config, dict) and agent.config.get("memory_enabled") is True:
            score += 5
            signals.append("config:memory_enabled")

        if load.outcome_hint_count:
            score += min(load.outcome_hint_count, 3) * 5
            signals.append(f"outcome_memory:{load.outcome_hint_count}")

        if load.penalty:
            score -= load.penalty
            signals.append(f"load_penalty:{load.penalty}")

        score = max(0, score)

        return RosterFit(
            agent_id=agent.id,
            name=agent.name,
            role=agent.role,
            capabilities=list(agent.capabilities or []),
            provider=agent.provider,
            model=agent.model,
            adapter_type=agent.adapter_type,
            work_function=profile.name,
            score=score,
            weak=score < WEAK_FIT_THRESHOLD,
            matched_signals=signals,
            load=load,
        )

    def _profile_for(self, work_function: str) -> WorkFunctionProfile:
        normalized = self._normalize_token(work_function)
        if normalized in PROFILES:
            return PROFILES[normalized]
        return WorkFunctionProfile(
            name=normalized,
            capability_terms=(normalized,),
            keywords=(normalized,),
        )

    @staticmethod
    def _normalize_text(value: Any) -> str:
        return str(value or "").strip().lower().replace("_", " ")

    @classmethod
    def _normalize_token(cls, value: Any) -> str:
        return cls._normalize_text(value).replace(" ", "_").replace("-", "_")

    @classmethod
    def _tokens(cls, value: Any) -> set[str]:
        text = cls._normalize_text(value)
        return {token for token in re.split(r"[^a-z0-9]+", text) if token}

    @classmethod
    def _matches_tokens(cls, normalized_term: str, tokens: set[str]) -> bool:
        if not normalized_term:
            return False
        term_tokens = set(normalized_term.split("_"))
        return normalized_term in tokens or term_tokens.issubset(tokens)
