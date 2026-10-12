"""Server-derived proof that an agent can actually execute a delegated remediation (SPR 180/184).

Only current executable seams count: the effective adapter/runtime, the command ``CliAdapter`` would
really build, executable and frozen-workspace validation, and ``effective_authorization``. Descriptive
``capabilities``, prompt text and ``config.tools`` never establish or restrict executable access.
"""
from __future__ import annotations

import shutil  # looked up at call time (shutil.which) so the executable check reflects the current host
from pathlib import Path
from uuid import UUID

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict

from huddleroom.adapters.cli_adapter import CliAdapter
from huddleroom.models.agent import Agent
from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.models.project import Project
from huddleroom.services.orchestration_authority_service import effective_authorization
from huddleroom.services.orchestration_recovery_configuration import RecoveryConfigurationService
from huddleroom.services.project_service import ProjectService

# Work functions a remediation task may run as; "implementation" is preferred (not ad-hoc gated).
REMEDIATION_WORK_FUNCTIONS = ("implementation", "investigation")
# Flags in the built command that hand the agent its local tools without per-call approval.
_UNRESTRICTED_FLAGS = frozenset({
    "--dangerously-skip-permissions", "--allow-all-tools", "--yes", "--auto", "--no-approve",
    "--yolo", "--dangerously-bypass-approvals-and-sandbox",
})


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ActualExecutionScopeV1(_Frozen):
    unrestricted_tools: bool
    workspace_write: bool
    network: bool
    shell_without_approval: bool


class AgentExecutionEvidenceV1(_Frozen):
    agent_id: UUID
    is_active: bool
    adapter_type: str
    cli_runtime: str | None
    executable: str | None
    executable_available: bool
    workspace_available: bool
    accepted_work_functions: list[str]
    authority_allowed: bool
    authority_reason: str | None
    workspace_mutation: str
    model_spend_remaining: dict[str, str]
    actual_execution_scope: ActualExecutionScopeV1
    qualified: bool
    gaps: list[str]


def scope_from_command(cmd: list[str] | None, *, custom: bool = False) -> ActualExecutionScopeV1:
    """Scope of the command as built; a custom script is arbitrary local code."""
    if not cmd:
        return ActualExecutionScopeV1(unrestricted_tools=False, workspace_write=False, network=False,
                                      shell_without_approval=False)
    unrestricted = custom or any(arg in _UNRESTRICTED_FLAGS for arg in cmd)
    return ActualExecutionScopeV1(
        unrestricted_tools=unrestricted,
        workspace_write=unrestricted or any(arg in {"workspace-write", "sandbox_mode=workspace-write"} for arg in cmd),
        network=unrestricted or "sandbox_workspace_write.network_access=true" in cmd,
        # Non-interactive `exec` (Codex) runs shell commands with no per-command approval prompt.
        shell_without_approval=unrestricted or cmd[1:2] == ["exec"],
    )


class AgentExecutionEvidenceService:
    async def inspect(self, db, goal: OrchestrationGoal, run: OrchestrationRun, agent_id: UUID) -> AgentExecutionEvidenceV1:
        agent = await db.get(Agent, agent_id)
        if agent is None:
            raise ValueError("out_of_scope_target")
        gaps: list[str] = []
        config = agent.config if isinstance(agent.config, dict) else {}
        runtime = config.get("cli_runtime", agent.cli_runtime or "claude_code") if agent.adapter_type == "cli" else None
        try:
            workspace = await ProjectService().require_runnable_project(db, goal.project_id)
            workspace_available = True
        except HTTPException:
            project = await db.get(Project, goal.project_id)
            workspace, workspace_available = Path((project.workspace_path if project else None) or "/"), False
            gaps.append("workspace_unavailable")
        cmd = None
        if runtime is not None:
            try:
                # Neutral task text: only the runtime's own flags may contribute to the scope.
                cmd = CliAdapter._build_command(runtime, "proceed", config, workspace, workspace / "task.md",
                                                model=agent.model)
            except RuntimeError:
                cmd = None
        else:
            gaps.append("adapter_has_no_registered_remediation_action")  # API/routine agents
        executable = cmd[0] if cmd else None
        executable_available = bool(executable and shutil.which(executable))
        if runtime is not None and not executable_available:
            gaps.append("executable_unavailable")
        scope = scope_from_command(cmd, custom=runtime == "custom")
        # Only the accepted (completed, current) definition review counts; a pending rerun grants nothing.
        surfaces = await RecoveryConfigurationService()._accepted_surfaces(db, goal)  # pylint: disable=protected-access
        accepted = [function for item in surfaces["eligibility"] if str(item["agent_id"]) == str(agent.id)
                    for function in item["work_functions"] if isinstance(function, str)]
        if not agent.is_active:
            gaps.append("agent_inactive")
        if not set(accepted) & set(REMEDIATION_WORK_FUNCTIONS):
            gaps.append("not_accepted_for_remediation")
        authority = await effective_authorization(db, goal, run)
        if not authority.allowed:
            gaps.append(f"authority_denied:{authority.reason_code}")
        if scope.workspace_write and authority.workspace_mutation == "read_only":
            gaps.append("workspace_write_exceeds_authority")
        return AgentExecutionEvidenceV1(
            agent_id=agent.id, is_active=bool(agent.is_active), adapter_type=agent.adapter_type, cli_runtime=runtime,
            executable=executable, executable_available=executable_available, workspace_available=workspace_available,
            accepted_work_functions=sorted(accepted), authority_allowed=authority.allowed,
            authority_reason=authority.reason_code, workspace_mutation=authority.workspace_mutation,
            model_spend_remaining={key: format(value.normalize(), "f") for key, value in authority.model_spend_remaining.items()},
            actual_execution_scope=scope, qualified=not gaps, gaps=gaps,
        )
