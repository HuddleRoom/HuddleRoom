import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.agent import Agent
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services.knowledge_service import KnowledgeService
from huddleroom.services.channel_service import ChannelService
from huddleroom.services.message_service import MessageService
from huddleroom.services.event_bus import emit_event
from huddleroom.services.memory_service import MemoryService
from huddleroom.services.project_service import ProjectService
from huddleroom.services.litellm_models import build_litellm_model_name
from huddleroom.services.tool_executor import get_memory_system_prompt, MEMORY_TOOLS, run_tool_loop
from huddleroom.services.secret_redaction import redact_secrets
from huddleroom.services.agent_response_stream import AgentResponseInvocation, InvocationContext

logger = logging.getLogger(__name__)
knowledge_service = KnowledgeService()
channel_service = ChannelService()
message_service = MessageService()
memory_service_instance = MemoryService()


class ApiAdapter:
    async def run(self, session_id: uuid.UUID, db: AsyncSession, runner_task_id: str | None = None) -> None:
        result = await db.execute(select(Session).where(Session.id == session_id))

        session = result.scalar_one_or_none()
        if not session or session.status not in {"pending", "running"}:
            return
        from huddleroom.workers.session_tasks import orchestration_lineage_state
        lineage = await orchestration_lineage_state(db, session)
        if lineage is False:
            return
        orchestration = lineage is True
        attempt = (session.metadata_ or {}).get("attempt")
        if orchestration and (session.status != "running" or not isinstance(attempt, dict)
                            or not runner_task_id or attempt.get("claimed_runner_task_id") != runner_task_id
                            or session.runner_task_id != runner_task_id):
            return

        result = await db.execute(select(Agent).where(Agent.id == session.agent_id))
        agent = result.scalar_one_or_none()
        if not agent:
            session.status = "failed"
            session.error = "agent_not_found"
            if orchestration:
                from huddleroom.workers.session_tasks import _mark_attempt_result
                if not await _mark_attempt_result(db, session.id, runner_task_id):
                    return
            await db.flush()
            await emit_event(db, session.project_id, "session.failed", {
                "session_id": str(session_id),
                "error": "agent_not_found",
                "project_id": str(session.project_id),
            })
            from huddleroom.services.session_sync import sync_task_from_session
            await sync_task_from_session(db, session)
            return

        task = None
        if session.task_id:
            result = await db.execute(select(Task).where(Task.id == session.task_id))
            task = result.scalar_one_or_none()

        messages, task_context, knowledge_count, channel_count = await self._build_messages(agent, task, db)

        session.input_context = {
            "system_prompt": agent.system_prompt or "You are a focused AI task executor. Complete assigned work efficiently.",
            "task_context": task_context,
            "knowledge_items_count": knowledge_count,
            "channel_messages_count": channel_count,
        }
        if session.status == "pending":
            session.status = "running"
            session.started_at = datetime.now(timezone.utc)
            await db.flush()
            await emit_event(db, session.project_id, "session.started", {
                "session_id": str(session_id), "agent_id": str(session.agent_id),
                "task_id": str(session.task_id) if session.task_id else None,
                "project_id": str(session.project_id),
            })
        # Release SQLite's writer lock before the outbound LLM call.
        await db.commit()

        if orchestration:
            await self._run_with_retry(agent, session, messages, task_context, db, runner_task_id=runner_task_id)
        else:
            await self._run_with_retry(agent, session, messages, task_context, db)

    async def _build_messages(
        self, agent: Agent, task: Task | None, db: AsyncSession
    ) -> tuple[list[dict], str, int, int]:
        knowledge_items = []
        if task and task.project_id:
            try:
                knowledge_items = await knowledge_service.search(
                    db, task.project_id, task.description or task.title, limit=10
                )
            except Exception as e:
                logger.warning("Knowledge search failed: %s", e)

        channel_messages = []
        if task:
            try:
                ch = await channel_service.get_or_create_for_task(db, task.project_id, task.id)
                window = agent.config.get("context_message_window", 20)
                msgs, _ = await message_service.list(db, ch.id, limit=window)
                channel_messages = msgs
            except Exception as e:
                logger.warning("Channel history fetch failed: %s", e)

        task_context = ""
        if task:
            task_context = f"Task: {task.title}\n"
            if task.description:
                task_context += f"Description: {task.description}\n"

        knowledge_context = ""
        if knowledge_items:
            knowledge_context = "\nRelevant knowledge:\n"
            for k in knowledge_items:
                knowledge_context += f"- {k.title or 'Untitled'}: {k.content[:200]}\n"

        memory_context = ""
        if agent.config.get("memory_enabled", False) and task:
            try:
                query = f"{task.title} {task.description or ''}"
                memory_results = await memory_service_instance.search(
                    db, agent.id, task.project_id, query, limit=5
                )
                if memory_results:
                    memory_context = "\nRelevant memories:\n"
                    for m in memory_results:
                        vis = "shared" if m.shared else "private"
                        tags_str = ", ".join(m.tags) if m.tags else ""
                        memory_context += f"- [{m.created_at}] ({tags_str}, {vis}) {m.content[:200]}\n"
            except Exception as e:
                logger.warning("Memory search failed: %s", e)

        channel_context = ""
        if channel_messages:
            channel_context = "\nRecent messages:\n"
            for m in reversed(channel_messages):
                channel_context += f"- {m.content[:100]}\n"

        full_user_content = (task_context + knowledge_context + memory_context + channel_context).strip()
        if not full_user_content:
            full_user_content = "Please proceed with your assigned task."

        system_prompt = f"You are {agent.name}, the {agent.role}."
        if agent.system_prompt:
            system_prompt = f"{system_prompt}\n{agent.system_prompt}"
        if agent.config.get("memory_enabled", False):
            system_prompt += "\n\n" + get_memory_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_user_content},
        ]
        return messages, task_context, len(knowledge_items), len(channel_messages)

    async def _run_with_retry(
        self,
        agent: Agent,
        session: Session,
        messages: list[dict],
        task_context: str,
        db: AsyncSession,
        runner_task_id: str | None = None,
        invocation: AgentResponseInvocation | None = None,
    ) -> None:
        import litellm

        orchestration = isinstance((session.metadata_ or {}).get("attempt"), dict)

        if invocation is None:
            invocation = AgentResponseInvocation(
                InvocationContext(
                    session.project_id,
                    "agent",
                    str(agent.id),
                    agent.name,
                    "api",
                    "task",
                    agent.model,
                    task_context or None,
                )
            )

        run_config = session.metadata_.get("_run_config", {}) if session.metadata_ else {}

        provider_option_keys: set[str] = set()

        async def request_settings() -> tuple[str, str, dict, set[str]]:
            nonlocal provider_option_keys
            await db.refresh(agent)
            effective_model = run_config.get("model_override") or agent.model
            model_name = build_litellm_model_name(agent.provider, effective_model)
            provider_kwargs: dict = {"temperature": agent.config.get("temperature", 0.7)}
            if agent.config.get("provider_extras"):
                provider_kwargs.update(agent.config["provider_extras"])
            provider_kwargs["max_tokens"] = run_config.get("max_tokens", agent.config.get("max_tokens", 4096))
            if model_name.startswith(("ollama/", "ollama_chat/")):
                provider_kwargs.setdefault("api_base", os.getenv("OLLAMA_API_BASE", "http://localhost:11434"))
            stale_provider_option_keys = provider_option_keys
            provider_option_keys = set(agent.config.get("provider_extras", {}))
            if "api_base" in provider_kwargs:
                provider_option_keys.add("api_base")
            return model_name, effective_model, provider_kwargs, stale_provider_option_keys

        retry_delays = [10, 30, 60]
        error = None
        deadline = asyncio.get_running_loop().time() + run_config.get("timeout", 10 ** 9)
        token_usage_enforced = bool(run_config.get("_roadmap_budget_enforced") and "max_tokens" in run_config)
        token_cap = run_config.get("max_tokens") if token_usage_enforced else None
        prior_token_in = int((session.metadata_ or {}).get("token_count_in", 0))
        prior_token_out = int((session.metadata_ or {}).get("token_count_out", 0))
        token_in_total = prior_token_in
        token_out_total = prior_token_out
        attempt_token_in = attempt_token_out = 0
        token_usage_complete = True

        def _store_usage() -> None:
            session.metadata_ = {
                **(session.metadata_ or {}),
                "token_count_in": token_in_total,
                "token_count_out": token_out_total,
                **({"token_usage_complete": token_usage_complete} if token_usage_enforced else {}),
            }

        seen_responses: list[object] = []

        def _record_usage(resp) -> None:
            nonlocal token_in_total, token_out_total, attempt_token_in, attempt_token_out, token_usage_complete
            if any(resp is seen for seen in seen_responses):
                return
            seen_responses.append(resp)
            usage = getattr(resp, "usage", None)
            prompt = getattr(usage, "prompt_tokens", None)
            completion = getattr(usage, "completion_tokens", None)
            if not isinstance(prompt, int) or not isinstance(completion, int) or prompt < 0 or completion < 0:
                if token_usage_enforced:
                    token_usage_complete = False
                    raise RuntimeError("authoritative token usage unavailable")
                return
            token_in_total += prompt
            token_out_total += completion
            attempt_token_in += prompt
            attempt_token_out += completion
            if token_cap is not None and attempt_token_in + attempt_token_out > token_cap:
                raise RuntimeError("claimed token budget exhausted")

        def _remaining_timeout() -> float:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("claimed session timeout exhausted")
            return remaining

        async def _run_tool_loop(**kwargs):
            return await asyncio.wait_for(run_tool_loop(**kwargs), timeout=_remaining_timeout())

        for attempt in range(4):
            try:
                model_name, effective_model, provider_kwargs, _ = await request_settings()
                tools = MEMORY_TOOLS if agent.config.get("memory_enabled", False) else None

                last_resp_holder: dict = {}

                async def _completion(**kwargs):
                    nonlocal model_name, effective_model, provider_kwargs, token_in_total, token_out_total, token_usage_complete
                    model_name, effective_model, provider_kwargs, stale_provider_option_keys = await request_settings()
                    for key in stale_provider_option_keys:
                        kwargs.pop(key, None)
                    kwargs.update(model=model_name, **provider_kwargs)
                    if token_cap is not None:
                        remaining_tokens = token_cap - attempt_token_in - attempt_token_out
                        try:
                            prompt_tokens = litellm.token_counter(model=model_name, messages=kwargs.get("messages", []))
                        except Exception as exc:
                            raise RuntimeError("claimed token prompt measurement unavailable") from exc
                        if prompt_tokens >= remaining_tokens:
                            raise RuntimeError("claimed token budget exhausted")
                        kwargs["max_tokens"] = min(kwargs.get("max_tokens", remaining_tokens), remaining_tokens - prompt_tokens)
                    if orchestration:
                        await ProjectService().require_runnable_project(db, session.project_id)
                        from huddleroom.workers.session_tasks import mark_attempt_effect_started
                        if not await mark_attempt_effect_started(db, session.id, runner_task_id):
                            raise asyncio.CancelledError("orchestration runner claim lost")
                    resp = await asyncio.wait_for(litellm.acompletion(**kwargs), timeout=_remaining_timeout())
                    last_resp_holder["resp"] = resp
                    if not hasattr(resp, "__aiter__"):
                        _record_usage(resp)
                    return resp

                try:
                    output = await _run_tool_loop(
                        completion_fn=_completion,
                        messages=messages,
                        tools=tools,
                        agent_id=agent.id,
                        project_id=session.project_id,
                        db=db,
                        invocation=invocation,
                        response_observer=lambda response: (last_resp_holder.update(resp=response), _record_usage(response)),
                        model=model_name,
                        **provider_kwargs,
                    )
                except Exception as tool_err:
                    if tools and "tool" in str(tool_err).lower():
                        logger.warning("Model %s may not support tools, retrying without: %s", model_name, tool_err)
                        model_name, effective_model, provider_kwargs, _ = await request_settings()
                        output = await _run_tool_loop(
                            completion_fn=_completion,
                            messages=messages,
                            tools=None,
                            agent_id=agent.id,
                            project_id=session.project_id,
                            db=db,
                            invocation=invocation,
                            response_observer=lambda response: (last_resp_holder.update(resp=response), _record_usage(response)),
                            model=model_name,
                            **provider_kwargs,
                        )
                    else:
                        raise

                last_resp = last_resp_holder.get("resp")
                usage = getattr(last_resp, "usage", None) if last_resp else None
                token_in = token_in_total
                token_out = token_out_total
                session.output = output
                session.status = "completed"
                session.ended_at = datetime.now(timezone.utc)
                _store_usage()
                self._store_roadmap_elapsed(session)
                session.metadata_["model_used"] = effective_model
                if orchestration:
                    from huddleroom.workers.session_tasks import _mark_attempt_result
                    if not await _mark_attempt_result(db, session.id, runner_task_id):
                        return
                await db.flush()
                # INVARIANT: session.completed emitters MUST call sync_task_from_session
                # (task->done) in this same transaction before it commits; the
                # orchestrator's canonical-report consumption in
                # orchestration_service._ingest_session_evidence keys off
                # task.status == "done" and silently drops the report otherwise.
                await emit_event(db, session.project_id, "session.completed", {
                    "session_id": str(session.id),
                    "task_id": str(session.task_id) if session.task_id else None,
                    "token_count_out": token_out,
                    "project_id": str(session.project_id),
                    **({"runner_task_id": runner_task_id} if orchestration else {}),
                })
                from huddleroom.services.session_sync import sync_task_from_session
                await sync_task_from_session(db, session)
                return
            except litellm.RateLimitError:
                token_usage_complete = False if token_usage_enforced else token_usage_complete
                if attempt < 3:
                    await asyncio.sleep(min(retry_delays[attempt], _remaining_timeout()))
                else:
                    error = "rate_limit_exceeded"
                    break
            except litellm.APIConnectionError:
                token_usage_complete = False if token_usage_enforced else token_usage_complete
                if attempt < 3:
                    await asyncio.sleep(min(retry_delays[attempt], _remaining_timeout()))
                else:
                    error = "api_connection_error"
                    break
            except litellm.ServiceUnavailableError:
                token_usage_complete = False if token_usage_enforced else token_usage_complete
                if attempt < 3:
                    await asyncio.sleep(min(retry_delays[attempt], _remaining_timeout()))
                else:
                    error = "service_unavailable"
                    break
            except litellm.InternalServerError:
                token_usage_complete = False if token_usage_enforced else token_usage_complete
                if attempt < 3:
                    await asyncio.sleep(min(retry_delays[attempt], _remaining_timeout()))
                else:
                    error = "internal_server_error"
                    break
            except litellm.AuthenticationError:
                token_usage_complete = False if token_usage_enforced else token_usage_complete
                error = "auth_failed"
                break
            except litellm.ContextWindowExceededError:
                token_usage_complete = False if token_usage_enforced else token_usage_complete
                if attempt == 0:
                    messages[1]["content"] = task_context or "Please proceed with your assigned task."
                else:
                    error = "context_too_long"
                    break
            except Exception as e:
                token_usage_complete = False if token_usage_enforced else token_usage_complete
                # Truncate to 200 chars to avoid storing API keys or request bodies
                error = redact_secrets(type(e).__name__ + ": " + str(e)[:200])
                break

        session.status = "failed"
        session.error = error or "unknown_error"
        session.ended_at = datetime.now(timezone.utc)
        session.resumable = True
        _store_usage()
        self._store_roadmap_elapsed(session)
        if orchestration:
            from huddleroom.workers.session_tasks import _mark_attempt_result
            if not await _mark_attempt_result(db, session.id, runner_task_id):
                return
        await db.flush()
        await emit_event(db, session.project_id, "session.failed", {
            "session_id": str(session.id),
            "task_id": str(session.task_id) if session.task_id else None,
            "error": session.error,
            "project_id": str(session.project_id),
            "resumable": True,
        })
        from huddleroom.services.session_sync import sync_task_from_session
        await sync_task_from_session(db, session)

    @staticmethod
    def _store_roadmap_elapsed(session: Session) -> None:
        """Keep prior attempts measurable when a resumable session reuses its row."""
        config = (session.metadata_ or {}).get("_run_config", {})
        if not config.get("_roadmap_budget_enforced"):
            return
        prior = config.get("_roadmap_prior_usage", {})
        prior_seconds = Decimal(str(prior.get("max_hours", "0"))) * Decimal("3600")
        elapsed = Decimal("0")
        if session.started_at is not None and session.ended_at is not None:
            elapsed = max(Decimal("0"), Decimal(str((session.ended_at - session.started_at).total_seconds())))
        session.metadata_ = {
            **(session.metadata_ or {}),
            "_roadmap_elapsed_seconds": format((prior_seconds + elapsed).normalize(), "f"),
            "_roadmap_turn_count": int(Decimal(str(prior.get("max_turns", "0")))) + 1,
        }
