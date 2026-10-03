from __future__ import annotations

import json
import logging
import re
from uuid import UUID

import litellm

from huddleroom.config import settings
from huddleroom.services.agent_response_stream import AgentResponseInvocation, InvocationContext
from huddleroom.services.llm_structured_repair import complete_with_repair
from huddleroom.services.orchestration_llm_decision_adapter import orchestrator_preamble

logger = logging.getLogger(__name__)
_DEFAULT_OPENAI_CONTROL_MODEL = "openai/gpt-4o-mini"


class MeetingIntelligenceService:
    _CONSENSUS_REPAIR_INSTRUCTION = (
        "Your previous output was invalid or truncated. Reply again with a single JSON object only. "
        "No markdown fences. No prose. No explanation."
    )
    _NEXT_SPEAKER_REPAIR_INSTRUCTION = (
        "Your previous output was invalid. Reply again with a single JSON object only using exactly these keys: "
        "next_speaker_id, reason, close_item, close_reason. No markdown fences. No prose. No explanation."
    )

    @staticmethod
    def _split_model_name(model_name: str) -> tuple[str | None, str]:
        if "/" in model_name:
            provider, model = model_name.split("/", 1)
            return provider, model
        return None, model_name

    @staticmethod
    def _control_model() -> str:
        return settings.meeting_control_model or settings.orchestration_model

    @staticmethod
    def _control_model_source() -> str:
        return "meeting_control_model" if settings.meeting_control_model else "orchestration_model"

    async def _emit_trace(self, trace_emit, trace: dict) -> None:
        if trace_emit is None:
            return
        try:
            await trace_emit(trace)
        except Exception as exc:
            logger.warning("Meeting trace emission failed: %s", exc)

    async def check_consensus(
        self,
        item_title: str,
        item_question: str | None,
        item_options: list[str] | None,
        expected_speakers: list[str],
        turns_this_round: list[dict],
        prior_rounds_summary: str | None,
        trace_emit=None,
        trace_context: dict | None = None,
        *,
        project=None,
        meeting=None,
        project_id: UUID | None = None,
    ) -> dict:
        formatted_turns = "\n".join(
            f"{t['speaker']}: {t['content']}" for t in turns_this_round
        )
        prior_block = (
            f"PRIOR ROUNDS SUMMARY: {prior_rounds_summary}\n\n"
            if prior_rounds_summary
            else ""
        )
        options_text = ", ".join(item_options) if item_options else "Open-ended"
        speakers_text = ", ".join(expected_speakers) if expected_speakers else "unknown"

        user_msg = (
            f"AGENDA ITEM: {item_title}\n"
            f"QUESTION: {item_question or 'N/A'}\n"
            f"OPTIONS: {options_text}\n"
            f"EXPECTED SPEAKERS: {speakers_text}\n\n"
            f"TURNS THIS ROUND:\n{formatted_turns}\n\n"
            f"{prior_block}"
            "CONFIDENCE RUBRIC:\n"
            "1.0 — All expected speakers explicitly agree on the same named option, no hedging.\n"
            "0.8 — All speakers agree on same option; minor hedging present.\n"
            "0.6 — Strong majority agrees; one speaker is ambiguous but not opposed.\n"
            "0.4 — Majority leans one way; at least one speaker clearly opposes.\n"
            "0.2 — No majority; multiple distinct positions expressed.\n"
            "0.0 — Outright disagreement or no substantive positions stated.\n\n"
            'Output JSON only: {"consensus": true|false, "agreed_position": "<exact option text if consensus=true, else null>", '
            '"confidence": 0.0-1.0, "dissenting_speakers": ["<name>", ...], "rationale": "<one sentence>"}'
        )
        control_model = self._control_model()
        provider, model = self._split_model_name(control_model)
        messages = [
            {
                "role": "system",
                "content": (
                    orchestrator_preamble(project, meeting=meeting) + "\n\n"
                    "You are a consensus detector for a structured AI deliberation system. "
                    "Analyze the provided meeting turns and determine whether participants have reached "
                    "genuine agreement. Silence does not count as agreement. Every expected speaker must "
                    "have explicitly stated support for the same option for consensus to be true."
                ),
            },
            {"role": "user", "content": user_msg},
        ]

        await self._emit_trace(
            trace_emit,
            {
                **(trace_context or {}),
                "kind": "consensus_check",
                "stage": "request",
                "provider": provider,
                "model": model,
                "message_count": len(messages),
                "prompt_chars": sum(len(m.get("content", "") or "") for m in messages),
                "messages": messages,
            },
        )

        default = {
            "consensus": False,
            "agreed_position": None,
            "confidence": 0.0,
            "dissenting_speakers": [],
            "rationale": "",
        }
        try:
            raw = ""
            parsed = default
            trace_parse_status = None
            max_tok = 600

            # Outer loop for max_tokens escalation; inner repair via complete_with_repair
            for token_attempt in range(2):
                try:
                    # Capture first parse failure status for tracing
                    first_status_holder = {"status": None}

                    def parse_consensus_response(content: str) -> dict:
                        """Raise on parse failure so complete_with_repair can retry."""
                        result, status = self._parse_json_response_with_status(content, default=default)
                        if first_status_holder["status"] is None and status != "ok":
                            first_status_holder["status"] = status
                        if status != "ok":
                            raise ValueError(f"consensus parse failed: {status}")
                        if not isinstance(result, dict):
                            raise ValueError("consensus response is not a dict")
                        return result

                    request = {
                        "model": control_model,
                        "messages": messages,
                        "temperature": 0.0,
                        "max_tokens": max_tok,
                    }

                    # Create invocation context for consensus check (only if project_id available)
                    consensus_invocation = None
                    if project_id is not None:
                        consensus_request_prompt = {
                            "operation": "meeting_consensus",
                            "title": item_title or "Unknown agenda item",
                        }
                        consensus_invocation_ctx = InvocationContext(
                            project_id,
                            actor_kind="system",
                            actor_id="orchestrator",
                            actor_label="Orchestrator",
                            invocation_kind="api",
                            operation="meeting_consensus",
                            model_or_runtime=model,
                            request_prompt=consensus_request_prompt,
                        )
                        consensus_invocation = AgentResponseInvocation(consensus_invocation_ctx)

                    parsed = await complete_with_repair(
                        litellm.acompletion,
                        request,
                        parse=parse_consensus_response,
                        max_attempts=3,
                        invocation=consensus_invocation,
                    )
                    # Success; use captured first failure status if any, else "ok"
                    if trace_parse_status is None:
                        trace_parse_status = first_status_holder["status"] or "ok"
                    break
                except (ValueError, json.JSONDecodeError) as exc:
                    # complete_with_repair exhausted; escalate max_tokens if first attempt
                    if trace_parse_status is None:
                        # Try to extract status from exc; fallback to error message
                        exc_str = str(exc).lower()
                        if "empty" in exc_str:
                            trace_parse_status = "empty"
                        elif "malformed" in exc_str:
                            trace_parse_status = "malformed"
                        else:
                            trace_parse_status = "error"
                    if token_attempt == 0:
                        logger.info("consensus_check: parse failed, escalating max_tokens from %d to %d", max_tok, max_tok * 2)
                        max_tok *= 2
                        # Continue to next token_attempt
                    else:
                        # Both token attempts exhausted; use default
                        parsed = default

            if trace_parse_status is None:
                trace_parse_status = "ok"

            # Validate agreed_position is one of the provided options
            if parsed.get("consensus") and item_options:
                agreed = (parsed.get("agreed_position") or "").strip()
                options_lower = [o.lower() for o in item_options]
                if agreed.lower() not in options_lower:
                    parsed["consensus"] = False
                    parsed["agreed_position"] = None
                    parsed["confidence"] = 0.0

            await self._emit_trace(
                trace_emit,
                {
                    **(trace_context or {}),
                    "kind": "consensus_check",
                    "stage": "response",
                    "provider": provider,
                    "model": model,
                    "raw_response": raw,
                    "parsed_result": parsed,
                    "parse_status": trace_parse_status,
                },
            )
            return parsed
        except Exception as exc:
            logger.warning("check_consensus LLM call failed: %s", exc)
            await self._emit_trace(
                trace_emit,
                {
                    **(trace_context or {}),
                    "kind": "consensus_check",
                    "stage": "response",
                    "provider": provider,
                    "model": model,
                    "error": str(exc),
                    "parse_status": "error",
                },
            )
            return default

    async def extract_positions(
        self,
        turns: list[dict],
        item_options: list[str] | None = None,
        trace_emit=None,
        trace_context: dict | None = None,
        *,
        project=None,
        meeting=None,
        project_id: UUID | None = None,
    ) -> list[dict]:
        formatted = "\n".join(f"{t['speaker']}: {t['content']}" for t in turns)
        options_text = ", ".join(item_options) if item_options else "none — open-ended"
        user_msg = (
            f"OPTIONS (canonical): {options_text}\n\n"
            f"TURNS:\n{formatted}\n\n"
            'Output JSON array: [{"speaker": "<name>", "option": "<exact option text or free-text if no options>", "summary": "<one sentence>"}]'
        )
        control_model = self._control_model()
        provider, model = self._split_model_name(control_model)
        messages = [
            {
                "role": "system",
                "content": (
                    orchestrator_preamble(project, meeting=meeting) + "\n\n"
                    "Extract each participant's stated position. If the agenda has explicit options, "
                    "map each position to the closest matching option name. Output canonical option "
                    "names, not paraphrases."
                ),
            },
            {"role": "user", "content": user_msg},
        ]
        await self._emit_trace(
            trace_emit,
            {
                **(trace_context or {}),
                "kind": "extract_positions",
                "stage": "request",
                "provider": provider,
                "model": model,
                "message_count": len(messages),
                "prompt_chars": sum(len(m.get("content", "") or "") for m in messages),
                "messages": messages,
            },
        )
        try:
            def parse_positions_response(content: str) -> list:
                """Raise on parse failure so complete_with_repair can retry."""
                result = self._parse_json_response(content, default=[])
                if not isinstance(result, list):
                    raise ValueError("positions response is not a list")
                return result

            request = {
                "model": control_model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 400,
            }

            # Create invocation context for positions extraction (only if project_id available)
            positions_invocation = None
            if project_id is not None:
                positions_request_prompt = {
                    "operation": "meeting_positions",
                }
                positions_invocation_ctx = InvocationContext(
                    project_id,
                    actor_kind="system",
                    actor_id="orchestrator",
                    actor_label="Orchestrator",
                    invocation_kind="api",
                    operation="meeting_positions",
                    model_or_runtime=model,
                    request_prompt=positions_request_prompt,
                )
                positions_invocation = AgentResponseInvocation(positions_invocation_ctx)

            parsed = await complete_with_repair(
                litellm.acompletion,
                request,
                parse=parse_positions_response,
                max_attempts=3,
                invocation=positions_invocation,
            )
            await self._emit_trace(
                trace_emit,
                {
                    **(trace_context or {}),
                    "kind": "extract_positions",
                    "stage": "response",
                    "provider": provider,
                    "model": model,
                    "raw_response": "",
                    "raw_response_chars": 0,
                    "parsed_result": parsed,
                    "parse_status": "ok",
                },
            )
            return parsed
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("extract_positions LLM call failed after retries: %s", exc)
            await self._emit_trace(
                trace_emit,
                {
                    **(trace_context or {}),
                    "kind": "extract_positions",
                    "stage": "response",
                    "provider": provider,
                    "model": model,
                    "error": str(exc),
                },
            )
            return []

    async def select_next_speaker(
        self,
        participant_ids: list[str],
        participant_names: dict[str, str],
        transcript_excerpt: str,
        item_title: str | None = None,
        item_options: list[str] | None = None,
        pending_signals: list[dict] | None = None,
        spoke_this_round: dict[str, bool] | None = None,
        trace_emit=None,
        trace_context: dict | None = None,
        *,
        project=None,
        meeting=None,
        project_id: UUID | None = None,
    ) -> dict:
        roster = "\n".join(
            f"- id={pid} name={participant_names.get(pid, pid)} "
            f"spoke_this_round={bool((spoke_this_round or {}).get(pid, False))}"
            for pid in participant_ids
        )
        signals_text = "none"
        if pending_signals:
            sig_lines = [
                f"- {s.get('agent_id', 'unknown')}: {s.get('message') or 'wants to speak'}"
                for s in pending_signals
            ]
            signals_text = "\n".join(sig_lines)
        options_text = ", ".join(item_options) if item_options else "open-ended"

        user_msg = (
            f"AGENDA ITEM: {item_title or 'current item'}\n"
            f"OPTIONS: {options_text}\n\n"
            f"PARTICIPANTS (spoke_this_round flag):\n{roster}\n\n"
            f"PENDING SPEAK SIGNALS:\n{signals_text}\n\n"
            f"DISCUSSION SO FAR (last 2000 chars):\n{transcript_excerpt}\n\n"
            "Selection rules:\n"
            "1. Prefer participants who have signaled they want to speak.\n"
            "2. Among non-signalers, prefer participants who have NOT yet spoken this round.\n"
            "3. Among those, prefer the participant whose expertise best matches the current options.\n\n"
            'Output JSON only: {"next_speaker_id": "<uuid>", "reason": "<string>", '
            '"close_item": true|false, "close_reason": "<string or null>"}'
        )
        control_model = self._control_model()
        candidate_models = [control_model]
        if control_model != _DEFAULT_OPENAI_CONTROL_MODEL:
            candidate_models.append(_DEFAULT_OPENAI_CONTROL_MODEL)

        provider, model = self._split_model_name(control_model)
        messages = [
            {
                "role": "system",
                "content": (
                    orchestrator_preamble(project, meeting=meeting) + "\n\n"
                    "You are a meeting moderator. After each speaker turn, decide who should speak next "
                    "and whether the current agenda item should close. Close the item only when the "
                    "question has been fully addressed or when further discussion is unlikely to produce "
                    "new information."
                ),
            },
            {"role": "user", "content": user_msg},
        ]
        await self._emit_trace(
            trace_emit,
            {
                **(trace_context or {}),
                "kind": "moderator_select_next_speaker",
                "stage": "request",
                "provider": provider,
                "model": model,
                "messages": messages,
            },
        )
        default = {
            "next_speaker_id": participant_ids[0] if participant_ids else None,
            "reason": "Moderator control model returned no valid JSON; falling back to participant order.",
            "close_item": False,
            "close_reason": None,
            "selector_type": "moderator",
            "selected_by": self._control_model_source(),
            "model_used": control_model,
            "messages": messages,
            "raw_response": "",
            "fallback_used": True,
        }
        # Create invocation context for speaker selection (only if project_id available)
        speaker_invocation = None
        if project_id is not None:
            speaker_request_prompt = {
                "operation": "meeting_speaker_selection",
                "title": item_title or "Unknown agenda item",
            }
            speaker_invocation_ctx = InvocationContext(
                project_id,
                actor_kind="system",
                actor_id="orchestrator",
                actor_label="Orchestrator",
                invocation_kind="api",
                operation="meeting_speaker_selection",
                model_or_runtime=control_model,
                request_prompt=speaker_request_prompt,
            )
            speaker_invocation = AgentResponseInvocation(speaker_invocation_ctx)

        try:
            parsed = default
            response_model = control_model
            success = False
            # Outer loop: model fallback (provider-level)
            for candidate_model in candidate_models:
                response_model = candidate_model
                try:
                    def parse_speaker_response(content: str) -> dict:
                        """Raise on parse failure so complete_with_repair can retry."""
                        result, status = self._parse_json_response_with_status(content, default=default)
                        if not isinstance(result, dict) or status != "ok":
                            raise ValueError(f"speaker parse failed: {status}")
                        return result

                    request = {
                        "model": candidate_model,
                        "messages": messages,
                        "temperature": 0.3,
                        "max_tokens": 200,
                    }
                    result = await complete_with_repair(
                        litellm.acompletion,
                        request,
                        parse=parse_speaker_response,
                        max_attempts=3,
                        invocation=speaker_invocation,
                    )
                    # Success; augment result with metadata
                    parsed = {
                        **result,
                        "selector_type": "moderator",
                        "selected_by": self._control_model_source(),
                        "model_used": candidate_model,
                        "messages": messages,
                        "fallback_used": False,
                    }
                    success = True
                    break
                except (ValueError, json.JSONDecodeError) as exc:
                    # This model failed; try next candidate model
                    logger.debug("select_next_speaker failed for model %s: %s", candidate_model, exc)
                    parsed = {
                        **default,
                        "model_used": candidate_model,
                    }
                    # Continue to next candidate_model

            await self._emit_trace(
                trace_emit,
                {
                    **(trace_context or {}),
                    "kind": "moderator_select_next_speaker",
                    "stage": "response",
                    "provider": self._split_model_name(response_model)[0],
                    "model": self._split_model_name(response_model)[1],
                    "raw_response": parsed.get("raw_response", ""),
                    "parsed_result": parsed,
                    "parse_status": "ok" if success else "fallback",
                },
            )
            return parsed
        except Exception as exc:
            logger.warning("select_next_speaker LLM call failed: %s", exc)
            await self._emit_trace(
                trace_emit,
                {
                    **(trace_context or {}),
                    "kind": "moderator_select_next_speaker",
                    "stage": "response",
                    "provider": provider,
                    "model": model,
                    "error": str(exc),
                },
            )
            return default

    def _parse_json_response_with_status(self, raw: str, default):
        if not raw:
            logger.warning("Failed to parse LLM JSON response: %s", raw[:200])
            return default, "empty"

        fenced = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", raw, re.DOTALL | re.IGNORECASE)
        candidates = [fenced.group(1)] if fenced else []
        candidates.append(raw)

        embedded = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
        if embedded:
            candidates.append(embedded.group(1))

        for candidate in candidates:
            try:
                return json.loads(candidate), "ok"
            except (TypeError, json.JSONDecodeError):
                continue
        logger.warning("Failed to parse LLM JSON response: %s", raw[:200])
        return default, "malformed"

    def _parse_json_response(self, raw: str, default):
        parsed, _ = self._parse_json_response_with_status(raw, default)
        return parsed
