from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from openai import OpenAI

from .schemas import AgentDecision, ExperimentTrace, ProposedAction, ResearchState, TaskSpec
from .top_models import model_key


ROOT = Path(__file__).resolve().parents[2]


SYSTEM = """You are an autonomous ML research planner.
You propose the next experiment inside a safe catalog. You are not the world model.
Qwen-AgentWorld will predict outcomes after your decision; Lucky Loop will execute real code and verify claims.

Return strict JSON only with exactly these keys:
- research_question: string
- working_hypothesis: string
- candidate_action_ids: array of strings from the provided catalog
- preferred_action_id: one action_id from the provided catalog
- rationale: string
- expected_evidence_needed: string
- claim_risk: string
- stop_or_continue: "continue" or "stop_and_report"

Rules:
- Never invent shell commands.
- Never choose an action_id outside the catalog.
- Prefer real evidence over score chasing.
- If top single-run models are close, request matched multi-seed verification before robust best-model claims.
- If budget is low, consolidate evidence instead of chasing marginal scores.
- If no experiments exist, establish a simple baseline first."""


class ResearchAgent(Protocol):
    backend: str
    model_name: str | None

    def propose_next_step(
        self,
        task: TaskSpec,
        state: ResearchState,
        candidate_catalog: list[ProposedAction],
        prior_traces: list[ExperimentTrace],
    ) -> tuple[AgentDecision, str]:
        ...


def _trace_summary(trace: ExperimentTrace) -> dict:
    return {
        "run_id": trace.run_id,
        "model": trace.proposed_action.model,
        "params": trace.proposed_action.params,
        "accuracy": trace.actual_result.accuracy,
        "f1": trace.actual_result.f1,
        "status": trace.actual_result.status,
        "prediction_match": trace.comparison.metric_match,
        "unexpected_events": trace.comparison.unexpected_events,
        "verification_status": trace.verification.status if trace.verification else None,
    }


def build_agent_prompt(
    task: TaskSpec,
    state: ResearchState,
    candidate_catalog: list[ProposedAction],
    prior_traces: list[ExperimentTrace],
) -> str:
    payload = {
        "task": task.model_dump(),
        "state": state.model_dump(),
        "candidate_catalog": [candidate.model_dump() for candidate in candidate_catalog],
        "prior_traces": [_trace_summary(trace) for trace in prior_traces],
        "planner_contract": {
            "safe_catalog_only": True,
            "no_freeform_commands": True,
            "must_return_preferred_action_id": True,
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def build_agent_request(
    task: TaskSpec,
    state: ResearchState,
    candidate_catalog: list[ProposedAction],
    prior_traces: list[ExperimentTrace],
) -> dict:
    return {
        "schema_version": "1.0",
        "task": task.model_dump(),
        "state": state.model_dump(),
        "candidate_catalog": [candidate.model_dump() for candidate in candidate_catalog],
        "prior_traces_summary": [_trace_summary(trace) for trace in prior_traces],
        "planner_contract": {
            "safe_catalog_only": True,
            "no_freeform_commands": True,
            "choose_action_id_only": True,
            "qwen_agentworld_is_world_model_not_planner": True,
        },
        "response_schema": {
            "research_question": "string",
            "working_hypothesis": "string",
            "candidate_action_ids": ["action_id"],
            "preferred_action_id": "action_id from candidate_catalog",
            "rationale": "string",
            "expected_evidence_needed": "string",
            "claim_risk": "string",
            "stop_or_continue": "continue | stop_and_report",
        },
    }


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _json_from_text(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("agent response did not contain a JSON object")
        return json.loads(text[start : end + 1])


def _catalog_ids(candidates: list[ProposedAction]) -> set[str]:
    return {candidate.action_id or "" for candidate in candidates}


def validate_agent_decision(decision: AgentDecision, candidates: list[ProposedAction]) -> AgentDecision:
    ids = _catalog_ids(candidates)
    if decision.preferred_action_id not in ids:
        raise ValueError(f"agent preferred unknown action_id: {decision.preferred_action_id}")
    unknown = [action_id for action_id in decision.candidate_action_ids if action_id not in ids]
    if unknown:
        raise ValueError(f"agent referenced unknown candidate_action_ids: {unknown}")
    return decision


class LLMResearchAgent:
    def __init__(self, backend: str = "openai_compatible") -> None:
        self.backend = backend
        self.base_url = os.getenv("LUCKYLOOP_AGENT_BASE_URL")
        self.model_name = os.getenv("LUCKYLOOP_AGENT_MODEL")
        self.api_key = os.getenv("LUCKYLOOP_AGENT_API_KEY")
        if not self.base_url or not self.model_name or not self.api_key:
            raise RuntimeError(
                "LLM planner requires LUCKYLOOP_AGENT_BASE_URL, "
                "LUCKYLOOP_AGENT_MODEL, and LUCKYLOOP_AGENT_API_KEY."
            )

    def propose_next_step(
        self,
        task: TaskSpec,
        state: ResearchState,
        candidate_catalog: list[ProposedAction],
        prior_traces: list[ExperimentTrace],
    ) -> tuple[AgentDecision, str]:
        prompt = build_agent_prompt(task, state, candidate_catalog, prior_traces)
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=900,
        )
        content = resp.choices[0].message.content or "{}"
        decision = AgentDecision(**_json_from_text(content))
        return validate_agent_decision(decision, candidate_catalog), prompt


@dataclass
class HandoffResearchAgent:
    backend: str = "generic_handoff"
    model_name: str | None = "external-agent-handoff"
    io_dir: Path = ROOT / "agent_io"
    timeout_seconds: int = 900
    poll_seconds: float = 2.0

    def propose_next_step(
        self,
        task: TaskSpec,
        state: ResearchState,
        candidate_catalog: list[ProposedAction],
        prior_traces: list[ExperimentTrace],
    ) -> tuple[AgentDecision, str]:
        request = build_agent_request(task, state, candidate_catalog, prior_traces)
        prompt = json.dumps(request, indent=2, sort_keys=True)
        task_dir = self.io_dir / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        request_path = task_dir / f"{state.state_id}.request.json"
        response_path = task_dir / f"{state.state_id}.response.json"
        if response_path.exists():
            response_path.unlink()
        request_path.write_text(prompt + "\n", encoding="utf-8")
        print(f"Waiting for external agent response: {response_path}")
        print(f"Agent request written to: {request_path}")
        deadline = time.time() + self.timeout_seconds
        while time.time() < deadline:
            if response_path.exists():
                data = json.loads(response_path.read_text(encoding="utf-8"))
                decision = AgentDecision(**data)
                return validate_agent_decision(decision, candidate_catalog), prompt
            time.sleep(self.poll_seconds)
        raise TimeoutError(f"Timed out waiting for agent response: {response_path}")


@dataclass
class CommandResearchAgent:
    backend: str = "generic_command"
    model_name: str | None = "external-agent-command"
    io_dir: Path = ROOT / "agent_io"
    timeout_seconds: int = 900

    def propose_next_step(
        self,
        task: TaskSpec,
        state: ResearchState,
        candidate_catalog: list[ProposedAction],
        prior_traces: list[ExperimentTrace],
    ) -> tuple[AgentDecision, str]:
        template = os.getenv("LUCKYLOOP_AGENT_COMMAND")
        if not template:
            raise RuntimeError("agent_command mode requires LUCKYLOOP_AGENT_COMMAND.")
        request = build_agent_request(task, state, candidate_catalog, prior_traces)
        prompt = json.dumps(request, indent=2, sort_keys=True)
        task_dir = self.io_dir / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        request_path = task_dir / f"{state.state_id}.request.json"
        response_path = task_dir / f"{state.state_id}.response.json"
        if response_path.exists():
            response_path.unlink()
        request_path.write_text(prompt + "\n", encoding="utf-8")
        command = template.format(
            request_path=shlex.quote(str(request_path)),
            response_path=shlex.quote(str(response_path)),
        )
        proc = subprocess.run(command, shell=True, cwd=str(ROOT), text=True, capture_output=True, timeout=self.timeout_seconds)
        if proc.returncode != 0:
            raise RuntimeError(
                "agent command failed "
                f"returncode={proc.returncode}\nstdout={proc.stdout[-2000:]}\nstderr={proc.stderr[-2000:]}"
            )
        if not response_path.exists():
            raise RuntimeError(f"agent command completed but did not write {response_path}")
        data = json.loads(response_path.read_text(encoding="utf-8"))
        decision = AgentDecision(**data)
        return validate_agent_decision(decision, candidate_catalog), prompt


@dataclass
class ReplayResearchAgent:
    backend: str = "replay"
    model_name: str | None = "replay-agent-v1"

    def propose_next_step(
        self,
        task: TaskSpec,
        state: ResearchState,
        candidate_catalog: list[ProposedAction],
        prior_traces: list[ExperimentTrace],
    ) -> tuple[AgentDecision, str]:
        prompt = build_agent_prompt(task, state, candidate_catalog, prior_traces)
        decision = _replay_decision(task, state, candidate_catalog, prior_traces)
        return validate_agent_decision(decision, candidate_catalog), prompt


def _choose(candidates: list[ProposedAction], predicate) -> ProposedAction:
    for candidate in candidates:
        if predicate(candidate):
            return candidate
    return candidates[0]


def _seen_keys(traces: list[ExperimentTrace]) -> set[str]:
    return {model_key(trace.proposed_action.model, trace.proposed_action.params) for trace in traces}


def _replay_decision(
    task: TaskSpec,
    state: ResearchState,
    candidates: list[ProposedAction],
    traces: list[ExperimentTrace],
) -> AgentDecision:
    if not candidates:
        raise ValueError("ReplayResearchAgent requires a non-empty candidate catalog")

    top_summary = state.top_model_summary
    if top_summary and top_summary.needs_robustness_verification:
        preferred = _choose(candidates, lambda c: c.model == "top_model_verification")
        hypothesis = "The best single-run model may not be robust across matched seeds."
        evidence = "Run matched multi-seed verification on the observed top models before allowing a robust winner claim."
        risk = "A single-split leaderboard can overstate a fragile best-model claim."
    elif not traces:
        preferred = _choose(
            candidates,
            lambda c: c.model == "logistic_regression" and not c.params.get("scale"),
        )
        hypothesis = "A simple unscaled linear baseline should anchor the search before interventions."
        evidence = "Establish baseline accuracy, runtime, and convergence behavior."
        risk = "No best-model claim is allowed from a baseline-only run."
    elif (
        traces[0].proposed_action.model == "logistic_regression"
        and not traces[0].proposed_action.params.get("scale")
        and not any(
            trace.proposed_action.model == "logistic_regression" and trace.proposed_action.params.get("scale")
            for trace in traces
        )
    ):
        preferred = _choose(
            candidates,
            lambda c: c.model == "logistic_regression" and c.params.get("scale"),
        )
        hypothesis = "Feature scaling should improve or stabilize scale-sensitive linear classification."
        evidence = "Compare scaled logistic regression against the unscaled baseline."
        risk = "A single improvement is observational until repeated-seed verification."
    elif any(trace.comparison.unexpected_events for trace in traces):
        preferred = _choose(
            candidates,
            lambda c: c.model in {"svc", "gradient_boosting", "hist_gradient_boosting"},
        )
        hypothesis = "A prediction miss should trigger exploration of a different inductive bias."
        evidence = "Run an untested model family and compare prediction error against actual metrics."
        risk = "Prediction misses must be logged, not hidden behind a better score."
    elif len(_seen_keys(traces)) >= 3 and state.budget_remaining is not None and state.budget_remaining <= 2:
        preferred = _choose(candidates, lambda c: c.model in {"verification_sweep", "top_model_verification"})
        hypothesis = "Remaining budget should consolidate evidence rather than chase another single split."
        evidence = "Run a multi-seed verifier sweep or top-model comparison."
        risk = "Score chasing near the end can create unsupported claims."
    else:
        tested = {trace.proposed_action.model for trace in traces}
        preferred = _choose(
            candidates,
            lambda c: c.model not in tested and c.model not in {"verification_sweep", "top_model_verification"},
        )
        hypothesis = "Testing a new model family can reveal whether the current best score depends on inductive bias."
        evidence = "Run a real single-model experiment from the safe catalog."
        risk = "The result is still an observation until robustness is checked."

    candidate_ids = [candidate.action_id or "" for candidate in candidates[:6]]
    if preferred.action_id and preferred.action_id not in candidate_ids:
        candidate_ids.insert(0, preferred.action_id)
    return AgentDecision(
        research_question=task.goal or f"Maximize {task.primary_metric} on {task.dataset} while avoiding unsupported claims.",
        working_hypothesis=hypothesis,
        candidate_action_ids=candidate_ids,
        preferred_action_id=preferred.action_id or "",
        rationale=(
            f"Given state {state.state_id} with budget_remaining={state.budget_remaining}, "
            f"the planner chooses {preferred.action_id} to collect the most decision-relevant evidence."
        ),
        expected_evidence_needed=evidence,
        claim_risk=risk,
        stop_or_continue="continue",
    )


def make_research_agent(
    planner_mode: str,
    agent_backend: str | None = None,
    agent_io_dir: str | Path = "agent_io",
    agent_timeout_seconds: int = 900,
    agent_poll_seconds: float = 2.0,
) -> ResearchAgent | None:
    if planner_mode == "llm":
        return LLMResearchAgent(backend=agent_backend or "openai_compatible")
    if planner_mode == "agent_handoff":
        return HandoffResearchAgent(
            backend=agent_backend or "generic_handoff",
            io_dir=ROOT / agent_io_dir,
            timeout_seconds=agent_timeout_seconds,
            poll_seconds=agent_poll_seconds,
        )
    if planner_mode == "agent_command":
        return CommandResearchAgent(
            backend=agent_backend or "generic_command",
            io_dir=ROOT / agent_io_dir,
            timeout_seconds=agent_timeout_seconds,
        )
    if planner_mode == "replay":
        return ReplayResearchAgent()
    if planner_mode == "selector":
        return None
    raise ValueError(f"unknown planner_mode: {planner_mode}")
