"""模擬GDシナリオ(YAML)を仮想時計で再生し、チェックポイントでエージェントに判断させる。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from config import Settings
from facilitator.agent.agent import FacilitatorAgent
from facilitator.llm.base import LLMClient
from facilitator.logging import TickLogger
from facilitator.models import TickRecord
from facilitator.state import DiscussionState


@dataclass
class CheckResult:
    at: float
    expect: list[str]
    forbid: list[str]
    target_any: list[str] | None
    why: str
    record: TickRecord
    passed: bool
    fail_reason: str = ""


@dataclass
class ScenarioResult:
    id: str
    title: str
    checks: list[CheckResult] = field(default_factory=list)


def load_scenario(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def judge(cp: dict[str, Any], rec: TickRecord) -> tuple[bool, str]:
    action = rec.decision.action.value
    if action in cp.get("forbid", []):
        return False, f"禁止アクション {action}"
    if action not in cp["expect"]:
        return False, f"期待外 {action}"
    targets = cp.get("target_any")
    if action == "invite_person" and targets and rec.decision.target_speaker not in targets:
        return False, f"指名対象が期待外 {rec.decision.target_speaker}"
    return True, ""


def run_scenario(
    sc: dict[str, Any], llm: LLMClient, settings: Settings, logger: TickLogger | None = None, section: Any = None
) -> ScenarioResult:
    """section(report.Section)を渡すと、判断・発言・評価結果をまとめログ用に記録する。"""
    state = DiscussionState(settings=settings, theme=sc["theme"], duration_sec=sc["duration_sec"])
    for sid, name in (sc.get("participants") or {}).items():
        state.set_speaker_name(sid, name, confirmed=sc.get("names_confirmed", True))
    if sc.get("start_elapsed", 0) > 0:
        state.started = True
        state.start_time = 0.0
        state.phase = sc.get("initial_phase", "ideation")

    agent = FacilitatorAgent(state, llm, settings, logger=logger)
    if section is not None:
        section.state, section.info = state, {"テーマ": sc["theme"]}
    pending = sorted(sc["utterances"], key=lambda u: u["t"] + u["d"])
    result = ScenarioResult(sc["id"], sc["title"])

    for cp in sorted(sc["checkpoints"], key=lambda c: c["at"]):
        now = float(cp["at"])
        # STT の確定結果は発話終了時に届く
        while pending and pending[0]["t"] + pending[0]["d"] <= now:
            u = pending.pop(0)
            state.add_utterance(u["s"], u["text"], u["t"], u["t"] + u["d"], u.get("c", 1.0))
        # VAD: 発話中の人がいるか
        state.speaking_now = any(u["t"] <= now < u["t"] + u["d"] for u in pending)
        m = llm.mark() if hasattr(llm, "mark") else 0
        rec = agent.tick(now, trigger="checkpoint")
        ok, reason = judge(cp, rec)
        if section is not None:
            section.decision(rec, llm.since(m) if hasattr(llm, "since") else None)
            d = rec.decision
            section.checks.append(
                {
                    "at": now,
                    "expect": cp["expect"],
                    "result": d.action.value + (f"({d.target_speaker})" if d.target_speaker else ""),
                    "passed": ok,
                    "fail_reason": reason,
                    "why": cp.get("why", ""),
                }
            )
        result.checks.append(
            CheckResult(now, cp["expect"], cp.get("forbid", []), cp.get("target_any"), cp.get("why", ""), rec, ok, reason)
        )
    return result
