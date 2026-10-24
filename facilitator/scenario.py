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
    expect_phase: list[str] | None = None
    phase_ok: bool | None = None  # フェーズ認識の評価(行動の評価とは別)。LLM が判断していない/期待なしなら None


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


def judge_phase(cp: dict[str, Any], rec: TickRecord) -> bool | None:
    """LLM がこのチェックポイントで判断した実際のフェーズが、期待(expect_phase)のいずれかか。"""
    if not cp.get("expect_phase") or rec.llm_raw_action is None:
        return None
    return rec.observed_phase in cp["expect_phase"]


def run_scenario(
    sc: dict[str, Any], llm: LLMClient, settings: Settings, logger: TickLogger | None = None
) -> ScenarioResult:
    state = DiscussionState(settings=settings, theme=sc["theme"], duration_sec=sc["duration_sec"])
    for sid, name in (sc.get("participants") or {}).items():
        state.set_speaker_name(sid, name, confirmed=sc.get("names_confirmed", True))
    agent = FacilitatorAgent(state, llm, settings, logger=logger)
    if sc.get("start_elapsed", 0) > 0:
        state.started = True
        state.start_time = 0.0
        # 途中から始めるシナリオは、それまでの LLM 判断を initial_phase として与える(省略時は未判断)
        state.observed_phase = sc.get("initial_phase", "")
        state.phase_observed_at = 0.0 if state.observed_phase else None
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
        rec = agent.tick(now, trigger="checkpoint")
        ok, reason = judge(cp, rec)
        result.checks.append(
            CheckResult(
                now, cp["expect"], cp.get("forbid", []), cp.get("target_any"), cp.get("why", ""), rec, ok, reason,
                cp.get("expect_phase"), judge_phase(cp, rec),
            )
        )
    return result
