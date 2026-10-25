"""AI 参加者にも、進行役と同じ議論タイプの流れ(いまの段階)を伝える。"""

from __future__ import annotations

import facilitator.engine as engine_mod
from config import load_settings
from facilitator.engine import Engine, SessionConfig
from facilitator.llm.mock_llm import MockLLM
from facilitator.participants import Persona, discussion_flow, generate_utterance
from tests.sim import SimClock


class CaptureLLM:
    name = "capture"

    def __init__(self) -> None:
        self.users: list[str] = []

    def generate_json(self, system, user, schema, context):
        self.users.append(user)
        return {"text": "原因は昼休みの時間が全員同じことだと思います。"}


def start(monkeypatch, tmp_path) -> Engine:
    monkeypatch.setattr(engine_mod, "make_llm", lambda settings: MockLLM())
    eng = Engine(load_settings(), clock=SimClock(), log_dir=tmp_path, tts_chars_per_sec=1000.0)
    eng.start(SessionConfig("学食の混雑", 10, ["たろう"], mode="text", intro=False, tts=False,
                            discussion_type="problem_solving", ai_count=1), background=False)
    eng._run_tick(eng._agent_step())  # 開始の説明
    return eng


def test_flow_tells_current_and_target_phase(monkeypatch, tmp_path):
    eng = start(monkeypatch, tmp_path)
    try:
        flow = discussion_flow(eng.agent, eng.now())
        assert "課題解決型" in flow and "課題整理 → 原因分析" in flow
        assert "いまの段階: 「課題整理」" in flow
        eng.state.observed_phase = "cause_analysis"
        eng.clock.sleep(10 * 60 * 0.7)  # 時間の目安では「評価・比較」の頃
        flow = discussion_flow(eng.agent, eng.now())
        assert "いまの段階: 「原因分析」" in flow and "時間の目安では" in flow
        assert "そろそろ解決策の検討に移りませんか" in flow  # 遅れているときは、段階を飛ばさず今の次を提案する
        assert "仕切ったり" in flow  # 進行役の役目は取らない
    finally:
        eng.stop()


def test_flow_reaches_bot_prompt():
    llm = CaptureLLM()
    p = Persona("speaker_2", "アオイ", "コスト重視", "お金をかけずに", "落ち着いて話す")
    generate_utterance(llm, p, "job", "学食の混雑", ["たろう: 混んでいます"], "間があいた", 2, flow="# 議論の流れ(議論タイプ: 課題解決型)\n")
    assert "# 議論の流れ(議論タイプ: 課題解決型)" in llm.users[0]
