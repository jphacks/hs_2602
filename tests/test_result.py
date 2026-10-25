"""結果画面のデータ: 全部の発言と、参加者ごとの発言量。"""

from __future__ import annotations

import facilitator.engine as engine_mod
from config import load_settings
from facilitator.engine import Engine, SessionConfig
from facilitator.llm.mock_llm import MockLLM
from tests.sim import SimClock


def test_result_has_full_log_and_amounts(monkeypatch, tmp_path):
    monkeypatch.setattr(engine_mod, "make_llm", lambda settings: MockLLM())
    eng = Engine(load_settings(), clock=SimClock(), log_dir=tmp_path, tts_chars_per_sec=1000.0)
    try:
        eng.start(SessionConfig("テーマ", 5, ["たろう", "はなこ"], mode="text", intro=False, tts=False, discussion_type="problem_solving"), background=False)
        eng._run_tick(eng._agent_step())  # 開始の説明
        for i in range(130):  # 画面に送る直近の発言(120件)より多い
            eng.clock.sleep(1.0)
            eng.add_text("speaker_1" if i % 3 else "speaker_2", f"発言{i}です")
        eng.stop()
        r = eng.result()
        assert r["ok"] and r["theme"] == "テーマ"
        assert len(r["log"]) == 131 and r["log"][0]["who"] == "AI ファシリテーター"
        people = {p["name"]: p for p in r["people"]}
        assert people["たろう"]["turns"] == 86 and people["はなこ"]["turns"] == 44
        assert people["たろう"]["sec"] > people["はなこ"]["sec"] > 0
        assert r["facilitator"]["turns"] == 1
    finally:
        eng.stop()


def test_self_introduction_is_not_counted_in_result(monkeypatch, tmp_path):
    monkeypatch.setattr(engine_mod, "make_llm", lambda settings: MockLLM())
    eng = Engine(load_settings(), clock=SimClock(), log_dir=tmp_path, tts_chars_per_sec=1000.0)
    try:
        eng.start(SessionConfig("テーマ", 5, ["たろう", "はなこ"], mode="text", intro=False, tts=False, discussion_type="problem_solving"), background=False)
        st = eng.state
        st.intro_ids.add(st.add_utterance("speaker_1", "たろうです。よろしくお願いします。", eng.now() - 9, eng.now() - 5, 1.0).id)
        eng._run_tick(eng._agent_step())  # 開始の説明
        eng.clock.sleep(2.0)
        eng.add_text("speaker_2", "本題の発言です")
        eng.stop()
        r = eng.result()
        people = {p["name"]: p for p in r["people"]}
        assert people["たろう"]["turns"] == 0 and people["はなこ"]["turns"] == 1  # 自己紹介は数えない
        assert any(u["intro"] for u in r["log"])  # ログには残る
    finally:
        eng.stop()
