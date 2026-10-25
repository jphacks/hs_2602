"""エンジン(実時間ループ)の回帰テスト。モック LLM・テキストモード・TTS なしで、API は呼ばない。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings  # noqa: E402
from facilitator.engine import Engine, SessionConfig  # noqa: E402


def _wait(cond, timeout: float) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


def _session(eng: Engine, minutes: float) -> None:
    eng.start(SessionConfig("テーマ", minutes, ["A", "B"], mode="text", intro=False, tts=False, discussion_type="problem_solving"))


def test_consecutive_sessions_on_same_engine():
    """前のセッションの「AI 発話後のミュート時刻」が、新しいセッションの時計に持ち越されないこと。"""
    eng = Engine(
        load_settings(),
        llm_provider="mock",
        tts_chars_per_sec=1000.0,  # 発話の待ち時間を短くする
        cooldown_sec=0.0,
        analysis_interval_sec=0.5,
    )
    try:
        # 1回目: 3秒のGDを開始から終了(close_gd)まで走らせる
        _session(eng, 0.05)
        assert _wait(lambda: eng.status == "finished", 10), "1回目のセッションが終了しない"
        assert eng._ai_mute_until > 1.0  # 終了の発話で、1回目の時計でのミュート時刻が残っている

        # 2回目: 同じエンジンで新しいセッションを開始すると、すぐに AI が開始の説明をする
        _session(eng, 1)
        assert _wait(lambda: eng.state.started, 1.5), "2回目のセッションで AI が開始しない"
        # 発言を入れると、ルール層を通って LLM(モック)の判断が行われる
        eng.add_text("speaker_1", "まず原因を整理しましょう。")
        assert _wait(lambda: any(r.gate == "llm" for r in eng.agent.history), 5), "2回目のセッションで LLM の判断が行われない"
    finally:
        eng.stop()
