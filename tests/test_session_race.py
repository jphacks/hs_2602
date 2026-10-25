"""セッションの終了と次の開始が重なったときに、古いセッションの処理が新しいセッションを変更しないことのテスト。

実時間の待ちに頼らず、threading.Event で順序を固定する(古い処理を途中で止め、その間に新しいセッションを開始し、再開させる)。
時計は疑似時計(tests/sim.py)なので、エンジン内部の待ちも実時間を使わない。API は呼ばない。
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import facilitator.engine as engine_mod  # noqa: E402
import facilitator.report as report_mod  # noqa: E402
from config import load_settings  # noqa: E402
from facilitator.engine import Engine, SessionConfig  # noqa: E402
from facilitator.llm.mock_llm import MockLLM  # noqa: E402
from tests.sim import SimClock  # noqa: E402

TIMEOUT = 5  # スレッドが止まったままにならないための上限(通常はすぐ終わる)


def cfg(theme: str) -> SessionConfig:
    return SessionConfig(theme, 5, ["A", "B"], mode="text", intro=False, tts=False, discussion_type="problem_solving")


class Gate:
    """古い処理を途中で止めて、合図で再開させる。"""

    def __init__(self) -> None:
        self.entered, self.release = threading.Event(), threading.Event()

    def pause_here(self) -> None:
        self.entered.set()
        assert self.release.wait(TIMEOUT)


class FakeDevice:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True

    def clear(self) -> None:
        self.stopped = True


@pytest.fixture
def eng(monkeypatch, tmp_path):
    monkeypatch.setattr(engine_mod, "make_llm", lambda settings: MockLLM())
    e = Engine(load_settings(), clock=SimClock(), log_dir=tmp_path, tts_chars_per_sec=1000.0)
    yield e
    e.stop()


@pytest.fixture
def blocked_report(monkeypatch):
    """まとめログの書き出しを途中で止める(書き出しは止めた後に本来の処理を行う)。"""
    gate, written = Gate(), []
    orig = report_mod.TestReport.write

    def write(self, *a, **k):
        gate.pause_here()
        path = orig(self, *a, **k)
        written.append(path)
        return path

    monkeypatch.setattr(report_mod.TestReport, "write", write)
    return gate, written


def run_in_thread(fn) -> threading.Thread:
    t = threading.Thread(target=fn, daemon=True)
    t.start()
    return t


# ---------- 今回の不具合: 古い stop() が新しいセッションを finished にする ----------
def test_old_stop_does_not_finish_new_session(eng, blocked_report):
    gate, written = blocked_report
    eng.start(cfg("1回目"), background=False)
    old_state = eng.state
    t = run_in_thread(eng.stop)  # 1回目の終了処理(まとめログの書き出しで止まる)
    assert gate.entered.wait(TIMEOUT)
    eng.start(cfg("2回目"), background=False)  # その間に2回目を開始
    new_state = eng.state
    gate.release.set()  # 1回目の終了処理を再開・完了させる
    t.join(TIMEOUT)
    assert not t.is_alive()
    assert eng.status == "running" and not eng._stop.is_set()  # 2回目は実行中のまま
    assert eng.state is new_state and eng.cfg.theme == "2回目" and eng._ended_at is None
    assert old_state is not new_state and len(written) == 1 and written[0].exists()  # 1回目のまとめログは書き出される


def test_old_stop_does_not_stop_new_session_devices(eng, blocked_report):
    gate, _ = blocked_report
    eng.start(cfg("1回目"), background=False)
    old_mic, old_stt = FakeDevice(), FakeDevice()
    eng._mic, eng._stt = old_mic, old_stt
    t = run_in_thread(eng.stop)
    assert gate.entered.wait(TIMEOUT)
    eng.start(cfg("2回目"), background=False)
    new_mic, new_stt = FakeDevice(), FakeDevice()
    eng._mic, eng._stt = new_mic, new_stt  # 2回目の機器(音声モード相当)
    gate.release.set()
    t.join(TIMEOUT)
    assert old_mic.stopped and old_stt.stopped  # 1回目の機器は止める
    assert not new_mic.stopped and not new_stt.stopped and eng._mic is new_mic  # 2回目の機器には触れない


# ---------- 終了の経路(close_gd)と、古い判断・ループ ----------
def test_close_path_does_not_finish_new_session(eng, blocked_report):
    """時間切れの終了(close_gd)の途中で次のセッションが始まっても、古い終了処理は新しいセッションを変えない。"""
    gate, _ = blocked_report
    eng.start(cfg("1回目"), background=False)
    eng._run_tick(eng._agent_step())  # 開始の説明
    eng.clock.sleep(2.0)  # 開始の説明の直後(発話中の扱い)を過ぎる
    eng.state.duration_sec = eng.state.elapsed(eng.now())  # 時間切れ
    trig = eng._agent_step()
    assert trig == "time_up"
    t = run_in_thread(lambda: eng._run_tick(trig))  # 終了の発話 → stop() のまとめログで止まる
    assert gate.entered.wait(TIMEOUT)
    eng.start(cfg("2回目"), background=False)
    gate.release.set()
    t.join(TIMEOUT)
    assert eng.status == "running" and eng.cfg.theme == "2回目" and not eng._stop.is_set()
    assert eng._run_tick(eng._agent_step()) is None and eng.state.started  # 2回目も開始の説明ができる


class BlockingLLM(MockLLM):
    """1回目の呼び出しを止める(LLM の応答待ちの間に次のセッションが始まる状況)。"""

    def __init__(self, gate: Gate) -> None:
        self.gate, self.calls = gate, 0

    def generate_json(self, system, user, schema, context):
        self.calls += 1
        if self.calls == 1:
            self.gate.pause_here()
        out = super().generate_json(system, user, schema, context)
        if len(context["allowed_actions"]) > 1:
            out["decision"].update(action="ask_question", speech="原因はどう整理できそうですか?")
        return out


def test_stale_decision_from_old_session_is_dropped(monkeypatch, tmp_path):
    gate = Gate()
    llms = iter([BlockingLLM(gate), MockLLM()])
    monkeypatch.setattr(engine_mod, "make_llm", lambda settings: next(llms))
    eng = Engine(load_settings(), clock=SimClock(), log_dir=tmp_path, tts_chars_per_sec=1000.0, cooldown_sec=0.0, analysis_interval_sec=0.5)
    try:
        eng.start(cfg("1回目"), background=False)
        eng._run_tick(eng._agent_step())  # 開始の説明(LLM なし)
        eng.add_text("speaker_1", "原因は昼の集中です。")
        eng.clock.sleep(3.0)
        trig = eng._agent_step()
        assert trig
        t = run_in_thread(lambda: eng._run_tick(trig))  # 1回目の判断(LLM の応答待ちで止まる)
        assert gate.entered.wait(TIMEOUT)
        eng.start(cfg("2回目"), background=False)
        eng._agent_busy = True  # 2回目の判断が進行中という状況
        gate.release.set()
        t.join(TIMEOUT)
        assert not eng.state.interventions and not any(u.source == "ai" for u in eng.state.utterances)  # 古い判断は話さない
        assert eng._agent_busy  # 古い判断の後始末は、2回目の判断中の印を消さない
        assert eng.status == "running"
    finally:
        eng.stop()


def test_old_agent_loop_exits_after_new_session_starts(eng):
    eng.start(cfg("1回目"), background=False)
    gen = eng._gen
    t = run_in_thread(lambda: eng._agent_loop(gen))  # 1回目の判断ループ(疑似時計なのですぐ回る)
    eng.start(cfg("2回目"), background=False)
    t.join(TIMEOUT)
    assert not t.is_alive()  # 古いループは止まる(以前は _stop が差し替わって止まらなかった)
    assert eng.status == "running"


# ---------- 通常の動作 ----------
def test_stop_twice_writes_report_once(eng, monkeypatch):
    written = []
    orig = report_mod.TestReport.write
    monkeypatch.setattr(report_mod.TestReport, "write", lambda self, *a, **k: written.append(orig(self, *a, **k)) or written[-1])
    eng.start(cfg("1回目"), background=False)
    eng.stop()
    eng.stop()
    assert eng.status == "finished" and len(written) == 1 and written[0].exists()


def test_consecutive_sessions_without_overlap(eng):
    for theme in ("1回目", "2回目", "3回目"):
        eng.start(cfg(theme), background=False)
        assert eng._run_tick(eng._agent_step()) is None and eng.state.started
        assert eng.status == "running" and eng.cfg.theme == theme
        eng.stop()
        assert eng.status == "finished"


# ---------- 発話の排他を待つ間に次のセッションが始まる ----------
class SignalLock:
    """発話の排他。別スレッドが待ち始めたら合図する(テスト側は内部のロックを直接持って塞ぐ)。"""

    def __init__(self) -> None:
        self.inner, self.waiting = threading.RLock(), threading.Event()

    def __enter__(self):
        self.waiting.set()
        self.inner.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.inner.release()


class FakeTTS:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text: str) -> float:
        self.spoken.append(text)
        return 0.0

    def stop(self) -> None:
        pass


def test_old_decision_waiting_for_speak_lock_does_not_speak_in_new_session(eng):
    """世代の確認を通った後、別の AI の発話(排他)を待つ間に次のセッションが始まっても、古い判断は話さない。"""
    eng.start(cfg("1回目"), background=False)
    lock = eng._speak_lock = SignalLock()
    lock.inner.acquire()  # 別の AI が話している状況
    trig = eng._agent_step()  # 開始の説明(強制イベント)
    t = run_in_thread(lambda: eng._run_tick(trig))
    assert lock.waiting.wait(TIMEOUT)
    eng.start(cfg("2回目"), background=False)
    tts = eng._tts = FakeTTS()  # 2回目の声
    lock.inner.release()
    t.join(TIMEOUT)
    assert not t.is_alive()
    assert not tts.spoken and not eng.state.interventions and not eng.state.utterances  # 2回目には何も残さない
    assert not eng.state.started and eng.status == "running"
    assert eng._run_tick(eng._agent_step()) is None and eng.state.started  # 2回目は自分の開始の説明ができる


def test_old_bot_speech_waiting_for_speak_lock_is_dropped(eng):
    eng.start(cfg("1回目"), background=False)
    old_gen = eng._gen
    lock = eng._speak_lock = SignalLock()
    lock.inner.acquire()
    result = []
    t = run_in_thread(lambda: result.append(eng._say("古い発言です。", speaker="speaker_3", gen=old_gen)))
    assert lock.waiting.wait(TIMEOUT)
    eng.start(cfg("2回目"), background=False)
    tts = eng._bot_tts["speaker_3"] = FakeTTS()
    lock.inner.release()
    t.join(TIMEOUT)
    assert result == [None] and not tts.spoken and not eng.state.utterances and not eng.ai_speaking


def test_say_with_current_gen_still_speaks(eng):
    eng.start(cfg("1回目"), background=False)
    tts = eng._tts = FakeTTS()
    utt = eng._say("では始めましょう。", gen=eng._gen)
    assert utt is not None and tts.spoken == ["では始めましょう。"] and eng.state.utterances[-1] is utt
