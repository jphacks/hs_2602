"""雑談モードの AI 司会(facilitator/chat/host.py)の分離・接続のテスト(第1段階)。API は呼ばない。

雑談の司会が雑談型のときだけ使われ、GD・GW では従来の進行役がそのまま使われることを確かめる
(プロンプト・行動・開始の話題は第3段階で雑談専用にした。tests/test_chat_actions.py)。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.server as server  # noqa: E402
from config import load_settings  # noqa: E402
from facilitator.agent.agent import FacilitatorAgent  # noqa: E402
from facilitator.chat.host import ChatHostAgent, agent_class_for  # noqa: E402
from facilitator.discussion_types import CASUAL_FREE_THEME, DISCUSSION_TYPES  # noqa: E402
from facilitator.engine import Engine  # noqa: E402
from facilitator.models import Action  # noqa: E402
from facilitator.policy import knob_defaults, preset_policy  # noqa: E402
from facilitator.scenario import run_scenario  # noqa: E402
from facilitator.llm.mock_llm import MockLLM  # noqa: E402
from facilitator.state import DiscussionState  # noqa: E402
from tests.test_content import StubLLM  # noqa: E402


# ---------- 司会のクラスの選び方 ----------
def test_only_casual_uses_chat_host():
    assert agent_class_for("casual") is ChatHostAgent
    for key in [k for k in DISCUSSION_TYPES if k != "casual"] + [None, ""]:
        assert agent_class_for(key) is FacilitatorAgent, key  # GD・GW は従来の進行役そのもの(継承したクラスでもない)


def test_chat_host_keeps_engine_interface():
    """engine が使う窓口(判断・トリガー・確定・生成しながら話す処理の検証・依頼・記録)をそのまま持つ。"""
    assert issubclass(ChatHostAgent, FacilitatorAgent)
    for name in ("tick", "should_tick", "commit", "commit_spoken", "record_spoken", "cancel", "live_ok", "live_sentence",
                 "has_pending_request", "phase_status"):
        assert callable(getattr(ChatHostAgent, name)), name


# ---------- 第1段階: 振る舞いは従来と同じ ----------
def _pair(theme: str = "休日の過ごし方"):
    """同じ状態・同じ設定(画面が雑談で送る GW用の初期値)で、雑談の司会と従来の進行役を作る。"""
    out = []
    for cls in (ChatHostAgent, FacilitatorAgent):
        s = load_settings()
        st = DiscussionState(settings=s, theme=theme, duration_sec=600)
        st.started, st.start_time = True, 0.0
        for sid, n in {"speaker_1": "たろう", "speaker_2": "はなこ"}.items():
            st.set_speaker_name(sid, n)
        st.add_utterance("speaker_1", "休みの日は寝てばかりです。", 100, 104, 0.9)
        out.append(cls(st, StubLLM("ask_question", "皆さんはどうですか?", health="stalled"), s, purpose="general",
                       discussion_type="casual", policy=preset_policy("general_gw")))
    return out


def test_chat_host_has_own_prompt_and_schema():
    """第3段階: プロンプトと出力の選択肢は雑談専用(GD・GW の進行役のものは変えない)。"""
    chat, old = _pair()
    assert chat.system_prompt != old.system_prompt and chat.output_schema != old.output_schema


@pytest.mark.parametrize("theme", ["休日の過ごし方", CASUAL_FREE_THEME])
def test_time_warning_and_close_templates_are_same_as_before(theme):
    """時間告知・終了の定型文は雑談型のもの(結論を求めない)のまま。開始だけ雑談の司会のあいさつと話題にした。"""
    chat, old = _pair(theme)
    for a in (Action.time_warning, Action.close_gd):
        assert chat._template(a, 300.0) == old._template(a, 300.0)
    assert chat._template(Action.start_gd, 0.0) != old._template(Action.start_gd, 0.0)


# ---------- 接続: 画面が雑談モードで送る内容で開始すると、雑談の司会で動く ----------
@pytest.fixture
def engine(monkeypatch, tmp_path):
    eng = Engine(load_settings(), llm_provider="mock", tts_chars_per_sec=1000.0, log_dir=tmp_path)
    monkeypatch.setattr(server, "engine", eng)
    yield eng
    eng.stop()


BASE = dict(minutes=5, names=["A", "B"], mode="text", intro=False, tts=False, preset="A", custom={})


def test_chat_mode_payload_starts_chat_host(engine):
    """画面の「雑談」モードが送る形: purpose general・議論タイプ雑談型・AI 参加者なし・就活以外の初期値・テーマは空でもよい。"""
    r = server.start(server.StartReq(theme="", purpose="general", discussion_type="casual", ai_count=0,
                                     knobs=knob_defaults()["general"], **BASE))
    assert r["ok"]
    assert type(engine.agent) is ChatHostAgent and engine.state.theme == CASUAL_FREE_THEME
    snap = engine.snapshot()
    assert snap["discussion_type_key"] == "casual" and snap["status"] == "running"


@pytest.mark.parametrize("purpose,dtype", [("job", "problem_solving"), ("job", "planning"), ("general", None), ("general", "idea_generation")])
def test_gd_gw_sessions_keep_facilitator_agent(engine, purpose, dtype):
    extra = {"discussion_type": dtype} if dtype else {}
    knobs = knob_defaults()[purpose]
    assert server.start(server.StartReq(theme="テーマ", purpose=purpose, knobs=knobs, **BASE, **extra))["ok"]
    assert type(engine.agent) is FacilitatorAgent


def test_scenario_runner_uses_chat_host_for_casual(monkeypatch):
    import facilitator.scenario as scenario_mod

    used = []
    monkeypatch.setattr(scenario_mod, "agent_class_for", lambda key: used.append(agent_class_for(key)) or used[-1])
    sc = {"id": "chat", "title": "雑談", "theme": "休日の過ごし方", "purpose": "general", "discussion_type": "casual",
          "duration_sec": 300, "start_elapsed": 60, "initial_phase": "warm_up",
          "participants": {"speaker_1": "たろう", "speaker_2": "はなこ"},
          "utterances": [{"t": 60, "d": 4, "s": "speaker_1", "text": "休みの日は寝てばかりです。"}],
          "checkpoints": [{"at": 80, "expect": ["no_intervention", "ask_question", "invite_all"], "require_llm": True}]}
    res = run_scenario(sc, MockLLM(), load_settings(llm_provider="mock"), policy=preset_policy("general_gw"))
    assert used == [ChatHostAgent] and res.checks and res.checks[0].record.llm_called
    run_scenario({**sc, "discussion_type": "idea_generation", "initial_phase": "divergence"}, MockLLM(),
                 load_settings(llm_provider="mock"), policy=preset_policy("general_gw"))
    assert used[-1] is FacilitatorAgent  # 雑談型以外は従来の進行役
