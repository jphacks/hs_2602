"""画面・API と介入ポリシーのテスト(プリセットと変更した項目、画面の「詳細」との互換、セッションへの反映、依頼待ちの解除)。
API は呼ばない(LLM はモック)。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.server as server  # noqa: E402
from config import POLICY_PRESETS, ROOT, STYLE_KNOBS, load_settings  # noqa: E402
from facilitator.agent.agent import FacilitatorAgent  # noqa: E402
from facilitator.agent.prompts import build_system_prompt  # noqa: E402
from facilitator.engine import Engine  # noqa: E402
from facilitator.llm.base import LLMError  # noqa: E402
from facilitator.models import Action  # noqa: E402
from facilitator.policy import (  # noqa: E402
    ITEM_LABELS,
    JOB_CONTENT_NOTICE,
    InterventionPolicy,
    Level,
    preset_policy,
    request_policy,
    ui_spec,
)
from facilitator.state import DiscussionState  # noqa: E402

HTML = ROOT / "app" / "static" / "index.html"


# ---------- 画面に渡す定義(値を画面に持たない) ----------
def test_ui_spec_matches_policy_definition():
    spec = ui_spec()
    assert [it["key"] for it in spec["items"]] == list(ITEM_LABELS)
    for it in spec["items"]:
        for o in it["options"]:  # 選択肢はどれもポリシーの有効な値
            InterventionPolicy.model_validate({it["key"]: o["v"], **({"content_scope": "hint"} if it["key"] == "content_initiative" else {})})
        if it["help"]:
            assert set(it["help"]) == {o["v"] for o in it["options"]}  # ⑥は全選択肢に説明がある
    assert {k: v["values"] for k, v in spec["presets"].items()} == POLICY_PRESETS
    assert spec["purpose_preset"] == {"job": "job_gd", "general": "general_gw"}
    assert spec["notices"]["job_content"] == JOB_CONTENT_NOTICE


def test_content_help_matches_actual_behavior():
    """「具体案まで + 積極的」の説明は、自発的な具体案に制限がある実際の動作と一致させる。"""
    help_ = ui_spec()["items"][-1]["help"]
    assert "具体案は頼まれたときか停滞時の段階を踏んだときだけ" in help_["proactive"]
    assert "質問 → ヒント → 具体案" in help_["stalled"]
    assert "頼まれたとき" in help_["none"]


# ---------- 画面(第4段階の前の UI に戻した)と API の互換 ----------
def test_ui_sends_legacy_knobs_that_api_accepts():
    """現在の画面(owaki さんの UI)は preset "A" と knobs を送る(スタイルが「カスタム」のときだけ「詳細」の5項目、
    それ以外は空)。項目の名前は /api/info の knobs から表示する。"""
    html = HTML.read_text(encoding="utf-8")
    assert "knobs:" in html and 'preset: "A"' in html and "ui.info?.knobs" in html
    assert "policy-items" not in html and "policyPayload" not in html  # 第4段階の「介入の設定」は撤去
    assert set(server.info()["knobs"]) == set(STYLE_KNOBS)


# ---------- API: 入力からポリシーを決める ----------
def test_request_policy_priority_and_validation():
    p, notes = request_policy("general", None, {"organize": "low"})
    assert p == preset_policy("general_gw").updated(organize="low") and notes == []
    p, notes = request_policy("job", "general_gw")  # 就活で GW用プリセット → 中立でない旨
    assert p == preset_policy("general_gw") and notes == [JOB_CONTENT_NOTICE]
    p, _ = request_policy("general", None, {"content_scope": "none"})
    assert p.content_initiative.value == "none"
    for bad in [("job", "nope", None), ("job", None, {"stall": "max"}), ("job", None, {"cooldown_sec": "30"})]:
        with pytest.raises(ValueError):
            request_policy(*bad)


def test_request_policy_legacy_inputs():
    p, notes = request_policy("job", None, None, "B")
    assert p.stall == Level.high and "旧進行スタイル B" in notes[0]
    p, notes = request_policy("job", None, None, "custom")
    assert p == InterventionPolicy() and "custom" in notes[0]
    p, _ = request_policy("general", None, {"organize": "low"}, "A", {"organize": 2, "suggest": 0})
    assert p.organize == Level.high and p.content_scope.value == "none"  # 旧「詳細」は最後に重ねる(旧画面の互換)
    p, _ = request_policy("job", None, None, "A", {"suggest": 2})
    assert p == InterventionPolicy()  # 旧画面と同じく、就活の提案度合いは標準まで


# ---------- API: セッションへの反映 ----------
@pytest.fixture
def engine(monkeypatch, tmp_path):
    eng = Engine(load_settings(), llm_provider="mock", tts_chars_per_sec=1000.0, log_dir=tmp_path)
    monkeypatch.setattr(server, "engine", eng)
    yield eng
    eng.stop()


BASE = dict(theme="テーマ", minutes=5, names=["A", "B"], mode="text", intro=False, tts=False)


def test_api_start_applies_preset_and_changed_items(engine):
    r = server.start(server.StartReq(purpose="general", policy_preset="general_gw", policy={"organize": "low"}, **BASE))
    assert r["ok"] and r["notices"] == []
    p = engine.policy
    assert p.organize == Level.low and p.turns == Level.high and p.stall == Level.high  # 未変更の項目はプリセットの値
    assert engine.agent._param("quiet_member_sec") == 60.0 and engine.agent._param("silence_trigger_sec") == 8.0
    assert engine.agent.system_prompt == build_system_prompt(
        "general", "generic", levels={"progress": "standard", "drift": "standard", "organize": "low", "turns": "high", "stall": "high"},
        content=("hint", "stalled"),
    )
    snap = engine.snapshot()
    assert snap["policy_preset"] == "general_gw" and snap["policy_overrides"] == {"organize": "low"}
    assert snap["policy"]["organize"] == "low" and snap["policy_notices"] == []


def test_api_start_job_with_content_shows_notice(engine):
    r = server.start(server.StartReq(discussion_type="planning", policy={"content_scope": "hint"}, **BASE))
    assert r["ok"] and r["notices"] == [JOB_CONTENT_NOTICE]
    assert engine.snapshot()["policy_notices"] == [JOB_CONTENT_NOTICE]
    assert engine._sec.info["注意"] == JOB_CONTENT_NOTICE
    assert "offer_hint" in engine.agent.output_schema["properties"]["decision"]["properties"]["action"]["enum"]


def test_api_start_rejects_invalid_policy(engine):
    r = server.start(server.StartReq(discussion_type="planning", policy={"stall": "max"}, **BASE))
    assert not r["ok"] and "設定の値" in r["error"]
    r = server.start(server.StartReq(discussion_type="planning", policy_preset="other", **BASE))
    assert not r["ok"] and "プリセット" in r["error"]


def test_api_start_job_default_is_unchanged_gd(engine):
    """就活GDの既定(新しい画面からも、旧画面の knobs からも)は、従来の GD 標準のまま。"""
    for extra in ({}, {"policy_preset": "job_gd"}, {"knobs": dict.fromkeys(["phase", "redirect", "organize", "turns", "suggest"], 1)}):
        assert server.start(server.StartReq(discussion_type="problem_solving", **BASE, **extra))["ok"]
        assert engine.policy == InterventionPolicy()
        assert engine.agent.system_prompt == build_system_prompt("job", "problem_solving")
        d = load_settings()
        assert (engine.settings.cooldown_sec, engine.settings.analysis_interval_sec, engine.settings.min_silence_for_intervention_sec) == (
            d.cooldown_sec, d.analysis_interval_sec, d.min_silence_for_intervention_sec
        )


def test_api_legacy_style_b_no_longer_changes_numbers(engine):
    r = server.start(server.StartReq(discussion_type="planning", preset="B", **BASE))
    assert r["ok"] and "旧進行スタイル B" in r["notices"][0]
    assert engine.settings.cooldown_sec == 45.0 and engine.settings.min_silence_for_intervention_sec == 2.0
    assert engine.agent._param("silence_trigger_sec") == 8.0  # ⑤ 積極的として扱う


def test_api_info_serves_knob_labels_and_policy_spec():
    info = server.info()
    assert info["knobs"] == {k: v["label"] for k, v in STYLE_KNOBS.items()}  # 画面の「詳細」
    assert info["policy"] == ui_spec() and "presets" not in info  # API の利用元向け(旧進行スタイルの数値は廃止)


@pytest.mark.parametrize(
    "purpose,knobs,check",
    [
        ("job", dict.fromkeys(["phase", "redirect", "organize", "turns", "suggest"], 1), lambda p: p == InterventionPolicy()),
        ("general", dict.fromkeys(["phase", "redirect", "organize", "turns", "suggest"], 1),
         lambda p: (p.organize, p.turns, p.stall, p.content_scope.value) == (Level.standard, Level.standard, Level.high, "hint")),
        ("general", {"phase": 1, "redirect": 1, "organize": 2, "turns": 1, "suggest": 0},
         lambda p: p.organize == Level.high and p.content_scope.value == "none"),
        ("job", {"phase": 1, "redirect": 1, "organize": 1, "turns": 1, "suggest": 2}, lambda p: p == InterventionPolicy()),
        ("job", {}, lambda p: p == InterventionPolicy()),  # 「就活」は詳細を送らない → GD用プリセットのまま
        ("general", {}, lambda p: p == preset_policy("general_gw")),  # 「就活以外」は詳細を送らない → GW用プリセットのまま
    ],
)
def test_api_start_with_current_ui_payload(engine, purpose, knobs, check):
    """現在の画面が送る形(preset "A"・custom {}・knobs)で開始すると、介入ポリシーに変換されてセッションに反映される。"""
    extra = {} if purpose == "general" else {"discussion_type": "planning"}
    r = server.start(server.StartReq(purpose=purpose, preset="A", custom={}, knobs=knobs, **BASE, **extra))
    assert r["ok"] and check(engine.policy)


# ---------- 依頼待ちの解除(AI 参加者が待ち続けない) ----------
def _agent(llm):
    s = load_settings()
    st = DiscussionState(settings=s, theme="次の学園祭の企画を決めよう", duration_sec=600)
    st.started, st.start_time = True, 0.0
    st.set_speaker_name("speaker_1", "たろう")
    st.add_utterance("speaker_1", "AIさん、何かヒントもらえますか？", 100, 103, 0.9)
    return st, FacilitatorAgent(st, llm, s, purpose="general", policy=InterventionPolicy(content_scope="hint"))


class FailingLLM:
    name = "fail"

    def generate_json(self, *a):
        raise LLMError("boom")


def test_pending_request_cleared_after_llm_error():
    st, agent = _agent(FailingLLM())
    rec = agent.tick(105.5, "request")
    assert rec.overridden.startswith("LLM失敗") and not agent.has_pending_request(105.6)


def test_pending_request_expires_even_if_people_keep_talking():
    """参加者が話し続けて判断が始まらなくても、30秒で待ちが解ける(AI 参加者の待ちも解ける)。"""
    from tests.test_content import StubLLM

    st, agent = _agent(StubLLM())
    agent._scan_requests(104.0)
    assert agent.has_pending_request(104.0)
    for t in range(104, 140, 3):  # 間をあけずに話し続ける
        st.add_utterance("speaker_1", "それでですね", t, t + 3, 0.9)
        assert agent.should_tick(t + 3.2) is None  # 2秒の間がないので判断しない
    assert not agent.has_pending_request(140.0)


def test_engine_bots_wait_only_while_request_pending(engine, monkeypatch):
    import facilitator.engine as engine_mod

    assert server.start(server.StartReq(purpose="general", **BASE))["ok"]
    calls = []
    monkeypatch.setattr(engine_mod, "pick_bot", lambda *a: calls.append(1) or None)
    engine.state.add_utterance("speaker_1", "AIさん、何かヒントもらえますか？", 1.0, 3.0, 1.0)
    engine.agent._scan_requests(3.5)
    assert engine._bot_pick(3.5) is None and not calls  # 依頼の間は AI 参加者を選ばない
    engine._bot_pick(40.0)  # 30秒たてば待ちは解ける
    assert calls == [1]


def test_forced_event_also_clears_pending_request():
    """時間告知などで LLM を呼んだ場合も、依頼の候補は1回の判断で見送る(待ち続けない)。"""
    from tests.test_content import StubLLM

    st, agent = _agent(StubLLM("time_warning", "残り2分です。"))
    st.add_utterance("speaker_1", "AIさん、何かヒントもらえますか？", 478, 481, 0.9)
    rec = agent.tick(483.5)
    assert rec.gate == "forced:time_warning" and rec.decision.action == Action.time_warning
    assert not agent.has_pending_request(483.6)
