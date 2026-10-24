"""画面・API と介入ポリシーのテスト(プリセットと変更した項目、旧 API の互換、セッションへの反映、依頼待ちの解除)。
API は呼ばない(LLM はモック)。画面のスクリプトは node があれば偽の DOM の上で動かす(tests/ui_policy_harness.js)。"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.server as server  # noqa: E402
from config import POLICY_PRESETS, ROOT, load_settings  # noqa: E402
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


# ---------- 画面のスクリプト(node) ----------
@pytest.fixture(scope="module")
def ui_steps(tmp_path_factory):
    node = shutil.which("node")
    if not node:
        pytest.skip("node がないため画面のスクリプトは確かめない")
    spec = tmp_path_factory.mktemp("ui") / "spec.json"
    spec.write_text(json.dumps(ui_spec(), ensure_ascii=False), encoding="utf-8")
    r = subprocess.run([node, str(ROOT / "tests" / "ui_policy_harness.js"), str(HTML), str(spec)],
                       capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr
    return {s["label"]: s for s in json.loads(r.stdout)}


def test_ui_preset_values_are_shown(ui_steps):
    assert ui_steps["job_default"]["values"] == POLICY_PRESETS["job_gd"]
    assert ui_steps["general_default"]["values"] == POLICY_PRESETS["general_gw"]
    assert ui_steps["job_default"]["payload"] == {"policy_preset": "job_gd", "policy": {}}


def test_ui_sends_only_changed_items(ui_steps):
    s = ui_steps["general_organize_low"]
    assert s["payload"] == {"policy_preset": "general_gw", "policy": {"organize": "low"}}
    assert s["values"] == {**POLICY_PRESETS["general_gw"], "organize": "low"}  # ほかの項目はプリセットの値のまま
    assert ui_steps["job_stall_back"]["payload"]["policy"] == {}  # 初期値に戻したら「未変更」
    assert ui_steps["general_same_purpose"]["payload"]["policy"] == {"organize": "low"}  # 同じスタイルの押し直しでは消えない


def test_ui_preset_switch_resets_changes(ui_steps):
    assert ui_steps["general_default"]["payload"]["policy"] == {}  # スタイルの変更でプリセットが変わった
    assert ui_steps["choose_job_gd"] == {**ui_steps["choose_job_gd"], "preset": "job_gd", "values": POLICY_PRESETS["job_gd"]}
    s = ui_steps["pinned_purpose_change"]  # プリセットを選んだ後はスタイルを変えてもプリセットと変更を保つ
    assert s["preset"] == "job_gd" and s["payload"]["policy"] == {"turns": "high"}


def test_ui_content_dependency(ui_steps):
    s = ui_steps["general_scope_none"]
    assert s["values"]["content_scope"] == "none" and s["values"]["content_initiative"] == "none"
    assert ui_steps["general_scope_back"]["values"]["content_initiative"] == "stalled"  # 戻すとプリセットの値に戻る


def test_ui_job_neutrality_notice(ui_steps):
    assert ui_steps["job_default"]["notice"] == ""
    assert ui_steps["job_content_hint"]["notice"] == JOB_CONTENT_NOTICE
    assert ui_steps["job_with_gw_preset"]["notice"] == JOB_CONTENT_NOTICE
    assert ui_steps["general_default"]["notice"] == ""  # 就活以外では出さない


def test_ui_no_longer_sends_legacy_settings():
    html = HTML.read_text(encoding="utf-8")
    assert "knobs:" not in html and 'preset: "A"' not in html and "custom: {}" not in html
    assert "...policyPayload()" in html and "policy_preset: s.policy_preset" in html  # 再読み込みでも引き継ぐ


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


def test_api_info_serves_policy_spec():
    info = server.info()
    assert info["policy"] == ui_spec() and "knobs" not in info and "presets" not in info


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
