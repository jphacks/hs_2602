"""詳細設定(owaki さんの UI の7項目: ①〜⑤、⑥-A・⑥-B)と介入ポリシーの接続のテスト。API は呼ばない。

7項目の効果は介入ポリシー(facilitator/policy.py)に一本化している。画面に返す項目と初期値はポリシーの定義から作り、
画面から送られた値はポリシーに変換する。ここでは、その変換と、画面に見える初期値と実際に適用される設定の一致を確かめる。
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.server as server  # noqa: E402
from config import load_settings  # noqa: E402
from facilitator.agent.prompts import JOB_CONTENT_RULES, JOB_NEUTRAL_RULES, build_system_prompt  # noqa: E402
from facilitator.engine import Engine  # noqa: E402
from facilitator.policy import (  # noqa: E402
    JOB_CONTENT_NOTICE,
    UI_KNOBS,
    InterventionPolicy,
    from_knobs,
    knob_defaults,
    preset_policy,
    request_policy,
    to_knobs,
    ui_knobs,
)

KEYS = list(UI_KNOBS)
HTML = Path(__file__).resolve().parents[1] / "app" / "static" / "index.html"


# ---------- 初期値(owaki さんの元のテストと同じ意図) ----------
def test_gd_defaults_are_all_standard_without_content_support():
    d = knob_defaults()["job"]
    assert d == {"progress": 1, "redirect": 1, "organize": 1, "turns": 1, "stall": 1, "content": 0, "content_when": 0}
    assert request_policy("job", knobs=d)[0] == InterventionPolicy()


def test_gw_defaults_follow_the_table():
    d = knob_defaults()["general"]
    assert d == {"progress": 1, "redirect": 1, "organize": 2, "turns": 2, "stall": 2, "content": 1, "content_when": 1}
    assert request_policy("general", knobs=d)[0] == preset_policy("general_gw")


def test_content_none_fixes_spontaneous_support_to_none():
    p = from_knobs({"content": 0, "content_when": 2}, preset_policy("general_gw"), "general")
    assert p.content_scope.value == "none" and p.content_initiative.value == "none"


def test_job_can_enable_content_support_and_prompt_is_consistent():
    p, notices = request_policy("job", knobs={**knob_defaults()["job"], "content": 2, "content_when": 1})
    assert (p.content_scope.value, p.content_initiative.value) == ("concrete", "stalled")
    assert notices == [JOB_CONTENT_NOTICE]  # 中立設定ではない旨(画面には出していない。残る不整合として報告)
    prompt = build_system_prompt("job", "problem_solving", content=("concrete", "stalled"))
    assert "propose_idea:" in prompt and JOB_CONTENT_RULES in prompt and JOB_NEUTRAL_RULES not in prompt


# ---------- 画面に返す定義 ----------
def test_info_returns_seven_knobs_and_defaults_from_policy():
    info = server.info()
    assert info["knobs"] == ui_knobs() and [k["key"] for k in info["knobs"]] == KEYS
    assert all(len(k["options"]) == 3 for k in info["knobs"])
    assert info["knob_defaults"] == knob_defaults()
    assert set(info["knob_defaults"]) == {"job", "general"}  # 画面は job 以外(就活以外・カスタム)に general を使う


def test_ui_uses_knob_defaults_and_sends_all_knobs():
    """画面(owaki さんの版、こちらでは編集しない)が、この API の形を使っていること。"""
    html = HTML.read_text(encoding="utf-8")
    assert "knob_defaults" in html and "knobs: { ...ui.knobs }" in html and "ui.info?.knobs" in html


# ---------- 3つのスタイル: 画面に見える初期値と、実際に適用される設定が一致する ----------
@pytest.mark.parametrize("style,purpose,preset", [("job", "job", "job_gd"), ("general", "general", "general_gw")])
def test_style_defaults_shown_equal_policy_applied(style, purpose, preset):
    shown = knob_defaults()[style]  # 画面がこの値を表示し、そのまま送る
    assert request_policy(purpose, knobs=shown)[0] == preset_policy(preset)
    assert to_knobs(preset_policy(preset)) == shown


def test_custom_style_applies_exactly_what_is_shown():
    """カスタム(用途は就活以外として送る)は、送った7項目がそのままポリシーになる(全組み合わせ)。"""
    seen = 0
    for values in itertools.product((0, 1, 2), repeat=len(KEYS)):
        knobs = dict(zip(KEYS, values))
        p = request_policy("general", knobs=knobs)[0]
        expected = {**knobs, "content_when": 0 if knobs["content"] == 0 else knobs["content_when"]}
        assert to_knobs(p) == expected, knobs
        seen += 1
    assert seen == 3**7


# ---------- 7項目を個別に変えると、対応する設定だけが変わる ----------
@pytest.mark.parametrize("style", ["job", "general"])
@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("value", [0, 1, 2])
def test_each_knob_changes_only_its_own_item(style, key, value):
    base = knob_defaults()[style]
    knobs = {**base, key: value}
    got = to_knobs(request_policy("job" if style == "job" else "general", knobs=knobs)[0])
    expected = dict(knobs)
    if expected["content"] == 0:
        expected["content_when"] = 0  # ⑥-A が「なし」なら ⑥-B も「なし」(依存関係の補正だけ)
    assert got == expected


# ---------- セッションへの反映(判断条件・出力の選択肢・プロンプト) ----------
@pytest.fixture
def engine(monkeypatch, tmp_path):
    eng = Engine(load_settings(), llm_provider="mock", tts_chars_per_sec=1000.0, log_dir=tmp_path)
    monkeypatch.setattr(server, "engine", eng)
    yield eng
    eng.stop()


BASE = dict(theme="テーマ", minutes=5, names=["A", "B"], mode="text", intro=False, tts=False, preset="A", custom={})


def _start(purpose, knobs, **kw):
    extra = {"discussion_type": "problem_solving"} if purpose == "job" else {}
    r = server.start(server.StartReq(purpose=purpose, knobs=knobs, **BASE, **extra, **kw))
    assert r["ok"], r
    return r


def test_ui_payload_job_default_is_unchanged_gd(engine):
    _start("job", knob_defaults()["job"])
    assert engine.policy == InterventionPolicy()
    assert engine.agent.system_prompt == build_system_prompt("job", "problem_solving")
    enum = engine.agent.output_schema["properties"]["decision"]["properties"]["action"]["enum"]
    assert "offer_hint" not in enum and "propose_idea" not in enum


@pytest.mark.parametrize("turns,stall", list(itertools.product((0, 1, 2), repeat=2)))
def test_turns_and_stall_keep_global_safety_settings(engine, turns, stall):
    """④・⑤をどう変えても、クールダウン45秒・分析間隔20秒・2秒の間・個人指名の安全条件は変わらない。"""
    _start("general", {**knob_defaults()["general"], "turns": turns, "stall": stall})
    a, d = engine.agent, load_settings()
    for k in ("cooldown_sec", "analysis_interval_sec", "min_silence_for_intervention_sec", "person_cooldown_sec",
              "speaker_reliability_threshold"):
        assert a._param(k) == getattr(d, k), k
    assert a._param("silence_trigger_sec") == {0: 18.0, 1: 12.0, 2: 8.0}[stall]  # ⑤の効果は沈黙トリガーだけ
    assert a._param("quiet_member_sec") == (60.0 if turns == 2 else 0.0)  # ④の効果は発言の少ない人の表示だけ


@pytest.mark.parametrize("content,expect_actions", [(0, []), (1, ["offer_hint"]), (2, ["offer_hint", "propose_idea"])])
def test_content_scope_is_enforced_in_output_schema(engine, content, expect_actions):
    _start("general", {**knob_defaults()["general"], "content": content})
    enum = engine.agent.output_schema["properties"]["decision"]["properties"]["action"]["enum"]
    assert [a for a in enum if a in ("offer_hint", "propose_idea")] == expect_actions
    assert ("# 内容支援なし" in engine.agent.system_prompt) == (content == 0)


def test_unknown_knob_keys_are_ignored(engine):
    _start("general", {**knob_defaults()["general"], "bogus": 2, "phase": 0})
    assert engine.policy.progress.value == "standard"  # progress(新しいキー)が旧キー phase より優先
