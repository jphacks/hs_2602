"""介入ポリシー(第1段階: 設定の保持・検証・互換変換)のテスト。API は呼ばない。

第1段階では介入の判断・頻度・発話を変えないので、従来の計算(旧進行スタイル + 旧カスタム + 「詳細」の knob_overrides)と
ポリシー経由の計算が、数値設定・追加の指示・システムプロンプトのすべてで一致することを確かめる。
"""

from __future__ import annotations

import hashlib
import itertools
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CUSTOM_KEYS, POLICY_PRESETS, PRESETS, PURPOSE_PRESET, STYLE_KNOBS, knob_overrides, load_settings  # noqa: E402
from facilitator.agent.prompts import build_system_prompt  # noqa: E402
from facilitator.engine import Engine, SessionConfig  # noqa: E402
from facilitator.discussion_types import SELECTABLE_TYPES  # noqa: E402
from facilitator.policy import (  # noqa: E402
    ITEM_LABELS,
    LEVEL_ITEMS,
    ContentInitiative,
    ContentScope,
    InterventionPolicy,
    Level,
    from_knobs,
    legacy_knobs,
    preset_policy,
    purpose_preset,
    session_overrides,
    session_policy,
)

KNOB_KEYS = list(STYLE_KNOBS)  # 画面が送る順(phase, redirect, organize, turns, suggest)


def _legacy(preset: str, custom: dict, knobs: dict, purpose: str) -> tuple[dict, list[str]]:
    """第1段階より前の engine.start の計算(比較の基準)。"""
    overrides = dict(PRESETS.get(preset, {}))
    if preset == "custom":
        overrides.update({k: float(v) for k, v in custom.items() if k in CUSTOM_KEYS})
    knob_settings, notes = knob_overrides(knobs, purpose)
    overrides.update(knob_settings)
    return overrides, notes


# ---------- プリセット ----------
def test_presets_match_agreed_values():
    assert preset_policy("job_gd") == InterventionPolicy()  # 就活GD用はすべて標準・内容支援なし
    gw = preset_policy("general_gw")
    assert (gw.progress, gw.drift, gw.organize, gw.turns, gw.stall) == (
        Level.standard, Level.standard, Level.high, Level.high, Level.high
    )
    assert (gw.content_scope, gw.content_initiative) == (ContentScope.hint, ContentInitiative.stalled)


def test_purpose_selects_preset_and_unknown_falls_back_to_job():
    assert PURPOSE_PRESET == {"job": "job_gd", "general": "general_gw"}
    assert purpose_preset("general") == preset_policy("general_gw")
    assert purpose_preset("other") == preset_policy("job_gd")


def test_every_preset_is_valid_and_complete():
    for name, values in POLICY_PRESETS.items():
        assert set(values) == set(ITEM_LABELS), name
        assert preset_policy(name).model_dump(exclude={"legacy_suggest"}) == values


# ---------- 検証と依存関係 ----------
def test_scope_none_fixes_initiative_none():
    p = InterventionPolicy(content_scope="none", content_initiative="proactive")
    assert p.content_initiative == ContentInitiative.none
    # updated() でも補正される(model_copy と違い検証を通す)
    gw = preset_policy("general_gw")
    assert gw.updated(content_scope="none").content_initiative == ContentInitiative.none
    assert gw.updated(content_scope="concrete").content_initiative == ContentInitiative.stalled


@pytest.mark.parametrize(
    "bad",
    [{"stall": "max"}, {"content_scope": "all"}, {"content_initiative": "always"}, {"legacy_suggest": 3}, {"unknown": "low"}],
)
def test_invalid_values_are_rejected(bad):
    with pytest.raises(ValidationError):
        InterventionPolicy.model_validate(bad)


def test_policy_is_immutable():
    with pytest.raises(ValidationError):
        InterventionPolicy().stall = Level.high  # type: ignore[misc]


def test_all_combinations_normalize_to_1701_valid_policies():
    """①〜⑤ 3段階 × ⑥ 3×3 のうち、A=なし なら B=なし に補正されて 3^5 × 7 = 1701 通りになる。"""
    levels, scopes, inits = list(Level), list(ContentScope), list(ContentInitiative)
    seen = set()
    for combo in itertools.product(levels, levels, levels, levels, levels, scopes, inits):
        p = InterventionPolicy(**dict(zip(LEVEL_ITEMS, combo[:5])), content_scope=combo[5], content_initiative=combo[6])
        assert p.content_scope != ContentScope.none or p.content_initiative == ContentInitiative.none
        seen.add(p)
    assert len(seen) == 3**5 * 7


def test_job_notice_when_content_support_enabled():
    assert InterventionPolicy().notices("job") == []
    assert InterventionPolicy(content_scope="hint").notices("job")  # 中立設定ではない旨を UI に出す
    assert InterventionPolicy(content_scope="hint").notices("general") == []


# ---------- 画面の「詳細」(STYLE_KNOBS)との互換 ----------
def test_knob_mapping_and_invalid_values_are_ignored():
    p = from_knobs({"phase": 0, "redirect": 2, "organize": 1, "turns": 2, "suggest": 2}, InterventionPolicy())
    assert (p.progress, p.drift, p.organize, p.turns) == (Level.low, Level.high, Level.standard, Level.high)
    assert p.legacy_suggest == 2
    assert p.stall == Level.standard  # ⑤は画面の項目がないので base のまま
    base = preset_policy("general_gw")
    assert from_knobs({"phase": 5, "nope": 0, "suggest": -1}, base) == base  # 従来どおり不正な値は無視
    assert from_knobs(None, base) == base


def test_session_policy_uses_screen_for_1_to_4_and_preset_for_5_6():
    """①〜④は画面の値(指定なしは標準=従来どおり)、⑤・⑥は用途のプリセットの値。"""
    gw = session_policy("general")
    assert (gw.progress, gw.drift, gw.organize, gw.turns) == (Level.standard,) * 4
    assert (gw.stall, gw.content_scope, gw.content_initiative) == (Level.high, ContentScope.hint, ContentInitiative.stalled)
    assert session_policy("job") == InterventionPolicy()
    assert session_policy("general", {"organize": 2}).organize == Level.high


def test_legacy_knobs_round_trip():
    for values in itertools.product((0, 1, 2), repeat=len(KNOB_KEYS)):
        knobs = dict(zip(KNOB_KEYS, values))
        assert legacy_knobs(from_knobs(knobs, InterventionPolicy())) == knobs


@pytest.mark.parametrize("purpose", ["job", "general"])
def test_effects_identical_to_legacy_for_all_knob_combinations(purpose):
    """画面が送りうる 3^5 通りの「詳細」すべてで、数値設定の上書きと追加の指示が従来と一致する(就活の提案度合いの上限も含む)。"""
    for values in itertools.product((0, 1, 2), repeat=len(KNOB_KEYS)):
        knobs = dict(zip(KNOB_KEYS, values))
        assert session_overrides("A", {}, session_policy(purpose, knobs), purpose) == _legacy("A", {}, knobs, purpose), knobs


@pytest.mark.parametrize(
    "preset,custom,knobs",
    [
        ("B", {}, {}),
        ("B", {}, {"turns": 0}),  # 「詳細」が旧進行スタイルを上書きする順序も従来どおり
        ("custom", {"cooldown_sec": 50, "silence_trigger_sec": 9, "bogus": 1}, {}),
        ("custom", {"cooldown_sec": 50}, {"phase": 0, "turns": 2}),
        ("A", {}, {"redirect": 0}),
        ("A", {}, {}),  # 画面を通さない呼び出し(knobs なし)
    ],
)
def test_effects_identical_to_legacy_with_old_presets(preset, custom, knobs):
    for purpose in ("job", "general"):
        assert session_overrides(preset, custom, session_policy(purpose, knobs), purpose) == _legacy(preset, custom, knobs, purpose)


def test_knob_order_other_than_screen_keeps_content_in_screen_order():
    """画面は STYLE_KNOBS の順で送る。別の順で送られた場合も、上書きと指示の内容は同じで、指示の並びは画面の順になる
    (従来は送られた順。画面からの利用では差は出ない)。"""
    knobs = {"turns": 2, "phase": 0}
    new = session_overrides("A", {}, session_policy("job", knobs), "job")
    old = _legacy("A", {}, knobs, "job")
    assert new[0] == old[0] and sorted(new[1]) == sorted(old[1])
    assert new == _legacy("A", {}, {"phase": 0, "turns": 2}, "job")


# ---------- 就活GDの標準: システムプロンプトと数値設定が従来どおり ----------
# 第1段階の時点のシステムプロンプト(sha256 の先頭16桁)。第2段階でプロンプトを項目ごとに分けても、
# 就活GD用の標準設定ではこの値が変わらないこと。意図してプロンプトを変えたときだけ更新する
PROMPT_SHA = {
    ("job", "problem_solving"): "95f1eec7203dca03",
    ("job", "idea_generation"): "efa93f2072780cf7",
    ("job", "decision_making"): "b229544b4a7b94c8",
    ("job", "planning"): "853b3068e08d954d",
    ("general", "problem_solving"): "bdd9dd00950b4bac",
    ("general", "idea_generation"): "2156328787dac7d2",
    ("general", "decision_making"): "9f6efd84b388f523",
    ("general", "planning"): "668c028747d0ccb6",
    ("general", "generic"): "e660af2776cd8f86",
}


def _start(tmp_path, purpose: str, dtype: str, knobs: dict | None = None, policy=None) -> Engine:
    eng = Engine(load_settings(), llm_provider="mock", log_dir=tmp_path)
    eng.start(
        SessionConfig("テーマ", 5, ["A", "B", "C"], mode="text", intro=False, tts=False, purpose=purpose,
                      discussion_type=dtype, knobs=knobs or {}, policy=policy),
        background=False,
    )
    return eng


@pytest.mark.parametrize("purpose,dtype", list(PROMPT_SHA))
def test_standard_session_keeps_prompt_and_timing(tmp_path, purpose, dtype):
    ui_knobs = dict.fromkeys(KNOB_KEYS, 1)  # 画面の既定(すべて標準)
    for knobs in ({}, ui_knobs):
        eng = _start(tmp_path, purpose, dtype, knobs)
        try:
            prompt = eng.agent.system_prompt
            assert prompt == build_system_prompt(purpose, dtype)
            assert hashlib.sha256(prompt.encode()).hexdigest()[:16] == PROMPT_SHA[(purpose, dtype)], "システムプロンプトが変わった"
            s, d = eng.settings, load_settings()
            for k in ("cooldown_sec", "analysis_interval_sec", "silence_trigger_sec", "min_silence_for_intervention_sec",
                      "redirect_wait_max_sec", "phase_behind_gap", "phase_behind_late_gap", "phase_behind_late_fraction"):
                assert getattr(s, k) == getattr(d, k), k
            assert eng.agent.policy == eng.policy == session_policy(purpose, knobs)
        finally:
            eng.stop()


def test_session_with_knobs_keeps_legacy_effects(tmp_path):
    """チームで追加した「詳細」の効果(数値設定と追加の指示)が従来どおり効く。"""
    knobs = {"phase": 2, "redirect": 0, "organize": 2, "turns": 2, "suggest": 2}
    eng = _start(tmp_path, "job", "planning", knobs)
    try:
        overrides, notes = _legacy("A", {}, knobs, "job")
        assert eng.agent.system_prompt == build_system_prompt("job", "planning", notes)
        assert all(getattr(eng.settings, k) == v for k, v in overrides.items())
        assert eng.settings.cooldown_sec == 30.0  # 「どれだけ話を回すか: 積極的」の従来の値
        assert not any("AIからの提案度合い" in n for n in notes)  # 就活では提案度合いは標準まで(JOB_MAX_LEVEL)
        assert "介入ポリシー" in eng._sec.info
    finally:
        eng.stop()


def test_explicit_policy_stage1_has_no_effect_for_5_and_6(tmp_path):
    """⑤・⑥はまだ効果を持たない(第2・第3段階から)。就活GDで内容支援を有効にしてもプロンプトは変わらない。"""
    p = InterventionPolicy(stall="high", content_scope="concrete", content_initiative="proactive")
    eng = _start(tmp_path, "job", "problem_solving", policy=p)
    try:
        assert eng.policy == p
        assert eng.agent.system_prompt == build_system_prompt("job", "problem_solving")
        assert eng.settings.silence_trigger_sec == load_settings().silence_trigger_sec
    finally:
        eng.stop()
