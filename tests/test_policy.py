"""介入ポリシーの設定(保持・検証・優先順位・旧設定との互換)のテスト。API は呼ばない。
①〜⑤の段階ごとの効果と項目の独立性は tests/test_levels.py。"""

from __future__ import annotations

import hashlib
import itertools
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import POLICY_PRESETS, PURPOSE_PRESET, STYLE_KNOBS, load_settings  # noqa: E402
from facilitator.agent.prompts import CONTENT_NONE_RULES, build_system_prompt  # noqa: E402
from facilitator.engine import Engine, SessionConfig  # noqa: E402
from facilitator.policy import (  # noqa: E402
    ITEM_LABELS,
    LEVEL_ITEMS,
    ContentInitiative,
    ContentScope,
    InterventionPolicy,
    Level,
    from_knobs,
    legacy_knobs,
    levels,
    preset_policy,
    purpose_preset,
    resolve_policy,
    session_policy,
)

KNOB_KEYS = list(STYLE_KNOBS)  # 画面が送る順(phase, redirect, organize, turns, suggest)
UI_DEFAULT_KNOBS = dict.fromkeys(KNOB_KEYS, 1)  # 画面の既定(すべて標準)
STANDARD_LEVELS = dict.fromkeys(LEVEL_ITEMS, "standard")


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
        assert preset_policy(name).model_dump() == values


# ---------- 検証と依存関係 ----------
def test_scope_none_fixes_initiative_none():
    p = InterventionPolicy(content_scope="none", content_initiative="proactive")
    assert p.content_initiative == ContentInitiative.none
    gw = preset_policy("general_gw")  # updated() でも補正される(model_copy と違い検証を通す)
    assert gw.updated(content_scope="none").content_initiative == ContentInitiative.none
    assert gw.updated(content_scope="concrete").content_initiative == ContentInitiative.stalled


@pytest.mark.parametrize(
    "bad",
    [{"stall": "max"}, {"content_scope": "all"}, {"content_initiative": "always"}, {"legacy_suggest": 1}, {"unknown": "low"}],
)
def test_invalid_values_are_rejected(bad):
    with pytest.raises(ValidationError):
        InterventionPolicy.model_validate(bad)


def test_policy_is_immutable():
    with pytest.raises(ValidationError):
        InterventionPolicy().stall = Level.high  # type: ignore[misc]


def test_all_combinations_normalize_to_1701_valid_policies():
    """①〜⑤ 3段階 × ⑥ 3×3 のうち、A=なし なら B=なし に補正されて 3^5 × 7 = 1701 通りになる。"""
    lv, scopes, inits = list(Level), list(ContentScope), list(ContentInitiative)
    seen = set()
    for combo in itertools.product(lv, lv, lv, lv, lv, scopes, inits):
        p = InterventionPolicy(**dict(zip(LEVEL_ITEMS, combo[:5])), content_scope=combo[5], content_initiative=combo[6])
        assert p.content_scope != ContentScope.none or p.content_initiative == ContentInitiative.none
        seen.add(p)
    assert len(seen) == 3**5 * 7


def test_job_notice_when_content_support_enabled():
    assert InterventionPolicy().notices("job") == []
    assert InterventionPolicy(content_scope="hint").notices("job")  # 中立設定ではない旨を UI に出す
    assert InterventionPolicy(content_scope="hint").notices("general") == []


# ---------- 設定の優先順位 ----------
def test_resolve_policy_priority():
    """プリセット < 個別の変更 < 旧画面の「詳細」。最後に依存関係を補正する。"""
    assert resolve_policy("general") == preset_policy("general_gw")  # 何も変えなければプリセットの値
    assert resolve_policy("job", preset="general_gw") == preset_policy("general_gw")  # プリセットを明示
    p = resolve_policy("general", overrides={"organize": "low", "content_scope": "none"})
    assert p.organize == Level.low and p.turns == Level.high  # 変えた項目だけ上書き
    assert p.content_initiative == ContentInitiative.none  # A=なし → B=なし
    p = resolve_policy("general", overrides={"organize": "low"}, knobs={"organize": 2})
    assert p.organize == Level.high  # knobs が最後
    with pytest.raises(ValidationError):
        resolve_policy("general", overrides={"stall": "max"})  # 個別の変更の不正な値はエラー


def test_session_policy_with_current_ui_knobs():
    """現在の画面は「詳細」を全部送るので、①〜④は画面の値(既定は標準)、⑤・⑥はプリセットの値になる
    (プリセットの③④の積極的が画面の既定で上書きされる。変更した項目だけ送るようにするのは第4段階)。"""
    gw = session_policy("general", UI_DEFAULT_KNOBS)
    assert (gw.progress, gw.drift, gw.organize, gw.turns) == (Level.standard,) * 4
    assert (gw.stall, gw.content_scope, gw.content_initiative) == (Level.high, ContentScope.hint, ContentInitiative.stalled)
    assert session_policy("general") == preset_policy("general_gw")  # 画面を通さない呼び出しはプリセットのまま
    assert session_policy("job", UI_DEFAULT_KNOBS) == session_policy("job") == InterventionPolicy()


# ---------- 画面の「詳細」(STYLE_KNOBS)との互換 ----------
def test_knob_mapping_and_invalid_values_are_ignored():
    p = from_knobs({"phase": 0, "redirect": 2, "organize": 1, "turns": 2}, InterventionPolicy())
    assert (p.progress, p.drift, p.organize, p.turns) == (Level.low, Level.high, Level.standard, Level.high)
    assert p.stall == Level.standard  # ⑤は画面の項目がないので base のまま
    base = preset_policy("general_gw")
    assert from_knobs({"phase": 5, "nope": 0, "suggest": -1}, base) == base  # 従来どおり不正な値は無視
    assert from_knobs(None, base) == base


def test_suggest_knob_maps_to_content_support():
    """「AIからの提案度合い」: 0 → 内容支援なし / 1 → プリセットのまま / 2 → ヒントまで・積極的。就活では 2 を受け付けない。"""
    gw = preset_policy("general_gw")
    assert from_knobs({"suggest": 1}, gw, "general") == gw
    p = from_knobs({"suggest": 0}, gw, "general")
    assert (p.content_scope, p.content_initiative) == (ContentScope.none, ContentInitiative.none)
    p = from_knobs({"suggest": 2}, gw, "general")
    assert (p.content_scope, p.content_initiative) == (ContentScope.hint, ContentInitiative.proactive)
    job = preset_policy("job_gd")
    assert from_knobs({"suggest": 2}, job, "job") == job  # 現在の画面と同じく就活は標準まで(JOB_MAX_LEVEL)
    assert from_knobs({"suggest": 0}, job, "job") == job


def test_legacy_knobs_round_trip():
    keys = [k for k in KNOB_KEYS if k != "suggest"]
    for values in itertools.product((0, 1, 2), repeat=len(keys)):
        knobs = dict(zip(keys, values))
        assert legacy_knobs(from_knobs(knobs, InterventionPolicy())) == knobs


def test_settings_have_no_legacy_numeric_presets():
    """旧進行スタイル(A/B/custom)の数値の上書きは廃止した(B は「⑤停滞への介入: 積極的」に変換。tests/test_ui_policy.py)。"""
    import config

    assert not hasattr(config, "PRESETS") and not hasattr(config, "CUSTOM_KEYS")


# ---------- 就活GDの標準: システムプロンプトと数値設定が従来どおり ----------
# システムプロンプト(sha256 の先頭16桁)。①〜⑤がすべて標準なら、プロンプトを項目ごとに組み立てても
# 第1段階の時点と同じであること。意図してプロンプトを変えたときだけ更新する
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


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@pytest.mark.parametrize("purpose,dtype", list(PROMPT_SHA))
def test_all_standard_prompt_unchanged(purpose, dtype):
    """就活GDは第1段階のプロンプトのまま。一般GWは「内容支援なし」の節(高-1)を末尾に加えただけで、ほかは同じ。"""
    p = build_system_prompt(purpose, dtype)
    if purpose == "general":
        assert p.endswith(CONTENT_NONE_RULES) and p.count(CONTENT_NONE_RULES) == 1
        p = p[: -len(CONTENT_NONE_RULES)]
    else:
        assert CONTENT_NONE_RULES not in p
    assert _sha(p) == PROMPT_SHA[(purpose, dtype)]
    assert build_system_prompt(purpose, dtype, levels=STANDARD_LEVELS) == build_system_prompt(purpose, dtype)


def _start(tmp_path, purpose: str, dtype: str, knobs: dict | None = None, policy=None) -> Engine:
    eng = Engine(load_settings(), llm_provider="mock", log_dir=tmp_path)
    eng.start(
        SessionConfig("テーマ", 5, ["A", "B", "C"], mode="text", intro=False, tts=False, purpose=purpose,
                      discussion_type=dtype, knobs=knobs or {}, policy=policy),
        background=False,
    )
    return eng


TIMING_KEYS = ("cooldown_sec", "analysis_interval_sec", "silence_trigger_sec", "min_silence_for_intervention_sec",
               "redirect_wait_max_sec", "phase_behind_gap", "phase_behind_late_gap", "phase_behind_late_fraction",
               "drift_confirm_ticks", "quiet_member_sec")


@pytest.mark.parametrize("dtype", ["problem_solving", "idea_generation", "decision_making", "planning"])
def test_job_session_keeps_prompt_and_timing(tmp_path, dtype):
    """就活GD(画面の既定・knobs なしの両方)は、システムプロンプトも判断条件も従来どおり。"""
    for knobs in ({}, UI_DEFAULT_KNOBS):
        eng = _start(tmp_path, "job", dtype, knobs)
        try:
            assert _sha(eng.agent.system_prompt) == PROMPT_SHA[("job", dtype)], "システムプロンプトが変わった"
            d = load_settings()
            for k in TIMING_KEYS:
                assert eng.agent._param(k) == getattr(d, k), k
        finally:
            eng.stop()


def test_ui_turns_high_keeps_cooldown_and_interval(tmp_path):
    """旧「どれだけ話を回すか: 積極的」はクールダウン 30秒・分析間隔 15秒・沈黙トリガー 8秒に変えていたが、
    ④は発言機会の判断だけを変える(発言の少ない参加者を示す)。"""
    eng = _start(tmp_path, "job", "planning", {**UI_DEFAULT_KNOBS, "turns": 2})
    try:
        a, d = eng.agent, load_settings()
        assert (a._param("cooldown_sec"), a._param("analysis_interval_sec"), a._param("silence_trigger_sec")) == (
            d.cooldown_sec, d.analysis_interval_sec, d.silence_trigger_sec
        )
        assert a._param("quiet_member_sec") == 60.0
        assert eng.policy.turns == Level.high and "介入ポリシー" in eng._sec.info
    finally:
        eng.stop()


def test_general_session_from_current_ui_applies_preset_stall_and_content(tmp_path):
    """一般GW を現在の画面から始めると、⑤・⑥はプリセットの値が効き、③④は画面の既定(標準)になる。"""
    eng = _start(tmp_path, "general", "generic", UI_DEFAULT_KNOBS)
    try:
        assert eng.agent._param("silence_trigger_sec") == 8.0
        assert eng.agent._param("quiet_member_sec") == 0.0
        assert eng.agent.system_prompt == build_system_prompt(
            "general", "generic", levels={**STANDARD_LEVELS, "stall": "high"}, content=("hint", "stalled")
        )
    finally:
        eng.stop()
