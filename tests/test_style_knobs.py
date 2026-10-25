"""詳細設定(①〜⑤、⑥-A・⑥-B)の初期値と、進行役への反映。"""

from __future__ import annotations

from config import STYLE_DEFAULTS, knob_overrides
from facilitator.agent.prompts import build_system_prompt


def test_gd_defaults_are_all_standard_without_content_support():
    overrides, notes = knob_overrides({}, "job")
    assert overrides == {} and notes == []
    assert STYLE_DEFAULTS["job"]["content"] == 0 and STYLE_DEFAULTS["job"]["content_when"] == 0


def test_gw_defaults_follow_the_table():
    overrides, notes = knob_overrides({}, "general")
    text = "\n".join(notes)
    for label in ("③ 論点・意見の整理", "④ 発言機会の調整", "⑤ 停滞への介入", "⑥-A 内容支援の範囲", "⑥-B 自発的な内容支援"):
        assert label in text
    assert "① 進行" not in text and "② 脱線" not in text  # 標準は指示なし
    assert "ヒント" in text and "停滞したときだけ" in text
    assert overrides["silence_trigger_sec"] == 8.0 and overrides["cooldown_sec"] == 30.0


def test_content_none_fixes_spontaneous_support_to_none():
    _, notes = knob_overrides({"content": 0, "content_when": 2}, "general")
    assert not any(n.startswith("- ⑥") for n in notes)


def test_job_can_enable_content_support_and_prompt_says_it_takes_priority():
    _, notes = knob_overrides({"content": 2, "content_when": 1}, "job")
    assert any("具体的な案の例" in n for n in notes)
    prompt = build_system_prompt("job", "problem_solving", notes)
    assert "に限り" in prompt and "優先します" in prompt and "具体的な案の例" in prompt
