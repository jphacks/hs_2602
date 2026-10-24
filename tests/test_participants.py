"""AI 参加者の立場の指定(主催者が設定画面で決めた立場を守る)。"""

from __future__ import annotations

from facilitator.llm.base import LLMError
from facilitator.participants import generate_personas


class MockName:
    name = "mock"


class FakeLLM:
    name = "fake"

    def __init__(self, personas: list[dict] | None = None, fail: bool = False) -> None:
        self.personas, self.fail, self.users = personas or [], fail, []

    def generate_json(self, system, user, schema, context):
        self.users.append(user)
        if self.fail:
            raise LLMError("x")
        return {"personas": self.personas}


def P(name: str, stance: str) -> dict:
    return {"name": name, "stance": stance, "opinion": f"{stance}の意見", "style": "落ち着いて話す"}


def test_mock_uses_given_stance_and_keeps_auto_for_blank():
    ps = generate_personas(MockName(), "テーマ", "job", 2, ["たろう"], 2, stances=["環境重視", ""])
    assert ps[0].stance == "環境重視" and "環境重視" in ps[0].opinion
    assert ps[1].stance and ps[1].stance != "環境重視"


def test_llm_prompt_lists_given_stances():
    llm = FakeLLM([P("ハルト", "環境重視"), P("サクラ", "コスト重視")])
    generate_personas(llm, "テーマ", "job", 2, [], 1, stances=["環境重視", ""])
    assert "「環境重視」(指定" in llm.users[0] and "2人目: 自由" in llm.users[0]


def test_given_stance_is_enforced_even_if_llm_changes_it():
    llm = FakeLLM([P("ハルト", "エコ志向"), P("サクラ", "コスト重視")])
    ps = generate_personas(llm, "テーマ", "job", 2, [], 1, stances=["環境重視"])
    assert ps[0].stance == "環境重視" and ps[0].opinion == "エコ志向の意見"
    assert ps[1].stance == "コスト重視"


def test_given_stance_survives_llm_failure():
    ps = generate_personas(FakeLLM(fail=True), "テーマ", "job", 1, [], 1, stances=["安全性重視"])
    assert ps[0].stance == "安全性重視" and "安全性重視" in ps[0].opinion


def test_no_stances_prompt_unchanged():
    llm = FakeLLM([P("ハルト", "環境重視")])
    generate_personas(llm, "テーマ", "job", 1, [], 1)
    assert "主催者が指定した立場" not in llm.users[0]
