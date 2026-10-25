"""振り返り(練習の補助)と発言の種類: 人間の発言だけを対象にし、AI と話者不明は評価しない。"""

from __future__ import annotations

from facilitator.feedback import build_feedback

LOG = [
    {"id": 0, "speaker_id": "ai", "who": "AI ファシリテーター", "text": "テーマは「学食」、時間は5分です。始めてください。", "source": "ai"},
    {"id": 1, "speaker_id": "speaker_1", "who": "たろう", "text": "昼休みをずらすのがいいと思います", "source": "human"},
    {"id": 2, "speaker_id": "speaker_3", "who": "アオイ", "text": "費用はどうでしょうか?", "source": "bot"},
    {"id": 3, "speaker_id": "speaker_2", "who": "はなこ", "text": "なぜそう思いますか?", "source": "human"},
    {"id": 4, "speaker_id": "unknown", "who": "不明", "text": "えっと", "source": "human"},
    {"id": 5, "speaker_id": "speaker_1", "who": "たろう", "text": "実際に授業が同じ時間に終わるからです", "source": "human"},
]
HUMANS = [{"id": "speaker_1", "name": "たろう"}, {"id": "speaker_2", "name": "はなこ"}]


class FakeLLM:
    name = "fake"

    def __init__(self) -> None:
        self.user = ""
        self.schema: dict = {}

    def generate_json(self, system, user, schema, context):
        self.user, self.schema = user, schema
        return {
            "utterances": [{"id": 1, "type": "proposal"}, {"id": 3, "type": "question"}, {"id": 5, "type": "reason"}],
            "people": [
                {"speaker_id": "speaker_1", "aspects": [{"aspect": "logic", "level": "great", "quote": "実際に授業が同じ時間に終わるからです"}],
                 "good": ["理由を添えられていました"], "next": ["他の人の意見にも触れてみましょう"]},
                {"speaker_id": "speaker_2", "aspects": [], "good": [], "next": ["自分の案も出してみましょう"]},
            ],
        }


def test_only_human_utterances_are_classified():
    llm = FakeLLM()
    fb = build_feedback(llm, "学食", "課題解決型", LOG, HUMANS)
    assert fb["ok"]
    ids = llm.schema["properties"]["utterances"]["items"]["properties"]["id"]["enum"]
    assert ids == [1, 3, 5]  # AI・AI 参加者・話者不明は分類しない
    assert "[-] アオイ(AI)" in llm.user and "[-] 不明" in llm.user  # 文脈としては渡す
    taro, hanako = fb["people"]
    assert taro["types"]["proposal"] == 1 and taro["types"]["reason"] == 1 and taro["turns"] == 2
    assert hanako["types"]["question"] == 1
    logic = next(a for a in taro["aspects"] if a["aspect"] == "logic")
    assert logic["level"] == "great" and logic["label"] == "論理性"
    assert all(a["level"] == "next" and not a["quote"] for a in hanako["aspects"])  # 根拠がない観点は「これから」


class MockName:
    name = "mock"


def test_mock_feedback_works_without_llm():
    fb = build_feedback(MockName(), "学食", "課題解決型", LOG, HUMANS)
    assert fb["ok"] and [p["name"] for p in fb["people"]] == ["たろう", "はなこ"]
    assert fb["people"][1]["types"]["question"] == 1
    assert fb["people"][0]["good"] and fb["people"][0]["next"]


def test_self_introduction_is_not_reviewed():
    """自己紹介(声の登録)の発言は、振り返り・発言の種類に含めない。"""
    llm = FakeLLM()
    log = [{"id": 9, "speaker_id": "speaker_2", "who": "はなこ", "text": "はなこです。よろしくお願いします。", "source": "human", "intro": True}] + LOG
    fb = build_feedback(llm, "学食", "課題解決型", log, HUMANS)
    assert 9 not in llm.schema["properties"]["utterances"]["items"]["properties"]["id"]["enum"]
    assert "よろしくお願いします" not in llm.user
    assert next(p for p in fb["people"] if p["name"] == "はなこ")["turns"] == 1
