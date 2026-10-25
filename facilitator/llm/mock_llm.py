"""API キーなしで動作確認するためのルールベースのモック LLM。判断品質は評価対象外。"""

from __future__ import annotations

from typing import Any, Callable

from facilitator.llm.stream import simulate_stream


MOCK_SUPPORT = [
    ("propose_idea", "例えば、時間帯ごとに割引をする案も考えられます。皆さんはどう思いますか?"),
    ("offer_hint", "利用する人の立場から見ると、どんな困りごとがありそうでしょう?"),
    ("ask_question", "ここまでの案の中で、一番効果が大きそうなのはどれだと思いますか?"),
]


class MockLLM:
    name = "mock"
    streams = True  # 画面の確認用に、生成途中の JSON を少しずつ渡す(ストリーミングの再現)
    stream_delay = 0.03

    def generate_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        context: dict[str, Any],
        on_partial: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        out = self._decide(context)
        if on_partial is not None:
            delay = self.stream_delay if out["decision"]["speech"] else 0.0
            simulate_stream(out, on_partial, delay=delay, slow_key="speech")
        return out

    def _decide(self, context: dict[str, Any]) -> dict[str, Any]:
        if "chat" in context:  # 雑談の司会(判断のタイミングはルール層が決めるので、呼ばれたら状況に合わせて話す)
            return self._decide_chat(context)
        allowed: list[str] = context["allowed_actions"]
        silence = context["silence_sec"]
        stall = context.get("stall_silence_sec", 12)  # 停滞とみなす沈黙(介入ポリシー⑤で変わる。標準は12秒)
        action, speech = "no_intervention", ""
        if len(allowed) == 1:
            action = allowed[0]
            speech = {
                "time_warning": "残り時間が少なくなってきました。そろそろ結論をまとめていきましょうか。",
                "close_gd": "時間になりました。グループとしての結論を確認させてください。",
            }.get(action, "")
        elif silence >= stall or context.get("request"):
            # 停滞時・依頼時は、許可されている中で一番踏み込んだ支援を選ぶ(内容支援の段階の配線確認用)
            for a, text in MOCK_SUPPORT:
                if a in allowed:
                    action, speech = a, text
                    break
        return {
            "analysis": {
                "phase": context["observed_phase"] or context["target_phase"],
                "phase_reason": "(mock)",
                "health": "stalled" if silence >= stall else "progressing",
                "issues": [],
                "opinions": [],
                "conflicts": [],
                "agreements": [],
                "missing_perspectives": [],
                "recent_summary": "(mock)",
            },
            "decision": {
                "action": action,
                "reasoning": "(mock) 沈黙時間と許可アクションのみで判断",
                "speech": speech,
                "urgency": 0.5 if action != "no_intervention" else 0.0,
                "target_speaker": None,
            },
        }

    def _decide_chat(self, context: dict[str, Any]) -> dict[str, Any]:
        allowed: list[str] = context["allowed_actions"]
        chat = context["chat"]
        topic = (chat["topic_candidates"] or ["最近ちょっとハマっていることはありますか?"])[0]
        action, speech = "no_intervention", ""
        if len(allowed) == 1:
            action = allowed[0]
        elif context.get("request") and "offer_topic" in allowed:
            action, speech = "offer_topic", f"では、{topic}"
        elif chat["answerers"] == 1 and "invite_all" in allowed:
            action, speech = "invite_all", "そうなんですね。ほかの皆さんはどうですか?"
        elif chat["answers"] and "ask_question" in allowed:
            action, speech = "ask_question", "いいですね。もう少し詳しく聞かせてもらえますか?"
        elif chat["unanswered"] >= 2 and "change_topic" in allowed:
            action, speech = "change_topic", f"話は変わりますが、{topic}"
        elif "invite_all" in allowed:
            action, speech = "invite_all", "どなたからでも大丈夫ですよ。思いついたことを気軽に聞かせてください。"
        return {
            "analysis": {"phase": context["observed_phase"] or "warm_up", "phase_reason": "(mock)",
                         "health": "stalled" if action != "no_intervention" else "progressing", "recent_summary": "(mock)"},
            "decision": {"action": action, "reasoning": "(mock) 雑談: 直前の AI の発言への反応だけで判断", "speech": speech,
                         "urgency": 0.5 if action != "no_intervention" else 0.0, "target_speaker": None},
        }
