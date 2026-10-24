"""API キーなしで動作確認するためのルールベースのモック LLM。判断品質は評価対象外。"""

from __future__ import annotations

from typing import Any


class MockLLM:
    name = "mock"

    def generate_json(
        self, system: str, user: str, schema: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
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
        elif silence >= stall and "ask_question" in allowed:
            action, speech = "ask_question", "ここまでの案の中で、一番効果が大きそうなのはどれだと思いますか?"
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
