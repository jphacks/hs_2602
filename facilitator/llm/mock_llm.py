"""API キーなしで動作確認するためのルールベースのモック LLM。判断品質は評価対象外。"""

from __future__ import annotations

from typing import Any, Callable

from facilitator.llm.stream import simulate_stream


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
        allowed: list[str] = context["allowed_actions"]
        silence = context["silence_sec"]
        action, speech = "no_intervention", ""
        if len(allowed) == 1:
            action = allowed[0]
            speech = {
                "time_warning": "残り時間が少なくなってきました。そろそろ結論をまとめていきましょうか。",
                "close_gd": "時間になりました。グループとしての結論を確認させてください。",
            }.get(action, "")
        elif silence >= 12 and "ask_question" in allowed:
            action, speech = "ask_question", "ここまでの案の中で、一番効果が大きそうなのはどれだと思いますか?"
        return {
            "analysis": {
                "phase": context["observed_phase"] or context["target_phase"],
                "phase_reason": "(mock)",
                "health": "stalled" if silence >= 12 else "progressing",
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
