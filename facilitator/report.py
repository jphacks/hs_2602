"""テスト・セッションの結果を1つの Markdown ファイルにまとめる(logs/report-*.md)。

- タイムライン: 発言ごとの開始〜終了時刻・話者・内容、進行役の判断と理由、AI 参加者が話した理由
- AI に渡したプロンプトと返事(折りたたみ)
- 発言量のまとめ、評価のチェックポイント結果
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from config import ROOT
from facilitator.llm.base import generate, supports_stream
from facilitator.models import TickRecord

ACTION_JA = {
    "no_intervention": "見守り",
    "start_gd": "開始の説明",
    "ask_question": "問いかけ",
    "redirect_topic": "軌道修正",
    "organize_issues": "論点整理",
    "compare_criteria": "比較基準の提示",
    "invite_all": "全体へ発言を促す",
    "invite_person": "個人へ発言を促す",
    "return_to_speaker": "発言を戻す",
    "time_warning": "時間の告知",
    "close_gd": "終了・結論の確認",
}
GATE_JA = {
    "blocked:speaking": "誰かが話している",
    "blocked:cooldown": "直前の介入から間がない(クールダウン)",
    "blocked:short_pause": "沈黙が短い(会話の途中)",
    "blocked:forced_gap": "時間告知を直前の介入から少しあける",
    "blocked:finished": "終了済み",
}


class RecordingLLM:
    """LLM 呼び出しの入力と返事を記録するラッパー(中身の LLM はそのまま使う)。"""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.name = inner.name
        self.calls: list[dict[str, Any]] = []

    @property
    def streams(self) -> bool:
        return supports_stream(self.inner)

    def generate_json(self, system: str, user: str, schema: dict, context: dict, on_partial: Any = None) -> dict:
        call: dict[str, Any] = {"kind": context.get("kind", "進行役の判断"), "system": system, "user": user}
        try:
            out = generate(self.inner, system, user, schema, context, on_partial)
            call["response"] = out
            return out
        except Exception as e:
            call["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            self.calls.append(call)

    def __getattr__(self, name: str) -> Any:
        """ラッパーにない属性(last_usage など)は中身の LLM のものを返す。"""
        return getattr(self.inner, name)

    def mark(self) -> int:
        return len(self.calls)

    def since(self, mark: int) -> dict[str, Any] | None:
        """mark 以降の最後の呼び出し(なければ None)。"""
        return self.calls[-1] if len(self.calls) > mark else None


def ts(sec: float) -> str:
    sec = max(0.0, sec)
    return f"{int(sec // 60):02d}:{sec % 60:04.1f}"


@dataclass
class Section:
    title: str
    state: Any = None  # DiscussionState
    decisions: list[tuple[TickRecord, dict | None]] = field(default_factory=list)
    bot_notes: dict[tuple[str, float], tuple[str, dict | None]] = field(default_factory=dict)
    personas: list[Any] = field(default_factory=list)
    persona_call: dict | None = None
    checks: list[dict[str, Any]] = field(default_factory=list)
    info: dict[str, str] = field(default_factory=dict)
    hide_human_text: bool = False

    def decision(self, rec: TickRecord, call: dict | None = None) -> None:
        self.decisions.append((rec, call))

    def bot(self, speaker_id: str, start: float, reason: str, call: dict | None = None) -> None:
        self.bot_notes[(speaker_id, round(start, 2))] = (reason, call)

    # ---------- 描画 ----------
    def _t(self, t: float) -> str:
        st = self.state
        if st is None or not st.started or t < st.start_time:
            return f"開始前 {ts(t)}"
        return ts(st.elapsed(t))

    def render(self) -> str:
        st = self.state
        out = [f"## {self.title}", ""]
        for k, v in self.info.items():
            out.append(f"- **{k}**: {v}")
        if self.personas:
            out += ["", "### AI 参加者に割り当てた立場", ""]
            for p in self.personas:
                out.append(f"- 🤖 **{p.name}**【{p.stance}】{p.opinion}(話し方: {p.style})")
            if self.persona_call:
                out.append(_details("立場の生成: AI に渡した入力と返事", self.persona_call))
        if st is None:
            return "\n".join(out)

        # タイムライン: 発言(開始時刻)と判断(判断時刻)を時刻順に並べる
        items: list[tuple[float, int, str]] = []
        for u in st.utterances:
            span = f"`{self._t(u.start_time)} – {self._t(u.end_time)}`({u.duration:.1f}秒)"
            if u.source == "ai":
                line = f"{span} 🎙 **進行役**: {u.text}"
            elif u.source == "bot":
                name = st.speaker_names.get(u.speaker_id, u.speaker_id)
                line = f"{span} 🤖 **{name}**(AI参加者): {u.text}"
                note = self.bot_notes.get((u.speaker_id, round(u.start_time, 2)))
                if note:
                    line += f"\n    - 話した理由: {note[0]}"
                    if note[1]:
                        line += "\n" + _details("AI に渡した入力と返事", note[1], indent="    ")
            else:
                name = st.speaker_names.get(u.speaker_id) or "不明"
                conf = "" if u.speaker_confidence >= 0.999 else f"(話者の確信度 {u.speaker_confidence:.2f})"
                text = "(本文は記録しない設定)" if self.hide_human_text else u.text
                line = f"{span} 👤 **{name}**{conf}: {text}"
            items.append((u.start_time, 1, "- " + line))
        for rec, call in self.decisions:
            d = rec.decision
            if rec.gate.startswith("blocked"):
                line = f"`{self._t(rec.time)}` ・ 進行役は見送り — ルール層: {GATE_JA.get(rec.gate, rec.gate)}"
            else:
                head = "🧭 **進行役の判断**" + ("(ルールで決定)" if rec.gate.startswith("forced") else "")
                line = f"`{self._t(rec.time)}` {head}: **{ACTION_JA.get(d.action.value, d.action.value)}**"
                line += f"(きっかけ: {rec.trigger}、沈黙 {rec.silence_sec:.0f}秒、状態: {rec.analysis.health})"
                if d.reasoning:
                    line += f"\n    - 理由: {d.reasoning}"
                if d.speech and d.action.value != "no_intervention":
                    line += f"\n    - 発話: 「{d.speech}」"
                if rec.overridden:
                    line += f"\n    - ルール層による変更: {rec.overridden}"
                if call:
                    line += "\n" + _details("AI に渡した入力と返事", call, indent="    ")
            items.append((rec.time, 0, "- " + line))
        out += ["", "### タイムライン(発言は 開始 – 終了)", ""]
        out += [x[2] for x in sorted(items, key=lambda x: (x[0], x[1]))]

        if self.checks:
            out += ["", "### 評価(チェックポイント)", "", "| 時刻 | 期待 | 結果 | 判定 | 想定 |", "|---|---|---|---|---|"]
            for c in self.checks:
                mark = "✅" if c["passed"] else f"❌ {c['fail_reason']}"
                out.append(f"| {ts(c['at'])} | {' / '.join(c['expect'])} | {c['result']} | {mark} | {c['why']} |")

        end = max([0.0] + [u.end_time for u in st.utterances])
        out += ["", "### 発言量のまとめ", "", "| 参加者 | 種類 | 発言回数 | 発言時間 |", "|---|---|---|---|"]
        for x in st.speaker_stats(end):
            out.append(f"| {x.label or x.speaker_id} | {'AI参加者' if x.is_bot else '人間'} | {x.turns}回 | {x.total_speech_sec:.0f}秒 |")
        ai = [u for u in st.utterances if u.source == "ai"]
        out.append(f"| 進行役 | AI | {len(ai)}回 | {sum(u.duration for u in ai):.0f}秒 |")
        return "\n".join(out)


def _details(summary: str, call: dict, indent: str = "") -> str:
    body = [f"<details><summary>{summary}({call.get('kind', '')})</summary>", "", "**入力(ユーザープロンプト)**", "", "```text", call["user"], "```"]
    if "response" in call:
        body += ["", "**返事**", "", "```json", json.dumps(call["response"], ensure_ascii=False, indent=1), "```"]
    if "error" in call:
        body += ["", f"**エラー**: {call['error']}"]
    body += ["", "</details>"]
    lines = "\n".join(body).split("\n")  # 複数行のプロンプトも含めて全行を字下げする(リスト内で表示を崩さない)
    return "\n".join(indent + x if x else "" for x in lines)


class TestReport:
    def __init__(self, kind: str, name: str, meta: dict[str, str] | None = None) -> None:
        self.kind, self.name = kind, name
        self.meta = meta or {}
        self.sections: list[Section] = []
        self.created = datetime.now()

    def section(self, title: str) -> Section:
        sec = Section(title)
        self.sections.append(sec)
        return sec

    def write(self, log_dir: Path | None = None) -> Path:
        d = Path(log_dir or ROOT / "logs")
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"report-{self.created:%Y%m%d-%H%M%S}-{self.name}.md"
        head = [f"# テストログ: {self.name}", "", f"- **種類**: {self.kind}", f"- **日時**: {self.created:%Y-%m-%d %H:%M:%S}"]
        head += [f"- **{k}**: {v}" for k, v in self.meta.items()]
        head += ["- 時刻は GD 開始からの経過時間(一時停止中は含めない)。発言は「開始 – 終了」", ""]
        path.write_text("\n".join(head) + "\n" + "\n\n".join(s.render() for s in self.sections) + "\n", encoding="utf-8")
        return path
