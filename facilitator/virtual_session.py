"""仮想時計で動く GD セッション(シミュレーターとチャット GD で共通)。

アプリ(engine.py)と同じロジックで、進行役の判断と AI 参加者の発言を時刻ごとに進める。
画面を使わず、起きたことを読みやすいテキスト行として out() に出力する。
"""

from __future__ import annotations

from collections import Counter
from typing import Callable

from config import Settings
from facilitator.agent.agent import FacilitatorAgent
from facilitator.llm.base import LLMError
from facilitator.logging import TickLogger
from facilitator.participants import (
    LEVELS,
    bot_mode_for,
    choose_and_generate,
    generate_personas,
    generate_utterance,
    level_interval,
    pick_bot,
    transcript_for_bot,
)
from facilitator.report import ACTION_JA, GATE_JA, RecordingLLM, TestReport
from facilitator.models import UNLIMITED_SEC
from facilitator.state import DiscussionState

MODE_JA = {"active": "練習相手モード(間があれば発言)", "reserved": "控えめモード(名指し・長い沈黙のみ)"}
INDENT = " " * 18


def tt(sec: float) -> str:
    sec = max(0.0, sec)
    return f"{int(sec // 60):02d}:{sec % 60:04.1f}"


class VirtualSession:
    def __init__(
        self,
        settings: Settings,
        llm: RecordingLLM,
        *,
        theme: str,
        humans: dict[str, str],
        ai_count: int,
        duration_sec: float,
        purpose: str = "job",
        title: str = "",
        report_kind: str = "",
        report_name: str = "session",
        out: Callable[[str], None] = print,
        show_blocked: bool = True,
        level: int | None = None,  # AI 参加者のレベル(1〜3)。思考力と発言の間隔が変わる
        personas: list | None = None,  # 立場を固定したいとき(比較実験用)。None なら生成する
    ) -> None:
        self.level = level
        self.interval = level_interval(level)
        self.threshold_cache: dict[int, float] = {}
        self.s, self.llm, self.out, self.purpose = settings, llm, out, purpose
        self.unlimited = duration_sec >= UNLIMITED_SEC
        limit_ja = "なし" if self.unlimited else f"{duration_sec / 60:g}分"
        self.show_blocked = show_blocked
        settings.num_participants = len(humans) + ai_count
        self.humans = humans
        st = self.st = DiscussionState(settings=settings, theme=theme, duration_sec=duration_sec)
        for sid, name in humans.items():
            st.set_speaker_name(sid, name, confirmed=True)
        self.report = TestReport(report_kind, report_name, {"LLM": llm.name})
        sec = self.sec = self.report.section(title or theme)
        sec.state = st

        m = llm.mark()
        if personas is not None:
            self.personas = personas
            ai_count = len(personas)
        else:
            self.personas = generate_personas(llm, theme, purpose, ai_count, list(humans.values()), len(humans) + 1, level)
        sec.personas, sec.persona_call = self.personas, llm.since(m)
        # 参加者の一覧(集計の対象)を、人間 + 実際の AI 参加者でそろえる(立場をファイルで渡した場合も)
        st.speaker_ids = list(humans) + [p.speaker_id for p in self.personas]
        settings.num_participants = len(st.speaker_ids)
        for p in self.personas:
            st.set_speaker_name(p.speaker_id, p.name, confirmed=True)
            st.bot_ids.add(p.speaker_id)
        self.mode = bot_mode_for(len(humans), ai_count)
        sec.info = {
            "テーマ": theme,
            "人数": f"人間 {len(humans)}人 / AI参加者 {ai_count}人",
            "AI参加者のモード": MODE_JA.get(self.mode, "なし"),
            "制限時間": limit_ja,
            **({"AI参加者のレベル": f"{LEVELS[level]['label']} / 発言の間隔 {self.interval[0]:g}〜{self.interval[1]:g}秒"} if level else {}),
        }
        self.agent = FacilitatorAgent(st, llm, settings, logger=TickLogger(settings.log_dir, report_name), purpose=purpose)
        self.t = 0.0
        self.busy_until = -1.0
        self.last_bot_end = -1e9
        self.answered = -1

        out(f"テーマ: {theme} / 制限時間 {limit_ja}")
        out(f"参加者: 人間 {'、'.join(humans.values()) or 'なし'} / AI参加者 {ai_count}人 → {MODE_JA.get(self.mode, 'AI参加者なし')}")
        if level:
            out(f"AI参加者のレベル: {LEVELS[level]['label']} / 発言の間隔 {self.interval[0]:g}〜{self.interval[1]:g}秒")
        for p in self.personas:
            out(f"  🤖 {p.name}【{p.stance}】{p.opinion}")
        out("")

    # ---------- 時刻 ----------
    def et(self, t: float) -> float:
        """GD 開始からの経過秒(開始前は 0)。"""
        return self.st.elapsed(t) if self.st.started else 0.0

    # ---------- 人間の発言 ----------
    def human(self, sid: str, text: str, start: float, end: float, conf: float = 1.0) -> None:
        self.st.add_utterance(sid, text, start, end, conf)
        name = self.humans.get(sid, sid)
        self.out(f"[{tt(self.et(start))}–{tt(self.et(end))}] 👤 {name}: {text}")

    # ---------- 1ステップ(進行役 → AI 参加者) ----------
    def step(self, t: float, speaking_now: bool = False) -> None:
        st, s, llm = self.st, self.s, self.llm
        self.t = t
        st.speaking_now = speaking_now
        if t < self.busy_until or st.finished:
            return
        # 1. 進行役
        trig = self.agent.should_tick(t)
        if trig:
            m = llm.mark()
            rec = self.agent.tick(t, trig)
            self.sec.decision(rec, llm.since(m))
            d = rec.decision
            when = tt(self.et(t))
            if d.action.value != "no_intervention":
                dur = len(d.speech) / s.tts_chars_per_sec
                self.busy_until = t + dur
                rule = "・ルールで決定" if rec.gate.startswith("forced") else ""
                self.out(f"[{when}–{tt(self.et(t + dur))}] 🎙 進行役({ACTION_JA.get(d.action.value, d.action.value)}{rule}): {d.speech}")
                if d.reasoning and not rec.gate.startswith("forced:start"):
                    self.out(f"{INDENT}└ 理由: {d.reasoning}")
                if rec.overridden:
                    self.out(f"{INDENT}└ ルール層による変更: {rec.overridden}")
            elif rec.gate == "llm":
                self.out(f"[{when}] 🧭 進行役: 見守り — {d.reasoning}")
            elif self.show_blocked and rec.gate.startswith("blocked"):
                self.out(f"[{when}] ・ 進行役は見送り(ルール: {GATE_JA.get(rec.gate, rec.gate)})")
        # 2. AI 参加者(進行役が話していないときだけ)
        if t < self.busy_until or st.finished or self.agent.pending_request is not None:
            return  # 進行役への依頼には進行役が答える(AI 参加者が先に答えない)
        picked = pick_bot(
            st, self.personas, self.mode, t, self.last_bot_end, self.answered, len(self.humans),
            self.interval, self.threshold_cache,
        )
        if not picked:
            return
        p, reason, addressed = picked
        m = llm.mark()
        why = ""
        try:
            if addressed >= 0:  # 名指しされた人が答える
                text = generate_utterance(llm, p, self.purpose, st.theme, transcript_for_bot(st), reason, self.level)
            else:  # 誰が話すかは文脈で選ぶ(直前の話者は除く)
                last_sid = st.utterances[-1].speaker_id if st.utterances else ""
                cands = [x for x in self.personas if x.speaker_id != last_sid] or self.personas
                counts = {x.speaker_id: x.spoken for x in self.personas}
                p, text, why = choose_and_generate(
                    llm, cands, self.purpose, st.theme, transcript_for_bot(st), reason, self.level, counts, p
                )
        except LLMError as e:
            self.out(f"{INDENT}(AI参加者 {p.name} の発言を生成できませんでした: {e})")
            return
        if not text:
            return
        dur = len(text) / s.tts_chars_per_sec
        st.add_utterance(p.speaker_id, text, t, t + dur, 1.0, source="bot")
        self.sec.bot(p.speaker_id, t, reason + (f" / {p.name}が話した理由: {why}" if why else ""), llm.since(m))
        p.spoken += 1
        self.busy_until = self.last_bot_end = t + dur
        if addressed >= 0:
            self.answered = addressed
        self.out(f"[{tt(self.et(t))}–{tt(self.et(t + dur))}] 🤖 {p.name}(AI参加者): {text}")
        self.out(f"{INDENT}└ 話した理由: {reason}")
        if why:
            self.out(f"{INDENT}└ {p.name}が話した理由: {why}")

    def run_until(self, t_end: float, step: float = 0.5) -> None:
        t = self.t
        while t <= t_end and not self.st.finished:
            self.step(t)
            t += step
        self.t = max(self.t, min(t, t_end))

    # ---------- 結果 ----------
    def summary(self) -> str:
        st = self.st
        end = max([self.t] + [u.end_time for u in st.utterances])
        acted = Counter(iv.action.value for iv in st.interventions)
        watched = sum(1 for r, _ in self.sec.decisions if r.gate == "llm" and r.decision.action.value == "no_intervention")
        blocked = Counter(r.gate for r, _ in self.sec.decisions if r.gate.startswith("blocked"))
        stats = st.speaker_stats(end)
        total = sum(x.total_speech_sec for x in stats) or 1
        lines = [
            "━━━━━━━━━━ 結果 ━━━━━━━━━━",
            f"テーマ: {st.theme}",
            f"経過時間: {tt(self.et(end))}"
            + (" / 制限時間なし" if self.unlimited else f" / 制限 {st.duration_sec / 60:g}分")
            + (" (終了済み)" if st.finished else ""),
            "",
            f"進行役の介入: {len(st.interventions)}回"
            + (f"(内訳: {'、'.join(f'{ACTION_JA.get(k, k)} {v}' for k, v in acted.items())})" if acted else ""),
            f"進行役が考えたうえで見守った: {watched}回",
            f"ルールで見送った: {sum(blocked.values())}回"
            + (f"(内訳: {'、'.join(f'{GATE_JA.get(k, k)} {v}' for k, v in blocked.items())})" if blocked else ""),
            "",
            "発言量:",
        ]
        for x in stats:
            kind = "AI参加者" if x.is_bot else "人間"
            share = x.total_speech_sec / total * 100
            bar = "█" * round(share / 5)
            lines.append(f"  {(x.label or x.speaker_id):<6}({kind}) {x.turns:>2}回 {x.total_speech_sec:>4.0f}秒 {share:>3.0f}% {bar}")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines)
