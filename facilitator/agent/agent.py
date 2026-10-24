"""Observe → Understand → Decide → Act のエージェント本体(ルール + LLM のハイブリッド)。"""

from __future__ import annotations

import re
import time
from typing import Callable

from pydantic import ValidationError

from config import Settings
from facilitator.agent.prompts import OUTPUT_SCHEMA, build_system_prompt, build_user_prompt
from facilitator.llm.base import LLMClient, LLMError
from facilitator.logging import TickLogger
from facilitator.models import AI_SPEAKER, UNKNOWN, UNLIMITED_SEC, Action, AgentOutput, Decision, TickRecord
from facilitator.state import DiscussionState

# ルール層だけが発動させるアクション(LLM が自発的に選ぶことはできない)
RULE_ONLY = {Action.start_gd, Action.time_warning, Action.close_gd}
FREE_ACTIONS = [
    Action.no_intervention,
    Action.ask_question,
    Action.redirect_topic,
    Action.organize_issues,
    Action.compare_criteria,
    Action.invite_all,
]
INVITE_ALL_FALLBACK = "まだお話しされていない方も、ここまでの議論についてどう考えているか聞かせてもらえますか?"


def _mmss(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def _limit_sentences(text: str, n: int = 3) -> str:
    parts = [p for p in re.split(r"(?<=[。!?!?])", text.strip()) if p]
    return "".join(parts[:n])


class FacilitatorAgent:
    def __init__(
        self,
        state: DiscussionState,
        llm: LLMClient,
        settings: Settings,
        speak: Callable[[str], float] | None = None,
        logger: TickLogger | None = None,
        purpose: str = "job",
    ) -> None:
        self.system_prompt = build_system_prompt(purpose)
        self.state = state
        self.llm = llm
        self.settings = settings
        # speak(text) -> 発話にかかる秒数。None なら推定値を使う(テキストモード)
        self.speak = speak
        self.logger = logger
        self.history: list[TickRecord] = []
        self.last_tick_time: float | None = None

    # ================= Observe =================
    def _gate(self, now: float) -> tuple[str, list[Action], str | None]:
        """ルール層。(gate 名, 許可アクション, LLM への注意書き) を返す。許可が空なら LLM を呼ばない。"""
        st, s = self.state, self.settings
        if st.finished:
            return "blocked:finished", [], None
        if not st.started:
            return "forced:start_gd", [Action.start_gd], None
        if st.is_speaking(now):
            return "blocked:speaking", [], None

        remaining = st.remaining(now)
        if remaining <= 0:
            return "forced:close_gd", [Action.close_gd], "制限時間になりました。close_gd を選んでください。"
        since = st.since_last_intervention(now)
        for th in self._warn_thresholds():
            if remaining <= th and th not in st.warned_thresholds:
                if since is not None and since < s.forced_min_gap_sec:
                    return "blocked:forced_gap", [], None
                return (
                    "forced:time_warning",
                    [Action.time_warning],
                    f"残り約{int(remaining // 60)}分{int(remaining % 60)}秒です。time_warning を選んでください。",
                )

        if since is not None and since < s.cooldown_sec:
            return "blocked:cooldown", [], None
        if st.silence_sec(now) < s.min_silence_for_intervention_sec:
            return "blocked:short_pause", [], None

        allowed = list(FREE_ACTIONS)
        if st.speaker_info_usable() and st.names_confirmed:
            allowed.append(Action.invite_person)
        if s.enable_return_to_speaker:
            allowed.append(Action.return_to_speaker)
        return "llm", allowed, None

    def _warn_thresholds(self) -> list[float]:
        cap = self.state.duration_sec * self.settings.time_warning_max_fraction
        return sorted(min(th, cap) for th in self.settings.time_warnings_remaining_sec)

    def _context(self, now: float, trigger: str, allowed: list[Action], note: str | None) -> dict:
        st = self.state
        transcript = []
        for u in st.utterances[-self.settings.transcript_window :]:
            if u.end_time > now:
                continue
            if u.source == "ai":
                who = "AI(ファシリテーター)"
            elif u.source == "bot":
                who = f"{st.speaker_names.get(u.speaker_id, u.speaker_id)}(AI参加者・{u.speaker_id})"
            elif u.speaker_id == UNKNOWN or not self.settings.use_speaker_info:
                who = "不明"
            elif u.speaker_id in st.names_confirmed:
                who = f"{st.speaker_names[u.speaker_id]}({u.speaker_id})"
            else:
                who = u.speaker_id
            transcript.append(f"[{_mmss(st.elapsed(u.start_time))}] {who}: {u.text}")

        usable = st.speaker_info_usable()
        return {
            "theme": st.theme,
            "phase": st.phase,
            "elapsed_sec": st.elapsed(now),
            "remaining_sec": st.remaining(now),
            "duration_sec": st.duration_sec,
            "silence_sec": st.silence_sec(now),
            "trigger": trigger,
            "since_last_intervention": st.since_last_intervention(now),
            "intervention_count": len(st.interventions),
            "recent_interventions": [
                f"{_mmss(st.elapsed(iv.time))} {iv.action.value}" for iv in st.interventions[-4:]
            ],
            "prev_analysis": st.analysis.model_dump(),
            "speaker_reliability": st.speaker_reliability(),
            "speaker_stats": [x.model_dump() for x in st.speaker_stats(now)] if usable else None,
            "transcript": transcript,
            "allowed_actions": [a.value for a in allowed],
            "note": note,
        }

    # ================= Decide (LLM 後の検証) =================
    def _validate(self, d: Decision, allowed: list[Action], now: float) -> tuple[Decision, str | None]:
        st, s = self.state, self.settings
        if d.action not in allowed:
            if len(allowed) == 1:  # 強制イベントは必ず処理する
                return Decision(action=allowed[0], reasoning=d.reasoning, speech=d.speech, urgency=1.0), (
                    f"許可外の {d.action.value} を強制アクションに置換"
                )
            return Decision(reasoning=d.reasoning), f"許可外の {d.action.value} を no_intervention に置換"

        if d.action == Action.no_intervention:
            return d.model_copy(update={"speech": "", "target_speaker": None}), None

        if d.action == Action.invite_person:
            t = d.target_speaker
            reason = None
            if t not in st.names_confirmed:
                reason = "指名対象の名前が未確認"
            elif not st.speaker_info_usable():
                reason = "話者識別の信頼度が閾値未満"
            else:
                last = st.last_invited_at(t)
                if last is not None and now - last < s.person_cooldown_sec:
                    reason = "同一人物への指名クールダウン中"
            if reason:
                return Decision(
                    action=Action.invite_all, reasoning=d.reasoning, speech=INVITE_ALL_FALLBACK, urgency=d.urgency
                ), f"invite_person→invite_all({reason})"

        speech = _limit_sentences(d.speech)
        if not speech:
            if d.action in RULE_ONLY:
                speech = self._template(d.action, now)
            else:
                return Decision(reasoning=d.reasoning), "発話内容が空のため no_intervention"
        return d.model_copy(update={"speech": speech}), None

    def _template(self, action: Action, now: float) -> str:
        st = self.state
        if action == Action.start_gd:
            mins = int(st.duration_sec // 60)
            bots = "、".join(f"{st.speaker_names[b]}さん" for b in sorted(st.bot_ids))
            limit = (
                "今回は制限時間を設けませんので、グループとしての結論がまとまるまで議論してください。"
                if st.duration_sec >= UNLIMITED_SEC
                else f"制限時間は{mins}分で、最後にグループとしての結論をまとめてください。"
            )
            return (
                f"それでは、グループディスカッションを始めます。テーマは「{st.theme}」です。"
                + limit
                + (f"今回はAI参加者の{bots}も一緒に議論します。" if bots else "")
                + "私は進行役として、ときどき論点の整理や質問をします。では、どなたからでもどうぞ。"
            )
        if action == Action.time_warning:
            return f"残り約{max(1, round(st.remaining(now) / 60))}分です。そろそろ結論に向けてまとめていきましょう。"
        return "時間になりました。皆さんの結論を最後に確認させてください。お疲れさまでした。"

    # ================= Act =================
    def _act(self, d: Decision, now: float) -> None:
        st = self.state
        if d.action == Action.no_intervention:
            return
        remaining_before = st.remaining(now)
        st.record_intervention(now, d)
        if d.action == Action.time_warning:
            for th in self._warn_thresholds():
                if remaining_before <= th:
                    st.warned_thresholds.add(th)
        dur = self.speak(d.speech) if self.speak else len(d.speech) / self.settings.tts_chars_per_sec
        st.add_utterance(AI_SPEAKER, d.speech, now, now + dur, 1.0, source="ai")

    # ================= 1 サイクル =================
    def tick(self, now: float, trigger: str = "periodic", act: bool = True) -> TickRecord:
        """1サイクル実行する。act=False のときは判断だけ行い、発話は commit() で確定させる
        (実時間モードで、LLM の応答待ちの間に誰かが話し始めていないかを確認するため)。"""
        self.last_tick_time = now
        st = self.state
        gate, allowed, note = self._gate(now)
        raw_action, overridden = None, None
        t0 = time.monotonic()

        if not allowed:
            decision = Decision(reasoning=f"ルール層により見送り({gate})")
        elif gate == "forced:start_gd":
            decision = Decision(
                action=Action.start_gd, reasoning="GD開始", speech=self._template(Action.start_gd, now), urgency=1.0
            )
        else:
            ctx = self._context(now, trigger, allowed, note)
            try:
                raw = self.llm.generate_json(self.system_prompt, build_user_prompt(ctx), OUTPUT_SCHEMA, ctx)
                out = AgentOutput.model_validate(raw)
                raw_action = out.decision.action.value
                st.analysis = out.analysis
                st.advance_phase(out.analysis.phase, now)
                decision, overridden = self._validate(out.decision, allowed, now)
            except (LLMError, ValidationError) as e:
                # パース失敗・API 失敗時は「介入しない」に倒す。強制イベントだけはテンプレートで処理
                if len(allowed) == 1 and allowed[0] in RULE_ONLY:
                    a = allowed[0]
                    decision = Decision(action=a, reasoning="LLM失敗のためテンプレート", speech=self._template(a, now))
                else:
                    decision = Decision(reasoning="LLM失敗のため見送り")
                overridden = f"LLM失敗: {type(e).__name__}: {str(e)[:80]}"

        latency = time.monotonic() - t0
        record = TickRecord(
            time=now,
            trigger=trigger,
            phase=st.phase,
            remaining_sec=st.remaining(now),
            silence_sec=st.silence_sec(now),
            gate=gate,
            allowed_actions=[a.value for a in allowed],
            analysis=st.analysis,
            decision=decision,
            llm_raw_action=raw_action,
            overridden=overridden,
            latency_sec=round(latency, 2),
        )
        if act:
            self._act(decision, now)
        self.history.append(record)
        if self.logger:
            self.logger.log_tick(record)
        return record

    def commit(self, record: TickRecord, now: float, duration: float | None = None) -> None:
        """tick(act=False) の判断を実行済みとして状態に反映する。"""
        d = record.decision
        if d.action == Action.no_intervention:
            return
        speak, self.speak = self.speak, (lambda _t: duration) if duration is not None else None
        try:
            self._act(d, now)
        finally:
            self.speak = speak

    def cancel(self, record: TickRecord, reason: str) -> None:
        """発話直前に取りやめた判断を記録する(介入回数・クールダウンには数えない)。"""
        record.overridden = f"発話取りやめ: {reason}"

    # ================= 実時間ループ用のトリガー判定 =================
    def should_tick(self, now: float) -> str | None:
        """定期分析 + イベント駆動(沈黙・時間閾値)。発火すべきトリガー名を返す。"""
        st, s = self.state, self.settings
        if st.finished:
            return None
        if not st.started:
            return "start"
        if st.is_speaking(now):
            return None
        last = self.last_tick_time
        remaining = st.remaining(now)
        if remaining <= 0:
            return "time_up"
        since = st.since_last_intervention(now)
        if any(remaining <= th and th not in st.warned_thresholds for th in self._warn_thresholds()) and (
            since is None or since >= s.forced_min_gap_sec
        ):
            return "time_threshold"
        silence = st.silence_sec(now)
        if silence < s.min_silence_for_intervention_sec:
            return None  # 短い間では分析しない(定期分析のタイマーを無駄に消費しない)
        if silence >= s.silence_trigger_sec and (last is None or now - last >= s.silence_trigger_sec):
            return "silence"
        if last is None or now - last >= s.analysis_interval_sec:
            return "periodic"
        return None
