"""Observe → Understand → Decide → Act のエージェント本体(ルール + LLM のハイブリッド)。"""

from __future__ import annotations

import re
import time
from typing import Any, Callable

from pydantic import ValidationError

from config import Settings
from facilitator.agent.prompts import PHASE_BEHIND_NOTES, build_output_schema, build_system_prompt, build_user_prompt
from facilitator.discussion_types import CASUAL_FREE_THEME, get_discussion_type
from facilitator.llm.base import LLMClient, LLMError, generate
from facilitator.logging import TickLogger
from facilitator.models import AI_SPEAKER, UNKNOWN, UNLIMITED_SEC, Action, AgentOutput, Decision, TickRecord, Utterance
from facilitator.agent.content_check import content_flags
from facilitator.agent.name_check import strip_names
from facilitator.agent.request_detect import detect_request
from facilitator.policy import CONTENT_ACTIONS, ContentInitiative, InterventionPolicy, level_settings, levels, session_policy
from facilitator.state import DiscussionState

# ルール層だけが発動させるアクション(LLM が自発的に選ぶことはできない)
RULE_ONLY = {Action.start_gd, Action.time_warning, Action.close_gd}
# 進行支援のアクション(停滞時の段階で「まず進行支援」にあたる)
PROCESS_ACTIONS = {
    Action.ask_question, Action.organize_issues, Action.compare_criteria, Action.invite_all, Action.invite_person,
}
# ⑤控えめで、停滞を理由とするときに沈黙の長さを待たせるアクション
STALL_ACTIONS = {Action.ask_question, Action.organize_issues}
FREE_ACTIONS = [
    Action.no_intervention,
    Action.ask_question,
    Action.redirect_topic,
    Action.organize_issues,
    Action.compare_criteria,
    Action.invite_all,
]
REQUEST_IN_COOLDOWN_NOTE = (
    "直前の介入から間がないため、いまは進行役への依頼に応じることだけができます。"
    "「進行役への依頼」の発言が、あなたに考えるヒントやアイデアを求める依頼なら、許可された内容支援で応じてください。"
    "依頼でない場合や、内容支援では応じられない依頼の場合は no_intervention を選んでください(ほかの介入はしません)。"
)
INVITE_ALL_FALLBACK ="まだお話しされていない方も、ここまでの議論についてどう考えているか聞かせてもらえますか?"
REDIRECT_MAX_SENTENCES = 2  # 軌道修正は短い区切りで話すので短くする
# 発話の文数の上限(既定は3文)。ヒントは問いの形の短いものにする
MAX_SENTENCES = {Action.redirect_topic: REDIRECT_MAX_SENTENCES, Action.offer_hint: 2, Action.propose_idea: 3}


def _mmss(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def _limit_sentences(text: str, n: int = 3, keep_last: bool = False) -> str:
    """先頭から n 文に切り詰める。keep_last=True なら最後の1文(締めの告知)を残す。"""
    parts = [p for p in re.split(r"(?<=[。!?!?])", text.strip()) if p]
    if keep_last and len(parts) > n:
        return "".join(parts[: n - 1] + parts[-1:])
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
        discussion_type: str | None = None,  # 省略時は用途の既定(就活GD=課題解決型、一般=汎用型)
        clock: Any = None,  # 応答時間の計測に使う時計(テストでは疑似時計)
        style_notes: list[str] | None = None,  # 画面の「詳細」で選んだ進行の調整
        policy: InterventionPolicy | None = None,  # 介入ポリシー。省略時は用途の既定のプリセット
    ) -> None:
        self.clock = clock or time
        self.policy = policy or session_policy(purpose)
        # ①〜⑤の段階で変わる判断条件。標準の項目は settings の値をそのまま使う(_param)
        self.level_settings = level_settings(self.policy)
        self.dtype = get_discussion_type(discussion_type, purpose)
        scope = self.policy.content_scope.value
        self.system_prompt = build_system_prompt(
            purpose, self.dtype.key, style_notes, levels(self.policy), (scope, self.policy.content_initiative.value)
        )
        self.output_schema = build_output_schema(self.dtype, scope)
        self.state = state
        self.llm = llm
        self.settings = settings
        # speak(text) -> 発話にかかる秒数。None なら推定値を使う(テキストモード)
        self.speak = speak
        self.logger = logger
        self.history: list[TickRecord] = []
        self.last_tick_time: float | None = None
        self.skipped_due: str | None = None  # 直近の should_tick で、定期分析の時刻なのに見送った理由
        self.off_topic_streak = 0  # 直近の LLM 判断で、続けて脱線(off_topic)と判断した回数
        self.content_log: list[tuple[float, Action, str | None]] = []  # 内容支援を行った (時刻, アクション, 許可の理由)
        self.content_reason: str | None = None  # 直近の _gate で内容支援を許可した理由
        self.support_step = 0  # 停滞が続く間の支援の段階(0: なし / 1: 進行支援をした / 2: ヒントを出した / 3: 具体案を出した)
        self.pending_request: Utterance | None = None  # まだ応じていない、進行役への依頼の候補
        self._scanned_id = -1  # 依頼を探し終えた発言 id

    def _param(self, key: str) -> Any:
        """介入ポリシーの段階で変わる判断条件。段階が上書きしていなければ settings の値。"""
        return self.level_settings.get(key, getattr(self.settings, key))

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

        in_cooldown = since is not None and since < s.cooldown_sec
        if in_cooldown and self.pending_request is None:
            return "blocked:cooldown", [], None
        if st.silence_sec(now) < s.min_silence_for_intervention_sec:
            return "blocked:short_pause", [], None
        if in_cooldown:
            # 進行役への依頼にだけは、クールダウン中でも応じる(1つの候補につき1回。発話中・2秒の間の条件は変えない)。
            # 許可するのは見送りと範囲の中の内容支援だけ。依頼と無関係な介入(質問・脱線修正・整理・発言促進)は
            # クールダウンを越えられない。LLM が依頼でないと判断すれば見送り、候補は消えて通常のクールダウンに戻る
            content, self.content_reason = self._content_actions(now)
            if not content:  # 内容支援が無効なら依頼を検知しないので通常は起きない(念のため)
                return "blocked:cooldown", [], None
            return "llm:request", [Action.no_intervention, *content], REQUEST_IN_COOLDOWN_NOTE

        allowed = list(FREE_ACTIONS)
        if self.off_topic_streak < self._param("drift_confirm_ticks"):
            allowed.remove(Action.redirect_topic)
        if st.speaker_info_usable() and st.names_confirmed:
            allowed.append(Action.invite_person)
        if s.enable_return_to_speaker:
            allowed.append(Action.return_to_speaker)
        content, self.content_reason = self._content_actions(now)
        allowed += content
        return "llm", allowed, None

    # ================= 内容支援(介入ポリシー⑥) =================
    def _content_actions(self, now: float) -> tuple[list[Action], str | None]:
        """いま許可する内容支援のアクションと、その理由。範囲(⑥-A)の外のアクションはどの理由でも許可しない。
        - request: 進行役への依頼があれば、範囲の中のアクションをすべて(自発性の設定によらない)
        - stalled: 自発性が「停滞時のみ」以上で、前回の判断が停滞なら、段階を踏む。進行支援をした後なら offer_hint、
          ヒントを出した後なら propose_idea(具体案まで のとき)
        - proactive: 自発性が「積極的」なら、停滞していなくても offer_hint だけ(具体案は依頼か停滞の段階を踏んだときだけ)
        自発的な支援(stalled / proactive)には、内容支援どうしの間隔と、1回の GD での上限をかける。"""
        p = self.policy
        if not p.content_support:
            return [], None
        if self.pending_request is not None:
            return p.content_actions, "request"
        if p.content_initiative == ContentInitiative.none or not self._spontaneous_ok(now):
            return [], None
        if self.state.analysis.health == "stalled":
            ladder = [a for a, step in ((Action.offer_hint, 1), (Action.propose_idea, 2)) if self.support_step >= step]
            ladder = [a for a in ladder if a in p.content_actions]
            if ladder:
                return ladder, "stalled"
        if p.content_initiative == ContentInitiative.proactive:
            return [Action.offer_hint], "proactive"
        return [], None

    def has_pending_request(self, now: float) -> bool:
        """応じていない依頼の候補があるか(AI 参加者が先に答えないようにするため)。古い候補はここでも見送るので、
        参加者が話し続けて判断が始まらない場合も、待ちは request_expire_sec で必ず解ける。"""
        r = self.pending_request
        if r is not None and now - r.end_time > self.settings.request_expire_sec:
            self.pending_request = None
        return self.pending_request is not None

    def _scan_requests(self, now: float) -> None:
        """新しい人間の発言から、進行役への依頼の候補を探す(内容支援が有効なときだけ)。古い依頼は見送る。"""
        st = self.state
        if not self.policy.content_support:
            return
        bots = [st.speaker_names.get(b, "") for b in st.bot_ids]
        for u in st.utterances:
            if u.id <= self._scanned_id or u.end_time > now:
                continue
            self._scanned_id = max(self._scanned_id, u.id)
            if u.source == "human" and detect_request(u.text, bots):
                self.pending_request = u
        self.has_pending_request(now)  # 古い候補は見送る

    def _spontaneous_ok(self, now: float) -> bool:
        spont = [t for t, _a, reason in self.content_log if reason != "request"]
        if len(spont) >= self.settings.content_max_spontaneous:
            return False
        last = max((t for t, _a, _r in self.content_log), default=None)
        return last is None or now - last >= self.settings.content_cooldown_sec

    def _warn_thresholds(self) -> list[float]:
        cap = self.state.duration_sec * self.settings.time_warning_max_fraction
        return sorted(min(th, cap) for th in self.settings.time_warnings_remaining_sec)

    def forced_event_pending(self, now: float) -> bool:
        """時間切れ、または未告知の時間閾値に達しているか(待機中の軌道修正より強制イベントを優先するため)。"""
        st = self.state
        remaining = st.remaining(now)
        return remaining <= 0 or any(remaining <= th and th not in st.warned_thresholds for th in self._warn_thresholds())

    def phase_status(self, now: float) -> tuple[str, int | None, bool]:
        """(目標フェーズ, 目標 - 実際のフェーズの段階差, 移行を促してよい程度に遅れているか)。
        実際のフェーズは前回の LLM 判断のままなので、遅れは LLM への判断材料にとどめ、ルールの介入条件には使わない。"""
        st = self.state
        frac = st.elapsed_fraction(now)
        target = self.dtype.target_phase(frac).key
        observed = self.dtype.index(st.observed_phase)
        if observed is None:
            return target, None, False
        gap = self.dtype.index(target) - observed
        behind = gap >= self._param("phase_behind_gap") or (
            gap >= self._param("phase_behind_late_gap") and frac >= self._param("phase_behind_late_fraction")
        )
        return target, gap, behind

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
        target, gap, behind = self.phase_status(now)
        observed = self.dtype.get(st.observed_phase)
        return {
            "theme": st.theme,
            "observed_phase": st.observed_phase,
            "observed_phase_label": observed.label if observed else "",
            "observed_age_sec": 0.0 if st.phase_observed_at is None else now - st.phase_observed_at,
            "target_phase": target,
            "target_phase_label": self.dtype.get(target).label,
            "phase_gap": gap,
            "phase_behind": behind,
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
            "phase_behind_note": PHASE_BEHIND_NOTES[self.policy.progress.value],
            "quiet_member_sec": self._param("quiet_member_sec"),
            "quiet_members": self._quiet_members(now) if usable else [],
            "stall_silence_sec": self._param("silence_trigger_sec"),  # モック LLM が停滞とみなす沈黙の長さ
            "request": self._request_line() if self.pending_request is not None else None,
        }

    def _request_line(self) -> str:
        st, u = self.state, self.pending_request
        who = st.speaker_names[u.speaker_id] if u.speaker_id in st.names_confirmed else u.speaker_id
        return f"[{_mmss(st.elapsed(u.start_time))}] {who}: {u.text}"

    def _quiet_members(self, now: float) -> list[str]:
        """④: quiet_member_sec 以上発言していない人間の参加者(AI 参加者は含めない)。0 なら示さない。
        一度も発言していない人は、GD の経過が quiet_member_sec 以上のときだけ含める。"""
        th, st = self._param("quiet_member_sec"), self.state
        if th <= 0:
            return []
        out = []
        for x in st.speaker_stats(now):
            if x.is_bot:
                continue
            since = x.since_last_spoke_sec
            if since is None and st.elapsed(now) < th or since is not None and since < th:
                continue
            who = f"{x.label}({x.speaker_id})" if x.label_confirmed else x.speaker_id
            out.append(f"{who}: " + ("一度も発言なし" if since is None else f"最後の発言から{since:.0f}秒"))
        return out

    # ================= Decide (LLM 後の検証) =================
    def _validate(self, d: Decision, allowed: list[Action], now: float) -> tuple[Decision, str | None]:
        st, s = self.state, self.settings
        if d.action not in allowed:
            if len(allowed) == 1:  # 強制イベントは必ず処理する
                return Decision(action=allowed[0], reasoning=d.reasoning, speech=d.speech, urgency=1.0), (
                    f"許可外の {d.action.value} を強制アクションに置換"
                )
            # 個人指名が許可されないのが「誰を指名すべきか確かでない」ためなら、全体への呼びかけで代替を検討する
            reason = self._no_naming_reason()
            if d.action == Action.invite_person and reason and Action.invite_all in allowed:
                return self._invite_all_instead(d, reason, now)
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
                return self._invite_all_instead(d, reason, now)

        # 終了は最後の一文で終了を告げるので、切り詰めても締めの告知を残す
        n = MAX_SENTENCES.get(d.action, 3)
        speech = _limit_sentences(d.speech, n, keep_last=d.action == Action.close_gd)
        note = f"発話を{n}文に短縮" if speech and speech != d.speech.strip() else None
        if not speech:
            if d.action in RULE_ONLY:
                speech = self._template(d.action, now)
            else:
                return Decision(reasoning=d.reasoning), "発話内容が空のため no_intervention"
        # 個人名で呼びかけてよいのは invite_person の指名対象だけ(agent/name_check.py)
        keep = st.speaker_names.get(d.target_speaker) if d.action == Action.invite_person else None
        fixed, left = strip_names(speech, list(st.speaker_names.values()), keep)
        if left:
            if d.action == Action.invite_person:
                return self._invite_all_instead(d, "発話に指名対象以外の個人名", now)
            if d.action in RULE_ONLY:
                return d.model_copy(update={"speech": self._template(d.action, now)}), "個人名を含むため定型文に置換"
            return Decision(reasoning=d.reasoning), f"個人名({'・'.join(left)})を含むため no_intervention"
        if fixed != speech:
            note = "; ".join(x for x in (note, "個人名を言い換え") if x)
            speech = fixed
        return d.model_copy(update={"speech": speech}), note

    def _hold_short_stall(
        self, d: Decision, overridden: str | None, analysis: Any, behind: bool, request: Any, now: float
    ) -> tuple[Decision, str | None]:
        """⑤停滞への介入(控えめ): 停滞を理由とする介入は、沈黙が stall_min_silence_sec に満たなければ見送る。
        対象は LLM 自身が停滞(stalled)と判断して選んだ ask_question / organize_issues と、停滞を理由に許可した内容支援だけ。
        膠着(conflicting)などほかの状態での整理・質問、①の遅れの判定が立っているときの進行の問いかけ、依頼への応答は対象外。
        「うーん」と言い続けるだけの停滞と、沈黙による停滞は LLM の出力からは区別できないので、どちらも沈黙の長さで待つ。"""
        th = self._param("stall_min_silence_sec")
        if th <= 0 or request is not None or behind or analysis.health != "stalled":
            return d, overridden
        stall_content = d.action in CONTENT_ACTIONS and self.content_reason == "stalled"
        if d.action not in STALL_ACTIONS and not stall_content:
            return d, overridden
        silence = self.state.silence_sec(now)
        if silence >= th:
            return d, overridden
        return Decision(reasoning=d.reasoning), (
            f"⑤停滞への介入(控えめ): 沈黙{silence:.0f}秒が{th:.0f}秒未満のため、停滞への {d.action.value} を見送り"
        )

    def _no_naming_reason(self) -> str | None:
        """個人指名を許可しない理由(話者の特定に関わるもの)。指名できる状態なら None。"""
        st, s = self.state, self.settings
        if not s.use_speaker_info:
            return "話者情報を使わない設定"
        if not st.speaker_info_usable():
            return "話者識別の信頼度が閾値未満"
        if not st.names_confirmed:
            return "名前が確認済みの参加者がいない"
        return None

    def _invite_all_instead(self, d: Decision, reason: str, now: float) -> tuple[Decision, str]:
        """個人指名の代わりに全体へ呼びかける(目的は発言機会の確保なので全体向けでも果たせる)。
        発言は個人名を含まない定型文にする。直近に全体への呼びかけをしていれば、繰り返さず見送る。"""
        last = next((iv.time for iv in reversed(self.state.interventions) if iv.action == Action.invite_all), None)
        if last is not None and now - last < self.settings.person_cooldown_sec:
            return Decision(reasoning=d.reasoning), f"invite_person→no_intervention({reason}。直近に全体への呼びかけ済み)"
        return Decision(
            action=Action.invite_all, reasoning=d.reasoning, speech=INVITE_ALL_FALLBACK, urgency=d.urgency
        ), f"invite_person→invite_all({reason})"

    def _template(self, action: Action, now: float) -> str:
        st = self.state
        if action == Action.start_gd:  # テーマと時間だけを短く伝える
            limit = "時間の制限はありません" if st.duration_sec >= UNLIMITED_SEC else f"時間は{int(st.duration_sec // 60)}分です"
            if self.dtype.key == "casual" and st.theme == CASUAL_FREE_THEME:  # 雑談でテーマを決めなかったとき
                return f"今日は自由に話しましょう。{limit}。始めてください。"
            return f"テーマは「{st.theme}」、{limit}。始めてください。"
        casual = self.dtype.key == "casual"  # 雑談型は結論を求めない
        if action == Action.time_warning:
            mins = max(1, round(st.remaining(now) / 60))
            return f"残り約{mins}分です。" + ("話してみて印象に残ったことも、よければ聞かせてください。" if casual else "そろそろ結論に向けてまとめていきましょう。")
        if casual:
            return "時間になりました。今日はここまでにしましょう。ありがとうございました。"
        return "時間になりました。皆さんの結論を最後に確認させてください。お疲れさまでした。"

    # ================= Act =================
    def _act(self, d: Decision, now: float, content_reason: str | None = None) -> None:
        st = self.state
        if d.action == Action.no_intervention:
            return
        self._record(d, now, content_reason)
        dur = self.speak(d.speech) if self.speak else len(d.speech) / self.settings.tts_chars_per_sec
        st.add_utterance(AI_SPEAKER, d.speech, now, now + dur, 1.0, source="ai")

    def _record(self, d: Decision, now: float, content_reason: str | None = None) -> None:
        st = self.state
        remaining_before = st.remaining(now)
        st.record_intervention(now, d)
        if d.action in CONTENT_ACTIONS:
            self.content_log.append((now, d.action, content_reason))
        if st.analysis.health == "stalled":  # 停滞が続く間の支援の段階: 進行支援 → ヒント → 具体案
            step = {Action.offer_hint: 2, Action.propose_idea: 3}.get(d.action, 1 if d.action in PROCESS_ACTIONS else 0)
            self.support_step = max(self.support_step, step)
        if d.action == Action.time_warning:
            for th in self._warn_thresholds():
                if remaining_before <= th:
                    st.warned_thresholds.add(th)

    # ================= 1 サイクル =================
    def tick(
        self,
        now: float,
        trigger: str = "periodic",
        act: bool = True,
        on_partial: Callable[[str, list[Action], str], None] | None = None,
    ) -> TickRecord:
        """1サイクル実行する。act=False のときは判断だけ行い、発話は commit() で確定させる
        (実時間モードで、LLM の応答待ちの間に誰かが話し始めていないかを確認するため)。
        on_partial を渡すと、生成途中の JSON テキストを (テキスト, 許可アクション, ゲート) で受け取れる
        (発言を一文できるたびに話し始めるため)。"""
        self.last_tick_time = now
        st = self.state
        self._scan_requests(now)
        request = self.pending_request
        gate, allowed, note = self._gate(now)
        raw_action, overridden = None, None
        _, gap, behind = self.phase_status(now)  # 判断前(前回判断のフェーズ)の値を記録する
        llm_called, usage = False, None
        last_utt = max((u.id for u in st.utterances if u.end_time <= now), default=None)
        t0 = self.clock.monotonic()

        if not allowed:
            decision = Decision(reasoning=f"ルール層により見送り({gate})")
        elif gate == "forced:start_gd":
            decision = Decision(
                action=Action.start_gd, reasoning="GD開始", speech=self._template(Action.start_gd, now), urgency=1.0
            )
        else:
            ctx = self._context(now, trigger, allowed, note)
            llm_called = True
            try:
                try:
                    partial = (lambda acc: on_partial(acc, allowed, gate)) if on_partial else None
                    raw = generate(self.llm, self.system_prompt, build_user_prompt(ctx), self.output_schema, ctx, partial)
                finally:
                    usage = getattr(self.llm, "last_usage", None)
                out = AgentOutput.model_validate(raw)
                raw_action = out.decision.action.value
                st.analysis = out.analysis
                self.off_topic_streak = self.off_topic_streak + 1 if out.analysis.health == "off_topic" else 0
                if out.analysis.health != "stalled":
                    self.support_step = 0  # 停滞が終わったら、支援の段階を最初に戻す
                st.update_phase(out.analysis.phase, self.dtype.keys, now)
                decision, overridden = self._validate(out.decision, allowed, now)
                decision, overridden = self._hold_short_stall(decision, overridden, out.analysis, behind, request, now)
            except (LLMError, ValidationError) as e:
                # パース失敗・API 失敗時は「介入しない」に倒す。強制イベントだけはテンプレートで処理
                if len(allowed) == 1 and allowed[0] in RULE_ONLY:
                    a = allowed[0]
                    decision = Decision(action=a, reasoning="LLM失敗のためテンプレート", speech=self._template(a, now))
                else:
                    decision = Decision(reasoning="LLM失敗のため見送り")
                overridden = f"LLM失敗: {type(e).__name__}: {str(e)[:80]}"

        if llm_called and request is not None:
            self.pending_request = None  # 1つの依頼には1回だけ判断させる(依頼でないと LLM が判断した場合も)
        latency = self.clock.monotonic() - t0
        record = TickRecord(
            time=now,
            trigger=trigger,
            observed_phase=st.observed_phase,
            target_phase=self.phase_status(now)[0],
            phase_gap=gap,
            phase_behind=behind,
            remaining_sec=st.remaining(now),
            silence_sec=st.silence_sec(now),
            gate=gate,
            allowed_actions=[a.value for a in allowed],
            analysis=st.analysis,
            decision=decision,
            llm_raw_action=raw_action,
            overridden=overridden,
            latency_sec=round(latency, 2),
            llm_called=llm_called,
            llm_usage=usage,
            context_last_utt_id=last_utt,
            content_reason=self.content_reason if any(a in CONTENT_ACTIONS for a in allowed) else None,
            content_flags=content_flags(decision.action, decision.speech),
        )
        if act:
            self._act(decision, now, record.content_reason)
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
            self._act(d, now, record.content_reason)
        finally:
            self.speak = speak

    def commit_spoken(self, record: TickRecord, now: float, action: Action, spoken: str) -> None:
        """生成しながら話した発言を、介入として記録する(発言は議論ログに追加済み)。
        検証で判断が変わった場合や途中で LLM が失敗した場合も、実際に話した内容と行動を記録に残す。"""
        d = record.decision
        if d.action != action or d.speech != spoken:
            if d.action != action:
                record.overridden = f"生成しながら {action.value} として話した(最終判断: {d.action.value})"
            d = Decision(action=action, reasoning=d.reasoning, speech=spoken, urgency=d.urgency)
            record.decision = d
            record.content_flags = content_flags(action, spoken)  # 実際に話した文で検査し直す
        self._record(d, now, record.content_reason)

    def record_spoken(self, now: float, action: Action, spoken: str) -> None:
        """判断の途中で(LLM 以外の理由で)失敗したが、生成しながら一部を話してしまった発言を介入として記録する
        (記録しないとクールダウンが効かず、すぐにまた話してしまうため)。"""
        reason = self.content_reason if action in CONTENT_ACTIONS else None
        self._record(Decision(action=action, reasoning="判断の途中で失敗(話した分を記録)", speech=spoken), now, reason)

    # ---------- 生成しながら話すときの検証(全文を待つ場合の _validate / _hold_short_stall と同じ方針) ----------
    def live_ok(self, action: Action, now: float) -> bool:
        """生成しながら話し始めてよいか。⑤控えめの見送り(沈黙が基準未満の停滞への介入)の対象になりうる行動は、
        全文を待って検証する(話し始めてから見送りが決まらないように)。"""
        th = self._param("stall_min_silence_sec")
        if th > 0 and (action in STALL_ACTIONS or action in CONTENT_ACTIONS) and self.state.silence_sec(now) < th:
            return False
        return True

    def live_sentence(self, action: Action, sentence: str, index: int) -> str | None:
        """生成しながら話す index 番目(0 始まり)の一文。文数の上限を超えるか、言い換えられない個人名を含めば None
        (その文から先は話さない)。呼びかけ・「〇〇さんの」は全文の場合と同じく言い換える。"""
        if index >= MAX_SENTENCES.get(action, 3):
            return None
        fixed, left = strip_names(sentence, list(self.state.speaker_names.values()))
        return None if left else fixed

    def cancel(self, record: TickRecord, reason: str) -> None:
        """発話直前に取りやめた判断を記録する(介入回数・クールダウンには数えない)。"""
        record.overridden = f"発話取りやめ: {reason}"

    # ================= 実時間ループ用のトリガー判定 =================
    def should_tick(self, now: float) -> str | None:
        """定期分析 + イベント駆動(沈黙・時間閾値)。発火すべきトリガー名を返す。
        定期分析の時刻なのに見送った場合は、その理由を skipped_due に残す(計測用。判定には使わない)。"""
        st, s = self.state, self.settings
        self.skipped_due = None
        if st.finished:
            return None
        if not st.started:
            return "start"
        last = self.last_tick_time
        due = last is None or now - last >= s.analysis_interval_sec
        if st.is_speaking(now):
            self.skipped_due = "speaking" if due else None
            return None
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
            self.skipped_due = "short_silence" if due else None
            return None  # 短い間では分析しない(定期分析のタイマーを無駄に消費しない)
        self._scan_requests(now)
        if self.pending_request is not None:
            return "request"  # 進行役への依頼(内容支援が有効なときだけ検出する)
        trigger_sec = self._param("silence_trigger_sec")
        if silence >= trigger_sec and (last is None or now - last >= trigger_sec):
            return "silence"
        if last is None or now - last >= s.analysis_interval_sec:
            return "periodic"
        return None
