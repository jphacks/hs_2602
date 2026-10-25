"""雑談モードの AI 司会(最初の画面で「雑談」を選んだとき = 議論タイプが雑談型のときに使う)。

役割: 会話の参加者ではなく、初対面などで少し気まずい人たちの間に入り、話題を出したり会話を回したりして、
人間どうしの交流をつくる司会者。最初は AI が積極的に会話を生み出し、参加者どうしの会話が始まったら自然に引く。
GD・GW のように議論を進めたり結論を出したりはしない。

構成: GD・GW の進行役(FacilitatorAgent)を継承する。engine は進行役の窓口(tick / should_tick / commit /
生成しながら話す処理の検証 / 依頼の検知 / 記録)をそのまま使うので、engine と GD・GW の進行役のコードを変えずに
差し替えられる。雑談専用の判断は、このクラスの中だけで上書きする(GD・GW の進行役、プロンプト、介入ポリシーには
手を入れない。上書きのための入口 _cooldown_sec / stale_reason だけを進行役に置いている)。

第2段階(現在): 会話の状態と介入のタイミング。
- AI 主導期(lead): 参加者が話すきっかけを積極的につくる。AI が話した後、
  - 誰も答えなければ、chat_no_answer_sec の沈黙を待ってから判断する(答えのない AI の発言が続くほど待ちを伸ばす)
  - 誰かが答えたら、ほかの人が反応する chat_reaction_wait_sec を待ってから判断する(その間に誰かが話せば数え直す)
- 自走期(free): 人間どうしの会話を優先し、自分からは判断しない(LLM も呼ばない)。依頼・時間告知・終了には応じる
- 主導期 → 自走期: AI の発言の後、人間どうしのつながった発言(別の人が chat_link_gap_sec 以内に続けて話す)が
  chat_free_min_links 回あるか、1回あって直近の LLM の判断も「会話が進んでいる」とき(ルールで決めきれない分を LLM で補う)
- 自走期 → 主導期: chat_lull_silence_sec の沈黙が続き、自走期に入ってから chat_free_min_sec たったときだけ
  (一時的な沈黙で頻繁に切り替えない)
- 判断を始めた後に人間の新しい発言があれば、その判断の発話はしない(stale_reason)
- 同じ状況(新しい発言がない)では、次の待ち時間がたつまで判断し直さない(待つことも選べる。LLM を呼びすぎない)
数値はすべて仮の値(config.Settings の chat_*)。

第3段階: 雑談専用のプロンプト・行動・開始の話題。
- プロンプトと出力スキーマは雑談専用(facilitator/chat/prompts.py)。GD・GW のプロンプトは使わない
- 行動: offer_topic(話題を出す)/ ask_question(直前の発言を深掘り)/ invite_all(ほかの人に広げる)/
  change_topic(話題を切り替える)/ no_intervention。invite_person は GD・GW と同じ安全条件のときだけ許可する
  (ルール層の許可と、名前の確認・信頼度・同一人物の間隔の検証は進行役のものをそのまま使う)。内容支援(offer_hint など)は使わない
- 依頼(「AIさん、何か話題ありますか」など)には、内容支援の設定によらず応じる。間がないときは話題の提供・転換と全体への返しだけ
- 開始: あいさつ + 用意した候補の話題(facilitator/chat/topics.py。テーマがあればテーマに沿ったもの)。行動は start_gd のまま
  (engine の開始の扱いを変えない)なので、開始の発言も AI の発言として第2段階の待ち(答えがなければ待つ)に入る
- 発話は短く(どの行動も2文まで)
話題の履歴の管理・話題転換の細かな条件・重複の防止は第4段階(いまは出した話題を記録し、候補から除くだけ)。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from facilitator.agent.agent import FacilitatorAgent
from facilitator.chat.prompts import CHAT_REQUEST_NOTE, CHAT_SYSTEM_PROMPT, build_chat_output_schema, build_chat_user_prompt
from facilitator.chat.topics import is_free, opening_speech, opening_topic, topic_candidates
from facilitator.discussion_types import CASUAL
from facilitator.models import Action, Decision, TickRecord, Utterance

LEAD, FREE = "lead", "free"  # AI 主導期 / 自走期
FLOWING = ("progressing",)  # LLM の analysis.health のうち「会話が進んでいる」(converging は話題が一段落した)
# 雑談の行動(許可の順)。invite_person は安全条件を満たすときだけ加える
CHAT_ACTIONS = [Action.no_intervention, Action.offer_topic, Action.ask_question, Action.invite_all, Action.change_topic]
# 間がない(chat_min_gap_sec)ときの依頼への応答: 話題の提供・転換と、意見を求められたときの全体への返しだけ
REQUEST_ACTIONS = [Action.no_intervention, Action.offer_topic, Action.change_topic, Action.invite_all]
TOPIC_ACTIONS = {Action.start_gd, Action.offer_topic, Action.change_topic}  # 話題を出す行動(出した話題を記録する)
CHAT_MAX_SENTENCES = 2  # 雑談の発話は短く(開始・時間告知・終了の定型文は除く)
CHAT_INVITE_ALL = "ほかの皆さんはどうですか?よければ聞かせてください。"
SCHEMA_ACTIONS = CHAT_ACTIONS + [Action.invite_person, Action.time_warning, Action.close_gd]


@dataclass
class ChatState:
    """ある時点の会話の状態(発言の記録から計算する)。"""

    last_ai: Utterance | None  # 直近の AI(司会)の発言
    since_ai: float | None  # AI が話し終えてからの秒数
    humans_after_ai: list[Utterance]  # AI の発言の後の人間の発言(AI が話し終えた後に話し終えたもの。話し始めの順)
    links_after_ai: int  # つながった発言の回数(別の人が短い間隔で続けて話した)。AI の発言と、主導期に入った時点の遅い方より後
    unanswered: int  # 人間の発言を挟まずに続いた、直近の AI の発言の数(直近の AI の発言の後に答えがあれば 0)


class ChatHostAgent(FacilitatorAgent):
    """雑談の司会(人間どうしの交流をつくる。会話の参加者にはならない)。"""

    invite_all_fallback = CHAT_INVITE_ALL

    def __init__(self, *args: Any, rng: random.Random | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.system_prompt = CHAT_SYSTEM_PROMPT  # GD・GW のプロンプト(介入ポリシーの文)は使わない
        self.output_schema = build_chat_output_schema(SCHEMA_ACTIONS)
        self.rng = rng or random.Random()
        self.opening: str | None = None  # 開始の話題(最初の判断で候補から選ぶ)
        self.topics_used: list[str] = []  # AI が出した話題(発言)。第4段階で話題の管理に使う
        self.chat_phase = LEAD
        self.chat_phase_since: float | None = None  # いまの時期に入った時刻
        self.chat_log: list[tuple[float, str, str]] = []  # 時期の切り替え (時刻, 新しい時期, 理由)

    # ================= 雑談専用の値 =================
    def _cooldown_sec(self) -> float:
        return self.settings.chat_min_gap_sec

    def _max_sentences(self, action: Action) -> int:
        return CHAT_MAX_SENTENCES

    def _detects_requests(self) -> bool:
        return True  # 話題を求める依頼には、内容支援の設定によらず応じる

    def _content_actions(self, now: float) -> tuple[list[Action], str | None]:
        return [], None  # 内容支援(offer_hint / propose_idea)は使わない。話題は offer_topic / change_topic で出す

    # ================= 開始の話題・発話 =================
    def _template(self, action: Action, now: float) -> str:
        if action == Action.start_gd:  # あいさつ + 用意した候補の話題(時間や進め方の説明はしない)
            if self.opening is None:
                self.opening = opening_topic(self.state.theme, self.rng)
            return opening_speech(self.state.theme, self.opening)
        return super()._template(action, now)  # 時間告知・終了は雑談型の定型文(結論を求めない)

    def _record(self, d: Decision, now: float, content_reason: str | None = None) -> None:
        super()._record(d, now, content_reason)
        if d.action in TOPIC_ACTIONS:
            self.topics_used.append(self.opening if d.action == Action.start_gd and self.opening else d.speech)

    def _context(self, now: float, trigger: str, allowed: list[Action], note: str | None) -> dict:
        ctx = super()._context(now, trigger, allowed, note)
        ctx["transcript"] = [line.replace("AI(ファシリテーター)", "AI(司会)") for line in ctx["transcript"]]
        cs = self.chat_state(now)
        ctx["chat"] = {
            "phase": self.chat_phase,
            "free_theme": is_free(self.state.theme),
            "since_ai": cs.since_ai,
            "answers": len(cs.humans_after_ai),
            "answerers": len({u.speaker_id for u in cs.humans_after_ai}),
            "unanswered": cs.unanswered,
            "topics_used": list(self.topics_used),
            "topic_candidates": topic_candidates(self.state.theme, self.topics_used)[:3],
        }
        return ctx

    def _user_prompt(self, ctx: dict) -> str:
        return build_chat_user_prompt(ctx)

    # ================= 会話の状態 =================
    def chat_state(self, now: float) -> ChatState:
        ended = [u for u in self.state.utterances if u.end_time <= now]
        ai = [u for u in ended if u.source == "ai"]
        last_ai = max(ai, key=lambda u: u.end_time) if ai else None
        after = last_ai.end_time if last_ai else float("-inf")
        # AI が話し終えた後に話し終えた発言を、AI の発言への応答とみなす(AI の言い終わりに重ねて答えた場合も含める)
        humans = sorted((u for u in ended if u.source == "human" and u.end_time > after), key=lambda u: u.start_time)
        # つながりは、主導期に戻った後の発言だけで数える(途切れる前のやり取りで、すぐ自走期に戻らないように)
        lead_since = self.chat_phase_since if self.chat_phase == LEAD and self.chat_phase_since is not None else after
        recent = [u for u in humans if u.end_time > max(after, lead_since)]
        gap = self.settings.chat_link_gap_sec
        links = sum(
            1 for a, b in zip(recent, recent[1:])
            if a.speaker_id != b.speaker_id and b.start_time - a.end_time <= gap
        )
        unanswered = 0
        for u in sorted(ended, key=lambda u: u.start_time, reverse=True):
            if u.source == "ai":
                unanswered += 1
            elif u.source == "human":
                break
        return ChatState(last_ai, None if last_ai is None else now - last_ai.end_time, humans, links, unanswered)

    def _llm_says_flowing(self, cs: ChatState) -> bool:
        """AI の発言の後の LLM の判断で、会話が進んでいるとされたか(ルールで決めきれないつながりを補う)。"""
        after = cs.last_ai.end_time if cs.last_ai else float("-inf")
        last = next((r for r in reversed(self.history) if r.llm_called and r.llm_raw_action is not None), None)
        return last is not None and last.time >= after and last.analysis.health in FLOWING

    def _set_phase(self, phase: str, now: float, why: str) -> None:
        if phase != self.chat_phase:
            self.chat_phase, self.chat_phase_since = phase, now
            self.chat_log.append((round(now, 2), phase, why))

    def update_phase(self, now: float, cs: ChatState | None = None) -> ChatState:
        """時期を更新して、その時点の会話の状態を返す。"""
        cs = cs or self.chat_state(now)
        s = self.settings
        if self.chat_phase_since is None:
            self.chat_phase_since = now
        if self.chat_phase == LEAD:
            if cs.links_after_ai >= s.chat_free_min_links:
                self._set_phase(FREE, now, f"人間どうしのつながった発言が{cs.links_after_ai}回")
            elif cs.links_after_ai >= 1 and self._llm_says_flowing(cs):
                self._set_phase(FREE, now, "つながった発言があり、LLM も会話が進んでいると判断")
        else:
            silence = self.state.silence_sec(now)
            if silence >= s.chat_lull_silence_sec and now - self.chat_phase_since >= s.chat_free_min_sec:
                self._set_phase(LEAD, now, f"会話が途切れた(沈黙{silence:.0f}秒)")
                cs = self.chat_state(now)  # 主導期に戻った時点からつながりを数え直す
        return cs

    def _need_silence(self, cs: ChatState) -> float:
        """主導期に、判断を始めるまでに必要な沈黙の長さ。"""
        s = self.settings
        if cs.humans_after_ai:  # 誰かが答えた: ほかの人が反応するのを待つ
            return s.chat_reaction_wait_sec
        backoff = min(max(cs.unanswered - 1, 0), s.chat_no_answer_backoff_max)  # 答えのない AI の発言が続くほど長く待つ
        return s.chat_no_answer_sec * (1 + backoff)

    def chat_wait_reason(self, now: float, cs: ChatState) -> str | None:
        """自分から判断を始めない理由(None なら判断してよい)。依頼・強制イベントはこの前に処理される。"""
        if self.chat_phase == FREE:
            return "chat_free"
        if cs.last_ai is None:
            return None
        if self.state.silence_sec(now) < self._need_silence(cs):
            return "chat_wait_reaction" if cs.humans_after_ai else "chat_wait_answer"
        return None

    # ================= トリガーとルール層 =================
    def should_tick(self, now: float) -> str | None:
        """雑談のトリガー。強制イベント・依頼は GD・GW と同じ。それ以外は会話の状態から決める
        (GD・GW の定期分析(20秒)と沈黙トリガーは使わない)。"""
        st, s = self.state, self.settings
        self.skipped_due = None
        if st.finished:
            return None
        if not st.started:
            return "start"
        cs = self.update_phase(now)  # 時期は毎回更新する(話している最中のつながった発言も反映する)
        if st.is_speaking(now):
            return None
        remaining = st.remaining(now)
        if remaining <= 0:
            return "time_up"
        since = st.since_last_intervention(now)
        if any(remaining <= th and th not in st.warned_thresholds for th in self._warn_thresholds()) and (
            since is None or since >= s.forced_min_gap_sec
        ):
            return "time_threshold"
        if st.silence_sec(now) < s.min_silence_for_intervention_sec:
            return None
        self._scan_requests(now)
        if self.pending_request is not None:
            return "request"  # 自走期でも応じる
        if self.chat_wait_reason(now, cs):
            return None
        if self._judged_same_situation(now, cs):
            return None
        return "chat_lead"

    def _judged_same_situation(self, now: float, cs: ChatState) -> bool:
        """前回の判断から新しい発言がなく、次の待ち時間もたっていない(同じ状況で何度も判断しない)。"""
        if self.last_tick_time is None:
            return False
        latest = max((u.id for u in self.state.utterances if u.end_time <= now), default=None)
        return latest == self._tick_context_id and now - self.last_tick_time < self._need_silence(cs)

    def _gate(self, now: float) -> tuple[str, list[Action], str | None]:
        """GD・GW と同じルール層(強制イベント・発話中・依頼など)のうえに、雑談の待ちの条件を重ねる。"""
        gate, allowed, note = super()._gate(now)
        if gate == "blocked:cooldown" and self.pending_request is not None:
            # 間がなくても依頼には応じる(進行役は内容支援があるときだけ応じるが、雑談では話題で応じる)
            return "llm:request", list(REQUEST_ACTIONS), CHAT_REQUEST_NOTE
        if gate != "llm":
            return gate, allowed, note  # 強制イベント・見送りはそのまま
        chat = list(CHAT_ACTIONS)
        if Action.invite_person in allowed:  # 個人指名は進行役と同じ条件(話者情報が使え、名前が確認済み)のときだけ
            chat.append(Action.invite_person)
        if self.pending_request is not None:
            return gate, chat, None  # 依頼には自走期でも応じる
        cs = self.update_phase(now)
        wait = self.chat_wait_reason(now, cs)
        if wait:
            return f"blocked:{wait}", [], None
        return gate, chat, None

    def stale_reason(self, now: float, context_utt_id: int | None = None) -> str | None:
        """判断を始めた後に人間の新しい発言があれば、その判断の発話はしない(古い会話の状況を前提に話さない)。"""
        cid = self._tick_context_id if context_utt_id is None else context_utt_id
        if cid is None:
            return None
        if any(u.source == "human" and u.id > cid and u.end_time <= now for u in self.state.utterances):
            return "判断中に会話が再開した"
        return None

    def tick(self, now: float, trigger: str = "periodic", act: bool = True, on_partial: Any = None) -> TickRecord:
        rec = super().tick(now, trigger, act=act, on_partial=on_partial)
        self.update_phase(now)  # LLM の判断(会話が進んでいるか)も踏まえて時期を更新する
        return rec


def agent_class_for(discussion_type: str | None) -> type[FacilitatorAgent]:
    """議論タイプに応じた司会のクラス。雑談型だけ雑談の司会、それ以外は GD・GW の進行役(従来どおり)。"""
    return ChatHostAgent if discussion_type == CASUAL.key else FacilitatorAgent
