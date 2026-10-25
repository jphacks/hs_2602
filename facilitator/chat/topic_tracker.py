"""雑談の話題の管理(第4段階)。いまの話題と最近の話題、話題ごとの AI の働きかけと参加者の反応を記録から計算する。

- 話題は、AI が出したもの(開始・offer_topic・change_topic)と、参加者が自分たちで始めたもの(LLM が
  analysis.topic_by = participants と判断したもの)の両方を記録する
- AI の働きかけ(時間告知・終了を除く AI の発言)ごとに、次の AI の発言までの人間の反応を分類する:
  none(答えなし)/ single(1人だけ)/ engaged(2人以上が話した = 人間どうしの会話になった)
- 反応が薄い(none / single)働きかけが同じ話題で続いた回数で、話題の切り替えを許可する(すぐには変えない。
  深掘り・橋渡しを試してからにする)。一度広がった話題が途切れた・一段落したときは、新しい話題を出してよい
- 同じ1人とだけの一問一答が続いた回数も数える(続きすぎたら全体へ広げる)
- 同じ働きかけ(深掘り・橋渡し)が薄い反応のまま続いたら、その働きかけは繰り返さない(質問攻めにしない)
数値は config.Settings の chat_topic_* / chat_one_on_one_max / chat_recent_topics(仮の値)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import Settings
from facilitator.chat.topics import similar_topic, topic_candidates
from facilitator.models import Action, Utterance
from facilitator.state import DiscussionState

NONE, SINGLE, ENGAGED = "none", "single", "engaged"
WEAK = (NONE, SINGLE)


@dataclass
class Topic:
    text: str
    source: str  # ai(AI が出した)/ participants(参加者が始めた)
    since: float  # 話題が始まった時刻(AI が出した時刻、または LLM が参加者の新しい話題と判断した時刻)


@dataclass
class TopicStatus:
    """いまの話題についての状況(判断のたびに記録から計算する)。"""

    topic: Topic | None
    reactions: list[str] = field(default_factory=list)  # この話題での AI の働きかけごとの反応(古い順)
    weak_streak: int = 0  # 直近から数えて、反応が薄い働きかけが続いた回数
    one_on_one: int = 0  # 直近から数えて、同じ1人だけが答えた働きかけが続いた回数
    repeated: Action | None = None  # 薄い反応のまま chat_same_nudge_max 回続いた同じ働きかけ(繰り返さない)
    partner: str | None = None  # 一問一答の相手(one_on_one > 0 のとき)
    engaged: bool = False  # この話題で人間どうしの会話になったことがあるか
    settled: bool = False  # この話題は一段落した(LLM が converging と判断、または広がった後に途切れた)
    can_change: bool = False  # change_topic を許可するか
    can_offer: bool = False  # offer_topic を許可するか


class TopicTracker:
    def __init__(self, settings: Settings, theme: str) -> None:
        self.settings, self.theme = settings, theme
        self.history: list[Topic] = []
        self.turns: list[tuple[float, Action]] = []  # AI の働きかけ (時刻, 行動)

    @property
    def current(self) -> Topic | None:
        return self.history[-1] if self.history else None

    # ---------- 記録 ----------
    def ai_topic(self, text: str, now: float) -> None:
        self.history.append(Topic(text, "ai", now))

    def participant_topic(self, text: str, now: float) -> bool:
        """参加者が始めた話題を記録する。いまの話題と同じなら記録しない。記録したら True。"""
        text = text.strip()
        cur = self.current
        if not text or cur is not None and self.similar(text, cur.text):
            return False
        self.history.append(Topic(text, "participants", now))
        return True

    def note_turn(self, now: float, action: Action) -> None:
        self.turns.append((now, action))

    # ---------- 参照 ----------
    def similar(self, a: str, b: str) -> bool:
        return similar_topic(a, b, self.theme, self.settings.chat_topic_similarity)

    def recent(self) -> list[str]:
        """最近扱った話題(新しい順ではなく古い順)。重複を避ける対象。"""
        return [t.text for t in self.history[-self.settings.chat_recent_topics:]]

    def is_recent(self, text: str) -> bool:
        return any(text == r or self.similar(text, r) for r in self.recent())

    def candidates(self) -> list[str]:
        return topic_candidates(self.theme, self.recent(), self.settings.chat_topic_similarity)

    def ai_topics(self) -> list[str]:
        return [t.text for t in self.history if t.source == "ai"]

    def _reaction(self, state: DiscussionState, i: int, now: float) -> tuple[str, set[str]]:
        """i 番目の働きかけへの反応(次の AI の働きかけまでの人間の発言)。"""
        t0 = self.turns[i][0]
        t1 = self.turns[i + 1][0] if i + 1 < len(self.turns) else float("inf")
        ai = next((u for u in state.utterances if u.source == "ai" and abs(u.start_time - t0) < 0.01), None)
        after = ai.end_time if ai else t0
        humans: list[Utterance] = [
            u for u in state.utterances
            if u.source == "human" and u.end_time > after and u.start_time < t1 and u.end_time <= now
        ]
        who = {u.speaker_id for u in humans}
        return (NONE if not humans else ENGAGED if len(who) >= 2 else SINGLE), who

    def status(self, state: DiscussionState, now: float, settled_by_llm: bool) -> TopicStatus:
        s, cur = self.settings, self.current
        if cur is None:
            return TopicStatus(None, can_offer=True)
        idx = [i for i, (t, _a) in enumerate(self.turns) if t >= cur.since]
        reacts = [self._reaction(state, i, now) for i in idx]
        kinds = [k for k, _w in reacts]
        weak = 0
        for k in reversed(kinds):
            if k not in WEAK:
                break
            weak += 1
        one, partner = 0, None
        for k, who in reversed(reacts):
            if k != SINGLE or partner is not None and who != {partner}:
                break
            partner, one = next(iter(who)), one + 1
        n = s.chat_same_nudge_max
        last = [self.turns[i][1] for i in idx[-n:]]
        repeated = last[0] if len(last) == n and len(set(last)) == 1 and weak >= n else None
        engaged = ENGAGED in kinds
        # 一度広がった話題が途切れた(主導期に戻るほどの沈黙)か、LLM が一段落と判断した
        settled = settled_by_llm or engaged and state.silence_sec(now) >= s.chat_lull_silence_sec
        can_change = weak >= s.chat_topic_change_after
        return TopicStatus(cur, reactions=kinds, weak_streak=weak, one_on_one=one, partner=partner if one else None,
                           repeated=repeated, engaged=engaged, settled=settled, can_change=can_change,
                           can_offer=can_change or settled)
