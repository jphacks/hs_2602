"""雑談の話題の管理(第4段階)。いまの話題と最近の話題、話題ごとの AI の働きかけと参加者の反応を記録から計算する。

- 話題は、AI が出したもの(開始・offer_topic・change_topic)と、参加者が自分たちで始めたもの(LLM が
  analysis.topic_by = participants と判断したもの)の両方を記録する
- AI の働きかけ(時間告知・終了を除く AI の発言)ごとに、次の AI の発言までの人間の反応を分類する:
  none(答えなし)/ passive(「特にないです」「そうですね」のような中身のない返事だけ)/ single(1人だけ)/
  joined(この話題でまだ答えていなかった人が1人加わった = 橋渡しが効いた)/
  engaged(1回の働きかけの後に2人以上が話した = 人間どうしの会話になった)
  発言が短いだけでは薄い反応とみなさない(「ラーメンです」は中身のある答え)。中身のない返事は passive_reply で判定する
- 反応が薄い(none / passive / single)働きかけが同じ話題で続いた回数で、話題の切り替えを許可する(すぐには変えない。
  深掘り・橋渡しを試してからにする)。一度広がった話題が途切れた・一段落したときは、新しい話題を出してよい
- 話題ごとに、中身のある答えをした人(answerers)も数える。順に1人ずつ答えたこと(joined)と、
  人間どうしの会話(engaged)は区別する(自走期への切り替えは engaged と同じく、つながった発言で決める)
- 答えなし・中身のない返事が続いた回数(話題をまたいで数える。quiet_streak)は、次の働きかけまでの待ちを伸ばすのに使う
- 同じ1人とだけの一問一答が続いた回数も数える(続きすぎたら全体へ広げる)
- 同じ働きかけ(深掘り・橋渡し)が薄い反応のまま続いたら、その働きかけは繰り返さない(質問攻めにしない)
数値は config.Settings の chat_topic_* / chat_one_on_one_max / chat_recent_topics(仮の値)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from config import Settings
from facilitator.chat.topics import similar_topic, topic_candidates
from facilitator.models import Action, Utterance
from facilitator.state import DiscussionState

NONE, PASSIVE, SINGLE, JOINED, ENGAGED = "none", "passive", "single", "joined", "engaged"
WEAK = (NONE, PASSIVE, SINGLE)  # 反応が薄い(JOINED は別の人が加わったので薄くない)
QUIET = (NONE, PASSIVE)  # 答えがない・中身のない返事だけ(待ちを伸ばす)

# 中身のない返事を作る語(長いものから順に消して、何も残らなければ中身のない返事とみなす。粗い判定)
_PASSIVE_PARTS = re.compile(
    "|".join([
        r"特に(何も|なにも)?(ないです|ないかも|ありません|ない|なし|は)?",
        r"(わから|分から|思いつか|思い浮かば|おもいつか)(ないです|ないかも|なくて|ない)",
        r"(何も|なにも|別に|べつに)(ないです|ありません|ない)?",
        r"(あんまり|あまり)(ないです|ありません|ない)?",
        r"どうだろう|どうでしょう|どうかな|普通|ふつう",
        r"そうですね|そうですか|なるほど|たしかに|確かに",
        r"ないです|ないかも|ありません|ない",
        r"うーん|ううん|んー|えーと|えっと|ええと|えー|あー|まあ|うん|はい|いいえ|いや",
        r"ですかね|ですね|です|かもです|かも|かな|けど|ね|よ|か",
    ])
)
_PUNCT = re.compile(r"[\s、。,.!?！？…・〜~「」『』()()]+")


def passive_reply(text: str) -> bool:
    """「特にないです」「そうですね」「うーん…」のような、中身のない返事か(短いだけの答えは含めない)。"""
    rest = _PASSIVE_PARTS.sub("", _PUNCT.sub("", text))
    return bool(_PUNCT.sub("", text)) and not rest


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
    answerers: int = 0  # この話題で、AI の働きかけに中身のある答えをした人の数(順に1人ずつ答えた人も数える)
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
        """i 番目の働きかけへの反応(次の AI の働きかけまでの人間の発言)と、中身のある答えをした人。
        JOINED はここでは返さない(話題の中で、前に答えた人と比べて status で決める)。"""
        t0 = self.turns[i][0]
        t1 = self.turns[i + 1][0] if i + 1 < len(self.turns) else float("inf")
        ai = next((u for u in state.utterances if u.source == "ai" and abs(u.start_time - t0) < 0.01), None)
        after = ai.end_time if ai else t0
        humans: list[Utterance] = [
            u for u in state.utterances
            if u.source == "human" and u.end_time > after and u.start_time < t1 and u.end_time <= now
        ]
        who = {u.speaker_id for u in humans if not passive_reply(u.text)}
        return (NONE if not humans else PASSIVE if not who else ENGAGED if len(who) >= 2 else SINGLE), who

    def quiet_streak(self, state: DiscussionState, now: float) -> int:
        """直近から数えて、答えがない・中身のない返事だけの AI の働きかけが続いた回数(話題をまたいで数える)。
        中身のある答えが1つでもあれば 0 に戻る。"""
        n = 0
        for i in reversed(range(len(self.turns))):
            if self._reaction(state, i, now)[0] not in QUIET:
                break
            n += 1
        return n

    def status(self, state: DiscussionState, now: float, settled_by_llm: bool) -> TopicStatus:
        s, cur = self.settings, self.current
        if cur is None:
            return TopicStatus(None, can_offer=True)
        idx = [i for i, (t, _a) in enumerate(self.turns) if t >= cur.since]
        reacts, seen = [], set()
        for i in idx:
            k, who = self._reaction(state, i, now)
            if k == SINGLE and seen and not who & seen:  # この話題でまだ答えていなかった人が加わった(橋渡しが効いた)
                k = JOINED
            seen |= who
            reacts.append((k, who))
        kinds = [k for k, _w in reacts]
        weak = 0
        for k in reversed(kinds):
            if k not in WEAK:
                break
            weak += 1
        one, partner = 0, None
        for k, who in reversed(reacts):
            if k not in (SINGLE, JOINED) or partner is not None and who != {partner}:
                break
            partner, one = next(iter(who)), one + 1
        n = s.chat_same_nudge_max
        last = [self.turns[i][1] for i in idx[-n:]]
        repeated = last[0] if len(last) == n and len(set(last)) == 1 and weak >= n else None
        engaged = ENGAGED in kinds
        # 一度広がった話題が途切れた(主導期に戻るほどの沈黙)か、LLM が一段落と判断した
        settled = settled_by_llm or engaged and state.silence_sec(now) >= s.chat_lull_silence_sec
        can_change = weak >= s.chat_topic_change_after
        return TopicStatus(cur, reactions=kinds, weak_streak=weak, answerers=len(seen), one_on_one=one, partner=partner if one else None,
                           repeated=repeated, engaged=engaged, settled=settled, can_change=can_change,
                           can_offer=can_change or settled)
