"""AI 参加者(議論に加わる仮想の参加者)。

- 開始時に、テーマに対する異なる立場を AI の人数分生成して1人ずつ割り当てる
- 発言は、割り当てられた立場と直近の議論を踏まえて1〜2文で生成する
- 「いつ話すか」はエンジン側(engine.py)が人数比に応じたモードで決める
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from facilitator.llm.base import LLMClient, LLMError
from facilitator.models import Action

MOCK_PERSONAS = [
    ("アオイ", "コスト重視", "まずはお金をかけずにすぐ実行できる案から試すべきだと考えている。", "落ち着いて現実的に話す"),
    ("ソラ", "利用者の体験重視", "使う人がどれだけ便利になるかを一番の判断基準にすべきだと考えている。", "明るく具体例を挙げて話す"),
    ("ハル", "根拠・データ重視", "思いつきで決めず、原因や数字を確かめてから判断すべきだと考えている。", "慎重に質問を交えて話す"),
]
MOCK_LINES = [
    "私は{stance}の立場から考えたいです。{opinion}",
    "今の意見に付け加えると、{stance}の観点も大事だと思います。",
    "なるほど。ただ、{stance}で考えると、少し別の見方もできそうです。",
    "ほかの人はどう思いますか？{stance}の私としては、まだ決めきれていません。",
]


@dataclass
class Persona:
    speaker_id: str
    name: str
    stance: str  # 立場の短い見出し(例: コスト重視)
    opinion: str  # 割り当てられた意見(1〜2文)
    style: str  # 話し方の特徴
    spoken: int = 0
    history: list[str] = field(default_factory=list)


# ---------------- 立場の生成 ----------------
PERSONA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "personas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "stance": {"type": "string"},
                    "opinion": {"type": "string"},
                    "style": {"type": "string"},
                },
                "required": ["name", "stance", "opinion", "style"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["personas"],
    "additionalProperties": False,
}

PERSONA_SYSTEM = """\
あなたはグループディスカッションに参加する「AI参加者」の設定を作ります。
テーマに対して、互いに異なる立場・意見を、指定された人数分つくってください。
- 立場は互いに重ならず、議論が深まるように観点や優先順位を分散させること(賛成/反対の二択ではなく、重視する観点の違いでもよい)
- 極端すぎず、実際の参加者が言いそうな現実的な意見にすること
- name はカタカナ2〜3文字のニックネーム。既存の参加者名と重ならないこと
- stance は10文字程度の見出し(例: コスト重視)
- opinion はその参加者の基本的な考えを1〜2文で
- style は話し方の特徴を短く(例: 落ち着いて具体例を挙げる)
"""


def generate_personas(
    llm: LLMClient, theme: str, purpose: str, count: int, taken_names: list[str], first_id: int
) -> list[Persona]:
    ids = [f"speaker_{first_id + i}" for i in range(count)]
    if count <= 0:
        return []
    if llm.name == "mock":
        return [Persona(sid, *MOCK_PERSONAS[i % len(MOCK_PERSONAS)]) for i, sid in enumerate(ids)]
    who = "就職活動の選考のグループディスカッションに参加する就活生" if purpose == "job" else "会議やワークショップの参加者"
    user = (
        f"テーマ: {theme}\n参加者の想定: {who}\nAI参加者の人数: {count}\n"
        f"既存の参加者名(使わないこと): {'、'.join(taken_names) or 'なし'}"
    )
    try:
        raw = llm.generate_json(PERSONA_SYSTEM, user, PERSONA_SCHEMA, {"kind": "AI参加者の立場の生成"})
        items = raw.get("personas", [])[:count]
    except LLMError:
        items = []
    personas = []
    for i, sid in enumerate(ids):
        if i < len(items):
            it = items[i]
            personas.append(Persona(sid, it["name"][:8], it["stance"][:20], it["opinion"], it["style"]))
        else:  # 生成に失敗した分は定型の立場で補う
            personas.append(Persona(sid, *MOCK_PERSONAS[i % len(MOCK_PERSONAS)]))
    # 名前の重複を避ける
    used = set(taken_names)
    for p in personas:
        while p.name in used:
            p.name += "2"
        used.add(p.name)
    return personas


# ---------------- 発言の生成 ----------------
UTTER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


def _utter_system(p: Persona, purpose: str) -> str:
    who = "就職活動の選考のグループディスカッションに参加している就活生" if purpose == "job" else "グループディスカッションの参加者"
    return f"""\
あなたは{who}の「{p.name}」です(AI参加者として議論に加わっています)。
あなたの立場: {p.stance} — {p.opinion}
話し方: {p.style}

# 発言のルール
- 1〜2文、話し言葉の自然な日本語(です・ます調)で短く話す。長い演説はしない
- 直前の流れに応答する(賛同・質問・反論・付け足し)。自分の立場を軸にしつつ、他の人の良い意見は認めてよい
- 進行役ではないので、議論をまとめたり仕切ったりしない。名指しで質問されたら、それに答える
- 他の参加者の発言機会を奪わないよう、ときどき他の人に意見を聞くのはよい
- 発言内容のみを出力する(名前や括弧書きは付けない)
"""


def generate_utterance(
    llm: LLMClient, p: Persona, purpose: str, theme: str, transcript: list[str], reason: str
) -> str:
    if llm.name == "mock":
        line = MOCK_LINES[p.spoken % len(MOCK_LINES)].format(stance=p.stance, opinion=p.opinion)
        return line
    user = (
        f"テーマ: {theme}\n\n# 直近の発言(古い順)\n"
        + ("\n".join(transcript[-20:]) or "(まだ発言なし)")
        + f"\n\n# 今あなたが話す理由\n{reason}\n\nあなた({p.name})の次の発言を作ってください。"
    )
    raw = llm.generate_json(_utter_system(p, purpose), user, UTTER_SCHEMA, {"kind": f"AI参加者({p.name})の発言"})
    return str(raw.get("text", "")).strip()


# ---------------- いつ話すか ----------------
# active(練習相手): 人間が2人以下、または AI が人間と同数以上のとき。間があれば積極的に話す
# reserved(控えめ): 人間が主役のとき。名指しされたときと長い沈黙のときだけ話す
BOT_ADDRESSED_PAUSE_SEC = 1.2  # 名指しされたら、この間のあとに答える
BOT_ACTIVE_PAUSE_SEC = 2.5
BOT_ACTIVE_GAP_SEC = 6.0  # AI 参加者どうしの発言の最低間隔
BOT_ACTIVE_MAX_CHAIN = 2  # 人間の発言を挟まずに AI 参加者が続けて話す上限
BOT_WAIT_FOR_INVITED_HUMAN_SEC = 6.0  # 進行役が人間を指名した直後は、その人の応答を待つ
BOT_RESERVED_PAUSE_SEC = 9.0
BOT_RESERVED_GAP_SEC = 40.0


def bot_mode_for(humans: int, bots: int) -> str:
    return "" if not bots else ("active" if humans <= 2 or bots >= humans else "reserved")


def pick_bot(
    state: Any,  # DiscussionState
    personas: list[Persona],
    mode: str,
    now: float,
    last_bot_end: float,
    answered_id: int,
    humans: int,
) -> tuple[Persona, str, int] | None:
    """今 AI 参加者が話すべきなら (誰が, 理由, 名指しされた発言 ID or -1) を返す。話さないなら None。"""
    st = state
    if not personas or not st.started or st.speaking_now or st.elapsed(now) < 3.0 or not st.utterances:
        return None
    silence = st.silence_sec(now)
    last = st.utterances[-1]
    if last.end_time > now:  # 誰か(AI を含む)がまだ話している
        return None
    by_id = {p.speaker_id: p for p in personas}

    # 1. 名指しされた(進行役の個人指名、または参加者の発言の中で名前を呼ばれた)
    if silence >= BOT_ADDRESSED_PAUSE_SEC and last.id != answered_id:
        target = None
        iv = st.last_intervention()
        if last.source == "ai" and iv and iv.action == Action.invite_person and iv.target_speaker in by_id:
            target = iv.target_speaker
        if target is None and last.source != "ai":  # 進行役の発言は個人指名のときだけ名指しとみなす
            for p in personas:
                if p.name in last.text and last.speaker_id != p.speaker_id:
                    target = p.speaker_id
                    break
        if target:
            who = "進行役" if last.source == "ai" else (st.speaker_names.get(last.speaker_id) or "参加者")
            return by_id[target], f"{who}から名指しで話を振られたので、それに答える", last.id

    # 進行役が人間を指名した直後は、その人が答えるのを待つ
    iv = st.last_intervention()
    if (
        last.source == "ai"
        and iv
        and iv.action == Action.invite_person
        and iv.target_speaker not in by_id
        and silence < BOT_WAIT_FOR_INVITED_HUMAN_SEC
    ):
        return None

    # 2. 間があいた
    if mode == "active":
        if silence < BOT_ACTIVE_PAUSE_SEC or now - last_bot_end < BOT_ACTIVE_GAP_SEC:
            return None
        chain = 0
        for u in reversed(st.utterances):
            if u.source == "human":
                break
            chain += u.source == "bot"
        if humans and chain >= BOT_ACTIVE_MAX_CHAIN:
            return None
        reason = "会話に間があいたので、直前の流れを受けて自分の立場から意見や質問を出す"
    else:
        if silence < BOT_RESERVED_PAUSE_SEC or now - last_bot_end < BOT_RESERVED_GAP_SEC:
            return None
        reason = "しばらく沈黙が続いたので、議論が進むように自分の考えを短く述べる"
    # 最近話していない AI 参加者を選ぶ(直前の話者は避ける)
    last_spoke = {
        p.speaker_id: max((u.end_time for u in st.utterances if u.speaker_id == p.speaker_id), default=-1e9)
        for p in personas
    }
    cands = [p for p in personas if p.speaker_id != last.speaker_id] or personas
    return min(cands, key=lambda p: last_spoke[p.speaker_id]), reason, -1


def transcript_for_bot(state: Any) -> list[str]:
    st = state
    lines = []
    for u in st.utterances[-20:]:
        if u.source == "ai":
            who = "進行役(AI)"
        elif u.source == "bot":
            who = f"{st.speaker_names.get(u.speaker_id, u.speaker_id)}(AI参加者)"
        else:
            who = st.speaker_names.get(u.speaker_id) or "不明"
        lines.append(f"{who}: {u.text}")
    return lines
