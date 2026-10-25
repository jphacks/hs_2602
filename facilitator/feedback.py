"""終了後の振り返り(練習の補助)と、発言の種類の分類。

- 振り返り: 人間の参加者ごとに、観点(主体性・論理性・傾聴/協調・議論への貢献)の目安と根拠の発言、良かった点・次に意識すること。
  練習用の参考で、選考の評価には使わない(画面にも明記する)。点数は出さない。
- 発言の種類: 人間の各発言を「提案・根拠/具体例・質問・反論/懸念・整理/まとめ・同意/受け止め・進行・その他」に分ける。
  良し悪しは付けない事実の分類で、選考の担当者が見るのはこちらまでを想定する。
AI(進行役・AI 参加者)の発言は文脈として渡すが、評価・分類の対象にしない。話者が分からない発言も対象外。
"""

from __future__ import annotations

import re
from typing import Any

from facilitator.llm.base import LLMClient, LLMError, generate

UTTERANCE_TYPES: dict[str, str] = {
    "proposal": "提案",
    "reason": "根拠・具体例",
    "question": "質問",
    "objection": "反論・懸念",
    "organize": "整理・まとめ",
    "agree": "同意・受け止め",
    "progress": "進行",
    "other": "その他",
}
ASPECTS: dict[str, str] = {
    "initiative": "主体性",
    "logic": "論理性",
    "listening": "傾聴・協調",
    "contribution": "議論への貢献",
}
LEVELS: dict[str, str] = {"great": "よくできた", "good": "できた", "next": "これから"}

SYSTEM = """\
あなたはグループディスカッションの練習を支援するコーチです。終わった議論のログを読み、2つのことをしてください。

1. 人間の参加者の各発言(ID 付き)を、次のどれか1つに分類する(良し悪しは付けない)
   - proposal: 案・アイデア・自分の意見を出す
   - reason: 根拠・理由・具体例・データを示す
   - question: 他の人に質問する、確認する
   - objection: 反論する、懸念やリスクを指摘する
   - organize: 出た意見を整理する、まとめる、比較する
   - agree: 同意する、受け止める、他の人の意見を広げる
   - progress: 時間や段階を意識して議論を進める、脱線を戻す
   - other: 相づちなど上のどれにも当たらない
2. 人間の参加者一人ずつに、練習のための振り返りを書く
   - 観点(initiative=主体性、logic=論理性、listening=傾聴・協調、contribution=議論への貢献)ごとに、目安(great / good / next)と、その根拠になったその人の実際の発言を1つ、ログからそのまま短く引用する
   - 良かった点(good)と、次に意識すること(next)をそれぞれ1〜2個、具体的に、本人に語りかける優しい口調で書く

注意
- 評価するのは人間の参加者だけ。進行役(AI)と AI参加者の発言は、流れをつかむための文脈としてだけ使う
- 発言が少ない人も否定せず、次にできる具体的な行動を示す。発言量だけで判断しない
- 引用は必ずその人の発言から。ログにない言葉を作らない。根拠になる発言がない観点は quote を空にし、level は next にする
- これは練習用の参考で、選考の評価ではない。点数は付けない
"""


def _schema(ids: list[int], speakers: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "utterances": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "enum": ids} if ids else {"type": "integer"},
                        "type": {"type": "string", "enum": list(UTTERANCE_TYPES)},
                    },
                    "required": ["id", "type"],
                    "additionalProperties": False,
                },
            },
            "people": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker_id": {"type": "string", "enum": speakers} if speakers else {"type": "string"},
                        "aspects": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "aspect": {"type": "string", "enum": list(ASPECTS)},
                                    "level": {"type": "string", "enum": list(LEVELS)},
                                    "quote": {"type": "string"},
                                },
                                "required": ["aspect", "level", "quote"],
                                "additionalProperties": False,
                            },
                        },
                        "good": {"type": "array", "items": {"type": "string"}},
                        "next": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["speaker_id", "aspects", "good", "next"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["utterances", "people"],
        "additionalProperties": False,
    }


def _mock_type(text: str) -> str:
    """モック用: 簡単な言葉の手がかりで分類する。"""
    t = text.strip()
    if re.search(r"[?？]$|どう思|ですか|ますか", t):
        return "question"
    if re.search(r"まとめ|整理|つまり|比べ", t):
        return "organize"
    if re.search(r"でも|しかし|ただ、|懸念|心配", t):
        return "objection"
    if re.search(r"なぜなら|からです|例えば|実際", t):
        return "reason"
    if re.search(r"確かに|賛成|なるほど|いいですね", t):
        return "agree"
    if re.search(r"時間|そろそろ|次に|話を戻", t):
        return "progress"
    return "proposal" if len(t) >= 8 else "other"


def _mock(humans: list[dict[str, Any]], utts: list[dict[str, Any]]) -> dict[str, Any]:
    types = [{"id": u["id"], "type": _mock_type(u["text"])} for u in utts]
    people = []
    for h in humans:
        mine = [u for u in utts if u["speaker_id"] == h["id"]]
        quote = mine[0]["text"][:40] if mine else ""
        people.append({
            "speaker_id": h["id"],
            "aspects": [{"aspect": a, "level": "good" if mine else "next", "quote": quote} for a in ASPECTS],
            "good": ["自分の考えを言葉にして議論に参加できていました。"] if mine else [],
            "next": ["次は、自分の意見に理由や具体例を一言添えてみましょう。"] if mine
            else ["まずは短くても一度、自分の考えを話してみましょう。"],
        })
    return {"utterances": types, "people": people}


def build_feedback(
    llm: LLMClient,
    theme: str,
    discussion_type: str,
    log: list[dict[str, Any]],
    humans: list[dict[str, Any]],
) -> dict[str, Any]:
    """log: 全発言 {id, speaker_id, who, text, source}。humans: 人間の参加者 {id, name}。
    返り値: {ok, people: [{id, name, turns, types: {種類: 回数}, aspects, good, next}], types, levels}"""
    human_ids = {h["id"] for h in humans}
    targets = [u for u in log if u["source"] == "human" and u["speaker_id"] in human_ids and u["text"].strip()]
    if llm.name == "mock":
        raw = _mock(humans, targets)
    else:
        lines = []
        for u in log:
            if u["source"] == "human" and u["speaker_id"] in human_ids:
                lines.append(f"[{u['id']}] {u['who']}({u['speaker_id']}): {u['text']}")
            elif u["source"] == "human":
                lines.append(f"[-] 不明: {u['text']}")
            else:
                lines.append(f"[-] {u['who']}(AI): {u['text']}")
        user = (
            f"テーマ: {theme}\n議論タイプ: {discussion_type}\n"
            f"人間の参加者: {'、'.join(f'{h['name']}({h['id']})' for h in humans)}\n\n"
            "# 議論のログ(古い順。[ID] は分類する発言、[-] は文脈のみ)\n" + "\n".join(lines)
        )
        try:
            raw = generate(llm, SYSTEM, user, _schema([u["id"] for u in targets], sorted(human_ids)), {"kind": "振り返り"})
        except LLMError as e:
            return {"ok": False, "error": f"振り返りを作れませんでした({type(e).__name__})"}

    kind = {int(x["id"]): x["type"] for x in raw.get("utterances", []) if x.get("type") in UTTERANCE_TYPES}
    by_person = {p["speaker_id"]: p for p in raw.get("people", [])}
    people = []
    for h in humans:
        mine = [u for u in targets if u["speaker_id"] == h["id"]]
        counts = {t: 0 for t in UTTERANCE_TYPES}
        for u in mine:
            counts[kind.get(u["id"], "other")] += 1
        fb = by_person.get(h["id"], {})
        aspects = {a["aspect"]: a for a in fb.get("aspects", []) if a.get("aspect") in ASPECTS}
        people.append({
            "id": h["id"],
            "name": h["name"],
            "turns": len(mine),
            "types": counts,
            "aspects": [
                {"aspect": key, "label": label, "level": aspects.get(key, {}).get("level", "next"),
                 "quote": aspects.get(key, {}).get("quote", "")}
                for key, label in ASPECTS.items()
            ],
            "good": [str(x) for x in fb.get("good", [])][:2],
            "next": [str(x) for x in fb.get("next", [])][:2],
        })
    return {"ok": True, "people": people, "types": UTTERANCE_TYPES, "levels": LEVELS}
