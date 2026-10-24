"""参加者から進行役(AI)への明示的な依頼を、LLM を呼ばずに検出する(保守的な判定)。

依頼とみなすのは、1つの発言に次の3つがそろうときだけ。参加者どうしの質問(「けんたさん、どう思いますか?」)は
進行役への呼びかけがないので検出しない。
1. 進行役への呼びかけ(「AIさん」「進行役さん」「ファシリテーターさん」「司会の方」など。「AIさんたち」は除く)
2. 依頼の表現(「?」「教えて」「ヒント」「どう思いますか」「ください」など)
3. 伝聞・言及ではない(「AIさんが言ってたように」「進行役の話だと」は除く)
AI 参加者がいるときに、その名前を含む発言は AI 参加者への呼びかけとみなして除く。

検出は依頼の「候補」で、進行役の LLM が発話ログの文脈から本当に自分への依頼かを確かめる(違えば no_intervention)。
限界: STT が呼びかけを聞き違えると検出できない(例:「AIさん」→「Aさん」)。テーマが AI に関するもので、
「AIさん」という呼び方で AI そのものを話題にしている場合は誤検出しうる(LLM の確認で抑える)。
"""

from __future__ import annotations

import re

ADDRESS = re.compile(
    r"(?:AI|ＡＩ|ai|エーアイ)\s*さん(?!たち|達|ら)|進行役(?:さん|の方)|ファシリ(?:テーター)?さん|司会(?:さん|の方)"
)
REQUEST = re.compile(
    r"[?？]|教えて|ヒント|アイデア|アドバイス|どう思|どうすれば|ありますか|ありません|ないですか|ください|もらえ|"
    r"お願い|助けて|手伝って|聞きたい"
)
REPORTED = re.compile(
    r"(?:(?:AI|ＡＩ|ai|エーアイ)\s*さん|進行役(?:さん|の方)?|ファシリ(?:テーター)?さん|司会(?:さん|の方)?)"
    r"(?:が|の)(?:言|出し|話|挙げ|くれ|質問|問い)"
)


def detect_request(text: str, bot_names: list[str] | tuple[str, ...] = ()) -> str | None:
    """進行役への依頼の候補なら、呼びかけの語を返す。そうでなければ None。"""
    m = ADDRESS.search(text)
    if not m or REPORTED.search(text) or not REQUEST.search(text):
        return None
    if any(n and n in text for n in bot_names):
        return None
    return m.group()
