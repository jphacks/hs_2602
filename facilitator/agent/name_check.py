"""進行役の発話に含まれる参加者の個人名を検査する(LLM を呼ばない)。

方針: 個人名で呼びかけてよいのは invite_person(指名の安全条件を満たしたときの指名対象)だけ。ほかのアクションでは
参加者の名前を使わない(話者識別の誤りで別の人の名前を出すおそれがあるため)。
- 参加者の名前 + 敬称(さん・くん・ちゃん・様)だけを個人名とみなす。テーマに出てくる人物名や、敬称のない引用は対象外
- 確実に言い換えられる形だけ直す: 呼びかけ(「はなこさん、」)は外し、「はなこさんの〜」は「先ほど出た〜」にする
- それ以外の形で名前が残れば、呼び出し側が発話させない(言い換えで文を壊さないため)
"""

from __future__ import annotations

import re

HONORIFIC = r"(?:さん|くん|君|ちゃん|様)"


def strip_names(speech: str, names: list[str], keep: str | None = None) -> tuple[str, list[str]]:
    """(言い換えた発話, 言い換えられずに残った名前)。keep は残してよい名前(invite_person の指名対象)。"""
    out = speech
    targets = sorted({n for n in names if n and n != keep}, key=len, reverse=True)  # 長い名前から(部分一致を避ける)
    for n in targets:
        name = re.escape(n) + HONORIFIC
        out = re.sub(rf"(^|(?<=[。!?！？]))\s*{name}[、,]\s*", r"\1", out)  # 文頭の呼びかけ
        out = re.sub(rf"{name}の", "先ほど出た", out)
    left = [n for n in targets if re.search(re.escape(n) + HONORIFIC, out)]
    return out, left
