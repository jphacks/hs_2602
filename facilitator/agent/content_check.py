"""発話本文に、許可されていない内容(案・意見・新しい観点)が混ざった疑いを、LLM を呼ばずに検出する(記録用)。

内容支援の許可はアクション(offer_hint / propose_idea)の許可・検証で制御するが、ほかのアクション(ask_question など)の
発話本文に案が混ざることはそれでは防げない。この検出は判断を変えず、TickRecord.content_flags に残して評価に使う。

限界: 言い回しのパターンだけを見る粗い判定。質問の形をとった誘導(「〜を増やせば解決しませんか?」の言い換え)や、
具体的な案を名詞だけで挙げる場合は見逃す。逆に、参加者が出した案を引用した整理を誤って検出することがある。
"""

from __future__ import annotations

import re

from facilitator.models import Action

# 具体的な案の提案(提案の形)
IDEA = [
    (re.compile(r"(?:して|すれば|したら|するのは|にするのは|のは)(?:どう|いかが)"), "案の提案"),
    (re.compile(r"という(?:案|方法|アイデア|手)"), "案の提示"),
]
# 進行役自身の意見・推奨(どの範囲でも言わない)
OPINION = [
    (re.compile(r"(?:おすすめ|お勧め|オススメ)"), "推奨"),
    (re.compile(r"べき(?:です|だ|では|でしょう)"), "当為"),
    (re.compile(r"(?:の方が|のほうが|が)(?:良い|よい|いい)と思"), "意見"),
    (re.compile(r"(?:私は|個人的には|私としては)"), "意見"),
]
# 新しい観点・例の持ち込み(内容支援なしのときだけ疑う)
VIEWPOINT = [
    (re.compile(r"例えば"), "例示"),
    (re.compile(r"(?:の観点|の視点|の立場)(?:から|で|では)"), "観点の提示"),
]


def content_flags(action: Action, speech: str) -> list[str]:
    """許可されていない内容が混ざった疑いのラベル。内容支援のアクションが許可されるのは範囲の内側だけなので、
    範囲によらず「そのアクションで言ってよいこと」を基準にする。"""
    if not speech or action in (Action.no_intervention, Action.start_gd, Action.time_warning, Action.close_gd):
        return []
    if action in (Action.propose_idea, Action.offer_topic, Action.change_topic):
        # 具体案・話題(雑談の司会)は出してよいが、勧めたり意見を言ったりはしない
        patterns = OPINION
    elif action == Action.offer_hint:  # ヒントは観点・例まで。具体案や意見は含めない
        patterns = IDEA + OPINION
    else:  # 進行支援のアクションの本文には、どの範囲でも案・意見・新しい観点を混ぜない
        patterns = IDEA + OPINION + VIEWPOINT
    out: list[str] = []
    for pat, label in patterns:
        if pat.search(speech) and label not in out:
            out.append(label)
    return out
