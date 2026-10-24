"""脱線の軌道修正を待つ間に、参加者が本題に戻ったかを LLM を呼ばずに推定する(保守的な簡易判定)。

戻ったと判断する根拠は次のどちらかだけ。新しい発言が増えたこと自体は理由にしない(脱線が続いているだけかもしれない)。
- 本題に戻ることを明示する言い回し(「話を戻すと」「本題」など)
- 1つの発言に、本題の語が2つ以上含まれる(1語だけでは断定しない)
本題の語 = テーマと、軌道修正を決めた LLM の分析(論点・意見・合意)に出る語から、脱線部分の発言に出る語を除いたもの。

限界: 本題の語を使わずに戻った場合(例:「じゃあ次に進みましょう」)は検出できない。分かち書きをしないので、
漢字・カタカナの連続を語とみなす粗い判定で、言い換えや表記ゆれにも弱い。
"""

from __future__ import annotations

import re

from facilitator.models import Analysis

RETURN_PHRASES = ("本題", "話を戻", "話、戻", "テーマに戻", "テーマの話", "脱線", "話がそれ", "話それ", "それはさておき", "戻りましょう")
_TERM = re.compile(r"[一-龥々]{2,}|[ァ-ヶー]{2,}|[A-Za-z]{3,}")
# 雑談にも本題にも出やすい一般的な語は、本題の語として数えない
_STOP = {"時間", "最近", "今度", "自分", "本当", "一番", "全部", "結構", "意見", "感じ", "普通", "大丈夫", "今日", "場合", "方法"}


def topic_terms(theme: str, analysis: Analysis, drift_texts: list[str]) -> set[str]:
    src = [theme, *analysis.issues, *analysis.opinions, *analysis.agreements]
    terms = {t for s in src for t in _TERM.findall(s)} - _STOP
    drift = {t for s in drift_texts for t in _TERM.findall(s)}
    return terms - drift


def detect_topic_return(theme: str, analysis: Analysis, drift_texts: list[str], new_texts: list[str]) -> str | None:
    """本題に戻ったと判断できる根拠があればその説明を返す。なければ None(脱線が続いているとみなす)。"""
    terms = topic_terms(theme, analysis, drift_texts)
    for text in new_texts:
        phrase = next((p for p in RETURN_PHRASES if p in text), None)
        if phrase:
            return f"言い回し「{phrase}」"
        hits = sorted(t for t in terms if t in text)
        if len(hits) >= 2:
            return f"本題の語「{'・'.join(hits[:3])}」"
    return None
