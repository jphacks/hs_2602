"""1つの文字起こし(ターン)を文ごとに分け、文ごとに話者を判定するための補助。

STT のターンは、複数人が間をあけずに続けて話すと1つにまとまることがある。そこで文(。!? で区切る)ごとに
音声の区間を見積もって声を照合し、同じ人が続く文はまとめ直して、別々の発言として記録する。
日本語では単語ごとの時刻がほとんど返らない(1ターン=1語)ため、文の区間は文字数の割合で見積もる。
単語の時刻が複数あるときは、それを使って見積もる。
"""

from __future__ import annotations

import re
from typing import Any

_SENT_END = re.compile(r"(?<=[。!?！？])")


def split_sentences(text: str) -> list[str]:
    return [p for p in _SENT_END.split(text.strip()) if p.strip()]


def sentence_spans(parts: list[str], start: float, end: float, words: list[dict[str, Any]] | None = None) -> list[tuple[float, float]]:
    """各文の (開始, 終了) 秒を見積もる。"""
    total = sum(len(p) for p in parts) or 1
    # 文字の位置 → 時刻 の対応(単語の時刻が2つ以上あれば単語ごと、なければターン全体を文字数で按分)
    marks: list[tuple[int, float, float]] = []  # (単語の先頭の文字位置, 開始, 終了)
    usable = [w for w in (words or []) if w.get("text") and w.get("start") is not None and w.get("end") is not None]
    if len(usable) >= 2:
        pos = 0
        for w in usable:
            marks.append((pos, w["start"] / 1000, w["end"] / 1000))
            pos += len(w["text"])
        scale = total / max(1, pos)  # 句読点の有無などで長さが少し違っても比で合わせる
        marks = [(int(p * scale), s, e) for p, s, e in marks]

    def at(char_pos: float) -> float:
        if not marks:
            return start + (end - start) * char_pos / total
        prev = marks[0]
        for m in marks:
            if m[0] > char_pos:
                break
            prev = m
        nxt = next((m for m in marks if m[0] > prev[0]), None)
        span_chars = (nxt[0] if nxt else total) - prev[0]
        frac = 0.0 if span_chars <= 0 else (char_pos - prev[0]) / span_chars
        return prev[1] + (prev[2] - prev[1]) * min(1.0, max(0.0, frac))

    spans, pos = [], 0
    for p in parts:
        # 文の終わりは最後の文字の直後(次の単語の始まりではなく、この文の最後の単語の終わり)
        spans.append((round(at(pos), 3), round(at(pos + len(p) - 1e-6), 3) if pos + len(p) < total else end))
        pos += len(p)
    return spans


def fill_labels(labels: list[tuple[str, float] | None], unknown: str = "unknown") -> list[tuple[str, float]]:
    """短すぎて判定できなかった文・判定が不確かな文は、前(なければ後ろ)の文の話者を引き継ぐ。"""
    known = [lb if lb and lb[0] != unknown else None for lb in labels]
    out: list[tuple[str, float] | None] = list(known)
    for i in range(len(out)):
        if out[i] is None and i > 0:
            out[i] = out[i - 1]
    for i in range(len(out) - 2, -1, -1):
        if out[i] is None:
            out[i] = out[i + 1]
    return [lb if lb is not None else (unknown, 0.0) for lb in out]


def group_by_speaker(parts: list[str], spans: list[tuple[float, float]], labels: list[tuple[str, float]]) -> list[tuple[str, float, float, str, float]]:
    """同じ話者が続く文をまとめる。(文, 開始, 終了, 話者, 確信度) のリスト。"""
    groups: list[list[Any]] = []
    for text, (s, e), (sid, conf) in zip(parts, spans, labels):
        if groups and groups[-1][3] == sid:
            g = groups[-1]
            g[0] += text
            g[2] = e
            g[4] = min(g[4], conf)
        else:
            groups.append([text, s, e, sid, conf])
    return [tuple(g) for g in groups]  # type: ignore[misc]
