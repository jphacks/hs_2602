"""生成途中の JSON(構造化出力)から文字列フィールドを取り出し、完成した文を順に渡す。

LLM の返事を全部待たずに、発言を一文できるたびに表示・読み上げるために使う。
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

_ESC = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f"}
_SENT_END = re.compile(r"(?<=[。!?!?\n])")


def partial_string(text: str, key: str) -> tuple[str | None, bool]:
    """途中までの JSON テキストから "key": "..." の値を取り出す。(値, 閉じたか) を返す。
    キーがまだ現れていなければ (None, False)。エスケープの途中で切れていれば、その手前までを返す。"""
    m = re.search(r'"%s"\s*:\s*"' % re.escape(key), text)
    if not m:
        return None, False
    out: list[str] = []
    i = m.end()
    while i < len(text):
        c = text[i]
        if c == '"':
            return "".join(out), True
        if c == "\\":
            if i + 1 >= len(text):
                break
            n = text[i + 1]
            if n == "u":
                if i + 6 > len(text):
                    break
                out.append(chr(int(text[i + 2 : i + 6], 16)))
                i += 6
                continue
            out.append(_ESC.get(n, n))
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out), False


class SentenceFeeder:
    """生成途中の JSON を受け取り、key の値から完成した文(。!? や改行で終わる)を on_sentence に順に渡す。"""

    def __init__(self, key: str, on_sentence: Callable[[str], None]) -> None:
        self.key = key
        self.on_sentence = on_sentence
        self._used = 0  # 渡し終えた文字数
        self._last: str | None = None

    def feed(self, acc: str) -> None:
        val, closed = partial_string(acc, self.key)
        if val is None:
            return
        self._last = val
        self._emit(val, closed)

    def finish(self) -> None:
        """生成が終わったら、句点のない最後の文も渡す。"""
        if self._last is not None:
            self._emit(self._last, True)

    def _emit(self, val: str, final: bool) -> None:
        parts = _SENT_END.split(val[self._used :])
        for p in parts if final else parts[:-1]:
            self._used += len(p)
            if p.strip():
                self.on_sentence(p.strip())


def simulate_stream(
    obj: dict[str, Any], on_partial: Callable[[str], None], chunk: int = 6, delay: float = 0.0, slow_key: str | None = None
) -> None:
    """モック用: JSON を少しずつ渡して、ストリーミングを再現する。slow_key を渡すと、その値の部分だけ delay ずつ待つ。"""
    text = json.dumps(obj, ensure_ascii=False)
    slow_from = text.find(f'"{slow_key}"') if slow_key else 0
    for i in range(chunk, len(text) + chunk, chunk):
        on_partial(text[:i])
        if delay and slow_from >= 0 and i >= slow_from:
            time.sleep(delay)
