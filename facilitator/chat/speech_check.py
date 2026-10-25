"""雑談の司会の発言の確認(第5段階の評価で見つかった、プロンプトだけでは防ぎにくい明確な問題だけを扱う)。

- 1回の発言に問いかけの文が2つ以上あれば、最初の問いかけだけを残す(生成しながら話す場合と同じ)(1文に2つの問いが入る場合は区別できない。プロンプトで防ぐ)
- 最近の AI の問いかけと実質的に同じ問いかけか(「ほかの皆さんは、」などの呼びかけと前置きを外して比べる)
- いまの話題で全員が話した後の「ほかの皆さん」「ほかの方」を「皆さん」にする
- 最近の AI の発言で使った定型の相づち(「盛り上がりましたね」など)の文を外す
どれも粗い判定で、正常な発言を止めすぎないよう、はっきりした場合だけにする。
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

SENTENCE = re.compile(r"[^。!?]+[。!?]*")
LEADS = ("話は変わりますが、", "ところで、", "では、", "まずは、")  # 話題の文・問いかけから外す前置き
_ADDRESS = re.compile(r"^((ほか|他)の)?(皆さん|みなさん|方|かた)(は|も|って)?、?")
_OTHERS = re.compile(r"(ほか|他)の(皆さん|みなさん|方々|方|かた)")
_PUNCT = re.compile(r"[\s、。,.!?…〜~「」『』]+")
FORMULAS = ("盛り上がりましたね", "盛り上がってますね", "盛り上がっていますね", "楽しそうですね", "素敵ですね")
SAME_QUESTION = 0.95  # 呼びかけと前置きを外した問いかけどうしの一致度(difflib)。これ以上なら同じ問い
CONTAINED = 0.8  # 一方がもう一方に含まれ、長さがこの割合以上なら同じ問い(「など」を足しただけ、など)


def normalize(text: str) -> str:
    return text.replace("？", "?").replace("！", "!").strip()


def sentences(text: str) -> list[str]:
    return [x.strip() for x in SENTENCE.findall(normalize(text)) if x.strip(" 　。!?")]


def is_question(sentence: str) -> bool:
    s = normalize(sentence)
    return "?" in s or s.rstrip("。").endswith("か")


def questions(text: str) -> list[str]:
    return [s for s in sentences(text) if is_question(s)]


def question_core(sentence: str) -> str:
    """問いかけの中身(前置き・「ほかの皆さんは、」などの呼びかけ・句読点を外したもの)。"""
    s = normalize(sentence)
    for lead in LEADS:
        s = s.split(lead)[-1]
    s = _ADDRESS.sub("", s.strip())
    return _PUNCT.sub("", s)


def same_question(a: str, b: str) -> bool:
    """呼びかけ・前置きを外すと同じか、ほぼ同じ(一方がもう一方をほぼ含む / 一致度が SAME_QUESTION 以上)。
    言い換え・絞り込み(「休みの日は」→「休みの日の朝は」)は止めない(プロンプトで防ぐ)。"""
    ca, cb = question_core(a), question_core(b)
    if not ca or not cb:
        return False
    short, long_ = sorted((ca, cb), key=len)
    return ca == cb or short in long_ and len(short) >= CONTAINED * len(long_) or (
        SequenceMatcher(None, ca, cb).ratio() >= SAME_QUESTION)


def repeated_question(speech: str, recent_ai: list[str]) -> str | None:
    """発言の問いかけのうち、最近の AI の発言の問いかけと同じもの(なければ None)。"""
    past = [q for r in recent_ai for q in questions(r)]
    return next((q for q in questions(speech) if any(same_question(q, p) for p in past)), None)


def one_question(speech: str) -> str:
    """問いかけの文が2つ以上あれば、最初の問いかけだけを残す(相づちなど問いでない文は残す。生成しながら話す場合も
    2つ目の問いの文から先は話さないので、それとそろえる)。"""
    sents = sentences(speech)
    qs = [i for i, s in enumerate(sents) if is_question(s)]
    if len(qs) <= 1:
        return speech
    return "".join(s for i, s in enumerate(sents) if not is_question(s) or i == qs[0])


def without_others(speech: str) -> str:
    """「ほかの皆さん」「ほかの方」を「皆さん」にする(全員が話した後の呼びかけ)。"""
    return _OTHERS.sub("皆さん", speech)


def without_repeated_formula(speech: str, recent_ai: list[str]) -> str:
    """最近の AI の発言で使った定型の相づちを含む、問いでない文を外す(残りがなければ空文字)。"""
    used = {f for f in FORMULAS if any(f in r for r in recent_ai)}
    if not used:
        return speech
    return "".join(s for s in sentences(speech) if is_question(s) or not any(f in s for f in used))
