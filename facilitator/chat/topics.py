"""雑談の話題の候補(開始の話題と、途中で LLM に参考として渡す候補)。

初対面でも答えやすい、身近で個人的すぎない話題だけを用意する(家族・恋愛・収入・健康・政治・宗教などは入れない)。
テーマが指定されていればテーマに沿った候補を、空(自由な雑談)なら一般的な候補を使う。
話題の履歴の管理・重複の防止は第4段階(ここでは、出した話題を除いて候補を返すだけ)。
"""

from __future__ import annotations

import random

from facilitator.discussion_types import CASUAL_FREE_THEME

# 自由な雑談の候補(問いかけの形。そのまま話せる1文)
FREE_TOPICS = [
    "最近ちょっとハマっていることはありますか?",
    "休みの日は、どんなふうに過ごすことが多いですか?",
    "最近食べたもので、おいしかったものはありますか?",
    "最近行ってよかった場所はありますか?",
    "朝型か夜型かでいうと、どちらですか?",
    "最近見て面白かった動画や番組はありますか?",
    "もし丸一日自由な日があったら、何をしたいですか?",
    "この季節になると、したくなることはありますか?",
]

# テーマの言葉で選ぶ候補(キーワードのどれかがテーマに含まれれば使う)
THEMED_TOPICS: list[tuple[tuple[str, ...], list[str]]] = [
    (("休日", "休み", "週末", "連休"), [
        "最近の休みの日は、どんなふうに過ごしましたか?",
        "休みの日は、外に出かける派ですか、家でゆっくりする派ですか?",
    ]),
    (("趣味", "ハマ", "好きなこと"), [
        "最近ちょっとハマっていることはありますか?",
        "昔から続けている趣味はありますか?",
    ]),
    (("食", "料理", "グルメ", "ご飯", "ごはん"), [
        "最近食べたもので、おいしかったものはありますか?",
        "よく作る料理や、よく行くお店はありますか?",
    ]),
    (("旅行", "旅", "観光", "おでかけ", "お出かけ"), [
        "これまでに行ってよかった場所はありますか?",
        "今行ってみたい場所はどこですか?",
    ]),
    (("学校", "大学", "学生", "授業", "サークル", "部活"), [
        "学生生活で、楽しかった思い出はありますか?",
        "授業やサークルで、最近あった出来事はありますか?",
    ]),
    (("仕事", "働", "職場", "バイト", "アルバイト"), [
        "お仕事やアルバイトで、最近ちょっとうれしかったことはありますか?",
        "仕事の合間の息抜きは、どんなことをしていますか?",
    ]),
    (("音楽", "映画", "本", "読書", "ゲーム", "アニメ", "ドラマ"), [
        "最近見たり聞いたりして、よかったものはありますか?",
        "昔から好きな作品はありますか?",
    ]),
]

# キーワードに当たらないテーマ用(テーマの言葉を入れて使う)
THEME_TEMPLATES = [
    "「{theme}」と聞いて、まず思い浮かぶことは何ですか?",
    "「{theme}」について、最近の身近な出来事があれば聞かせてください。",
]


def is_free(theme: str) -> bool:
    return not theme.strip() or theme == CASUAL_FREE_THEME


def topic_candidates(theme: str, used: list[str] | None = None) -> list[str]:
    """テーマに沿った話題の候補(出した話題は除く)。テーマの候補を使い切ったら、一般的な候補を足す。"""
    used_set = set(used or [])
    if is_free(theme):
        pool = list(FREE_TOPICS)
    else:
        pool = [q for keys, qs in THEMED_TOPICS if any(k in theme for k in keys) for q in qs]
        pool = pool or [t.format(theme=theme) for t in THEME_TEMPLATES]
        pool += FREE_TOPICS
    out = []
    for q in pool:
        if q not in used_set and q not in out:
            out.append(q)
    return out


def opening_topic(theme: str, rng: random.Random | None = None) -> str:
    """開始の話題(候補の先頭の数個から選ぶ。テーマがあればテーマに沿ったもの)。"""
    rng = rng or random.Random()
    cands = topic_candidates(theme)
    if not is_free(theme):  # テーマに沿った候補だけから選ぶ(一般的な候補は後ろに足してある)
        themed = [q for q in cands if q not in FREE_TOPICS] or cands
        return rng.choice(themed)
    return rng.choice(cands)


def opening_speech(theme: str, topic: str) -> str:
    """開始のあいさつと最初の話題(時間や進め方の堅い説明はしない)。"""
    if is_free(theme):
        return f"こんにちは、今日はよろしくお願いします。気軽にお話ししましょう。まずは、{topic}"
    return f"こんにちは、今日はよろしくお願いします。今日は「{theme}」について、気軽にお話ししましょう。まずは、{topic}"
