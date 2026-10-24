"""議論タイプとフェーズの定義。

フェーズは議論の状況を理解するための目安で、順番どおりに進ませるものではない。
- observed_phase: LLM が議論内容から判断する実際のフェーズ(後戻りあり)
- target_phase: 経過時間の割合から計算する目標フェーズ(強制的な移行時刻ではない)
新しい議論タイプ(アイデア創出型・意思決定型・計画立案型など)は DISCUSSION_TYPES に追加する。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseDef:
    key: str  # 出力スキーマの選択肢になる
    label: str  # プロンプトと画面に出す名前
    signs: str  # このフェーズにいると判断する目安(いま参加者が主に何をしようとしているか)
    policy: str  # 介入を検討する状況の例(このフェーズにいるだけでは介入しない)
    target_end: float  # 目標の終了時点(制限時間に対する割合)


@dataclass(frozen=True)
class DiscussionType:
    key: str
    label: str
    phases: tuple[PhaseDef, ...]

    @property
    def keys(self) -> list[str]:
        return [p.key for p in self.phases]

    def get(self, key: str) -> PhaseDef | None:
        return next((p for p in self.phases if p.key == key), None)

    def index(self, key: str) -> int | None:
        return self.keys.index(key) if key in self.keys else None

    def target_phase(self, elapsed_fraction: float) -> PhaseDef:
        """経過時間の割合から目標フェーズを返す(目標の終了時点を過ぎていない最初のフェーズ)。"""
        return next((p for p in self.phases if elapsed_fraction < p.target_end), self.phases[-1])


PROBLEM_SOLVING = DiscussionType(
    key="problem_solving",
    label="課題解決型",
    phases=(
        PhaseDef(
            "problem_definition",
            "課題整理",
            "テーマの解釈、目指す状態、前提や範囲を確認している",
            "前提や目的の認識が食い違ったまま話がかみ合わないとき、何を課題とするかの確認を促す",
            0.15,
        ),
        PhaseDef(
            "cause_analysis",
            "原因分析",
            "なぜ問題が起きているかを挙げ、分類・絞り込みをしている",
            "原因が整理されず議論が止まっているとき、参加者が挙げた原因の整理を促す(AI は原因を挙げない)",
            0.35,
        ),
        PhaseDef(
            "solution_ideation",
            "解決策の検討",
            "対策案を出し、具体化している",
            "案が出ずに停滞したとき、既出の原因を手がかりに質問する(AI は案を出さない)",
            0.60,
        ),
        PhaseDef(
            "evaluation",
            "評価・比較",
            "案の長所・短所を比べ、優先順位をつけようとしている",
            "比較の方法が定まらず膠着したとき、どう比べるかの進め方を問う(基準を決めるのは参加者)",
            0.80,
        ),
        PhaseDef(
            "conclusion",
            "結論・まとめ",
            "結論を合意し、発表や確認の準備をしている",
            "結論が決まらないまま時間が迫るとき、未決事項を整理して参加者自身の決定を促す。結論が出たら見守る",
            1.0,
        ),
    ),
)

# 一般モード用。既存の3段階(案出し・比較検討・結論)をそのまま残した汎用型
GENERIC = DiscussionType(
    key="generic",
    label="汎用型",
    phases=(
        PhaseDef("ideation", "案出し", "意見やアイデアを出し合っている", "意見が出ずに停滞したとき、直前の発言を手がかりに質問する", 0.55),
        PhaseDef("comparison", "比較検討", "出た意見を比べ、絞り込もうとしている", "比べ方が定まらず膠着したとき、どう比べるかを問う", 0.80),
        PhaseDef("conclusion", "結論", "結論を合意し、まとめている", "結論が決まらないまま時間が迫るとき、未決事項を整理して決定を促す", 1.0),
    ),
)

DISCUSSION_TYPES = {t.key: t for t in (PROBLEM_SOLVING, GENERIC)}
DEFAULT_TYPE_BY_PURPOSE = {"job": PROBLEM_SOLVING.key, "general": GENERIC.key}


def get_discussion_type(key: str | None = None, purpose: str = "job") -> DiscussionType:
    """key を省略すると用途の既定タイプ(就活GDは課題解決型、一般は汎用型)を返す。"""
    if key in DISCUSSION_TYPES:
        return DISCUSSION_TYPES[key]
    return DISCUSSION_TYPES[DEFAULT_TYPE_BY_PURPOSE.get(purpose, PROBLEM_SOLVING.key)]
