"""議論タイプとフェーズの定義。

フェーズは議論の状況を理解するための目安で、順番どおりに進ませるものではない。
- observed_phase: LLM が議論内容から判断する実際のフェーズ(後戻りあり)
- target_phase: 経過時間の割合から計算する目標フェーズ(強制的な移行時刻ではない)
就活GDではユーザーが4種類(課題解決型・アイデア創出型・意思決定型・計画立案型)から選ぶ。一般GWは省略時に汎用型。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PhaseDef:
    key: str  # 出力スキーマの選択肢になる
    label: str  # プロンプトと画面に出す名前
    signs: str  # このフェーズにいると判断する目安(いま参加者が主に何をしようとしているか)
    policy: str  # 介入を検討する状況の例(このフェーズにいるだけでは介入しない)
    target_end: float  # 目標の終了時点(制限時間に対する割合)
    description: str = ""  # フェーズの目的(プロンプトに出す。課題解決型・汎用型は従来どおり空)


@dataclass(frozen=True)
class DiscussionType:
    key: str
    label: str
    phases: tuple[PhaseDef, ...]
    guidance: str = ""  # このタイプで進行を支援するときの要点(プロンプトに出す)
    # 内容支援(介入ポリシー⑥)が有効なときの要点。範囲(hint / concrete)ごと。なければ guidance を使う
    support_guidance: dict[str, str] = field(default_factory=dict, hash=False, compare=False)

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


IDEA_GENERATION = DiscussionType(
    key="idea_generation",
    label="アイデア創出型",
    guidance="アイデアを出すのは参加者です。発散中は量と多様さを大切にし、評価や絞り込みを急がせないでください。",
    support_guidance={
        "hint": "アイデアを出すのは主に参加者です。発散中は量と多様さを大切にし、評価や絞り込みを急がせないでください。"
                "案が出ずに止まったときは、考える切り口のヒント(誰の立場から、どんな場面で、など)を示してよいです。",
        "concrete": "アイデアを出すのは主に参加者です。発散中は量と多様さを大切にし、評価や絞り込みを急がせないでください。"
                    "案が出ずに止まったときは、呼び水として具体的なアイデアの例を1つ示してよいです(採用するかは参加者が決めます)。",
    },
    phases=(
        PhaseDef("purpose_check", "目的確認", "誰のために何を実現したいか、条件や前提を確かめている",
                 "目的や対象の認識がずれたまま案出しに入ってかみ合わないとき、目的の確認を促す", 0.15,
                 "アイデアの目的・対象・条件をそろえる"),
        PhaseDef("divergence", "アイデア発散", "評価せずにアイデアを次々に出し、他の人の案に乗っている",
                 "案が出ずに止まったとき、参加者の既出の案や目的を手がかりに別の切り口を問う(AI は案を出さない)", 0.55,
                 "できるだけ多様なアイデアを出す"),
        PhaseDef("development", "整理・発展", "出た案をまとめたり組み合わせたり、良い案を膨らませている",
                 "案が散らばったまま整理されず進まないとき、参加者の案をもとに整理の仕方を問う", 0.85,
                 "出た案を整理し、組み合わせて発展させる"),
        PhaseDef("wrap_up", "まとめ", "提案する案を選び、内容と理由をまとめている",
                 "まとめるべき案が決まらないまま時間が迫るとき、未決事項を整理して参加者自身の決定を促す", 1.0,
                 "グループとして提案する案をまとめる"),
    ),
)

DECISION_MAKING = DiscussionType(
    key="decision_making",
    label="意思決定型",
    guidance="選択肢・評価基準・比較・決定はすべて参加者が行います。AI は選択肢や基準を示さず、参加者が自分で基準を立てて比べられるよう支援してください。",
    support_guidance={
        "hint": "評価基準・比較・決定は参加者が行います。基準が出ずに止まったときは、考える観点のヒントを示してよいですが、"
                "どの選択肢がよいかは示しません。",
        "concrete": "比較と決定は参加者が行います。止まったときは、評価基準や比べ方の具体例を1つ示してよいですが、"
                    "どの選択肢を選ぶべきかは示しません。",
    },
    phases=(
        PhaseDef("options_review", "選択肢確認", "選べる選択肢と、それぞれの中身や前提を確かめている",
                 "選択肢の中身の理解がずれたまま話がかみ合わないとき、各選択肢の確認を促す", 0.2,
                 "比べる選択肢とその中身をそろえる"),
        PhaseDef("criteria_setting", "評価基準", "何を重視して選ぶか(基準と優先度)を話し合っている",
                 "基準がないまま好みの言い合いになるとき、どんな基準で比べるかを参加者に問う(AI は基準を示さない)", 0.45,
                 "何を基準に選ぶかを決める"),
        PhaseDef("comparison", "比較", "決めた基準で選択肢の長所・短所を比べている",
                 "同じ主張の繰り返しで平行線になったとき、参加者が決めた基準に沿って比べ直すよう促す", 0.8,
                 "基準に沿って選択肢を比べる"),
        PhaseDef("decision", "決定", "1つに決め、理由と懸念への対応をまとめている",
                 "決め切れないまま時間が迫るとき、決めるべき事項を中立に示して参加者自身の決定を促す", 1.0,
                 "グループとしての選択と理由を決める"),
    ),
)

PLANNING = DiscussionType(
    key="planning",
    label="計画立案型",
    guidance="作業の洗い出し・担当・日程の具体化は参加者が行います。AI は作業や計画の案を示さず、抜けや順序を参加者自身が確かめられるよう支援してください。",
    support_guidance={
        "hint": "作業の洗い出し・担当・日程は参加者が決めます。抜けを確かめる観点のヒント(準備・当日・事後、など)を示してよいです。",
        "concrete": "担当・日程と計画全体は参加者が決めます。作業が出ずに止まったときは、必要になりそうな作業の具体例を1つ示してよいです。",
    },
    phases=(
        PhaseDef("goal_setting", "目標整理", "何を、いつまでに、どの水準で達成するかを確かめている",
                 "目標や期限の認識がずれたまま作業の話に入ってかみ合わないとき、目標の確認を促す", 0.15,
                 "達成したい目標・期限・制約をそろえる"),
        PhaseDef("task_listing", "作業洗い出し", "必要な作業を挙げ、抜けがないかを確かめている",
                 "作業が出ずに止まったとき、参加者が決めた目標を手がかりに必要なことを問う(AI は作業を挙げない)", 0.45,
                 "必要な作業を漏れなく挙げる"),
        PhaseDef("plan_detailing", "計画具体化", "作業の順序・担当・日程・必要な資源を決めている",
                 "担当や順序が決まらず堂々巡りになるとき、決めるべき項目を参加者の言葉で整理するよう促す", 0.85,
                 "順序・担当・日程を決める"),
        PhaseDef("review", "確認", "計画全体を見直し、無理や抜け、リスクを確かめている",
                 "見直しが不十分なまま時間が迫るとき、未確認の点を中立に示して参加者自身の確認を促す", 1.0,
                 "計画全体の実現可能性を確かめる"),
    ),
)

# 雑談型: モード選択の「雑談」で使う(議論タイプの選択肢には出さない)。結論を出さない会話の練習・交流向けで、
# 話しやすさと発言機会の偏りへの配慮を中心にする。テーマを入れなかったときは CASUAL_FREE_THEME を使う
CASUAL_FREE_THEME = "自由な雑談"
CASUAL = DiscussionType(
    key="casual",
    label="雑談型",
    guidance="結論を出す必要はありません。何を話すかは参加者が決めます。話題がテーマから多少広がっても、雑談として自然なら軌道修正しないでください。"
             "全員が話しやすい雰囲気づくりと、話す機会の偏りへの配慮を中心にしてください。",
    support_guidance={
        "hint": "結論を出す必要はなく、何を話すかは参加者が決めます。話題が途切れたときは、話のきっかけになる問いかけや身近な切り口のヒントを示してよいです。",
        "concrete": "結論を出す必要はなく、何を話すかは参加者が決めます。話題が途切れたときは、具体的な話題の例を1つ示してよいです。",
    },
    phases=(
        PhaseDef("warm_up", "話題出し", "身近な話や経験から話し始め、話したい話題を出し合っている",
                 "話し始めが止まったとき、テーマに関する身近な問いかけで話のきっかけを作る", 0.25,
                 "話しやすい雰囲気で話題を出し合う"),
        PhaseDef("expand", "広げる", "話題を掘り下げたり別の話題へ自然に移ったりしながら、互いの話を聞き合っている",
                 "一部の人だけで盛り上がり、話せていない人がいるとき、全体に話を振る", 0.9,
                 "話題を広げ、互いの考えや経験を聞き合う"),
        PhaseDef("closing", "締め", "話してみた感想や、印象に残った話を共有している",
                 "時間が迫ったとき、印象に残った話を一言ずつ共有するよう促す(結論は求めない)", 1.0,
                 "話した内容を振り返り、気持ちよく終える"),
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

DISCUSSION_TYPES = {t.key: t for t in (PROBLEM_SOLVING, IDEA_GENERATION, DECISION_MAKING, PLANNING, CASUAL, GENERIC)}
SELECTABLE_TYPES = [PROBLEM_SOLVING.key, IDEA_GENERATION.key, DECISION_MAKING.key, PLANNING.key]  # ユーザーが選べる4種類
DEFAULT_TYPE_BY_PURPOSE = {"job": PROBLEM_SOLVING.key, "general": GENERIC.key}


def get_discussion_type(key: str | None = None, purpose: str = "job") -> DiscussionType:
    """key を省略すると用途の既定タイプ(就活GDは課題解決型、一般は汎用型)を返す(評価・内部用)。"""
    if key in DISCUSSION_TYPES:
        return DISCUSSION_TYPES[key]
    return DISCUSSION_TYPES[DEFAULT_TYPE_BY_PURPOSE.get(purpose, PROBLEM_SOLVING.key)]


def resolve_session_type(key: str | None, purpose: str) -> DiscussionType:
    """セッション開始時の議論タイプ。省略(画面の「指定なし」)は就活GD・一般GWとも汎用型(2026-10-25 ユーザー決定)。
    雑談型は「雑談」モード(一般)だけ。不明な値はエラー。"""
    if not key:
        return GENERIC
    allowed = SELECTABLE_TYPES + [GENERIC.key] + ([] if purpose == "job" else [CASUAL.key])
    if key not in allowed:
        raise ValueError(f"選択できない議論タイプです: {key}")
    return DISCUSSION_TYPES[key]
