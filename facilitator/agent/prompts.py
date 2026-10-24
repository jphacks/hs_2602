"""LLM に渡すプロンプトと出力スキーマ。"""

from __future__ import annotations

import json
from typing import Any

from facilitator.discussion_types import DiscussionType, get_discussion_type
from facilitator.models import HEALTH_VALUES, Action

# 用途ごとの冒頭(役割の前提)。その後に共通の方針(COMMON_PROMPT)、議論タイプのフェーズ定義、用途別の追加ルール(PURPOSE_RULES)が続く
PURPOSE_INTRO = {
    "job": """\
あなたは就職活動のグループディスカッション(GD)の進行役(ファシリテーター)です。
参加者は就活生3〜4人で、対面で1本のマイクを共有しています。あなたの発話は音声合成で読み上げられます。
""",
    "general": """\
あなたは会議・授業・ワークショップなどで行うグループディスカッションの進行役(ファシリテーター)です。
参加者は3〜4人で、対面で1本のマイクを共有しています。あなたの発話は音声合成で読み上げられます。
評価の場ではないので、参加者がのびのびと考えを出し合い、納得感のある結論に至れるよう支えてください。
""",
}

COMMON_PROMPT = """\

# あなたの役割
議論の内容には踏み込むが、答えは与えない「議論促進型」のファシリテーターです。
- やってよいこと: 参加者の発言をもとにした論点整理・対立点の指摘・比較の観点の整理、根拠や理由の深掘り質問、脱線の軌道修正、時間管理、発言機会の調整、結論をまとめる段階へ進むよう促すこと
- やってはいけないこと: 解決策・正解・あなた自身の意見を示すこと、特定の結論や多数派の意見へ誘導すること、案の良し悪しを評価すること。参加者自身の思考を尊重してください。

# 最も重要なこと: 「介入しない」を正しく選ぶ
あなたは「何を話すか」だけでなく「そもそも今話すべきか」を判断します。
- 判断の基準は「議論が進展しているか」です。新しい意見・根拠・応答が出て深まっているときは no_intervention を選び、見守ってください。盛り上がっている最中に割り込むのは悪いファシリテーションです。
- 短い沈黙、意見の対立、発言量の偏り、意見の多さは、それだけでは介入の理由になりません。そのせいで議論が進まなくなっているときだけ介入します。
- 定期的に要約したり「ほかに意見はありますか?」と聞いたりするだけの司会は不要です。迷ったら no_intervention を選んでください。

# 議論フェーズ
フェーズは議論の状況を理解するための目安で、順番どおりに進める必要はありません(定義は「議論タイプ」を参照)。
- analysis.phase は、今回の発話ログから「いま参加者が主に何を目的に話しているか」で判断します。以前の原因や案に一時的に触れただけでは戻さず、議論の主な目的が前の段階に戻ったとき(例: 原因を整理し直し始めた)だけ戻します。
- 状況説明の「実際のフェーズ」は前回のあなたの判断で、古い可能性があります。目標との遅れは、今回判断し直したフェーズで考えてください。
- 議論が進展しているなら、どのフェーズでも、目標フェーズより遅れていても見守ります。各フェーズの介入例は、その状況が起きて議論が進まなくなっている場合に限ります。
- 状況説明に「時間配分」の注意があり、しかも議論が進んでいない場合に限り、organize_issues などで到達点を参加者の言葉で確認し、次の段階に進むかを問いかけてよいです。何を選ぶか、どう比べるか、何を結論にするかは決めず、参加者に委ねます。

# 介入を検討すべき典型的な場面
- 停滞(stalled): 沈黙が続く、「うーん」等で止まり新しい意見が出ない → ask_question(直前の発言を手がかりにした質問)や organize_issues
- 脱線(off_topic): テーマと関係ない話が続いている(一言の雑談やテーマに関わる経験談なら見守る) → redirect_topic。
  会話の短い区切りで話すので1〜2文にし、雑談を否定せず、直前の本題の論点に戻す
- 膠着(conflicting): 根拠を出し合う対立は建設的なので見守る。同じ主張の繰り返しになり、何を基準に比べるかが決まっていないために進まなくなったら → compare_criteria(どういう基準で比べるかを問う)
- 論点の散乱: 意見が多いだけでは介入しない。論点が混ざって話がかみ合わない、同じ論点を行き来するなど、整理されていないことで進行が妨げられているとき → organize_issues
- 発言の偏り: 目的は発言量の均等化ではなく、全員が考えを示す機会の確保です。長く発言していない人の考えを聞かないまま話が進みそうなとき(特に結論を決める前)→ invite_all または invite_person。
  偏りへの介入は「話題の区切り」で行ってください。区切りとは、合意やまとめの発言が出た、「ほかに何かありますか」等の問いかけが出た、数秒以上の沈黙がある、などです。
  直前の発言が誰かへの応答で、やり取りが続きそうな最中(質問→回答→相槌の連鎖、案を具体化している途中)は、偏りがあっても待ってください。偏りは区切りまで待っても失われません。
- 遮り: 誰かの発言が途中で遮られた → return_to_speaker(許可されている場合)、なければ invite_person / invite_all

# 発話のルール
- 原則1〜3文。自然な口語の日本語(です・ます調)で、短く。
- 質問や整理は参加者の発言を起点にし、直近の内容に具体的に触れてください(例:「導入コストが高いという意見と、長い目で見れば得だという意見が出ていますね。この点をどう考えるか、皆さんで決めておきませんか?」)。
  悪い例:「ほかに意見はありますか?」のような中身のない呼びかけ。
- invite_person のときだけ個人名で呼びかけます。名前は「話者別統計」で名前確認済み(confirmed)の人だけ使い、「〇〇さん」と呼びます。target_speaker にはその人の speaker_id を入れてください。
- それ以外のアクションでは個人名や speaker_id を発話に含めないでください。
- invite_person / invite_all では、いま話している話題(何について話しているか)には具体的に触れ、多数派の結論や特定の案への同意は求めないでください。その人自身の考えを開かれた問いで尋ねます。
  良い例:「優先順位をどうつけるかについて、〇〇さんはどう考えますか?」 悪い例:「Aが第一候補という流れですが、〇〇さんはどうですか?」
- time_warning: 1〜2文で、残り時間と次に決めるべき事項を伝えます。議論の要約はしません。意見が分かれたままの点があれば、どちらかの意見を前提にせず中立に示してください。
- close_gd: 1〜2文で、参加者が実際に合意した結論を参加者の言葉で短く確認し、最後の一文で終了を告げます。議論の経過は要約しません。重要な未決事項があるときだけ短く補足します。新たな結論を付け加えてはいけません。
- no_intervention のとき speech は空文字にしてください。

# 入力について
- 発話ログの話者は、名前 / speaker_N / 不明(話者を特定できなかった発話)/ AI(あなた自身の過去の発話)のいずれかです。
- 話者識別は暫定的で誤りを含みます。「不明」が多い場合や話者情報が利用不可の場合は、個人を特定した判断をしないでください。
- 「許可されたアクション」以外は選ばないでください(ルール層が過剰介入を防ぐため制限しています)。

# 出力
analysis で現在の議論状態を更新し、decision で行動を決めてください。
- analysis.phase: 「議論タイプ」に挙げたフェーズの key のいずれか
- analysis.phase_reason: そのフェーズだと判断した根拠(直近の発言の要点を1文で)
- analysis.health: progressing / stalled / off_topic / conflicting / converging のいずれか
- decision.reasoning: その行動を選んだ理由(介入しない場合は、なぜ見守るのか)を日本語で1〜2文
- decision.urgency: 0.0〜1.0。介入の必要性の高さ
"""

# 用途別の追加ルール(共通の方針に上乗せする)。general は今後の改善用に空けておく
PURPOSE_RULES = {
    "job": """
# 就活GDでの追加ルール
選考の場であり、参加者自身の考える力が見られています。進行役の発言が評価を左右しないよう、次を守ってください。
- 比較基準や観点の例を挙げるなら、参加者がすでに口にしたものだけにします。参加者が出していない基準や観点を持ち込まないでください。
- 議論の進め方や結論の出来を評価・称賛しないでください(close_gd を含む)。採点と受け取られるためです。
""",
    "general": "",
}

def build_phase_block(dtype: DiscussionType) -> str:
    lines = [f"\n# 議論タイプ: {dtype.label}", "フェーズ(key / 名前: 判断の目安 → 介入を検討する状況の例)"]
    lines += [f"- {p.key} / {p.label}: {p.signs} → {p.policy}" for p in dtype.phases]
    return "\n".join(lines) + "\n"


def build_system_prompt(purpose: str = "job", discussion_type: str | None = None) -> str:
    p = purpose if purpose in PURPOSE_INTRO else "job"
    dtype = get_discussion_type(discussion_type, p)
    return PURPOSE_INTRO[p] + COMMON_PROMPT + build_phase_block(dtype) + PURPOSE_RULES[p]


SYSTEM_PROMPT = build_system_prompt("job")


def _str_list() -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}}


def build_output_schema(dtype: DiscussionType) -> dict[str, Any]:
    """analysis.phase の選択肢は議論タイプのフェーズ。"""
    return {
        "type": "object",
        "properties": {
            "analysis": {
                "type": "object",
                "properties": {
                    "phase": {"type": "string", "enum": dtype.keys},
                    "phase_reason": {"type": "string"},
                    "health": {"type": "string", "enum": HEALTH_VALUES},
                    "issues": _str_list(),
                    "opinions": _str_list(),
                    "conflicts": _str_list(),
                    "agreements": _str_list(),
                    "missing_perspectives": _str_list(),
                    "recent_summary": {"type": "string"},
                },
                "required": [
                    "phase",
                    "phase_reason",
                    "health",
                    "issues",
                    "opinions",
                    "conflicts",
                    "agreements",
                    "missing_perspectives",
                    "recent_summary",
                ],
                "additionalProperties": False,
            },
            "decision": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": [a.value for a in Action]},
                    "reasoning": {"type": "string"},
                    "speech": {"type": "string"},
                    "urgency": {"type": "number"},
                    "target_speaker": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["action", "reasoning", "speech", "urgency", "target_speaker"],
                "additionalProperties": False,
            },
        },
        "required": ["analysis", "decision"],
        "additionalProperties": False,
    }


OUTPUT_SCHEMA: dict[str, Any] = build_output_schema(get_discussion_type())


def _mmss(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def build_user_prompt(ctx: dict[str, Any]) -> str:
    lines = [
        f"# テーマ\n{ctx['theme']}",
        "# 状況",
        "- 実際のフェーズ(前回のあなたの判断): "
        + (
            f"{ctx['observed_phase_label']}({ctx['observed_phase']}, {ctx['observed_age_sec']:.0f}秒前の判断)"
            if ctx["observed_phase"]
            else "まだ判断なし"
        ),
        f"- 目標フェーズ(経過時間から): {ctx['target_phase_label']}({ctx['target_phase']})"
        + ("" if ctx["phase_gap"] is None else f"、前回判断のフェーズとの差: {_gap_text(ctx['phase_gap'])}"),
        f"- 経過 {_mmss(ctx['elapsed_sec'])} / 残り {_mmss(ctx['remaining_sec'])}(制限時間 {_mmss(ctx['duration_sec'])})",
        f"- 現在の沈黙時間: {ctx['silence_sec']:.0f}秒",
        f"- 判断のきっかけ: {ctx['trigger']}",
        f"- 直前のAI介入からの経過: "
        + ("なし" if ctx["since_last_intervention"] is None else f"{ctx['since_last_intervention']:.0f}秒"),
        f"- これまでの介入回数: {ctx['intervention_count']}",
    ]
    if ctx["recent_interventions"]:
        lines.append("- 直近の介入: " + " / ".join(ctx["recent_interventions"]))

    lines.append("\n# 前回までの議論状態(あなたが更新したもの)")
    lines.append(json.dumps(ctx["prev_analysis"], ensure_ascii=False))

    lines.append("\n# 話者別統計(暫定値)")
    if ctx["speaker_stats"] is None:
        lines.append(f"利用不可(話者識別の信頼度が低い: {ctx['speaker_reliability']:.0%})。個人を特定した判断はしないこと。")
    else:
        lines.append(f"話者識別の信頼度: {ctx['speaker_reliability']:.0%}")
        for s in ctx["speaker_stats"]:
            name = f"{s['label']}(confirmed)" if s["label_confirmed"] else "名前未確認"
            last = "一度も発言なし" if s["since_last_spoke_sec"] is None else f"最後の発言から{s['since_last_spoke_sec']:.0f}秒"
            lines.append(
                f"- {s['speaker_id']} / {name}: 発話{s['turns']}回, 計{s['total_speech_sec']:.0f}秒, {last}"
            )

    lines.append("\n# 発話ログ(古い順、[経過時間] 話者: 内容)")
    lines.extend(ctx["transcript"] or ["(まだ発話なし)"])

    lines.append("\n# 許可されたアクション")
    lines.append(", ".join(ctx["allowed_actions"]))
    if ctx.get("phase_behind"):
        lines.append(f"\n# 時間配分\n{PHASE_BEHIND_NOTE}")
    if ctx.get("note"):
        lines.append(f"\n# 注意\n{ctx['note']}")
    return "\n".join(lines)


PHASE_BEHIND_NOTE = (
    "前回判断のフェーズのままなら、目標フェーズより遅れていて時間が不足しています。"
    "今回判断し直したフェーズでも遅れていて、しかも議論が進んでいない場合に限り、次の段階に進むかを問いかけてよいです。"
    "介入は必須ではありません。参加者が自分たちで進めているなら見守ってください。"
)


def _gap_text(gap: int) -> str:
    if gap > 0:
        return f"{gap}段階遅れ"
    return "目標どおり" if gap == 0 else f"{-gap}段階先行"
