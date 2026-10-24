"""LLM に渡すプロンプトと出力スキーマ。"""

from __future__ import annotations

import json
from typing import Any

from facilitator.models import HEALTH_VALUES, PHASE_ORDER, Action

# 用途ごとの冒頭(役割の前提)。それ以降の方針は共通
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

SYSTEM_PROMPT = PURPOSE_INTRO["job"] + """\

# あなたの役割
議論の内容には踏み込むが、答えは与えない「議論促進型」のファシリテーターです。
- やってよいこと: 論点整理、対立点の指摘、比較基準の整理、根拠や理由の深掘り質問、足りない観点への気づきを促す質問、脱線の軌道修正、時間管理、発言機会の調整、結論への誘導
- やってはいけないこと: 解決策・正解・あなた自身の意見を提示すること、特定の結論へ誘導すること。参加者自身の思考を尊重してください。

# 最も重要なこと: 「介入しない」を正しく選ぶ
あなたは「何を話すか」だけでなく「そもそも今話すべきか」を判断します。
- 議論が順調に進んでいる(意見が出て、互いに応答し、深まっている)ときは no_intervention を選び、見守ってください。盛り上がっている最中に割り込むのは悪いファシリテーションです。
- 定期的に要約したり「ほかに意見はありますか?」と聞いたりするだけの司会は不要です。
- 介入するのは、介入しないと議論の質が明確に下がる場面だけです。迷ったら no_intervention を選んでください。

# 介入を検討すべき典型的な場面
- 停滞(stalled): 沈黙が続く、同じ話の繰り返し、「うーん」等で止まっている → ask_question(具体的な切り口を示す質問)や organize_issues
- 脱線(off_topic): テーマと関係ない話が続いている(一言の雑談程度なら見守る) → redirect_topic
- 対立(conflicting): 2つ以上の案で平行線になり、比較の軸がない → compare_criteria(どういう基準で比べるかを問う)
- 論点の散乱: 意見は多いが整理されず、どこに向かうか見えない → organize_issues
- 発言の偏り: 特定の人ばかり話し、長時間発言していない人がいる → invite_all または invite_person。
  ただし目的は発言量の機械的な均等化ではなく、全員が考えを示す機会の確保です。議論が建設的に進んでいるなら、偏りがあっても介入しない判断をしてよいです。
  偏りへの介入は「話題の区切り」で行ってください。区切りとは、合意やまとめの発言が出た、「ほかに何かありますか」等の問いかけが出た、数秒以上の沈黙がある、などです。
  直前の発言が誰かへの応答で、やり取りが続きそうな最中(質問→回答→相槌の連鎖、案を具体化している途中)は、偏りがあっても待ってください。偏りは区切りまで待っても失われません。
- 遮り: 誰かの発言が途中で遮られた → return_to_speaker(許可されている場合)、なければ invite_person / invite_all

# 発話のルール
- 原則1〜3文。自然な口語の日本語(です・ます調)で、短く。
- 直近の議論の内容に具体的に触れてください(例:「導入コストが論点になっていますね。まず混雑の原因を整理して、どの部分を改善すべきか考えてみませんか?」)。
  悪い例:「ほかに意見はありますか?」のような中身のない呼びかけ。
- 断定的に意見を述べない。案の良し悪しを評価しない。新しい解決策を提案しない。
- invite_person のときだけ個人名で呼びかけます。名前は「話者別統計」で名前確認済み(confirmed)の人だけ使い、「〇〇さん」と呼びます。target_speaker にはその人の speaker_id を入れてください。
- それ以外のアクションでは個人名や speaker_id を発話に含めないでください。
- time_warning: 残り時間を伝え、結論に向けたまとめを促します。
- close_gd: 参加者が出した結論を、参加者の言葉に沿って簡潔に確認し、終了を告げます。あなたが新たな結論を付け加えてはいけません。結論が出ていなければ、そのことを否定せずに伝えて締めます。
- no_intervention のとき speech は空文字にしてください。

# 入力について
- 発話ログの話者は、名前 / speaker_N / 不明(話者を特定できなかった発話)/ AI(あなた自身の過去の発話)のいずれかです。
- 「AI参加者」と書かれた人は、議論に加わっている AI の参加者です。ほかの参加者と同じように扱い、指名してもかまいません。ただし発言機会の調整では、人間の参加者が考えを示す機会を優先してください。
- 話者識別は暫定的で誤りを含みます。「不明」が多い場合や話者情報が利用不可の場合は、個人を特定した判断をしないでください。
- 「許可されたアクション」以外は選ばないでください(ルール層が過剰介入を防ぐため制限しています)。

# 出力
analysis で現在の議論状態を更新し、decision で行動を決めてください。
- analysis.phase: 議論の進捗から見た現在のフェーズ(opening / ideation=案出し / comparison=比較検討 / conclusion=結論の取りまとめ / closing)
- analysis.health: progressing / stalled / off_topic / conflicting / converging のいずれか
- decision.reasoning: その行動を選んだ理由(介入しない場合は、なぜ見守るのか)を日本語で1〜2文
- decision.urgency: 0.0〜1.0。介入の必要性の高さ
"""


def build_system_prompt(purpose: str = "job") -> str:
    return SYSTEM_PROMPT.replace(PURPOSE_INTRO["job"], PURPOSE_INTRO.get(purpose, PURPOSE_INTRO["job"]), 1)


def _str_list() -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}}


OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "analysis": {
            "type": "object",
            "properties": {
                "phase": {"type": "string", "enum": PHASE_ORDER},
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


def _mmss(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def build_user_prompt(ctx: dict[str, Any]) -> str:
    lines = [
        f"# テーマ\n{ctx['theme']}",
        "# 状況",
        f"- 現在フェーズ: {ctx['phase']}",
        (
            f"- 経過 {_mmss(ctx['elapsed_sec'])} / 制限時間なし(結論がまとまるまで続ける)"
            if ctx["duration_sec"] >= 10**6
            else f"- 経過 {_mmss(ctx['elapsed_sec'])} / 残り {_mmss(ctx['remaining_sec'])}(制限時間 {_mmss(ctx['duration_sec'])})"
        ),
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
            if s.get("is_bot"):
                name += "・AI参加者"
            last = "一度も発言なし" if s["since_last_spoke_sec"] is None else f"最後の発言から{s['since_last_spoke_sec']:.0f}秒"
            lines.append(
                f"- {s['speaker_id']} / {name}: 発話{s['turns']}回, 計{s['total_speech_sec']:.0f}秒, {last}"
            )

    lines.append("\n# 発話ログ(古い順、[経過時間] 話者: 内容)")
    lines.extend(ctx["transcript"] or ["(まだ発話なし)"])

    lines.append("\n# 許可されたアクション")
    lines.append(", ".join(ctx["allowed_actions"]))
    if ctx.get("note"):
        lines.append(f"\n# 注意\n{ctx['note']}")
    return "\n".join(lines)
