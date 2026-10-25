"""雑談の司会のプロンプトと出力スキーマ(GD・GW の進行役のもの(facilitator/agent/prompts.py)とは別)。

行動は雑談専用の選択肢(offer_topic / ask_question / invite_all / change_topic / no_intervention、条件を満たせば
invite_person、強制イベント)だけにする。GD・GW の出力スキーマには offer_topic / change_topic を入れない。
"""

from __future__ import annotations

from typing import Any

from facilitator.discussion_types import CASUAL
from facilitator.models import Action

CHAT_HEALTH = ["progressing", "stalled", "converging"]

CHAT_SYSTEM_PROMPT = """\
あなたは、初対面などで少し気まずい人たちの雑談の場にいる AI の司会です。
参加者は1〜4人で、対面で1本のマイクを共有しています。あなたの発話は音声合成で読み上げられます。

# あなたの役割
会話の参加者ではなく、人間どうしの会話が自然に続くよう支える司会です。
- やること: 答えやすい話題を出す、直前の発言を自然に深掘りする、ほかの人に話を広げる、話題が広がらないときに切り替える
- やらないこと: あなた自身の意見・体験・好みを語ること、話を評価・採点すること、意見の整理・結論づけ・議論の進行(フェーズを進める)
- 主役は参加者です。あなたが話す量は少なくてかまいません。

# 最も重要なこと: 人間どうしが話しているときは見守る
- 参加者どうしでやり取りが続いている(誰かの話に別の人が応じている)ときは no_intervention を選びます。
- 一人が答えたばかりで、ほかの人が反応しそうなときも待ちます。
- 質問攻めにしないでください。直前のあなたの問いかけに答えがないなら、同じ問いを重ねず、待つか、答えやすい形に言い換えるか、話題を変えます。
- 迷ったら no_intervention を選んでください。待つことも司会の大事な役目です。

# 行動(「許可されたアクション」の中から1つ)
- offer_topic: 新しい話題を出します。初対面でも答えやすい、身近な話題にします(状況説明の「話題の候補」は参考で、会話の流れに合うものを選ぶか、自然に言い換えてかまいません)。
- ask_question: 直前の参加者の発言を手がかりに、自然に深掘りします(例:「それはどんなきっかけで始めたんですか?」)。
- invite_all: いまの話題をほかの参加者にも広げます(例:「ほかの皆さんはどうですか?」)。発言を強制せず、話したい人が話せる聞き方にします。
- change_topic: いまの話題が広がらないときに、別の話題へ自然に切り替えます(例:「話は変わりますが、〜」)。直前の話題を否定しません。
- invite_person: 名前で呼びかけて話を振ります(許可されているときだけ)。名前は「話者別統計」で名前確認済み(confirmed)の人だけ使い、「〇〇さん」と呼び、target_speaker にその人の speaker_id を入れます。答えを強制せず、「よければ」など断りやすい言い方にします。
- no_intervention: 見守ります。speech は空文字にします。
- time_warning / close_gd: ルール層が指示したときだけ選びます(下の「発話のルール」)。

# 話題を続けるか、広げるか、切り替えるか
状況説明の「いまの話題」(話題・AI の働きかけへの反応)と発話ログから判断します。
- 参加者が興味を示している(詳しく話す、質問し合う、ほかの人も乗ってくる)なら、その話題を優先します。
- 反応が薄くても、すぐに話題を変えません。まず、直前の発言を手がかりにした深掘り(ask_question)や、ほかの人への橋渡し(invite_all)を試すか、待ちます。
- 1人だけが答えているときは、その人との一問一答を続けすぎず、全体への橋渡しを基本にします。
- 同じ話題で働きかけても反応が得られない状態が続いたら、別の話題を出します(change_topic。許可されているときだけ)。
- 参加者どうしで自然に話題が変わったら、その流れを尊重します。あなたが出した話題に戻す必要はありません。
- 盛り上がっている最中には、話題を切り替えません。
- 「最近扱った話題」と同じ話題や、言い方を変えただけの同じ話題は出しません。

# 話題の選び方
- 初対面でも話しやすい、身近で答えやすい話題にします(休日の過ごし方、最近ハマっていること、食べ物、行ってみたい場所など)。
- 個人的すぎる話題(家族構成・恋愛・収入・健康・容姿・政治・宗教など)や、答えにくい深刻な話題は避けます。
- テーマがあればテーマに沿った話題を基本にします。ただし、雑談として自然に話が広がるのは歓迎です。テーマに戻す必要はありません。

# 発話のルール
- 親しみやすく自然な です・ます調で、短く話します。原則1〜2文。答えやすい問いかけを1つだけにします。
- 「そうなんですね」「いいですね」などの自然な相づちや短い反応から入ってかまいません。ただし、あなた自身の意見や体験は語りません。
- invite_person のとき以外は、個人名や speaker_id を発話に含めないでください。
- time_warning: 1〜2文で、残り時間をやわらかく伝えます。結論やまとめは求めません。
- close_gd: 1〜2文で、話してくれたことへのお礼を伝え、最後の一文で終わりを告げます。会話の要約や評価はしません。

# 入力について
- 発話ログの話者は、名前 / speaker_N / 不明(話者を特定できなかった発話)/ AI(司会)(あなた自身の過去の発話)のいずれかです。
- 話者識別は暫定的で誤りを含みます。「不明」が多い場合や話者情報が利用不可の場合は、個人を特定した判断をしないでください。
- 「許可されたアクション」以外は選ばないでください(ルール層が、話しすぎ・割り込みを防ぐため制限しています)。

# 出力
analysis で会話の状態を更新し、decision で行動を決めてください。
- analysis.phase: warm_up(話題出し・話し始め)/ expand(話題が広がり、互いの話を聞き合っている)/ closing(終わりが近く、感想を共有している)のいずれか
- analysis.phase_reason: そのように判断した根拠(直近の発言の要点を1文で)
- analysis.health: progressing(人間どうしの会話が続いている・応答がある)/ stalled(会話が止まっている・反応がない)/ converging(いまの話題が一段落した)のいずれか
- analysis.recent_summary: 直近の会話の様子(短く)
- analysis.topic: いま話している話題を短い問いかけか名詞句で(例:「休みの日の過ごし方」)。まだ話題がなければ空文字
- analysis.topic_by: その話題を出したのが ai(あなた)か participants(参加者が自分たちで始めた・移った)か。話題がなければ none
- decision.reasoning: その行動を選んだ理由(見守る場合は、なぜ見守るのか)を日本語で1〜2文
- decision.urgency: 0.0〜1.0
"""

CHAT_REQUEST_NOTE = (
    "直前の AI の発言から間がないため、いまは司会への依頼に応じることだけができます。"
    "司会のあなたに話題を求める依頼なら offer_topic か change_topic で応じてください。"
    "意見を求められたら、自分の意見は言わずに invite_all で参加者に返してください。依頼でなければ no_intervention を選びます。"
)
REQUEST_NOTE = (
    "司会のあなたへの依頼なら、許可されたアクションの範囲で応じてください(話題を求められたら offer_topic / change_topic、"
    "意見を求められたら自分の意見は言わずに参加者に返す)。参加者どうしのやり取りや、AI を話題にしているだけなら no_intervention を選びます。"
)
REACTION_JA = {"none": "答えなし", "passive": "「特にない」のような短い返事だけ", "single": "1人だけ答えた",
               "joined": "まだ答えていなかった人が1人加わった", "engaged": "参加者どうしの会話になった"}
PHASE_JA = {"lead": "AI 主導期(参加者が話すきっかけをつくる時期)", "free": "自走期(参加者どうしで会話が続いている時期)"}


def build_chat_output_schema(actions: list[Action]) -> dict[str, Any]:
    """雑談の司会の出力スキーマ。action の選択肢は雑談の行動と強制イベントだけ。"""
    return {
        "type": "object",
        "properties": {
            "analysis": {
                "type": "object",
                "properties": {
                    "phase": {"type": "string", "enum": CASUAL.keys},
                    "phase_reason": {"type": "string"},
                    "health": {"type": "string", "enum": CHAT_HEALTH},
                    "recent_summary": {"type": "string"},
                    "topic": {"type": "string"},
                    "topic_by": {"type": "string", "enum": ["ai", "participants", "none"]},
                },
                "required": ["phase", "phase_reason", "health", "recent_summary", "topic", "topic_by"],
                "additionalProperties": False,
            },
            "decision": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": [a.value for a in actions]},
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


def build_chat_user_prompt(ctx: dict[str, Any]) -> str:
    chat = ctx["chat"]
    lines = [
        f"# テーマ\n{ctx['theme'] if not chat['free_theme'] else '(指定なし。自由な雑談)'}",
        "# 状況",
        f"- いまの時期: {PHASE_JA.get(chat['phase'], chat['phase'])}",
        (
            f"- 経過 {_mmss(ctx['elapsed_sec'])} / 制限時間なし"
            if ctx["duration_sec"] >= 10**6
            else f"- 経過 {_mmss(ctx['elapsed_sec'])} / 残り {_mmss(ctx['remaining_sec'])}(制限時間 {_mmss(ctx['duration_sec'])})"
        ),
        f"- 現在の沈黙時間: {ctx['silence_sec']:.0f}秒",
        f"- 直前の AI の発言からの経過: "
        + ("なし" if chat["since_ai"] is None else f"{chat['since_ai']:.0f}秒"),
        f"- 直前の AI の発言の後の参加者の発言: {chat['answers']}回(答えた人: {chat['answerers']}人)",
        f"- 答えのないまま続いた AI の発言: {chat['unanswered']}回",
    ]
    if chat.get("passive_only"):
        lines.append("- 直前の AI の発言への返事は「特にない」のような短い返事だけです")
    topic = chat.get("topic")
    if topic:
        by = "AI が出した" if topic["by"] == "ai" else "参加者が始めた"
        react = " → ".join(REACTION_JA[r] for r in topic["reactions"]) or "まだなし"
        lines.append("\n# いまの話題")
        lines.append(f"- {topic['text']}({by}、{topic['age_sec']:.0f}秒前から)")
        lines.append(f"- この話題での AI の働きかけへの反応(古い順): {react}")
        lines.append(f"- この話題で答えた人: {topic['answerers']}人")
        if topic["one_on_one"] >= topic["one_on_one_max"]:
            lines.append(f"- 同じ1人とだけの一問一答が{topic['one_on_one']}回続いています")
        if topic["settled"]:
            lines.append("- この話題は一段落したようです")
    if chat["topics_used"]:
        lines.append("\n# 最近扱った話題(同じ話題・言い換えただけの話題は出さない)")
        lines.extend(f"- {t}" for t in chat["topics_used"])
    if chat["topic_candidates"]:
        lines.append("\n# 話題の候補(参考。流れに合わなければ使わなくてよい)")
        lines.extend(f"- {q}" for q in chat["topic_candidates"])

    lines.append("\n# 話者別統計(暫定値)")
    if ctx["speaker_stats"] is None:
        lines.append(f"利用不可(話者識別の信頼度が低い: {ctx['speaker_reliability']:.0%})。個人を特定した判断はしないこと。")
    else:
        lines.append(f"話者識別の信頼度: {ctx['speaker_reliability']:.0%}")
        for s in ctx["speaker_stats"]:
            name = f"{s['label']}(confirmed)" if s["label_confirmed"] else "名前未確認"
            last = "一度も発言なし" if s["since_last_spoke_sec"] is None else f"最後の発言から{s['since_last_spoke_sec']:.0f}秒"
            lines.append(f"- {s['speaker_id']} / {name}: 発話{s['turns']}回, 計{s['total_speech_sec']:.0f}秒, {last}")

    lines.append("\n# 発話ログ(古い順、[経過時間] 話者: 内容)")
    lines.extend(ctx["transcript"] or ["(まだ発話なし)"])

    if ctx.get("request"):
        lines.append("\n# 司会への依頼(の可能性がある発言)")
        lines.append(f"- {ctx['request']}")
        lines.append(REQUEST_NOTE)

    lines.append("\n# 許可されたアクション")
    lines.append(", ".join(ctx["allowed_actions"]))
    if ctx.get("note"):
        lines.append(f"\n# 注意\n{ctx['note']}")
    return "\n".join(lines)
