"""全設定をここに集約する。値の調整は基本的にこのファイルだけで行う。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


@dataclass
class Settings:
    # --- GD 基本設定 ---
    num_participants: int = 4
    gd_duration_sec: float = 600.0

    # --- エージェント実行タイミング ---
    analysis_interval_sec: float = 20.0  # 定期分析の間隔
    silence_trigger_sec: float = 12.0  # これ以上沈黙したらイベント駆動で分析
    min_gap_after_speech_sec: float = 1.0  # 発話終了後、最低この秒数あけてから話す(強制イベント含む)
    min_silence_for_intervention_sec: float = 2.0  # 任意の介入はこの秒数以上の沈黙が必要(会話の途中に割り込まない)
    # 脱線の軌道修正(redirect_topic)だけは、判断後に短い発話の区切りを待って話す(分析のタイミングは変えない)
    redirect_pause_sec: float = 0.8  # この長さの静けさがあれば話す(発話中には話さない)
    redirect_wait_max_sec: float = 15.0  # これ以上区切りがなければ取り消す(割り込まない)

    # --- 過剰介入の防止 ---
    cooldown_sec: float = 45.0  # 直前の介入からの最低間隔
    person_cooldown_sec: float = 120.0  # 同一人物への連続指名を避ける間隔
    time_warnings_remaining_sec: list[float] = field(default_factory=lambda: [120.0])
    time_warning_max_fraction: float = 0.35  # 短いGDでは「残り○秒」を制限時間のこの割合までに抑える
    forced_min_gap_sec: float = 20.0  # 時間告知などの強制イベントも、直前の介入からこの秒数はあける(終了は除く)

    # --- 話者識別 ---
    speaker_conf_threshold: float = 0.7  # これ未満の発話は個人別集計から除外
    speaker_reliability_threshold: float = 0.7  # 全体の信頼度がこれ未満なら invite_person 禁止
    use_speaker_info: bool = True  # UI から切り替え可能(False で全体呼びかけのみ)
    enable_return_to_speaker: bool = False  # 遮り検出は Step 2 の検証次第

    # --- フェーズ(実際のフェーズは LLM が判断し、時間では動かさない) ---
    # 目標フェーズより遅れていると見なす条件。移行を促してよい判断材料で、介入を強制するものではない
    phase_behind_gap: int = 2  # この段階数以上遅れている
    phase_behind_late_gap: int = 1  # または、この段階数以上遅れていて
    phase_behind_late_fraction: float = 0.6  # 経過がこの割合以上

    # --- 介入ポリシーの段階で変わる判断条件(既定値=標準。控えめ/積極的の値は LEVEL_SETTINGS) ---
    drift_confirm_ticks: int = 0  # 直前にこの回数続けて脱線と判断していなければ redirect_topic を許可しない(0=条件なし)
    quiet_member_sec: float = 0.0  # この秒数以上発言していない人間の参加者を「発言の少ない参加者」として LLM に示す(0=示さない)
    # LLM が停滞(stalled)と判断して選んだ ask_question / organize_issues(と停滞を理由に許可した内容支援)は、
    # 沈黙がこの秒数に満たなければ見送る(0=条件なし)。①の遅れの判定が立っているときと、依頼に答えるときは除く
    stall_min_silence_sec: float = 0.0

    # --- 雑談モードの AI 司会(facilitator/chat/host.py)。GD・GW のクールダウン・定期分析とは独立。値はすべて仮の値で、
    # 実際の音声テストで調整する ---
    chat_min_gap_sec: float = 6.0  # AI の発言どうしの最低間隔(GD・GW の cooldown_sec の代わり。依頼・強制イベントは従来どおり)
    chat_reaction_wait_sec: float = 6.0  # 人が話し終えてから、ほかの人が反応するのを待つ静けさの長さ(5〜8秒が目安)
    chat_no_answer_sec: float = 10.0  # AI の問いかけに誰も答えないとき、次の判断までに待つ沈黙
    chat_no_answer_backoff_max: int = 3  # 答えのない AI の発言が続くほど待ちを伸ばす(chat_no_answer_sec × (1 + 続いた回数)、この回数まで)
    chat_passive_wait_sec: float = 10.0  # AI の発言への返事が「特にないです」のような中身のない返事だけのとき、次の判断までに待つ沈黙
    chat_passive_backoff_max: int = 3  # 答えなし・中身のない返事が続くほど待ちを伸ばす(chat_passive_wait_sec × (1 + 続いた回数)、この回数まで)
    chat_link_gap_sec: float = 4.0  # 別の人の発言がこの秒数以内に続けば、つながった発言とみなす
    chat_free_min_links: int = 2  # AI の発言の後、つながった発言がこの回数あれば自走期に移る(1回でも LLM が進んでいると判断すれば移る)
    chat_lull_silence_sec: float = 15.0  # 自走期にこの沈黙が続いたら、会話が途切れたとみなす
    chat_free_min_sec: float = 20.0  # 自走期に移ってからこの秒数は、途切れても AI 主導期に戻さない(一時的な沈黙で切り替えないため)
    # 話題の管理(facilitator/chat/topic_tracker.py)。値はすべて仮の値
    chat_topic_change_after: int = 2  # 同じ話題で、AI の働きかけへの反応が薄い(答えなし・中身のない返事だけ・同じ人だけ)ことがこの回数続いたら話題を切り替えてよい
    chat_one_on_one_max: int = 2  # 同じ1人とだけの一問一答がこの回数続いたら、深掘りの質問(ask_question)を許可せず全体へ広げる
    chat_same_nudge_max: int = 2  # 同じ話題で、同じ働きかけ(深掘り・橋渡し)が薄い反応のままこの回数続いたら、その働きかけは繰り返さない
    chat_recent_topics: int = 5  # 重複を避ける「最近扱った話題」の数(AI が出したもの・参加者が始めたもの)
    chat_topic_similarity: float = 0.4  # 話題の種類で決まらないとき、同じ話題とみなす2文字の組の重なり

    # --- 内容支援(介入ポリシー⑥)。値はすべて仮の値 ---
    content_cooldown_sec: float = 120.0  # 自発的な内容支援どうしの最低間隔(参加者からの依頼への応答は除く)
    content_max_spontaneous: int = 3  # 1回の GD で自発的に行う内容支援の上限(依頼への応答は数えない)
    request_expire_sec: float = 30.0  # 進行役への依頼に、この秒数のうちに応じられなければ見送る

    # --- LLM ---
    llm_provider: str = os.getenv("LLM_PROVIDER", "mock")  # mock-test ブランチ: 既定はモック(anthropic | mock | handoff)
    llm_model: str = os.getenv("LLM_MODEL", "claude-opus-5-5")
    llm_effort: str = os.getenv("LLM_EFFORT", "low")
    llm_max_tokens: int = 8000
    llm_timeout_sec: float = 30.0
    transcript_window: int = 40  # LLM に渡す直近発話数

    # --- AI 発話 ---
    tts_chars_per_sec: float = 5.6  # 発話時間の見積もり(say -v Kyoko -r 210 の実測値)

    # --- ログ ---
    log_dir: Path = ROOT / "logs"
    log_transcript_text: bool = False  # 個人情報保護のためデフォルトで発話本文はログに残さない


# 旧画面の「詳細」の5項目(knobs: 0=控えめ / 1=標準 / 2=積極的)。旧 API の互換のため受け付けて介入ポリシーに変換する
# (現在の画面は介入ポリシーの項目を直接送る)。phase / redirect / organize / turns → ①〜④、
# suggest → ⑥(0 → 内容支援なし / 1 → プリセットのまま / 2 → ヒントまで・積極的)。facilitator/policy.py の from_knobs
STYLE_KNOBS: dict[str, dict[str, Any]] = {
    "phase": {"label": "次のフェーズへの進行度合い"},
    "redirect": {"label": "脱線修正の積極度"},
    "organize": {"label": "話題整理の積極度"},
    "turns": {"label": "どれだけ話を回すか"},
    "suggest": {"label": "AIからの提案度合い"},
}


# 旧「詳細」(knobs)で送られた就活の「AIからの提案度合い」は、旧画面と同じく標準までに抑える(旧 API の互換)。
# 現在の画面は介入ポリシーの⑥を直接送り、就活GDでも内容支援を有効にできる(中立設定ではない旨を表示する)
JOB_MAX_LEVEL = {"suggest": 1}


# 介入ポリシー①〜⑤の段階ごとの判断条件(Settings の上書き)。標準は上書きなし(=現行の値)。
# 値はすべて仮の値で、シナリオ評価(第5段階)で調整する。各項目は自分の判断条件だけを変え、
# クールダウン(45秒)・分析間隔(20秒)・割り込まないための間(2秒)はどの項目・段階でも変えない
LEVEL_SETTINGS: dict[str, dict[str, dict[str, Any]]] = {
    "progress": {  # ① 目標フェーズより「遅れている」とみなす条件(LLM に時間配分の注意を出す条件)
        "low": {"phase_behind_gap": 3, "phase_behind_late_gap": 2, "phase_behind_late_fraction": 0.75},
        "high": {"phase_behind_gap": 1, "phase_behind_late_gap": 1, "phase_behind_late_fraction": 0.45},
    },
    "drift": {  # ② 控えめ: 前回の判断でも脱線と見ていた(2回続けて脱線)ときだけ redirect_topic を許可
        "low": {"drift_confirm_ticks": 1},
        "high": {},
    },
    "organize": {},  # ③ 判断基準(プロンプト)だけを変える
    "turns": {  # ④ 積極的: 60秒以上発言していない人間の参加者を LLM に示す(個人指名の条件は変えない)
        "high": {"quiet_member_sec": 60.0},
    },
    "stall": {  # ⑤ 沈黙で判断を始めるまでの秒数
        # 控えめは、定期分析(20秒ごと)が短い沈黙のうちに判断しても、停滞への介入は 18 秒の沈黙まで待つ
        "low": {"silence_trigger_sec": 18.0, "stall_min_silence_sec": 18.0},
        "high": {"silence_trigger_sec": 8.0},
    },
}


# 介入ポリシーのプリセット(facilitator/policy.py の InterventionPolicy の初期値)。利用者は選んだ後に各項目を変更できる
# ①〜⑤: low=控えめ / standard=標準(現行の就活GD相当) / high=積極的
# ⑥-A content_scope: none=なし / hint=ヒントまで / concrete=具体案まで
# ⑥-B content_initiative: none=依頼されたときだけ / stalled=停滞時のみ / proactive=積極的(content_scope が none なら none に固定)
POLICY_PRESETS: dict[str, dict[str, str]] = {
    "job_gd": {  # 就活GD用: 参加者の自主性を重視し、必要なときだけ進行を支援する。内容は出さない
        "progress": "standard",
        "drift": "standard",
        "organize": "standard",
        "turns": "standard",
        "stall": "standard",
        "content_scope": "none",
        "content_initiative": "none",
    },
    "general_gw": {  # 一般GW用: 話しやすい場をつくり、人間の発言を引き出す。必要に応じて考え方のヒントを出せる
        "progress": "standard",
        "drift": "standard",
        "organize": "high",
        "turns": "high",
        "stall": "high",
        "content_scope": "hint",
        "content_initiative": "stalled",
    },
}
PURPOSE_PRESET = {"job": "job_gd", "general": "general_gw"}  # 用途ごとの既定のプリセット
POLICY_PRESET_LABELS = {"job_gd": "GD用(就活)", "general_gw": "GW用(一般)"}  # 画面に出す名前


def load_settings(**overrides) -> Settings:
    s = Settings()
    for k, v in overrides.items():
        setattr(s, k, v)
    return s
