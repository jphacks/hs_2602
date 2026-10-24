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

    # --- LLM ---
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic")  # anthropic | mock | handoff(チャット経由)
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


# 画面で選ぶ進行スタイル。A は Step 1 で評価した既定値
PRESETS: dict[str, dict[str, float]] = {
    "A": {},  # 標準(推奨)
    "B": {  # 積極サポート: 沈黙や停滞により早く介入する
        "cooldown_sec": 30.0,
        "silence_trigger_sec": 8.0,
        "analysis_interval_sec": 15.0,
        "min_silence_for_intervention_sec": 1.5,
    },
}
# カスタム設定で画面から変更できる項目
CUSTOM_KEYS = {"cooldown_sec", "silence_trigger_sec", "analysis_interval_sec", "min_silence_for_intervention_sec"}


# 画面の「詳細」で選ぶ進行の調整。0=控えめ / 1=標準(既定・これまでの評価値) / 2=積極的
# settings: 数値設定の上書き / prompt: 進行役への指示(システムプロンプトの末尾に足す)
STYLE_KNOBS: dict[str, dict[str, Any]] = {
    "phase": {
        "label": "次のフェーズへの進行度合い",
        "levels": [
            {"settings": {"phase_behind_gap": 3, "phase_behind_late_gap": 2, "phase_behind_late_fraction": 0.75},
             "prompt": "フェーズの移行は急がない。参加者が十分に話し合うことを優先し、移行を促すのは大きく遅れたときだけにする。"},
            {"settings": {}, "prompt": ""},
            {"settings": {"phase_behind_gap": 1, "phase_behind_late_gap": 1, "phase_behind_late_fraction": 0.45},
             "prompt": "時間配分を重視し、目標フェーズより遅れ始めたら、区切りで次のフェーズへの移行を早めに促す。"},
        ],
    },
    "redirect": {
        "label": "脱線修正の積極度",
        "levels": [
            {"settings": {"redirect_wait_max_sec": 10.0},
             "prompt": "脱線には寛容にする。雑談が長く続き、議論の時間が明らかに失われているときだけ軌道修正する。"},
            {"settings": {}, "prompt": ""},
            {"settings": {"redirect_wait_max_sec": 25.0},
             "prompt": "テーマから話がそれ始めたら、続く前に早めに軌道修正する。"},
        ],
    },
    "organize": {
        "label": "話題整理の積極度",
        "levels": [
            {"settings": {}, "prompt": "論点整理や比較基準の提示は控えめにし、参加者自身が整理するのを待つ。"},
            {"settings": {}, "prompt": ""},
            {"settings": {}, "prompt": "意見が増えてきたら、区切りで論点を整理したり比較の基準を示したりして、議論の見通しを良くする。"},
        ],
    },
    "turns": {
        "label": "どれだけ話を回すか",
        "levels": [
            {"settings": {"cooldown_sec": 60.0, "silence_trigger_sec": 15.0, "analysis_interval_sec": 25.0},
             "prompt": "発言機会の調整は控えめにし、参加者どうしの自然なやり取りに任せる。"},
            {"settings": {}, "prompt": ""},
            {"settings": {"cooldown_sec": 30.0, "silence_trigger_sec": 8.0, "analysis_interval_sec": 15.0},
             "prompt": "発言の少ない人に積極的に話を振り、全員が考えを示せるように回す。"},
        ],
    },
    "suggest": {
        "label": "AIからの提案度合い",
        "levels": [
            {"settings": {}, "prompt": "問いかけだけで進行し、観点や切り口を自分からは示さない。"},
            {"settings": {}, "prompt": ""},
            {"settings": {},
             "prompt": "議論に足りない観点(例: 費用、利用者の立場、実現性)を具体的に示してよい。ただし解決策・結論・自分の意見は言わない。"},
        ],
    },
}


# 就活GDでは選考の公平性のため、進行役が参加者の出していない観点を持ち込まない(prompts.PURPOSE_RULES["job"])。
# そのため「AIからの提案度合い」は就活では標準までに抑える
JOB_MAX_LEVEL = {"suggest": 1}


def knob_overrides(knobs: dict[str, int] | None, purpose: str = "job") -> tuple[dict[str, Any], list[str]]:
    """詳細の選択から、数値設定の上書きと進行役への追加の指示を作る。
    介入ポリシー(facilitator/policy.py)の①〜④の効果として使う(第2段階でポリシーの項目ごとの効果に置き換える)。"""
    overrides: dict[str, Any] = {}
    notes: list[str] = []
    for key, level in (knobs or {}).items():
        spec = STYLE_KNOBS.get(key)
        if not spec or level not in (0, 1, 2):
            continue
        if purpose == "job":
            level = min(level, JOB_MAX_LEVEL.get(key, 2))
        choice = spec["levels"][level]
        overrides.update(choice["settings"])
        if choice["prompt"]:
            notes.append(f"- {spec['label']}: {choice['prompt']}")
    return overrides, notes


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


def load_settings(**overrides) -> Settings:
    s = Settings()
    for k, v in overrides.items():
        setattr(s, k, v)
    return s
