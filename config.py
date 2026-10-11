"""全設定をここに集約する。値の調整は基本的にこのファイルだけで行う。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
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

    # --- 過剰介入の防止 ---
    cooldown_sec: float = 45.0  # 直前の介入からの最低間隔
    person_cooldown_sec: float = 120.0  # 同一人物への連続指名を避ける間隔
    time_warnings_remaining_sec: list[float] = field(default_factory=lambda: [120.0])

    # --- 話者識別 ---
    speaker_conf_threshold: float = 0.7  # これ未満の発話は個人別集計から除外
    speaker_reliability_threshold: float = 0.7  # 全体の信頼度がこれ未満なら invite_person 禁止
    use_speaker_info: bool = True  # UI から切り替え可能(False で全体呼びかけのみ)
    enable_return_to_speaker: bool = False  # 遮り検出は Step 2 の検証次第

    # --- フェーズ(経過時間の割合による強制移行の下限) ---
    phase_time_floor: dict[str, float] = field(
        default_factory=lambda: {"comparison": 0.55, "conclusion": 0.8}
    )

    # --- LLM ---
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic")  # anthropic | mock
    llm_model: str = os.getenv("LLM_MODEL", "claude-opus-5-5")
    llm_effort: str = os.getenv("LLM_EFFORT", "low")
    llm_max_tokens: int = 8000
    llm_timeout_sec: float = 30.0
    transcript_window: int = 40  # LLM に渡す直近発話数

    # --- AI 発話 ---
    tts_chars_per_sec: float = 7.0  # 発話時間の見積もり(日本語のおおよその読み上げ速度)

    # --- ログ ---
    log_dir: Path = ROOT / "logs"
    log_transcript_text: bool = False  # 個人情報保護のためデフォルトで発話本文はログに残さない


def load_settings(**overrides) -> Settings:
    s = Settings()
    for k, v in overrides.items():
        setattr(s, k, v)
    return s
