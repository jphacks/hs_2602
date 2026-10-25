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
LEVEL3 = ["控えめ", "標準", "積極的"]

# 画面の「詳細設定」。①〜⑤は 控えめ / 標準 / 積極的、⑥-A・⑥-B は内容支援(答えに近い手助け)をどこまでするか。
# 各段階は 数値設定の上書き(settings) と 進行役への追加の指示(prompt) を持つ
STYLE_KNOBS: dict[str, dict[str, Any]] = {
    "progress": {
        "label": "① 進行・時間管理",
        "options": LEVEL3,
        "levels": [
            {"settings": {"phase_behind_gap": 3, "phase_behind_late_gap": 2, "phase_behind_late_fraction": 0.75},
             "prompt": "フェーズの移行は急がない。参加者が十分に話し合うことを優先し、移行を促すのは大きく遅れたときだけにする。"},
            {"settings": {}, "prompt": ""},
            {"settings": {"phase_behind_gap": 1, "phase_behind_late_gap": 1, "phase_behind_late_fraction": 0.45},
             "prompt": "時間配分を重視し、目標フェーズより遅れ始めたら、区切りで次のフェーズへの移行を早めに促す。"},
        ],
    },
    "redirect": {
        "label": "② 脱線への対応",
        "options": LEVEL3,
        "levels": [
            {"settings": {"redirect_wait_max_sec": 10.0},
             "prompt": "脱線には寛容にする。雑談が長く続き、議論の時間が明らかに失われているときだけ軌道修正する。"},
            {"settings": {}, "prompt": ""},
            {"settings": {"redirect_wait_max_sec": 25.0},
             "prompt": "テーマから話がそれ始めたら、続く前に早めに軌道修正する。"},
        ],
    },
    "organize": {
        "label": "③ 論点・意見の整理",
        "options": LEVEL3,
        "levels": [
            {"settings": {}, "prompt": "論点整理や比較基準の提示は控えめにし、参加者自身が整理するのを待つ。"},
            {"settings": {}, "prompt": ""},
            {"settings": {}, "prompt": "意見が増えてきたら、区切りで論点や出た意見を整理し、議論の見通しを良くする。"},
        ],
    },
    "turns": {
        "label": "④ 発言機会の調整",
        "options": LEVEL3,
        "levels": [
            {"settings": {"person_cooldown_sec": 180.0, "cooldown_sec": 60.0},
             "prompt": "発言機会の調整は控えめにし、参加者どうしの自然なやり取りに任せる。"},
            {"settings": {}, "prompt": ""},
            {"settings": {"person_cooldown_sec": 75.0, "cooldown_sec": 30.0},
             "prompt": "発言の少ない人に積極的に話を振り、全員が考えを示せるように回す。"},
        ],
    },
    "stall": {
        "label": "⑤ 停滞への介入",
        "options": LEVEL3,
        "levels": [
            {"settings": {"silence_trigger_sec": 18.0, "analysis_interval_sec": 25.0},
             "prompt": "沈黙や行き詰まりにはすぐに介入せず、参加者が自力で立て直すのを待つ。"},
            {"settings": {}, "prompt": ""},
            {"settings": {"silence_trigger_sec": 8.0, "analysis_interval_sec": 15.0},
             "prompt": "議論が止まりかけたら、早めに問いかけて流れを作る。"},
        ],
    },
    "content": {
        "label": "⑥-A 内容支援の範囲",
        "options": ["なし", "ヒントまで", "具体案まで"],
        "levels": [
            {"settings": {}, "prompt": ""},
            {"settings": {},
             "prompt": "考える手がかりになるヒント(観点・切り口・身近な例)まで示してよい。具体的な解決策や結論、あなた自身の意見は示さない。"},
            {"settings": {},
             "prompt": "具体的な案の例まで示してよい。ただし複数の選択肢として示し、どれにするかは参加者に任せる。特定の結論へ誘導しない。"},
        ],
    },
    "content_when": {
        "label": "⑥-B 自発的な内容支援",
        "options": ["なし", "停滞時のみ", "積極的"],
        "levels": [
            {"settings": {}, "prompt": "内容の支援は自分からはせず、参加者から求められたときだけ行う。"},
            {"settings": {}, "prompt": "内容の支援は、議論が停滞したときだけ自分から行う。"},
            {"settings": {}, "prompt": "議論を深めるのに役立つと判断したら、停滞していなくても自分から内容の支援をしてよい。"},
        ],
    },
}

# スタイルごとの詳細設定の初期値(GD用 = 就活、GW用 = 就活以外。カスタムは GW用から始める)
STYLE_DEFAULTS: dict[str, dict[str, int]] = {
    "job": {"progress": 1, "redirect": 1, "organize": 1, "turns": 1, "stall": 1, "content": 0, "content_when": 0},
    "general": {"progress": 1, "redirect": 1, "organize": 2, "turns": 2, "stall": 2, "content": 1, "content_when": 1},
}
CONTENT_KEYS = ("content", "content_when")


def knob_overrides(knobs: dict[str, int] | None, purpose: str = "job") -> tuple[dict[str, Any], list[str]]:
    """詳細設定の選択から、数値設定の上書きと進行役への追加の指示を作る。選ばれていない項目は用途の初期値を使う。
    ⑥-A が「なし」なら ⑥-B も「なし」に固定し、内容支援の指示は出さない。"""
    chosen = {**STYLE_DEFAULTS.get(purpose, STYLE_DEFAULTS["job"]),
              **{k: v for k, v in (knobs or {}).items() if k in STYLE_KNOBS and v in (0, 1, 2)}}
    if chosen["content"] == 0:
        chosen["content_when"] = 0
    overrides: dict[str, Any] = {}
    notes: list[str] = []
    for key, spec in STYLE_KNOBS.items():
        if key in CONTENT_KEYS and chosen["content"] == 0:
            continue
        choice = spec["levels"][chosen[key]]
        overrides.update(choice["settings"])
        if choice["prompt"]:
            notes.append(f"- {spec['label']}: {choice['prompt']}")
    return overrides, notes


def content_support_allowed(knobs: dict[str, int] | None, purpose: str = "job") -> bool:
    """内容支援(⑥-A)が「なし」以外か。"""
    chosen = {**STYLE_DEFAULTS.get(purpose, STYLE_DEFAULTS["job"]), **(knobs or {})}
    return chosen.get("content", 0) in (1, 2)


def load_settings(**overrides) -> Settings:
    s = Settings()
    for k, v in overrides.items():
        setattr(s, k, v)
    return s
