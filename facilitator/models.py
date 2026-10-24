"""データ構造の定義。"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

UNKNOWN = "unknown"
AI_SPEAKER = "ai"


class Utterance(BaseModel):
    id: int
    speaker_id: str  # speaker_1..N / unknown / ai。ダイアライゼーションの後修正で差し替わる
    speaker_label: str | None = None  # ニックネーム(紐づけ済みの場合)
    text: str
    start_time: float
    end_time: float
    speaker_confidence: float = 0.0
    source: Literal["human", "ai"] = "human"

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


class Action(str, Enum):
    no_intervention = "no_intervention"
    start_gd = "start_gd"
    ask_question = "ask_question"
    redirect_topic = "redirect_topic"
    organize_issues = "organize_issues"
    compare_criteria = "compare_criteria"
    invite_all = "invite_all"
    invite_person = "invite_person"
    return_to_speaker = "return_to_speaker"
    time_warning = "time_warning"
    close_gd = "close_gd"


HEALTH_VALUES = ["progressing", "stalled", "off_topic", "conflicting", "converging"]


class Analysis(BaseModel):
    """Understand の結果(LLM が更新する議論状態)。"""

    phase: str = ""  # 議論タイプのフェーズの key(facilitator/discussion_types.py)
    phase_reason: str = ""  # そのフェーズだと判断した根拠(短く)
    health: str = "progressing"
    issues: list[str] = Field(default_factory=list)
    opinions: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    agreements: list[str] = Field(default_factory=list)
    missing_perspectives: list[str] = Field(default_factory=list)
    recent_summary: str = ""


class Decision(BaseModel):
    action: Action = Action.no_intervention
    reasoning: str = ""
    speech: str = ""
    urgency: float = 0.0
    target_speaker: str | None = None


class AgentOutput(BaseModel):
    analysis: Analysis
    decision: Decision


class SpeakerStats(BaseModel):
    speaker_id: str
    label: str | None
    label_confirmed: bool
    total_speech_sec: float
    turns: int
    since_last_spoke_sec: float | None  # None = 一度も発言なし


class TickRecord(BaseModel):
    """1回の判断の記録(デモで「なぜ介入した/しなかったか」を見せるため)。"""

    time: float
    trigger: str
    observed_phase: str  # LLM が判断した実際のフェーズ(この判断後の値)
    target_phase: str  # 経過時間から計算した目標フェーズ
    phase_gap: int | None = None  # 目標 - 判断前の実際のフェーズ(段階数)。判断前のフェーズがなければ None
    phase_behind: bool = False  # 判断前のフェーズで見て、移行を促してよい程度に遅れていたか
    remaining_sec: float
    silence_sec: float
    gate: str  # ルール層の判定(llm / forced:... / blocked:...)
    allowed_actions: list[str]
    analysis: Analysis
    decision: Decision
    llm_raw_action: str | None = None
    overridden: str | None = None  # ルール層で書き換えた場合の理由
    latency_sec: float = 0.0
