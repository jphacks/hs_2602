"""議論状態の管理(Observe の土台)。発話の追加・話者の後差し替え・話者別集計を担当する。"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import Settings
from facilitator.models import (
    AI_SPEAKER,
    PHASE_ORDER,
    UNKNOWN,
    Action,
    Analysis,
    Decision,
    Phase,
    SpeakerStats,
    Utterance,
)


@dataclass
class Intervention:
    time: float
    action: Action
    speech: str
    target_speaker: str | None = None


@dataclass
class DiscussionState:
    settings: Settings
    theme: str
    duration_sec: float
    start_time: float = 0.0  # GD 開始時刻(時計の値)
    started: bool = False
    finished: bool = False
    phase: str = Phase.opening.value
    utterances: list[Utterance] = field(default_factory=list)
    analysis: Analysis = field(default_factory=Analysis)
    interventions: list[Intervention] = field(default_factory=list)
    speaker_ids: list[str] = field(default_factory=list)
    speaker_names: dict[str, str] = field(default_factory=dict)  # speaker_id -> ニックネーム
    names_confirmed: set[str] = field(default_factory=set)
    warned_thresholds: set[float] = field(default_factory=set)
    speaking_now: bool = False  # VAD による発話中フラグ
    _next_id: int = 0

    def __post_init__(self) -> None:
        if not self.speaker_ids:
            self.speaker_ids = [f"speaker_{i + 1}" for i in range(self.settings.num_participants)]

    # ---------- 発話 ----------
    def add_utterance(
        self,
        speaker_id: str,
        text: str,
        start_time: float,
        end_time: float,
        speaker_confidence: float = 1.0,
        source: str = "human",
    ) -> Utterance:
        if source == "human" and speaker_confidence < self.settings.speaker_conf_threshold:
            speaker_id = UNKNOWN
        u = Utterance(
            id=self._next_id,
            speaker_id=speaker_id,
            speaker_label=self.speaker_names.get(speaker_id),
            text=text,
            start_time=start_time,
            end_time=end_time,
            speaker_confidence=speaker_confidence,
            source=source,  # type: ignore[arg-type]
        )
        self._next_id += 1
        self.utterances.append(u)
        return u

    def update_speaker(self, utterance_id: int, speaker_id: str, confidence: float) -> None:
        """ダイアライゼーションの後修正を反映する。"""
        for u in self.utterances:
            if u.id == utterance_id and u.source == "human":
                if confidence < self.settings.speaker_conf_threshold:
                    speaker_id = UNKNOWN
                u.speaker_id = speaker_id
                u.speaker_confidence = confidence
                u.speaker_label = self.speaker_names.get(speaker_id)
                return

    def set_speaker_name(self, speaker_id: str, name: str, confirmed: bool = True) -> None:
        self.speaker_names[speaker_id] = name
        if confirmed:
            self.names_confirmed.add(speaker_id)
        for u in self.utterances:
            if u.speaker_id == speaker_id:
                u.speaker_label = name

    def human_utterances(self) -> list[Utterance]:
        return [u for u in self.utterances if u.source == "human"]

    # ---------- 時間 ----------
    def elapsed(self, now: float) -> float:
        return max(0.0, now - self.start_time) if self.started else 0.0

    def remaining(self, now: float) -> float:
        return max(0.0, self.duration_sec - self.elapsed(now)) if self.started else self.duration_sec

    def silence_sec(self, now: float) -> float:
        """最後の発話(人間・AI問わず)終了からの経過秒数。"""
        ended = [u.end_time for u in self.utterances if u.end_time <= now]
        last = max(ended) if ended else (self.start_time if self.started else now)
        return max(0.0, now - last)

    def is_speaking(self, now: float) -> bool:
        """VAD の発話中フラグ、または最後の発話終了直後の短い間。"""
        if self.speaking_now:
            return True
        return self.silence_sec(now) < self.settings.min_gap_after_speech_sec and bool(self.utterances)

    # ---------- 介入履歴 ----------
    def last_intervention(self) -> Intervention | None:
        return self.interventions[-1] if self.interventions else None

    def since_last_intervention(self, now: float) -> float | None:
        last = self.last_intervention()
        return None if last is None else now - last.time

    def record_intervention(self, now: float, decision: Decision) -> None:
        self.interventions.append(
            Intervention(now, decision.action, decision.speech, decision.target_speaker)
        )
        if decision.action == Action.start_gd:
            self.started = True
            self.start_time = now
            self.phase = Phase.ideation.value
        elif decision.action == Action.close_gd:
            self.phase = Phase.closing.value
            self.finished = True

    def last_invited_at(self, speaker_id: str) -> float | None:
        for iv in reversed(self.interventions):
            if iv.action == Action.invite_person and iv.target_speaker == speaker_id:
                return iv.time
        return None

    # ---------- フェーズ ----------
    def advance_phase(self, suggested: str, now: float) -> None:
        """フェーズは前進のみ。LLM の提案と、経過時間による下限(安全弁)の大きい方を採用。"""
        if self.finished or not self.started:
            return
        idx = PHASE_ORDER.index(self.phase)
        if suggested in PHASE_ORDER and suggested != Phase.closing.value:
            idx = max(idx, PHASE_ORDER.index(suggested))
        frac = self.elapsed(now) / self.duration_sec
        for phase, floor in self.settings.phase_time_floor.items():
            if frac >= floor:
                idx = max(idx, PHASE_ORDER.index(phase))
        self.phase = PHASE_ORDER[idx]

    # ---------- 話者別統計 ----------
    def speaker_reliability(self) -> float:
        """直近の人間発話のうち、話者が確信度十分で特定できた割合。"""
        recent = self.human_utterances()[-20:]
        if not recent:
            return 0.0
        ok = [u for u in recent if u.speaker_id not in (UNKNOWN, AI_SPEAKER)]
        return len(ok) / len(recent)

    def speaker_stats(self, now: float) -> list[SpeakerStats]:
        stats = []
        for sid in self.speaker_ids:
            mine = [u for u in self.human_utterances() if u.speaker_id == sid and u.end_time <= now]
            stats.append(
                SpeakerStats(
                    speaker_id=sid,
                    label=self.speaker_names.get(sid),
                    label_confirmed=sid in self.names_confirmed,
                    total_speech_sec=round(sum(u.duration for u in mine), 1),
                    turns=len(mine),
                    since_last_spoke_sec=round(now - max(u.end_time for u in mine), 1) if mine else None,
                )
            )
        return stats

    def speaker_info_usable(self) -> bool:
        return (
            self.settings.use_speaker_info
            and self.speaker_reliability() >= self.settings.speaker_reliability_threshold
        )
