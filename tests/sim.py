"""テスト用の疑似時計と、エンジンをスレッドなしで時系列に回す仕組み(実時間を待たない)。

- SimClock.sleep(dt) は仮想時刻を進め、その間に予定された出来事(参加者の発言など)を順に実行する
- SimSession.run() は本番の _agent_loop と同じく 0.25 秒ごとに _agent_step() を呼び、判断はその場で実行する
  (LLM の応答待ち・区切りの待機・読み上げも疑似時計で進むので、その間の出来事も時刻どおりに起きる)
"""

from __future__ import annotations

import heapq
import itertools
from typing import Callable

import facilitator.engine as engine_mod
from config import load_settings
from facilitator.engine import Engine, SessionConfig

THEME = "大学の学食の混雑を改善するには？"


class SimClock:
    def __init__(self) -> None:
        self.t = 1000.0
        self._events: list[tuple[float, int, Callable[[], None]]] = []
        self._seq = itertools.count()

    def monotonic(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.advance(self.t + max(0.0, dt))

    def at(self, t: float, fn: Callable[[], None]) -> None:
        heapq.heappush(self._events, (t, next(self._seq), fn))

    def advance(self, until: float) -> None:
        while self._events and self._events[0][0] <= until:
            t, _, fn = heapq.heappop(self._events)
            self.t = max(self.t, t)
            fn()
        self.t = max(self.t, until)


class SimSession:
    """疑似時計で動くエンジン(テキストモード)。play() で台本を流し、VAD の発話中・声の終わりも模擬する。"""

    def __init__(self, monkeypatch, llm, minutes: float = 10.0, names=("A", "B", "C"), **overrides) -> None:
        self.clock = SimClock()
        if hasattr(llm, "clock"):
            llm.clock = self.clock
        monkeypatch.setattr(engine_mod, "make_llm", lambda settings: llm)
        self.eng = Engine(load_settings(), clock=self.clock, **overrides)
        self.eng.start(SessionConfig(THEME, minutes, list(names), mode="text", intro=False, tts=False), background=False)
        self.spans: list[tuple[float, float]] = []  # 人が話していた区間(エンジン時刻)

    def now(self) -> float:
        return self.eng.now()

    def at(self, t: float, fn: Callable[[], None]) -> None:
        """エンジン時刻 t に fn を実行する。"""
        self.clock.at(self.eng._t0 + t, fn)

    def say(self, t: float, sid: str, text: str, dur: float = 1.0) -> None:
        """t に話し始め、dur 秒後に話し終えて発言が確定する(VAD と STT を模擬)。"""
        st, eng = self.eng.state, self.eng

        def begin() -> None:
            st.speaking_now = True

        def end() -> None:
            st.speaking_now = False
            eng._voice_end = eng.now()
            eng.add_text(sid, text)
            self.spans.append((t, eng.now()))

        self.at(t, begin)
        self.at(t + dur, end)

    def play(self, script: list[tuple[float, str, str, float]]) -> None:
        for t, sid, text, dur in script:
            self.say(t, sid, text, dur)

    def run(self, until: float, stop: Callable[[], bool] | None = None) -> None:
        """エンジン時刻 until まで回す。stop() が真になったら止める。"""
        eng = self.eng
        while eng.now() < until and not eng._stop.is_set():
            self.clock.sleep(0.25)
            trig = eng._agent_step()
            if trig:
                eng._run_tick(trig)
            if stop and stop():
                return

    def ai(self, part: str):
        return [u for u in self.eng.state.utterances if u.source == "ai" and part in u.text]


def chat_script(texts, start=0.0, gap=1.0, dur=1.0, speakers=("speaker_1", "speaker_2", "speaker_3")):
    out, t = [], start
    for i, text in enumerate(texts):
        out.append((t, speakers[i % len(speakers)], text, dur))
        t += dur + gap
    return out
