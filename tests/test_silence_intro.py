"""沈黙の数え方(認識中は止める)と、自己紹介の発言を発言量に数えないこと。"""

from __future__ import annotations

from config import load_settings
from facilitator.state import DiscussionState


def make() -> DiscussionState:
    st = DiscussionState(load_settings(), "テーマ", 300.0)
    st.speaker_ids = ["speaker_1", "speaker_2"]
    return st


def test_silence_does_not_count_while_transcribing():
    st = make()
    st.start_time, st.started = 0.0, True
    st.add_utterance("speaker_1", "最初の発言です", 0.0, 2.0, 1.0)
    assert st.silence_sec(10.0) == 8.0
    st.heard_at = 9.5  # 誰かが話していて、文字起こしの途中経過が届いている
    assert st.silence_sec(10.0) == 0.5


def test_self_introduction_is_not_counted_in_speaking_amount():
    st = make()
    st.intro_ids.add(st.add_utterance("speaker_1", "たろうです。よろしくお願いします。", -20.0, -15.0, 1.0).id)  # 自己紹介
    st.start_time, st.started = 0.0, True
    st.add_utterance("speaker_2", "はなこです、と言いつつ本題です", 1.0, 3.0, 1.0)
    stats = {x.speaker_id: x for x in st.speaker_stats(10.0)}
    assert stats["speaker_1"].total_speech_sec == 0 and stats["speaker_1"].turns == 0
    assert stats["speaker_2"].total_speech_sec == 2.0
