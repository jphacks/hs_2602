"""雑談の司会のプロンプト・行動・開始の話題(第3段階)のテスト。LLM はスタブ・モックで、API は呼ばない。"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import facilitator.engine as engine_mod  # noqa: E402
from config import load_settings  # noqa: E402
from facilitator.agent.agent import FREE_ACTIONS, INVITE_ALL_FALLBACK, FacilitatorAgent  # noqa: E402
from facilitator.agent.prompts import build_output_schema, build_system_prompt  # noqa: E402
from facilitator.chat.host import CHAT_ACTIONS, CHAT_INVITE_ALL, REQUEST_ACTIONS, ChatHostAgent  # noqa: E402
from facilitator.chat.prompts import CHAT_SYSTEM_PROMPT  # noqa: E402
from facilitator.chat.topics import FREE_TOPICS, opening_speech, opening_topic, topic_candidates  # noqa: E402
from facilitator.discussion_types import CASUAL_FREE_THEME, DISCUSSION_TYPES  # noqa: E402
from facilitator.engine import Engine, SessionConfig  # noqa: E402
from facilitator.llm.mock_llm import MockLLM  # noqa: E402
from facilitator.models import CHAT_ONLY_ACTIONS, Action, Decision  # noqa: E402
from facilitator.policy import InterventionPolicy, preset_policy  # noqa: E402
from facilitator.state import DiscussionState  # noqa: E402
from tests.sim import SimClock  # noqa: E402

NAMES = {"speaker_1": "たろう", "speaker_2": "はなこ", "speaker_3": "けんた"}


class StubLLM:
    """決まった行動と発話を返す。受け取った状況説明と許可アクションを記録する。"""

    name = "stub"

    def __init__(self, action: str = "no_intervention", speech: str = "", target: str | None = None) -> None:
        self.action, self.speech, self.target = action, speech, target
        self.ctx: dict | None = None
        self.user = ""

    def generate_json(self, system, user, schema, ctx):
        self.ctx, self.user, self.system, self.schema = ctx, user, system, schema
        return {"analysis": {"phase": "warm_up", "phase_reason": "", "health": "stalled", "recent_summary": ""},
                "decision": {"action": self.action, "reasoning": "", "speech": self.speech, "urgency": 0.5,
                             "target_speaker": self.target}}


def make_chat(llm=None, theme="休日の過ごし方", settings=None, policy=None, names=True):
    s = settings or load_settings()
    st = DiscussionState(settings=s, theme=theme, duration_sec=600)
    if names:
        for sid, n in NAMES.items():
            st.set_speaker_name(sid, n)
    ag = ChatHostAgent(st, llm or StubLLM(), s, purpose="general", discussion_type="casual",
                       policy=policy or preset_policy("general_gw"), rng=random.Random(0))
    return st, ag


def started(st, ag, answer_at: float | None = 20.0):
    """開始の発言のあと、答えが1つあり、反応の待ちが過ぎた状態にする。返り値は判断する時刻。"""
    ag.tick(0.0, "start")
    end = st.utterances[-1].end_time
    if answer_at is None:
        return end + 60.0
    st.add_utterance("speaker_1", "休みの日は寝てばかりです。", end + 1, end + 4, 0.9)
    return end + 4 + 10.0


# ---------- 1. 雑談専用のプロンプト ----------
def test_chat_prompt_is_separate_and_covers_policy():
    _st, ag = make_chat()
    assert ag.system_prompt == CHAT_SYSTEM_PROMPT
    for phrase in ("です・ます調", "質問攻めにしない", "個人的すぎる話題", "no_intervention を選びます", "意見・体験・好み",
                   "テーマに戻す必要はありません", "意見の整理・結論づけ"):
        assert phrase in CHAT_SYSTEM_PROMPT, phrase
    for gd_word in ("organize_issues", "compare_criteria", "redirect_topic", "offer_hint", "ファシリテーター"):
        assert gd_word not in CHAT_SYSTEM_PROMPT, gd_word


def test_user_prompt_uses_chat_context():
    llm = StubLLM("offer_topic", "最近行ってよかった場所はありますか?")
    st, ag = make_chat(llm)
    now = started(st, ag)
    ag.tick(now, "chat_lead")
    u = llm.user
    assert "AI 主導期" in u and "答えた人: 1人" in u and "これまでに AI が出した話題" in u and "話題の候補" in u
    assert "AI(司会)" in u and "ファシリテーター" not in u and "目標フェーズ" not in u
    assert ag.opening not in llm.ctx["chat"]["topic_candidates"]  # 出した話題は候補から除く


def test_gd_gw_prompts_and_schemas_unchanged_by_chat_actions():
    """GD・GW の出力スキーマに雑談の行動は入らない(全議論タイプ・内容支援の範囲)。プロンプトも雑談の文を含まない。"""
    for dtype in DISCUSSION_TYPES.values():
        for scope in ("none", "hint", "concrete"):
            enum = build_output_schema(dtype, scope)["properties"]["decision"]["properties"]["action"]["enum"]
            assert not set(enum) & {a.value for a in CHAT_ONLY_ACTIONS}
    for purpose in ("job", "general"):
        assert "offer_topic" not in build_system_prompt(purpose) and "change_topic" not in build_system_prompt(purpose)
    gd = FacilitatorAgent(DiscussionState(settings=load_settings(), theme="t", duration_sec=600), StubLLM(), load_settings())
    assert gd.system_prompt == build_system_prompt("job") and gd.invite_all_fallback == INVITE_ALL_FALLBACK


def test_chat_schema_has_only_chat_actions():
    _st, ag = make_chat()
    enum = ag.output_schema["properties"]["decision"]["properties"]["action"]["enum"]
    assert set(enum) == {"no_intervention", "offer_topic", "ask_question", "invite_all", "change_topic", "invite_person",
                         "time_warning", "close_gd"}
    assert ag.output_schema["properties"]["analysis"]["properties"]["phase"]["enum"] == ["warm_up", "expand", "closing"]


# ---------- 2. 行動の許可と検証 ----------
def test_gate_allows_chat_actions_and_person_only_when_safe():
    st, ag = make_chat()
    now = started(st, ag)
    gate, allowed, _ = ag._gate(now)
    assert gate == "llm" and allowed == CHAT_ACTIONS + [Action.invite_person]  # 名前確認済み・信頼度十分
    st2, ag2 = make_chat(names=False)
    gate, allowed, _ = ag2._gate(started(st2, ag2))
    assert allowed == CHAT_ACTIONS  # 名前が確認されていなければ個人指名は許可しない
    gd = FacilitatorAgent(st, StubLLM(), load_settings(), purpose="general", policy=preset_policy("general_gw"))
    later = now + 60  # GD・GW のクールダウン(45秒)の後
    assert not {Action.offer_topic, Action.change_topic} & set(gd._gate(later)[1])  # GD・GW の許可は従来どおり
    assert set(FREE_ACTIONS) <= set(gd._gate(later)[1])


@pytest.mark.parametrize("action", ["organize_issues", "redirect_topic", "compare_criteria", "offer_hint", "propose_idea"])
def test_gd_actions_are_rejected_in_chat(action):
    st, ag = make_chat(StubLLM(action, "ここまでの論点を整理しましょう。"))
    rec = ag.tick(started(st, ag), "chat_lead")
    assert rec.decision.action == Action.no_intervention and "許可外" in rec.overridden


@pytest.mark.parametrize("action", ["offer_topic", "ask_question", "invite_all", "change_topic"])
def test_chat_actions_are_spoken_and_recorded(action):
    st, ag = make_chat(StubLLM(action, "そうなんですね。休みの日はどこかに出かけますか?最近だとどうですか?"))
    rec = ag.tick(started(st, ag), "chat_lead")
    assert rec.decision.action.value == action
    assert rec.decision.speech == "そうなんですね。休みの日はどこかに出かけますか?"  # 2文まで
    assert st.interventions[-1].action.value == action and st.utterances[-1].source == "ai"
    assert rec.content_flags == []


def test_names_and_person_invite_follow_existing_safety():
    st, ag = make_chat(StubLLM("offer_topic", "はなこさんは、最近行ってよかった場所はありますか?"))
    rec = ag.tick(started(st, ag), "chat_lead")
    assert "はなこ" not in rec.decision.speech  # 個人名は invite_person 以外では使わない(文頭の呼びかけは外す)
    # 名前が確認されていない人への指名は、雑談用の全体への呼びかけにする
    st2, ag2 = make_chat(StubLLM("invite_person", "よければ、どうですか?", target="speaker_9"))
    rec2 = ag2.tick(started(st2, ag2), "chat_lead")
    assert rec2.decision.action == Action.invite_all and rec2.decision.speech == CHAT_INVITE_ALL
    # 確認済みで信頼度も十分なら、名前で呼びかけられる
    st3, ag3 = make_chat(StubLLM("invite_person", "はなこさんは、休みの日はどう過ごしていますか?", target="speaker_2"))
    rec3 = ag3.tick(started(st3, ag3), "chat_lead")
    assert rec3.decision.action == Action.invite_person and "はなこさん" in rec3.decision.speech


def test_live_sentence_limits_chat_to_two_sentences():
    st, ag = make_chat()
    assert ag.live_sentence(Action.offer_topic, "最近どうですか?", 1) == "最近どうですか?"
    assert ag.live_sentence(Action.offer_topic, "最近どうですか?", 2) is None
    assert ag.live_sentence(Action.ask_question, "はなこさん、どうですか?", 0) == "どうですか?"
    gd = FacilitatorAgent(st, StubLLM(), load_settings())
    assert gd.live_sentence(Action.ask_question, "どうですか?", 2) == "どうですか?"  # GD・GW は従来どおり3文


def test_request_answered_with_topic_even_right_after_ai_and_without_content_support():
    """「AIさん、何か話題ありますか」には、AI が話した直後でも、内容支援が「なし」の設定でも、話題で応じる。"""
    for policy in (preset_policy("general_gw"), InterventionPolicy()):
        llm = StubLLM("offer_topic", "では、最近行ってよかった場所はありますか?")
        st, ag = make_chat(llm, policy=policy, settings=load_settings(tts_chars_per_sec=1000.0))
        ag.tick(0.0, "start")
        end = st.utterances[-1].end_time
        st.add_utterance("speaker_2", "AIさん、何か話題ありますか？", end + 0.5, end + 3, 0.9)
        assert ag.should_tick(end + 5.5) == "request"
        rec = ag.tick(end + 5.5, "request")  # AI の発言から6秒(chat_min_gap_sec)たっていない
        assert rec.gate == "llm:request" and llm.ctx["allowed_actions"] == [a.value for a in REQUEST_ACTIONS]
        assert rec.decision.action == Action.offer_topic and ag.pending_request is None


# ---------- 3. 開始の話題 ----------
def test_opening_uses_theme_topic_and_no_time_explanation():
    st, ag = make_chat(theme="休日の過ごし方")
    rec = ag.tick(0.0, "start")
    text = rec.decision.speech
    assert rec.decision.action == Action.start_gd and "「休日の過ごし方」" in text and ag.opening in text
    assert ag.opening in topic_candidates("休日の過ごし方") and ag.opening not in FREE_TOPICS  # テーマに沿った候補
    assert "時間は" not in text and "始めてください" not in text
    assert ag.topics_used == [ag.opening]


def test_opening_for_free_chat_and_unknown_theme():
    st, ag = make_chat(theme=CASUAL_FREE_THEME)
    text = ag.tick(0.0, "start").decision.speech
    assert ag.opening in FREE_TOPICS and "自由な雑談" not in text and "気軽にお話ししましょう" in text
    t = opening_topic("宇宙の話", random.Random(1))
    assert "「宇宙の話」" in t  # キーワードに当たらないテーマは、テーマの言葉を入れた問いかけ
    assert opening_speech("宇宙の話", t).endswith(t)


def test_opening_waits_for_answer_then_for_reaction():
    """開始の発言も AI の発言として第2段階の待ちに入る(答えがなければ10秒、答えがあれば反応を6秒待つ)。"""
    s = load_settings()
    st, ag = make_chat(StubLLM("invite_all", "どなたからでも大丈夫ですよ。"), settings=s)
    st.started = False
    ag.tick(0.0, "start")
    end = st.utterances[-1].end_time
    assert end > 5  # 実際の長さの見積もり(あいさつ + 話題)
    assert ag.should_tick(end + s.chat_no_answer_sec - 0.5) is None
    assert ag.should_tick(end + s.chat_no_answer_sec + 0.01) == "chat_lead"
    st2, ag2 = make_chat(settings=s)
    ag2.tick(0.0, "start")
    end2 = st2.utterances[-1].end_time
    st2.add_utterance("speaker_1", "寝てばかりです。", end2 + 1, end2 + 3, 0.9)
    assert ag2.should_tick(end2 + 3 + s.chat_reaction_wait_sec - 0.5) is None
    assert ag2.should_tick(end2 + 3 + s.chat_reaction_wait_sec + 0.01) == "chat_lead"


# ---------- 4. 既存の engine(生成しながら話す処理・記録)との接続 ----------
class FastMock(MockLLM):
    stream_delay = 0.0

    def __init__(self) -> None:
        self.partials = 0

    def generate_json(self, system, user, schema, context, on_partial=None):
        def spy(acc: str) -> None:
            self.partials += 1
            on_partial(acc)
            time.sleep(0.002)

        return super().generate_json(system, user, schema, context, spy if on_partial else None)


def test_engine_streams_chat_actions_with_mock(monkeypatch, tmp_path):
    llm = FastMock()
    monkeypatch.setattr(engine_mod, "make_llm", lambda settings: llm)
    eng = Engine(load_settings(), clock=SimClock(), log_dir=tmp_path, tts_chars_per_sec=1000.0)
    try:
        eng.start(SessionConfig(CASUAL_FREE_THEME, 10, ["たろう", "はなこ"], mode="text", intro=False, tts=False, purpose="general",
                                discussion_type="casual"), background=False)
        assert type(eng.agent) is ChatHostAgent and eng.state.theme == CASUAL_FREE_THEME
        eng._run_tick(eng._agent_step())  # 開始のあいさつと話題
        first = eng.state.utterances[0]
        assert first.source == "ai" and eng.agent.opening in first.text
        eng.clock.sleep(1.0)
        eng.add_text("speaker_1", "最近は料理にハマっています。")
        eng.clock.sleep(7.0)  # 反応の待ちが過ぎた
        trig = eng._agent_step()
        assert trig == "chat_lead"
        eng._run_tick(trig)
        ai = [u for u in eng.state.utterances if u.source == "ai"]
        assert len(ai) == 2 and ai[1].text == "そうなんですね。ほかの皆さんはどうですか?" and llm.partials > 0
        assert eng.state.interventions[-1].action == Action.invite_all  # 生成しながら話し、介入として記録された
        eng.add_text("speaker_2", "AIさん、何か話題ありますか？")
        eng.clock.sleep(2.5)
        trig = eng._agent_step()
        assert trig == "request"
        eng._run_tick(trig)
        assert eng.state.interventions[-1].action == Action.offer_topic
        assert eng.state.utterances[-1].text.startswith("では、")
        assert eng.agent.topics_used[-1] == eng.state.utterances[-1].text
    finally:
        eng.stop()


def test_mock_changes_topic_after_repeated_silence():
    st, ag = make_chat(MockLLM(), settings=load_settings(tts_chars_per_sec=1000.0))
    ag.tick(0.0, "start")
    r1 = ag.tick(20.0, "chat_lead")
    assert r1.decision.action == Action.invite_all  # 答えがない: まず答えやすく呼びかける
    r2 = ag.tick(60.0, "chat_lead")
    assert r2.decision.action == Action.change_topic and r2.decision.speech.startswith("話は変わりますが、")
    assert ag.topics_used[0] not in r2.decision.speech


def test_mock_decision_for_gd_unchanged():
    """モックの GD・GW の判断は従来どおり(雑談の情報がない状況では雑談用の分岐に入らない)。"""
    ctx = {"allowed_actions": ["no_intervention", "ask_question"], "silence_sec": 15, "observed_phase": "",
           "target_phase": "ideation"}
    d = MockLLM()._decide(ctx)["decision"]
    assert d["action"] == "ask_question" and d["speech"].startswith("ここまでの案の中で")


def test_decision_model_accepts_chat_actions():
    assert Decision(action="offer_topic").action == Action.offer_topic
