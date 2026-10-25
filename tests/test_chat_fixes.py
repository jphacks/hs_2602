"""第5段階の実 Claude 評価で見つかった問題(①〜④)の回帰テスト。LLM はモック・スタブで、API は呼ばない。

① 話題の文の取り出し(問いの後に補足の文が続くと「?」になっていた)
② 判断の後の再確認で、LLM の health が converging → stalled に変わると offer_topic を誤って見送っていた
③ 「特にないです」のような返事が続くと、待ちが毎回6秒に戻って質問攻めになっていた
④ 順に別の人が答えても、働きかけごとに「1人だけ」と数えていた
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings  # noqa: E402
from facilitator.chat.host import FREE, LEAD, ChatHostAgent  # noqa: E402
from facilitator.chat.topic_tracker import ENGAGED, JOINED, NONE, PASSIVE, SINGLE, passive_reply  # noqa: E402
from facilitator.llm.mock_llm import MockLLM  # noqa: E402
from facilitator.models import Action, Decision  # noqa: E402
from facilitator.policy import preset_policy  # noqa: E402
from facilitator.state import DiscussionState  # noqa: E402

S = load_settings(tts_chars_per_sec=1000.0)  # AI の発話はほぼ一瞬
NAMES = {"speaker_1": "たろう", "speaker_2": "はなこ", "speaker_3": "けんた"}


def make(llm=None, theme="自由な雑談"):
    st = DiscussionState(settings=S, theme=theme, duration_sec=900)
    for sid, n in NAMES.items():
        st.set_speaker_name(sid, n)
    ag = ChatHostAgent(st, llm or MockLLM(), S, purpose="general", discussion_type="casual",
                       policy=preset_policy("general_gw"), rng=random.Random(0))
    return st, ag


def run(st, ag, responder, until):
    """0.25 秒ごとに判断する(tests/test_chat_topics.py と同じ)。返り値は AI の発言 [(時刻, 行動, 発言)]。"""
    pending: list[tuple[float, float, str, str]] = []
    out: list[tuple[float, str, str]] = []
    t = 0.0
    while t <= until:
        for item in sorted(pending, key=lambda x: x[1]):
            if item[1] <= t:
                pending.remove(item)
                st.add_utterance(item[2], item[3], item[0], item[1], 0.9)
        st.speaking_now = any(a <= t < b for a, b, _s, _x in pending)
        if not any(u.source == "ai" and u.end_time > t for u in st.utterances):
            trig = ag.should_tick(t)
            if trig:
                d = ag.tick(t, trig).decision
                if d.action != Action.no_intervention:
                    out.append((round(t, 2), d.action.value, d.speech))
                    end = st.utterances[-1].end_time
                    for delay, dur, sid, text in responder(len(out) - 1, end, d.action.value) or []:
                        pending.append((end + delay, end + delay + dur, sid, text))
        t += 0.25
    return out


def output(action, speech="", health="stalled", topic="", by="ai"):
    return {"analysis": {"phase": "expand", "phase_reason": "", "health": health, "recent_summary": "",
                         "topic": topic, "topic_by": by},
            "decision": {"action": action, "reasoning": "", "speech": speech, "urgency": 0.5, "target_speaker": None}}


class SeqLLM:
    """決めた順に出力を返す(尽きたら見送り)。"""

    name = "seq"

    def __init__(self, outs):
        self.outs, self.calls = list(outs), []

    def generate_json(self, system, user, schema, ctx):
        self.calls.append(ctx)
        return self.outs.pop(0) if self.outs else output("no_intervention")


class Eager:
    """許可されていれば必ず話す(最悪の場合。質問攻めをルール層が止めるかを見る)。"""

    name = "eager"
    ORDER = [("ask_question", "それはどんなときですか?"), ("invite_all", "ほかの皆さんはどうですか?"),
             ("change_topic", "話は変わりますが、最近おいしかったものはありますか?"),
             ("offer_topic", "では、休みの日の朝は何をしていますか?")]

    def __init__(self):
        self.calls = []

    QS = ["休みの日の朝はどう過ごしていますか?", "最近よく行くお店はありますか?", "最近見て面白かったものはありますか?",
          "行ってみたい場所はどこですか?", "子どものころ好きだった遊びは何ですか?", "最近買ってよかったものは何ですか?",
          "好きな季節はいつですか?", "朝ごはんは何派ですか?", "最近覚えたことはありますか?", "得意な料理は何ですか?"]

    def generate_json(self, system, user, schema, ctx):
        self.calls.append(ctx)
        q = self.QS[len(self.calls) % len(self.QS)]  # 毎回違う問い(同じ問いは検証で止まる)
        for a, sp in self.ORDER:
            if a in ctx["allowed_actions"]:
                return output(a, sp if a in ("change_topic", "offer_topic") else q)
        return output("no_intervention")


# ================= ① 話題の文の取り出し =================
@pytest.mark.parametrize("speech, topic", [
    ("最近ハマっていることはありますか？ちょっとしたことでも大丈夫です。", "最近ハマっていることはありますか?"),
    ("最近ハマっていることはありますか?ちょっとしたものでも大丈夫です。", "最近ハマっていることはありますか?"),
    ("話は変わりますが、最近食べておいしかったものはありますか?", "最近食べておいしかったものはありますか?"),
    ("いいですね!では、休みの日はどう過ごしていますか?気軽にどうぞ。", "休みの日はどう過ごしていますか?"),
    ("パンの話、楽しいですね。ところで、朝ごはんは何派ですか。", "朝ごはんは何派ですか?"),  # 疑問符なしの「〜か。」
    ("では、好きな季節について聞かせてください。", "好きな季節について聞かせてください"),  # 問いの形でない
    ("旅行は好きですか?行ってみたい場所はありますか?", "行ってみたい場所はありますか?"),  # 問いが2つなら最後
    ("こんにちは、今日はよろしくお願いします。気軽にお話ししましょう。まずは、最近ちょっとハマっていることはありますか?",
     "最近ちょっとハマっていることはありますか?"),
    ("好きな食べ物はなんですか", "好きな食べ物はなんですか?"),  # 句点も疑問符もない「〜か」も問いとみなす
])
def test_topic_text_handles_multiple_sentences(speech, topic):
    assert ChatHostAgent._topic_text(speech) == topic


def test_recorded_topic_is_never_just_a_question_mark():
    st, ag = make()
    ag.tick(0.0, "start")
    ag._act(Decision(action=Action.change_topic, speech="話は変わりますが、最近見た映画はありますか?どんなものでも大丈夫です。"), 30.0)
    assert ag.topics.current.text == "最近見た映画はありますか?"
    assert all(t.text.strip("?") for t in ag.topics.history)
    # 重複の確認も、取り出した話題の文で効く
    d, note = ag._validate(Decision(action=Action.offer_topic, speech="では、最近見た映画はありますか?なんでもどうぞ。"),
                           [Action.offer_topic], 40.0)
    assert "差し替え" in (note or "") and "映画" not in d.speech


# ================= ② 許可した offer_topic を誤って取り消さない =================
def _settled_conversation(llm):
    """開始の話題に1人が答えた状態をつくる(反応はまだ薄いので、一段落かどうかは LLM の判断だけで決まる)。"""
    st, ag = make(llm)
    ag.tick(0.0, "start")
    st.add_utterance("speaker_1", "最近はラーメンの食べ歩きにハマってます。", 1, 4, 0.9)
    return st, ag


def test_offer_topic_is_not_cancelled_when_health_flips():
    """判断の前は converging(一段落)で offer_topic を許可し、同じ判断で LLM が stalled と答えても取り消さない
    (評価のシナリオ5・7)。"""
    llm = SeqLLM([output("no_intervention", health="converging"),
                  output("offer_topic", "では、休みの日の朝は何をしていますか?", health="stalled")])
    st, ag = _settled_conversation(llm)
    ag.tick(11.0, "chat_lead")  # converging と判断(見送り)
    rec = ag.tick(18.0, "chat_lead")
    assert "offer_topic" in llm.calls[-1]["allowed_actions"]
    assert rec.decision.action == Action.offer_topic, rec.overridden
    assert not rec.overridden or "参加者が始めた話題" not in rec.overridden
    assert ag.topics.current.text == "休みの日の朝は何をしていますか?"


def test_offer_topic_kept_when_llm_names_current_topic_as_participants():
    """LLM が topic_by=participants と答えても、いまの話題と同じ(新しい話題として記録しない)なら取り消さない。"""
    llm = SeqLLM([output("no_intervention", health="converging"),
                  output("offer_topic", "では、休みの日の朝は何をしていますか?", health="stalled",
                         topic="最近ちょっとハマっていること", by="participants")])
    st, ag = _settled_conversation(llm)
    ag.opening = None
    ag.topics.history[-1].text = "最近ちょっとハマっていることはありますか?"
    ag.tick(11.0, "chat_lead")
    rec = ag.tick(18.0, "chat_lead")
    assert rec.decision.action == Action.offer_topic, rec.overridden


def test_new_participant_topic_still_blocks_topic_actions():
    """同じ判断で参加者の新しい話題を記録したときは、従来どおり話題の提供・転換を見送る(理由も正しい)。"""
    llm = SeqLLM([output("no_intervention", health="converging"),
                  output("offer_topic", "では、休みの日の朝は何をしていますか?", topic="最近観た映画", by="participants")])
    st, ag = _settled_conversation(llm)
    ag.tick(11.0, "chat_lead")
    st.add_utterance("speaker_2", "そういえば昨日映画を観ました。", 12, 15, 0.9)
    rec = ag.tick(22.0, "chat_lead")
    assert rec.llm_raw_action == "offer_topic" and rec.decision.action == Action.no_intervention
    assert "参加者が始めた話題" in rec.overridden and ag.topics.current.source == "participants"


# ================= ③ 短い・気のない返事が続いても質問攻めにしない =================
@pytest.mark.parametrize("text, passive", [
    ("特にないです。", True), ("あんまり…。", True), ("普通です。", True), ("どうだろう…。", True), ("特には。", True),
    ("うーん、特にないかもです。", True), ("そうですね。", True), ("えー、わからないです", True),
    ("ラーメンです。", False), ("パン!", False), ("うーん、寝てるくらいです。", False), ("私もです。", False),
    ("はい、そうなんです。", False), ("いいですね。", False), ("", False),
])
def test_passive_reply(text, passive):
    assert passive_reply(text) is passive


PASSIVE_LINES = [("speaker_1", "特にないです。"), ("speaker_2", "あんまり…。"), ("speaker_3", "普通です。"),
                 ("speaker_1", "どうだろう…。"), ("speaker_2", "特には。"), ("speaker_3", "うーん、特にないかもです。")]


def test_passive_replies_do_not_cause_rapid_questions():
    """評価のシナリオ2: 毎回1人が気のない返事をするだけ。待ちが段階的に伸び、質問を連発しない。"""
    def resp(i, _end, _a):
        sid, text = PASSIVE_LINES[i % len(PASSIVE_LINES)]
        return [(2, 1.5, sid, text)]

    st, ag = make(Eager())
    out = run(st, ag, resp, 120)
    times = [t for t, _a, _s in out]
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert all(g >= S.chat_passive_wait_sec for g in gaps), gaps
    assert gaps == sorted(gaps) and gaps[-1] > gaps[0]  # 続くほど長く待つ
    assert len(out) <= 5  # 2分で5回まで(修正前は約17秒ごとに7回以上話していた)
    assert max(gaps) <= S.chat_passive_wait_sec * (2 + S.chat_passive_backoff_max)  # 待ちには上限がある
    assert ag.chat_phase == LEAD


def test_short_but_engaged_answers_are_not_held_back():
    """短くても中身のある答えは、通常の反応の待ち(chat_reaction_wait_sec)のまま。"""
    st, ag = make()
    ag.tick(0.0, "start")
    st.add_utterance("speaker_1", "ラーメンです。", 1, 2, 0.9)
    cs = ag.chat_state(3.0)
    assert not cs.passive_only and ag._need_silence(cs) == S.chat_reaction_wait_sec
    assert ag.chat_wait_reason(3.0, cs) == "chat_wait_reaction"


def test_active_reply_releases_passive_backoff():
    """気のない返事が続いた後でも、参加者が中身のある話をすれば、すぐ通常の待ちに戻る。"""
    st, ag = make()
    ag.tick(0.0, "start")
    t = 0.0
    for k, (sid, text) in enumerate(PASSIVE_LINES[:3]):
        st.add_utterance(sid, text, t + 1, t + 2, 0.9)
        if k < 2:
            t += 30
            ag._act(Decision(action=Action.invite_all, speech="ほかの皆さんはどうですか?"), t)
    cs = ag.chat_state(t + 3)
    assert cs.passive_only and cs.quiet_streak == 3
    assert ag._need_silence(cs) == S.chat_passive_wait_sec * 3
    assert ag.chat_wait_reason(t + 3, cs) == "chat_wait_passive"
    st.add_utterance("speaker_2", "あ、でも最近パンを焼き始めました。", t + 4, t + 7, 0.9)
    cs = ag.chat_state(t + 8)
    assert not cs.passive_only and cs.quiet_streak == 0
    assert ag._need_silence(cs) == S.chat_reaction_wait_sec


def test_passive_replies_do_not_make_links():
    """気のない返事が別の人から続いても、人間どうしの会話(自走期)とはみなさない。"""
    st, ag = make()
    ag.tick(0.0, "start")
    for k, (sid, text) in enumerate(PASSIVE_LINES[:3]):
        st.add_utterance(sid, text, 1 + 2.5 * k, 2.5 + 2.5 * k, 0.9)
    ag.should_tick(10.0)
    assert ag.chat_state(10.0).links_after_ai == 0 and ag.chat_phase == LEAD
    assert ag.topic_status(10.0).reactions == [PASSIVE]


def test_no_answer_backoff_unchanged():
    """答えがないときの待ち(第2段階)は従来どおり。"""
    st, ag = make()
    ag.tick(0.0, "start")
    assert ag._need_silence(ag.chat_state(1.0)) == S.chat_no_answer_sec
    ag._act(Decision(action=Action.invite_all, speech="どなたからでも大丈夫ですよ。"), 11.0)
    assert ag._need_silence(ag.chat_state(12.0)) == S.chat_no_answer_sec * 2


# ================= ④ 話題ごとの参加人数 =================
def test_people_joining_one_by_one_are_recognized():
    """評価のシナリオ1・9: 開始の問いにたろう、橋渡しにはなこが答えた。はなこの答えは「薄い反応」ではない。
    ただし1人ずつ AI に答えただけなので、人間どうしの会話(自走期)にはしない。"""
    st, ag = make()
    ag.tick(0.0, "start")
    st.add_utterance("speaker_1", "最近は家でパンを焼くのにハマってます。", 1, 4, 0.9)
    ag._act(Decision(action=Action.invite_all, speech="ほかの皆さんはどうですか?"), 12.0)
    st.add_utterance("speaker_2", "私は食べる専門です。", 14, 16, 0.9)
    ag.should_tick(24.0)
    ts = ag.topic_status(24.0)
    assert ts.reactions == [SINGLE, JOINED] and ts.answerers == 2
    assert ts.weak_streak == 0 and not ts.can_change and not ts.engaged
    assert ag.chat_phase == LEAD
    gate, allowed, _ = ag._gate(24.0)
    assert Action.change_topic not in allowed  # 橋渡しが効いているので、まだ切り替えない


def test_same_person_answering_again_is_still_weak():
    st, ag = make()
    ag.tick(0.0, "start")
    st.add_utterance("speaker_1", "パンを焼いてます。", 1, 3, 0.9)
    ag._act(Decision(action=Action.invite_all, speech="ほかの皆さんはどうですか?"), 12.0)
    st.add_utterance("speaker_1", "最近はベーグルも焼きます。", 14, 16, 0.9)
    ts = ag.topic_status(24.0)
    assert ts.reactions == [SINGLE, SINGLE] and ts.answerers == 1 and ts.weak_streak == 2 and ts.can_change
    assert ts.one_on_one == 2 and ts.partner == "speaker_1"


def test_joined_then_silence_counts_from_there():
    """加わった後に答えがなくなれば、そこから薄い反応を数え直す。"""
    st, ag = make()
    ag.tick(0.0, "start")
    st.add_utterance("speaker_1", "パンを焼いてます。", 1, 3, 0.9)
    ag._act(Decision(action=Action.invite_all, speech="ほかの皆さんはどうですか?"), 12.0)
    st.add_utterance("speaker_2", "私は食べる専門です。", 14, 16, 0.9)
    ag._act(Decision(action=Action.ask_question, speech="どんなパンが好きですか?"), 26.0)
    ag._act(Decision(action=Action.invite_all, speech="どなたからでも大丈夫ですよ。"), 40.0)
    ts = ag.topic_status(60.0)
    assert ts.reactions == [SINGLE, JOINED, NONE, NONE] and ts.weak_streak == 2 and ts.can_change
    assert ts.answerers == 2


def test_conversation_among_humans_is_still_engaged_and_goes_free():
    """1回の働きかけの後に人間どうしのやり取りが続けば、従来どおり engaged で自走期に移る。"""
    st, ag = make()
    ag.tick(0.0, "start")
    for k, (sid, text) in enumerate([("speaker_1", "キャンプです。"), ("speaker_2", "どこに行くんですか?"),
                                     ("speaker_1", "近くの湖です。")]):
        st.add_utterance(sid, text, 1 + 4 * k, 4 + 4 * k, 0.9)
    ag.should_tick(13.0)
    ts = ag.topic_status(13.0)
    assert ts.reactions == [ENGAGED] and ts.engaged and ts.answerers == 2
    assert ag.chat_phase == FREE


def test_prompt_shows_answerers_and_new_reaction_kinds():
    llm = SeqLLM([])
    st, ag = make(llm)
    ag.tick(0.0, "start")
    st.add_utterance("speaker_1", "パンを焼いてます。", 1, 3, 0.9)
    ag._act(Decision(action=Action.invite_all, speech="ほかの皆さんはどうですか?"), 12.0)
    st.add_utterance("speaker_2", "特にないです。", 14, 15, 0.9)
    ctx = ag._context(40.0, "chat_lead", [Action.no_intervention], None)
    prompt = ag._user_prompt(ctx)
    assert "この話題で答えた人: 1人" in prompt
    assert "「特にない」のような短い返事だけ" in prompt
