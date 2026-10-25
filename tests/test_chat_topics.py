"""雑談の話題の管理と、続ける・広げる・切り替えるの判断(第4段階)の会話シナリオ。LLM はモック・スタブで、API は呼ばない。

参加者は AI の発言に反応して話す(responder)。AI の判断のタイミングは第2段階のルール(待ち時間・自走期)で決まる。
"""

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings  # noqa: E402
from facilitator.chat.host import FREE, ChatHostAgent  # noqa: E402
from facilitator.chat.topic_tracker import ENGAGED, NONE, SINGLE  # noqa: E402
from facilitator.chat.topics import similar_topic  # noqa: E402
from facilitator.llm.mock_llm import MockLLM  # noqa: E402
from facilitator.models import Action, Decision  # noqa: E402
from facilitator.policy import preset_policy  # noqa: E402
from facilitator.state import DiscussionState  # noqa: E402

S = load_settings(tts_chars_per_sec=1000.0)  # AI の発話はほぼ一瞬(時刻を読みやすくする)
NAMES = {"speaker_1": "たろう", "speaker_2": "はなこ", "speaker_3": "けんた"}


def make(llm=None, theme="自由な雑談"):
    st = DiscussionState(settings=S, theme=theme, duration_sec=900)
    for sid, n in NAMES.items():
        st.set_speaker_name(sid, n)
    ag = ChatHostAgent(st, llm or MockLLM(), S, purpose="general", discussion_type="casual",
                       policy=preset_policy("general_gw"), rng=random.Random(0))
    return st, ag


def run(st, ag, responder, until):
    """0.25 秒ごとに判断する。AI が話すたびに responder(何回目, 話し終えた時刻, 行動) が参加者の発言
    [(話し始めまでの秒数, 長さ, 話者, 発言)] を返し、それを台本に加える。返り値は AI の発言 [(時刻, 行動, 発言)]。"""
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
                rec = ag.tick(t, trig)
                d = rec.decision
                if d.action != Action.no_intervention:
                    out.append((round(t, 2), d.action.value, d.speech))
                    end = st.utterances[-1].end_time
                    for delay, dur, sid, text in responder(len(out) - 1, end, d.action.value) or []:
                        pending.append((end + delay, end + delay + dur, sid, text))
        t += 0.25
    return out


def actions(out):
    return [a for _t, a, _s in out]


def silent(_i, _end, _a):
    return []


# ---------- AI の話題に1人だけ答え、会話が続かない → まず橋渡し、それでも続かなければ切り替える ----------
def test_one_answer_then_bridge_then_change():
    def resp(i, _end, _a):
        return [(1, 3, "speaker_1", "最近は料理にハマっています。")] if i == 0 else []

    st, ag = make()
    out = run(st, ag, resp, 40)
    assert actions(out)[:3] == ["start_gd", "invite_all", "change_topic"]  # すぐには変えず、橋渡しを試してから
    assert ag.topics.history[0].text == ag.opening and len(ag.topics.history) == 2
    assert out[2][0] - out[1][0] >= S.chat_no_answer_sec  # 橋渡しの後は答えを待ってから


# ---------- 深掘りで会話が広がる → その話題を続け、見守る ----------
def test_deepening_widens_conversation_and_topic_is_kept():
    def resp(i, _end, a):
        if i == 0:  # 2人が間をあけて答える(まだ人間どうしの会話にはなっていない)
            return [(1, 3, "speaker_1", "最近は料理にハマっています。"), (9, 3, "speaker_2", "私はキャンプです。")]
        if a == "ask_question":  # 深掘りで、3人のやり取りが続く
            return [(1, 3, "speaker_2", "道具をそろえるのが楽しくて。"), (5, 3, "speaker_1", "どこに行くんですか?"),
                    (9, 3, "speaker_2", "近くの湖です。"), (13, 3, "speaker_3", "いいですね、行ってみたいです。")]
        return []

    st, ag = make()
    out = run(st, ag, resp, 45)  # 会話が途切れる(15秒の沈黙)前まで
    assert "ask_question" in actions(out)
    assert ag.chat_phase == FREE and actions(out)[-1] == "ask_question"  # 広がったら見守る(その後は話さない)
    assert "change_topic" not in actions(out) and "offer_topic" not in actions(out)
    assert ag.topic_status(45).engaged


# ---------- 橋渡しでほかの参加者が会話に入る ----------
def test_bridge_brings_others_in():
    def resp(i, _end, a):
        if i == 0:
            return [(1, 3, "speaker_1", "休みの日は寝てばかりです。")]
        if a == "invite_all":
            return [(1, 3, "speaker_2", "わかります、私もです。"), (5, 3, "speaker_3", "僕は映画を観に行きます。"),
                    (9, 3, "speaker_1", "何の映画ですか?"), (13, 3, "speaker_3", "アニメ映画です。")]
        return []

    st, ag = make()
    out = run(st, ag, resp, 40)  # 会話が途切れる前まで
    assert actions(out) == ["start_gd", "invite_all"]
    ts = ag.topic_status(30)
    assert ts.reactions == [SINGLE, ENGAGED] and ag.chat_phase == FREE and not ts.can_change


# ---------- 反応が薄い状態が続いたら切り替える / 同じ話題を繰り返さない / 無反応でも連発しない ----------
def test_silence_changes_topic_without_repeating_or_rapid_fire():
    st, ag = make()
    out = run(st, ag, silent, 300)
    acts = actions(out)
    assert acts[1] != "change_topic"  # 1回答えがなかっただけでは変えない
    assert "change_topic" in acts
    times = [t for t, _a, _s in out]
    assert all(b - a >= S.chat_no_answer_sec - 0.01 for a, b in zip(times, times[1:]))  # 連発しない
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert gaps[-1] > gaps[0]  # 答えのない発言が続くほど長く待つ
    assert len(out) <= 10  # 5分の無反応で 10 回まで
    topics = [h.text for h in ag.topics.history]
    for a, b in itertools.combinations(topics, 2):  # 同じ話題・言い換えただけの話題を出さない
        assert a != b and not similar_topic(a, b, st.theme), (a, b)


class ScriptedLLM:
    """行動と、LLM が判断した話題(analysis.topic / topic_by)を決めて返す。"""

    name = "scripted"

    def __init__(self, plan):
        self.plan, self.calls = plan, []

    def generate_json(self, system, user, schema, ctx):
        self.calls.append(ctx)
        action, speech, topic, by = self.plan(ctx)
        return {"analysis": {"phase": "expand", "phase_reason": "", "health": "stalled", "recent_summary": "",
                             "topic": topic, "topic_by": by},
                "decision": {"action": action, "reasoning": "", "speech": speech, "urgency": 0.5, "target_speaker": None}}


def test_duplicate_topic_from_llm_is_replaced():
    """LLM が最近の話題と実質的に同じ話題を出そうとしたら、重ならない候補に差し替える。"""
    st, ag = make(ScriptedLLM(lambda ctx: ("change_topic", "話は変わりますが、最近ハマっているものってありますか?", "", "ai")))
    ag.opening = "最近ちょっとハマっていることはありますか?"
    ag.tick(0.0, "start")
    assert "ハマ" in ag.opening
    ag._act(Decision(action=Action.invite_all, speech="どなたからでも大丈夫ですよ。"), 12.0)
    rec = ag.tick(40.0, "chat_lead")
    assert rec.decision.action == Action.change_topic and "差し替え" in rec.overridden
    assert "ハマ" not in rec.decision.speech and rec.decision.speech.startswith("話は変わりますが、")
    assert not ag.live_ok(Action.change_topic, 40.0) and ag.live_ok(Action.invite_all, 40.0)  # 話題の行動は全文を待つ


# ---------- 参加者が自然に別の話題へ移る → その流れを尊重する ----------
def test_participants_shift_topic_and_ai_follows():
    def plan(ctx):
        if any("映画" in line for line in ctx["transcript"]):  # 参加者が映画の話に移った
            return ("change_topic" if "change_topic" in ctx["allowed_actions"] else "invite_all",
                    "ほかの皆さんは、最近どんな映画を観ましたか?", "最近観た映画", "participants")
        return "invite_all", "どなたからでも大丈夫ですよ。", "", "ai"

    llm = ScriptedLLM(plan)
    st, ag = make(llm)
    ag.tick(0.0, "start")
    ag._act(Decision(action=Action.invite_all, speech="どなたからでも大丈夫ですよ。"), 11.0)
    st.add_utterance("speaker_1", "そういえば、昨日映画を観ました。", 13, 16, 0.9)
    rec = ag.tick(23.0, "chat_lead")
    cur = ag.topics.current
    assert cur.source == "participants" and cur.text == "最近観た映画"
    # 反応が薄い状態が続いていたのは AI の話題。参加者の話題は始まったばかりなので、同じ判断で切り替えを選んでも見送る
    assert rec.llm_raw_action == "change_topic" and rec.decision.action == Action.no_intervention
    assert "参加者が始めた話題" in rec.overridden
    st.add_utterance("speaker_2", "何の映画ですか?", 24, 26, 0.9)
    ag.tick(40.0, "chat_lead")
    assert "change_topic" not in llm.calls[-1]["allowed_actions"]  # 次の判断では、参加者の話題で反応を数え直す
    assert not any("映画" in c or "番組" in c for c in ag.topics.candidates())  # 参加者の話題と同じ候補は出さない
    assert "最近観た映画" in ag.topics.recent()


# ---------- 人間どうしで盛り上がっているときは見守る ----------
def test_lively_conversation_is_left_alone():
    def resp(i, _end, _a):
        if i:
            return []
        lines = ["最近は料理にハマっています。", "何を作るんですか?", "カレーが多いです。", "スパイスからですか?",
                 "そうなんです。", "すごいですね。", "今度教えてください。", "ぜひ。"]
        return [(1 + 4 * k, 3, f"speaker_{k % 3 + 1}", text) for k, text in enumerate(lines)]

    llm = MockLLM()
    st, ag = make(llm)
    out = run(st, ag, resp, 45)
    assert actions(out) == ["start_gd"] and ag.chat_phase == FREE


# ---------- 一人との一問一答が続きすぎない ----------
class QuestionLover:
    """許可されていれば深掘りの質問を選び、なければ全体へ広げる(最悪の場合)。"""

    name = "q"

    def generate_json(self, system, user, schema, ctx):
        allowed = ctx["allowed_actions"]
        if "ask_question" in allowed:
            a, sp = "ask_question", "それはどんなきっかけですか?"
        else:
            a, sp = "invite_all", "ほかの皆さんはどうですか?"
        return {"analysis": {"phase": "warm_up", "phase_reason": "", "health": "stalled", "recent_summary": "",
                             "topic": "", "topic_by": "ai"},
                "decision": {"action": a, "reasoning": "", "speech": sp, "urgency": 0.5, "target_speaker": None}}


def test_one_on_one_does_not_continue_too_long():
    def resp(_i, _end, _a):
        return [(1, 3, "speaker_1", "はい、そうなんです。")]  # いつもたろうだけが答える

    st, ag = make(QuestionLover())
    out = run(st, ag, resp, 60)
    # 1人とだけの一問一答が2回続いたら深掘りをやめて全体へ広げ、それも2回続けて薄ければ繰り返さない
    # (このスタブは切り替えを選ばないので、そのあとは見送る。実際の LLM は話題の切り替えか待つかを選ぶ)
    assert actions(out) == ["start_gd", "ask_question", "invite_all", "invite_all"]
    ts = ag.topic_status(60)
    assert ts.partner == "speaker_1" and ts.one_on_one >= S.chat_one_on_one_max
    # その人への個人指名も、全体への呼びかけにする
    d, note = ag._validate(Decision(action=Action.invite_person, speech="たろうさん、どうですか?", target_speaker="speaker_1"),
                           [Action.invite_person, Action.invite_all], 61.0)
    assert d.action in (Action.invite_all, Action.no_intervention) and "一問一答" in note


def test_topic_settled_allows_new_topic_after_lull():
    """一度広がった話題が途切れたら、新しい話題を出してよい(話題の提供は、いまの話題が続いている間は許可しない)。"""
    st, ag = make()
    ag.tick(0.0, "start")
    for k, (sid, text) in enumerate([("speaker_1", "料理です。"), ("speaker_2", "私もです。"), ("speaker_3", "いいですね。")]):
        st.add_utterance(sid, text, 1 + 3 * k, 3 + 3 * k, 0.9)
    assert not ag.topic_status(9.5).can_offer  # 話題が続いている
    ts = ag.topic_status(9 + S.chat_lull_silence_sec)
    assert ts.engaged and ts.settled and ts.can_offer and not ts.can_change
    assert ag.topic_status(9.5).reactions == [ENGAGED]
    assert NONE not in ts.reactions
