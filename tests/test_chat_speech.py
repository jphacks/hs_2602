"""第5段階の実 Claude 評価で見つかった発言の問題(⑤)と topic_by の基準(⑥)の回帰テスト。LLM はスタブで、API は呼ばない。

⑤ invite_all で開始の問いを繰り返す(シナリオ7・9)/ 全員が話した後の「ほかの皆さんは」(シナリオ3・7)/
   1回の発言に問いが2つ(シナリオ8)/ 定型の相づちの繰り返し / 自分の話題への引き戻し・人間のような感覚(プロンプト)
⑥ topic_by: AI の問いへの答えは ai、参加者が自分から別の話題に移ったときだけ participants
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings  # noqa: E402
from facilitator.chat import speech_check as sc  # noqa: E402
from facilitator.chat.host import ChatHostAgent  # noqa: E402
from facilitator.chat.prompts import CHAT_SYSTEM_PROMPT  # noqa: E402
from facilitator.models import Action, Decision  # noqa: E402
from facilitator.policy import preset_policy  # noqa: E402
from facilitator.state import DiscussionState  # noqa: E402

S = load_settings(tts_chars_per_sec=1000.0, num_participants=3)
NAMES = {"speaker_1": "たろう", "speaker_2": "はなこ", "speaker_3": "けんた"}
OPENING = "最近ちょっとハマっていることはありますか?"


def output(action, speech="", health="stalled", topic="", by="ai"):
    return {"analysis": {"phase": "expand", "phase_reason": "", "health": health, "recent_summary": "",
                         "topic": topic, "topic_by": by},
            "decision": {"action": action, "reasoning": "", "speech": speech, "urgency": 0.5, "target_speaker": None}}


class SeqLLM:
    name = "seq"

    def __init__(self, outs):
        self.outs, self.calls = list(outs), []

    def generate_json(self, system, user, schema, ctx):
        self.calls.append((ctx, user))
        return self.outs.pop(0) if self.outs else output("no_intervention")


def make(llm=None):
    st = DiscussionState(settings=S, theme="自由な雑談", duration_sec=900)
    for sid, n in NAMES.items():
        st.set_speaker_name(sid, n)
    ag = ChatHostAgent(st, llm or SeqLLM([]), S, purpose="general", discussion_type="casual",
                       policy=preset_policy("general_gw"), rng=random.Random(0))
    ag.opening = OPENING
    ag.tick(0.0, "start")
    return st, ag


def talk(st, lines, start=1.0):
    t = start
    for sid, text in lines:
        st.add_utterance(sid, text, t, t + 3, 0.9)
        t += 4
    return t


# ================= speech_check の単体 =================
def test_one_question_keeps_first_question_and_other_sentences():
    assert sc.one_question("どちらに行ったんですか?何を食べたんですか?") == "どちらに行ったんですか?"
    assert sc.one_question("いいですね。どんなパンですか?") == "いいですね。どんなパンですか?"  # 1つなら変えない
    assert sc.one_question("旅行は好きですか?気軽にどうぞ。行ってみたい場所はありますか?") == "旅行は好きですか?気軽にどうぞ。"


def test_proposal_is_not_a_question():
    """実 Claude の評価(最終)で見つかった: 「別の話題にしましょうか。」を問いと数えて、本当の問いを落としていた。"""
    speech = "では、別の話題にしましょうか。最近食べたもので、おいしかったものはありますか?"
    assert not sc.is_question("では、別の話題にしましょうか。")
    assert sc.one_question(speech) == speech
    assert ChatHostAgent._topic_text(speech) == "最近食べたもので、おいしかったものはありますか?"
    assert sc.is_question("どうでしょうか。") and sc.is_question("朝ごはんは何派ですか。")


def test_request_answer_keeps_real_question():
    """シナリオ8: 依頼に応じた話題の転換で、誘いかけの文だけが残らない。"""
    llm = SeqLLM([output("change_topic", "では、別の話題にしましょうか。最近食べたもので、おいしかったものはありますか?")])
    st, ag = make(llm)
    st.add_utterance("speaker_1", "うーん、特にないかもです。", 1, 3, 0.9)
    st.add_utterance("speaker_2", "AIさん、何か話題ある？", 5, 7, 0.9)
    rec = ag.tick(9.5, "request")
    assert rec.decision.action == Action.change_topic, rec.overridden
    assert rec.decision.speech.endswith("おいしかったものはありますか?") and not rec.overridden
    assert ag.topics.current.text == "最近食べたもので、おいしかったものはありますか?"
    assert ag.live_sentence(Action.ask_question, "では、少し聞いてみましょうか。", 0)
    assert ag.live_sentence(Action.ask_question, "最近よく行くお店はありますか?", 1)


@pytest.mark.parametrize("a, b, same", [
    (OPENING, "ほかの皆さんは、最近ちょっとハマっていることはありますか?", True),  # 呼びかけを足しただけ(シナリオ7)
    (OPENING, "最近ちょっとハマっていることはありますか?", True),
    ("ほかの皆さんはどうですか?", "皆さんはどうですか?", True),
    ("休みの日はどう過ごしていますか?", "ほかの方も、休みの日はどう過ごしていますか?", True),
    ("休みの日はどう過ごしていますか?", "休みの日の朝はどう過ごしていますか?", False),  # 絞り込みは止めない
    ("どんなパンを焼くんですか?", "どんなパンが好きですか?", False),  # 深掘りの続きは止めない
    (OPENING, "最近ちょっとしたことで嬉しかったことはありますか?", False),
    ("ほかの皆さんはどうですか?", "どなたか、思いついたことはありますか?", False),
])
def test_same_question(a, b, same):
    assert sc.same_question(a, b) is same


def test_without_others_and_formula():
    assert sc.without_others("ほかの皆さんは、外とおうちのどちらが好きですか?") == "皆さんは、外とおうちのどちらが好きですか?"
    assert sc.without_others("ほかの方も、よければどうぞ。") == "皆さんも、よければどうぞ。"
    recent = ["キャンプの話、盛り上がりましたね。ほかに好きな過ごし方はありますか?"]
    assert sc.without_repeated_formula("ラーメンの話も盛り上がりましたね。最近行ったお店はありますか?", recent) == "最近行ったお店はありますか?"
    assert sc.without_repeated_formula("盛り上がりましたね。最近行ったお店はありますか?", []) == \
        "盛り上がりましたね。最近行ったお店はありますか?"  # 初めて使うなら外さない


# ================= ⑤ シナリオの回帰 =================
def test_invite_all_does_not_repeat_opening_question():
    """シナリオ7: 開始の話題で盛り上がった後、橋渡しで開始の問いを繰り返さない(見送る)。"""
    llm = SeqLLM([output("invite_all", "ほかの皆さんは、最近ちょっとハマっていることはありますか?", health="converging")])
    st, ag = make(llm)
    end = talk(st, [("speaker_1", "ラーメンの食べ歩きです。"), ("speaker_2", "どこがおすすめですか?"),
                    ("speaker_3", "あそこ並びますよね。")])  # 全員が話した → 自走期
    ag.should_tick(end)
    rec = ag.tick(end + S.chat_free_min_sec + 1, "chat_lead")  # 途切れて主導期に戻った
    assert rec.llm_raw_action == "invite_all" and rec.decision.action == Action.no_intervention
    assert "同じ問い" in rec.overridden


def test_bridge_with_same_question_to_others_is_allowed():
    """1人が答えた後に、まだ話していない人へ同じ問いを向ける橋渡しは自然なので止めない。
    誰も答えていない問いの繰り返しと、全員が話した後の繰り返しは止める。"""
    bridge = "ほかの皆さんは、最近ちょっとハマっていることはありますか?"
    st, ag = make()
    d, note = ag._validate(Decision(action=Action.invite_all, speech=bridge), [Action.invite_all], 11.0)
    assert d.action == Action.no_intervention and "同じ問い" in note  # 誰も答えていない
    st.add_utterance("speaker_1", "パンを焼くことです。", 12, 14, 0.9)
    d, note = ag._validate(Decision(action=Action.invite_all, speech=bridge), [Action.invite_all], 21.0)
    assert d.action == Action.invite_all and d.speech == bridge
    d, note = ag._validate(Decision(action=Action.ask_question, speech="最近ちょっとハマっていることはありますか?"),
                           [Action.ask_question], 21.0)
    assert d.action == Action.no_intervention  # 呼びかけのない、同じ問いの繰り返し
    talk(st, [("speaker_2", "私は映画です。"), ("speaker_3", "僕はゲームです。")], start=22)
    d, note = ag._validate(Decision(action=Action.invite_all, speech=bridge), [Action.invite_all], 40.0)
    assert d.action == Action.no_intervention  # 全員が話した後


def test_topic_action_repeating_recent_question_is_replaced():
    llm = SeqLLM([output("change_topic", "話は変わりますが、最近ちょっとハマっていることはありますか?")])
    st, ag = make(llm)
    ag._act(Decision(action=Action.invite_all, speech="どなたからでも大丈夫ですよ。"), 12.0)
    rec = ag.tick(40.0, "chat_lead")
    assert rec.decision.action == Action.change_topic and "ハマ" not in rec.decision.speech


def test_no_others_when_everyone_has_spoken():
    """シナリオ3・7: いまの話題で全員が話した後は「ほかの皆さんは」と言わない。まだ話していない人がいれば変えない。"""
    speech = "ほかの皆さんは、外で過ごすのとおうちでのんびりするの、どちらが好きですか?"
    st, ag = make()
    talk(st, [("speaker_1", "キャンプです。"), ("speaker_2", "焚き火いいですね。")])
    d, note = ag._validate(Decision(action=Action.invite_all, speech=speech), [Action.invite_all], 20.0)
    assert d.speech == speech and not note  # けんたさんはまだ話していない
    st.add_utterance("speaker_3", "僕も行ってみたいです。", 20, 22, 0.9)
    assert ag.everyone_spoke(23.0)
    d, note = ag._validate(Decision(action=Action.invite_all, speech=speech), [Action.invite_all], 23.0)
    assert d.speech.startswith("皆さんは、") and "ほかの" not in d.speech and "皆さん" in note


def test_everyone_spoke_counts_only_current_topic():
    st, ag = make()
    talk(st, [("speaker_1", "キャンプです。"), ("speaker_2", "いいですね。"), ("speaker_3", "僕もです。")])
    assert ag.everyone_spoke(14.0)
    ag._act(Decision(action=Action.change_topic, speech="話は変わりますが、朝ごはんは何派ですか?"), 30.0)
    st.add_utterance("speaker_1", "パン派です。", 31, 33, 0.9)
    assert not ag.everyone_spoke(34.0)


def test_two_questions_become_one():
    """シナリオ8: 1回の発言で問いを2つしない(最初の問いを残す)。"""
    st, ag = make()
    st.add_utterance("speaker_3", "この前ちょっと遠出しました。", 1, 4, 0.9)
    d, note = ag._validate(Decision(action=Action.ask_question, speech="どちらに行ったんですか?楽しかったですか?"),
                           [Action.ask_question], 12.0)
    assert d.speech == "どちらに行ったんですか?" and "問いかけを1つ" in note


def test_long_speech_keeps_question():
    """実 Claude の評価(最終)で見つかった: 3文の発言を2文に切ると、問いかけが落ちて相づちだけになっていた(シナリオ9)。"""
    speech = ("もちろん大丈夫です。休みの日の過ごし方、人によっていろいろですよね。"
              "ほかにも、こんなふうに過ごしているという方がいれば、ぜひ聞かせてもらえますか?")
    st, ag = make()
    st.add_utterance("speaker_1", "休みの日は実家に帰ることが多いです。", 1, 4, 0.9)
    d, note = ag._validate(Decision(action=Action.invite_all, speech=speech), [Action.invite_all], 12.0)
    assert d.speech == "もちろん大丈夫です。ほかにも、こんなふうに過ごしているという方がいれば、ぜひ聞かせてもらえますか?"
    assert "問いかけを残して" in note
    assert sc.keep_question("いいですね。どんなパンですか?気軽にどうぞ。", 2) == "いいですね。どんなパンですか?気軽にどうぞ。"


def test_repeated_formula_is_dropped():
    st, ag = make()
    ag._act(Decision(action=Action.invite_all, speech="キャンプの話、盛り上がりましたね。ほかに好きな過ごし方はありますか?"), 20.0)
    st.add_utterance("speaker_1", "ラーメンも好きです。", 21, 23, 0.9)
    d, note = ag._validate(Decision(action=Action.ask_question, speech="ラーメンの話も盛り上がりましたね。どんな味が好きですか?"),
                           [Action.ask_question], 30.0)
    assert d.speech == "どんな味が好きですか?" and "定型の相づち" in note


def test_normal_deepening_is_not_blocked():
    """正常な深掘り・言い換えは止めない。"""
    st, ag = make()
    st.add_utterance("speaker_1", "最近は家でパンを焼くのにハマってます。", 1, 4, 0.9)
    for speech in ("いいですね。どんなパンを焼くことが多いんですか?", "パンを焼き始めたきっかけは何だったんですか?"):
        d, note = ag._validate(Decision(action=Action.ask_question, speech=speech), [Action.ask_question], 12.0)
        assert d.action == Action.ask_question and d.speech == speech and not note, (speech, note)


def test_live_speech_applies_same_checks():
    """生成しながら話す場合も、2つ目の問い・同じ問いの文から先は話さず、「ほかの皆さん」を言い換える。"""
    st, ag = make()
    assert ag.live_sentence(Action.ask_question, "いいですね。", 0) == "いいですね。"
    assert ag.live_sentence(Action.ask_question, "どこに行ったんですか?", 1) == "どこに行ったんですか?"
    assert ag.live_sentence(Action.ask_question, "楽しかったですか?", 2) is None  # 文数でも止まるが、2つ目の問いでも止まる
    assert ag.live_sentence(Action.ask_question, "どこですか?", 0) == "どこですか?"
    assert ag.live_sentence(Action.ask_question, "楽しかったですか?", 1) is None
    assert ag.live_sentence(Action.invite_all, "ほかの皆さんは、最近ちょっとハマっていることはありますか?", 0) is None
    talk(st, [("speaker_1", "キャンプです。"), ("speaker_2", "いいですね。"), ("speaker_3", "僕もです。")])
    assert ag.live_sentence(Action.invite_all, "ほかの皆さんは、外とおうちのどちらが好きですか?", 0) == \
        "皆さんは、外とおうちのどちらが好きですか?"


def test_prompt_guides_natural_hosting():
    for phrase in ("問いかけは1回の発言で1つだけ", "あなたの最近の発言", "おなかがすいてきますね", "前の話題に結びつけたり",
                   "「ほかの皆さんは」とは言いません", "同じ相づち", "迷ったら no_intervention"):
        assert phrase in CHAT_SYSTEM_PROMPT, phrase


def test_user_prompt_shows_recent_ai_and_who_spoke():
    llm = SeqLLM([output("no_intervention")])
    st, ag = make(llm)
    talk(st, [("speaker_1", "キャンプです。"), ("speaker_2", "いいですね。")])
    ag.tick(25.0, "chat_lead")
    user = llm.calls[-1][1]
    assert "# あなたの最近の発言" in user and OPENING in user.split("# あなたの最近の発言")[1]
    assert "この話題でまだ話していない人: 1人" in user
    st.add_utterance("speaker_3", "僕もです。", 26, 28, 0.9)
    llm.outs.append(output("no_intervention"))
    ag.tick(40.0, "chat_lead")
    assert "この話題では全員が話しました" in llm.calls[-1][1]


# ================= ⑥ topic_by =================
def test_topic_by_definition_in_prompt():
    for phrase in ("新しい具体例や言葉(店・場所・作品の名前など)が出ても ai", "参加者が自分から、それまでと違う話題を始め",
                   "短い相づちや、一言だけそれた発言は含めません"):
        assert phrase in CHAT_SYSTEM_PROMPT, phrase


def test_answer_with_new_details_keeps_ai_topic():
    """AI の問いへの答え(キャンプ)を LLM が ai と判断すれば、AI の話題のまま(話題の切り替えの数え直しをしない)。"""
    llm = SeqLLM([output("ask_question", "どちらでキャンプをしたんですか?", topic="キャンプ", by="ai")])
    st, ag = make(llm)
    st.add_utterance("speaker_1", "最近キャンプにハマってて、本栖湖に行きました。", 1, 4, 0.9)
    rec = ag.tick(12.0, "chat_lead")
    assert rec.decision.action == Action.ask_question
    assert ag.topics.current.source == "ai" and ag.topics.current.text == OPENING


def test_participant_shift_is_followed_in_prompt():
    """参加者が別の話題に移ったら、プロンプトで「引き戻さない」と伝え、AI の話題の行動は同じ判断では見送る。"""
    llm = SeqLLM([output("change_topic", "話は変わりますが、最近おいしかったものはありますか?", topic="最近観た映画",
                         by="participants"),
                  output("no_intervention", topic="最近観た映画", by="participants")])
    st, ag = make(llm)
    ag._act(Decision(action=Action.invite_all, speech="どなたからでも大丈夫ですよ。"), 11.0)
    talk(st, [("speaker_1", "そういえば昨日映画を観ました。"), ("speaker_2", "何の映画ですか?")], start=13)
    rec = ag.tick(28.0, "chat_lead")
    assert rec.decision.action == Action.no_intervention and ag.topics.current.source == "participants"
    ag.tick(40.0, "chat_lead")
    assert "引き戻さないでください" in llm.calls[-1][1]
