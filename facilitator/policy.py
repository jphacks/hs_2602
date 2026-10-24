"""介入ポリシー: 進行役がどの程度介入するかの6項目の設定。

就活GD用・一般GW用のプリセット(config.POLICY_PRESETS)を初期値にし、利用者が各項目を変更できる。
GD用・GW用で別々の進行役を作らず、共通の進行役がこの設定に応じて動く。

効果:
- ①〜⑤ は、項目ごとの判断条件(config.LEVEL_SETTINGS)と判断基準の文(prompts.LEVEL_TEXT)を変える。標準は現行どおり
- ⑥-A 内容支援の範囲は、使える内容支援のアクション(offer_hint / propose_idea)とプロンプトの内容支援の方針を決める。
  「なし」なら内容支援のアクションは出力の選択肢にも入らず、プロンプトも従来の中立の方針のまま
- ⑥-B 自発的な内容支援は、依頼されていないときに内容支援を許可する条件を決める(FacilitatorAgent._content_actions)

設定の優先順位(resolve_policy): プリセット(指定がなければ用途の既定)< 個別の変更 < 旧画面の「詳細」(knobs)。
最後に依存関係を補正する。現在の画面は「詳細」の5項目を常に全部送るため、①〜④はプリセットの値ではなく画面の値
(既定は標準)になる。画面が変更した項目だけを送るようにするのは第4段階。
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from config import (
    JOB_MAX_LEVEL,
    LEVEL_SETTINGS,
    POLICY_PRESET_LABELS,
    POLICY_PRESETS,
    PURPOSE_PRESET,
    STYLE_KNOBS,
    Settings,
)
from facilitator.models import Action


class Level(str, Enum):
    low = "low"  # 控えめ
    standard = "standard"  # 標準(現行の就活GD相当)
    high = "high"  # 積極的


class ContentScope(str, Enum):
    none = "none"  # 進行支援のみ
    hint = "hint"  # 新しい視点や考え方のヒントまで
    concrete = "concrete"  # 具体的なアイデアや解決策まで


class ContentInitiative(str, Enum):
    none = "none"  # 参加者から依頼されたときだけ
    stalled = "stalled"  # 停滞したときにも自発的に
    proactive = "proactive"  # 必要だと判断したときにも自発的に


LEVEL_ITEMS = ("progress", "drift", "organize", "turns", "stall")  # ①〜⑤
ITEM_LABELS = {
    "progress": "① 進行・時間管理",
    "drift": "② 脱線への対応",
    "organize": "③ 論点・意見の整理",
    "turns": "④ 発言機会の調整",
    "stall": "⑤ 停滞への介入",
    "content_scope": "⑥-A 内容支援の範囲",
    "content_initiative": "⑥-B 自発的な内容支援",
}
VALUE_LABELS = {
    "low": "控えめ", "standard": "標準", "high": "積極的",
    "none": "なし", "hint": "ヒントまで", "concrete": "具体案まで", "stalled": "停滞時のみ", "proactive": "積極的",
}
# 画面の「詳細」の既存項目(STYLE_KNOBS)と対応する項目。suggest は⑥に変換する(from_knobs)
KNOB_FOR_ITEM = {"progress": "phase", "drift": "redirect", "organize": "organize", "turns": "turns"}
LEVEL_INDEX = {Level.low: 0, Level.standard: 1, Level.high: 2}  # STYLE_KNOBS の段階番号
INDEX_LEVEL = {v: k for k, v in LEVEL_INDEX.items()}
# 範囲ごとに使える内容支援のアクション(ヒント → 具体案の順)
CONTENT_ACTIONS_BY_SCOPE = {
    ContentScope.none: [],
    ContentScope.hint: [Action.offer_hint],
    ContentScope.concrete: [Action.offer_hint, Action.propose_idea],
}
CONTENT_ACTIONS = {Action.offer_hint, Action.propose_idea}


class InterventionPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    progress: Level = Level.standard
    drift: Level = Level.standard
    organize: Level = Level.standard
    turns: Level = Level.standard
    stall: Level = Level.standard
    content_scope: ContentScope = ContentScope.none
    content_initiative: ContentInitiative = ContentInitiative.none

    @model_validator(mode="before")
    @classmethod
    def _fix_initiative(cls, data: Any) -> Any:
        """内容支援の範囲が「なし」なら、自発的な内容支援も「なし」に固定する。"""
        if isinstance(data, dict) and data.get("content_scope", ContentScope.none) == ContentScope.none:
            if "content_initiative" in data:
                ContentInitiative(data["content_initiative"])  # 不正な値は補正で隠さずにエラーにする
            data = {**data, "content_initiative": ContentInitiative.none}
        return data

    def updated(self, **changes: Any) -> InterventionPolicy:
        """一部の項目を変えた新しいポリシー(model_copy と違い、検証と依存関係の補正を通す)。"""
        return InterventionPolicy.model_validate({**self.model_dump(), **changes})

    @property
    def content_support(self) -> bool:
        """⑥の内容支援が有効か。"""
        return self.content_scope != ContentScope.none

    @property
    def content_actions(self) -> list[Action]:
        """範囲の上で使える内容支援のアクション(実際に許可するかは自発性と状況で決まる)。"""
        return list(CONTENT_ACTIONS_BY_SCOPE[self.content_scope])

    def notices(self, purpose: str) -> list[str]:
        """画面に出す注意。"""
        return [JOB_CONTENT_NOTICE] if purpose == "job" and self.content_support else []

    def overrides_from(self, base: InterventionPolicy) -> dict[str, str]:
        """base(プリセット)から変えた項目だけ(画面の再読み込みで、変更した項目を復元するため)。"""
        mine, theirs = self.model_dump(mode="json"), base.model_dump(mode="json")
        return {k: v for k, v in mine.items() if v != theirs[k]}

    def summary(self) -> str:
        """まとめログ用の1行表示。"""
        return " ".join(f"{ITEM_LABELS[k].split()[0]}{VALUE_LABELS[getattr(self, k).value]}" for k in ITEM_LABELS)


def preset_policy(name: str) -> InterventionPolicy:
    return InterventionPolicy.model_validate(POLICY_PRESETS[name])


def purpose_preset(purpose: str) -> InterventionPolicy:
    """用途の既定のプリセット(不明な用途は就活GD用)。"""
    return preset_policy(PURPOSE_PRESET.get(purpose, PURPOSE_PRESET["job"]))


def from_knobs(knobs: dict[str, Any] | None, base: InterventionPolicy, purpose: str = "job") -> InterventionPolicy:
    """画面の「詳細」の値(0=控えめ / 1=標準 / 2=積極的)を base に重ねる。不明な項目・値は無視する(従来どおり)。
    「AIからの提案度合い」(suggest)は⑥に変換する: 0 → 内容支援なし / 1 → base のまま / 2 → ヒントまで・積極的
    (旧来の意味「足りない観点を示してよい。解決策・結論・意見は言わない」に合わせる)。
    就活では現在の画面と同じく 2 を受け付けない(JOB_MAX_LEVEL。中立でない旨の表示は第4段階の画面で行う)。"""
    knobs = knobs or {}
    changes: dict[str, Any] = {}
    for item, key in KNOB_FOR_ITEM.items():
        v = knobs.get(key)
        if v in INDEX_LEVEL:
            changes[item] = INDEX_LEVEL[v]
    v = knobs.get("suggest")
    if v in (0, 1, 2) and purpose == "job":
        v = min(v, JOB_MAX_LEVEL["suggest"])
    if v == 0:
        changes["content_scope"] = ContentScope.none
    elif v == 2:
        changes.update(content_scope=ContentScope.hint, content_initiative=ContentInitiative.proactive)
    return base.updated(**changes) if changes else base


def resolve_policy(
    purpose: str,
    preset: str | None = None,
    overrides: dict[str, Any] | None = None,
    knobs: dict[str, Any] | None = None,
) -> InterventionPolicy:
    """セッションのポリシー。優先順位: プリセット(preset、指定がなければ用途の既定)< 個別の変更(overrides)
    < 旧画面の「詳細」(knobs)。overrides の不正な値はエラーにし、knobs の不正な値は従来どおり無視する。"""
    base = preset_policy(preset) if preset else purpose_preset(purpose)
    if overrides:
        base = base.updated(**overrides)
    return from_knobs(knobs, base, purpose)


def session_policy(purpose: str, knobs: dict[str, Any] | None = None) -> InterventionPolicy:
    """用途の既定のプリセットに、画面の「詳細」を重ねたポリシー。"""
    return resolve_policy(purpose, knobs=knobs)


def levels(policy: InterventionPolicy) -> dict[str, str]:
    """①〜⑤の段階(項目 → low / standard / high)。プロンプトの差し替えに使う。"""
    return {item: getattr(policy, item).value for item in LEVEL_ITEMS}


def level_settings(policy: InterventionPolicy) -> dict[str, Any]:
    """①〜⑤の段階で変わる判断条件(Settings の上書き)。標準の項目は何も上書きしない。"""
    out: dict[str, Any] = {}
    for item, level in levels(policy).items():
        out.update(LEVEL_SETTINGS[item].get(level, {}))
    return out


def legacy_knobs(policy: InterventionPolicy) -> dict[str, int]:
    """ポリシーの①〜④を STYLE_KNOBS の形(画面と同じ項目順)に戻す(第4段階で画面の初期値に使う)。"""
    out: dict[str, int] = {}
    for key in STYLE_KNOBS:
        item = next((i for i, k in KNOB_FOR_ITEM.items() if k == key), None)
        if item:
            out[key] = LEVEL_INDEX[getattr(policy, item)]
    return out


# ---------- 画面と API ----------
JOB_CONTENT_NOTICE = "内容支援が有効なため、厳格な就活GD向けの中立設定ではありません(選考の場では「なし」を推奨)"
# 旧進行スタイル(A / B / custom)。数値の上書きは廃止し、B は「⑤停滞への介入: 積極的」として扱う
LEGACY_STYLE = {"A": {}, "B": {"stall": "high"}, "custom": {}}
OPTION_LABELS: dict[str, list[tuple[str, str]]] = {
    **{k: [("low", "控えめ"), ("standard", "標準"), ("high", "積極的")] for k in LEVEL_ITEMS},
    "content_scope": [("none", "なし"), ("hint", "ヒントまで"), ("concrete", "具体案まで")],
    "content_initiative": [("none", "なし"), ("stalled", "停滞時のみ"), ("proactive", "積極的")],
}
# 選択肢の説明(画面に出す)。実際の動作(FacilitatorAgent._content_actions と config の値)と合わせること
OPTION_HELP: dict[str, dict[str, str]] = {
    "content_scope": {
        "none": "進行の支援だけ。新しい案や観点は出しません",
        "hint": "考える視点のヒントを、問いの形で1つずつ示せます。答えや解決策は言いません",
        "concrete": "ヒントに加えて、具体的なアイデアの例も示せます。決めるのは参加者です",
    },
    "content_initiative": {
        "none": "参加者から進行役に頼まれたとき(「AIさん、ヒントありますか?」など)だけ",
        "stalled": "頼まれたときに加えて、停滞が続くと 質問 → ヒント → 具体案(具体案まで のとき)の順に段階的に",
        "proactive": "停滞時に加えて、必要と判断すればヒントを自発的に。具体案は頼まれたときか停滞時の段階を踏んだときだけ",
    },
}


def ui_spec() -> dict[str, Any]:
    """画面の設定項目・プリセット・説明。画面はこれを表示するだけにする(値を画面側に持たない)。"""
    return {
        "items": [
            {"key": k, "label": ITEM_LABELS[k], "options": [{"v": v, "label": lb} for v, lb in OPTION_LABELS[k]],
             "help": OPTION_HELP.get(k, {})}
            for k in ITEM_LABELS
        ],
        "presets": {name: {"label": POLICY_PRESET_LABELS[name], "values": dict(values)} for name, values in POLICY_PRESETS.items()},
        "purpose_preset": dict(PURPOSE_PRESET),
        "notices": {"job_content": JOB_CONTENT_NOTICE},
        "spontaneous_limits": {"gap_sec": Settings.content_cooldown_sec, "max_per_session": Settings.content_max_spontaneous},
    }


def request_policy(
    purpose: str,
    policy_preset: str | None = None,
    overrides: dict[str, Any] | None = None,
    legacy_style: str | None = None,
    knobs: dict[str, Any] | None = None,
) -> tuple[InterventionPolicy, list[str]]:
    """画面・API の入力からセッションのポリシーを決める。(ポリシー, 注意) を返す。不正な値は ValueError。
    優先順位: プリセット(policy_preset、省略時は用途の既定)< 旧進行スタイル B < 変更した項目(overrides)< 旧「詳細」(knobs)。"""
    if policy_preset and policy_preset not in POLICY_PRESETS:
        raise ValueError(f"不明なプリセットです: {policy_preset}")
    unknown = set(overrides or {}) - set(ITEM_LABELS)
    if unknown:
        raise ValueError(f"不明な設定項目です: {', '.join(sorted(unknown))}")
    warnings: list[str] = []
    changes = dict(LEGACY_STYLE.get(legacy_style or "A", {}))
    if legacy_style == "B":
        warnings.append("旧進行スタイル B は「⑤停滞への介入: 積極的」として扱います(数値の変更は廃止)")
    elif legacy_style == "custom":
        warnings.append("旧進行スタイル custom の数値の変更は廃止しました(介入ポリシーの項目で調整してください)")
    changes.update(overrides or {})
    try:
        policy = resolve_policy(purpose, preset=policy_preset, overrides=changes, knobs=knobs)
    except ValidationError as e:
        raise ValueError(f"設定の値が正しくありません: {e.errors()[0].get('loc')} {e.errors()[0].get('msg')}") from e
    return policy, warnings + policy.notices(purpose)
