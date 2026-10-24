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

from pydantic import BaseModel, ConfigDict, model_validator

from config import CUSTOM_KEYS, JOB_MAX_LEVEL, LEVEL_SETTINGS, POLICY_PRESETS, PRESETS, PURPOSE_PRESET, STYLE_KNOBS
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
        """画面に出す注意(第4段階の UI で使う)。"""
        out = []
        if purpose == "job" and self.content_support:
            out.append("内容支援が有効なため、厳格な就活GD向けの中立設定ではありません")
        return out

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


def session_overrides(preset: str, custom: dict[str, float] | None) -> dict[str, Any]:
    """旧進行スタイル(A/B)と旧カスタムによる数値設定の上書き(第4段階で廃止する)。①〜⑥の効果は含まない。"""
    overrides: dict[str, Any] = dict(PRESETS.get(preset, {}))
    if preset == "custom":
        overrides.update({k: float(v) for k, v in (custom or {}).items() if k in CUSTOM_KEYS})
    return overrides
