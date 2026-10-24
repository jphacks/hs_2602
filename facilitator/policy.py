"""介入ポリシー: 進行役がどの程度介入するかの6項目の設定。

就活GD用・一般GW用のプリセット(config.POLICY_PRESETS)を初期値にし、利用者が各項目を変更できる。
GD用・GW用で別々の進行役を作らず、共通の進行役がこの設定に応じて動く。

第1段階(現在)は設定の保持・検証・互換変換だけを行い、介入の判断・頻度・発話は変えない。
- ①〜④ の段階の効果は、画面の「詳細」の既存の4項目(config.STYLE_KNOBS の phase / redirect / organize / turns)の効果をそのまま使う
- ⑤ 停滞と ⑥ 内容支援は値を持つだけで、まだ効果はない(第2・第3段階で反映する)
- 旧「AIからの提案度合い」(STYLE_KNOBS["suggest"])は⑥と1対1に対応しないので変換せず、legacy_suggest として旧挙動を保つ
- セッションの①〜④は画面の「詳細」の値(指定なしは標準=従来どおり)、⑤・⑥は用途のプリセットの値を使う(session_policy)。
  画面の5項目を6項目に対応させる第4段階で、①〜④にもプリセットの値を使う
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from config import CUSTOM_KEYS, POLICY_PRESETS, PRESETS, PURPOSE_PRESET, STYLE_KNOBS, knob_overrides


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
# 画面の「詳細」の既存項目(STYLE_KNOBS)と対応する項目。suggest は⑥と1対1に対応しないので含めない
KNOB_FOR_ITEM = {"progress": "phase", "drift": "redirect", "organize": "organize", "turns": "turns"}
LEVEL_INDEX = {Level.low: 0, Level.standard: 1, Level.high: 2}  # STYLE_KNOBS の段階番号
INDEX_LEVEL = {v: k for k, v in LEVEL_INDEX.items()}


class InterventionPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    progress: Level = Level.standard
    drift: Level = Level.standard
    organize: Level = Level.standard
    turns: Level = Level.standard
    stall: Level = Level.standard
    content_scope: ContentScope = ContentScope.none
    content_initiative: ContentInitiative = ContentInitiative.none
    # 旧「AIからの提案度合い」(0〜2)。⑥に置き換える第3段階までの互換用。None なら指定なし
    legacy_suggest: int | None = Field(default=None, ge=0, le=2)

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
        """⑥の内容支援が有効か(旧 suggest は含めない)。"""
        return self.content_scope != ContentScope.none

    def notices(self, purpose: str) -> list[str]:
        """画面に出す注意(第4段階の UI で使う)。"""
        out = []
        if purpose == "job" and self.content_support:
            out.append("内容支援が有効なため、厳格な就活GD向けの中立設定ではありません")
        return out

    def summary(self) -> str:
        """まとめログ用の1行表示。"""
        parts = [f"{ITEM_LABELS[k].split()[0]}{VALUE_LABELS[getattr(self, k).value]}" for k in ITEM_LABELS]
        if self.legacy_suggest is not None:
            parts.append(f"旧提案度合い{self.legacy_suggest}")
        return " ".join(parts)


def preset_policy(name: str) -> InterventionPolicy:
    return InterventionPolicy.model_validate(POLICY_PRESETS[name])


def purpose_preset(purpose: str) -> InterventionPolicy:
    """用途の既定のプリセット(不明な用途は就活GD用)。"""
    return preset_policy(PURPOSE_PRESET.get(purpose, PURPOSE_PRESET["job"]))


def from_knobs(knobs: dict[str, Any] | None, base: InterventionPolicy) -> InterventionPolicy:
    """画面の「詳細」の値(0=控えめ / 1=標準 / 2=積極的)を base に重ねる。不明な項目・値は無視する(従来どおり)。"""
    changes: dict[str, Any] = {}
    for item, key in KNOB_FOR_ITEM.items():
        v = (knobs or {}).get(key)
        if v in INDEX_LEVEL:
            changes[item] = INDEX_LEVEL[v]
    v = (knobs or {}).get("suggest")
    if v in (0, 1, 2):
        changes["legacy_suggest"] = v
    return base.updated(**changes) if changes else base


def session_policy(purpose: str, knobs: dict[str, Any] | None = None) -> InterventionPolicy:
    """セッションのポリシー(第1段階)。①〜④は画面の「詳細」の値で、指定がなければ標準(従来どおり)。
    ⑤・⑥は画面からまだ変えられないので、用途のプリセットの値を使う(効果は第2・第3段階から)。"""
    base = purpose_preset(purpose).updated(**{item: Level.standard for item in KNOB_FOR_ITEM})
    return from_knobs(knobs, base)


def legacy_knobs(policy: InterventionPolicy) -> dict[str, int]:
    """ポリシーを STYLE_KNOBS の形(画面と同じ項目順)に戻す。①〜④と旧 suggest だけで、⑤・⑥は対応する項目がない。"""
    out: dict[str, int] = {}
    for key in STYLE_KNOBS:
        if key == "suggest":
            if policy.legacy_suggest is not None:
                out[key] = policy.legacy_suggest
            continue
        item = next(i for i, k in KNOB_FOR_ITEM.items() if k == key)
        out[key] = LEVEL_INDEX[getattr(policy, item)]
    return out


def policy_effects(policy: InterventionPolicy, purpose: str) -> tuple[dict[str, Any], list[str]]:
    """ポリシーから、数値設定の上書きと進行役への追加の指示を作る。
    第1段階は既存の「詳細」の効果(config.knob_overrides、就活の提案度合いの上限を含む)をそのまま使う。"""
    return knob_overrides(legacy_knobs(policy), purpose)


def session_overrides(
    preset: str, custom: dict[str, float] | None, policy: InterventionPolicy, purpose: str
) -> tuple[dict[str, Any], list[str]]:
    """セッションの数値設定の上書きと、進行役への追加の指示。
    優先順位は従来どおり、旧進行スタイル(A/B)< 旧カスタム < 介入ポリシー。旧進行スタイルは第4段階で廃止する。"""
    overrides: dict[str, Any] = dict(PRESETS.get(preset, {}))
    if preset == "custom":
        overrides.update({k: float(v) for k, v in (custom or {}).items() if k in CUSTOM_KEYS})
    eff, notes = policy_effects(policy, purpose)
    overrides.update(eff)
    return overrides, notes
