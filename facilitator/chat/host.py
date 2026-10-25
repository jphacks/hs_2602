"""雑談モードの AI 司会(最初の画面で「雑談」を選んだとき = 議論タイプが雑談型のときに使う)。

役割: 会話の参加者ではなく、初対面などで少し気まずい人たちの間に入り、話題を出したり会話を回したりして、
人間どうしの交流をつくる司会者。最初は AI が積極的に会話を生み出し、参加者どうしの会話が始まったら自然に引く。
GD・GW のように議論を進めたり結論を出したりはしない。

構成: GD・GW の進行役(FacilitatorAgent)を継承する。engine は進行役の窓口(tick / should_tick / commit /
生成しながら話す処理の検証 / 依頼の検知 / 記録)をそのまま使うので、engine と GD・GW の進行役のコードを変えずに
差し替えられる。雑談専用の判断(会話の状態の計算・介入のタイミング・プロンプト・行動)は、このクラスの中だけで
上書きしていく(GD・GW の進行役、プロンプト、介入ポリシーには手を入れない)。

第1段階(現在): 分離して接続しただけで、振る舞いは雑談型の議論タイプを使う GD・GW の進行役と同じ。
"""

from __future__ import annotations

from facilitator.agent.agent import FacilitatorAgent
from facilitator.discussion_types import CASUAL


class ChatHostAgent(FacilitatorAgent):
    """雑談の司会(人間どうしの交流をつくる。会話の参加者にはならない)。"""


def agent_class_for(discussion_type: str | None) -> type[FacilitatorAgent]:
    """議論タイプに応じた司会のクラス。雑談型だけ雑談の司会、それ以外は GD・GW の進行役(従来どおり)。"""
    return ChatHostAgent if discussion_type == CASUAL.key else FacilitatorAgent
