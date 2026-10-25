# 模擬GDシナリオ

- `t` は GD 開始からの経過秒、`d` は発話の長さ(秒)、`s` は話者、`c` は話者識別の確信度。
- 発話は終了時刻(t+d)に STT の確定結果として届いたものとして扱う。発話中の時刻は VAD が発話中と判定する。
- `checkpoints` の各時刻でエージェントに判断させ、`expect`(許容アクション)・`forbid`(禁止アクション)・`target_any`(invite_person 時の許容対象)と比較する。
- `start_elapsed` が 0 より大きい場合、GD はすでに開始済み(AI の開始説明は済んでいる)として扱う。
- `require_llm: true` は LLM 自身の判断を評価する時点(ルール層が先に見送ったら失敗)。`expect_gate`(例: `blocked:short_pause`)はルール層の判定を確かめる時点で、LLM の見守りは評価しない。「見守り」を期待する時点はどちらかを付ける
- `policy`(省略可)で介入ポリシーを指定する。例: `policy: {preset: general_gw, stall: low}`。省略時は用途(`purpose`、省略時は job)の既定のプリセット。`scripts/evaluate.py --preset ... --set stall=high` で上書きできる。
- チェックポイントの `when`(例: `when: {stall: high}` / `when: {drift: [low, standard]}`)は、ポリシーがその段階のときだけ評価する(合わないチェックポイントでは判断させない)。18〜22 は段階ごとの違いを見るシナリオで、期待は仮説(第5段階で Claude で確かめる)。
