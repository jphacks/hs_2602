# SHIKAI — AI Facilitator Agent

就活のグループディスカッション・グループワーク・雑談で、AI が進行役(ファシリテーター)を自律的に務めるプロトタイプ(JPHACKS 2026)。
議論を聞き取り、「介入するか/しないか」を自分で判断して、必要なときだけ質問・論点整理・時間管理などを音声で行います。

## セットアップ
```bash
cd hs_2602
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # ANTHROPIC_API_KEY / ASSEMBLYAI_API_KEY を記入
# 話者照合モデル(約28MB)
mkdir -p models && curl -L -o models/campplus_zh_en_advanced.onnx \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx
```
マイクを使うスクリプトは、macOS の「ターミナル」アプリで実行してください(初回はマイクの許可を求められます)。

## 画面の使い方
1. **モード選択**: 「人間だけ」「AI参加者あり」「雑談」を選びます。
2. **設定**: テーマ(「自動生成」で AI がテーマを作ります)、制限時間、参加者の人数と名前、AI参加者の人数、スタイル(就活/就活以外)を入力します。「詳細」をオンにすると、進行の調整(フェーズ進行・脱線修正・話題整理・話を回す度合い・AIからの提案度合いを 控えめ/標準/積極的)と、その他の設定(入力方法・AIのレベル・声の登録・AIの声・話者情報)を変えられます。
3. **進行**: 「開始」で GD が始まります。参加者カードの幅は発言量に応じて変わります。

## AI 参加者
設定画面で人間と AI 参加者の人数を選ぶと、AI 参加者にはテーマに対する異なる立場が自動で割り当てられ、議論に加わります。人数比に応じて「練習相手モード」(積極的に発言)か「控えめモード」(名指しや長い沈黙のときだけ発言)になります。AI 参加者のレベル(1: 高校生程度・発言の間隔4〜6秒 / 2: 大学生程度・3〜5秒 / 3: 大学生程度・2〜4秒)を選べます(既定はレベル2。間隔は練習相手モードで使われます)。誰が話すかは会話の文脈で選ばれます。AI 参加者の発言でも LLM を使うため、Claude を使う場合はその分のトークンを消費します。

## チャット経由モード(API を使わずに判断の流れを確かめる)
`--llm handoff`(または `LLM_PROVIDER=handoff`)で実行すると、AI に問い合わせる場面でプロンプトを `exchange/request_NNN.md` に書き出し、`exchange/response_NNN.json` に返事が書かれるまで待ちます。Claude Code のチャットで「シナリオ06をチャット経由で流して」のように頼むと、チャットの AI がリクエストを読んで判断を書き込みます。
```bash
.venv/bin/python scripts/evaluate.py --llm handoff -s 06 -v
.venv/bin/python scripts/run_scenario.py scenarios/02_stalled.yaml --llm handoff
```
- ルール層(発話中・クールダウン・短い沈黙など)で止まった場面ではリクエストは出ません。AI が判断する場面だけ、何が渡され何を返したかをファイルで確認できます。
- 返事を待つ間は処理が止まるため、仮想時計で進むシナリオ再生向けです(実時間のアプリでは使いません)。

### AI 参加者入りのシミュレーション
```bash
.venv/bin/python scripts/simulate.py scenarios/bots/b1_practice.yaml --llm handoff   # 人間1人 + AI 2人(練習相手モード)
.venv/bin/python scripts/simulate.py scenarios/bots/b2_reserved.yaml --llm handoff   # 人間3人 + AI 1人(控えめモード)
```
人間の発言は台本どおりに流れ、AI 参加者の「立場の生成」「発言」と進行役の判断が、アプリと同じロジック(`participants.pick_bot`)で動きます。リクエストのタイトルに種類(進行役の判断 / AI参加者の立場の生成 / AI参加者(名前)の発言)が出ます。台本の人間は AI の発言に反応しないため、流れが噛み合わないことがあります。

## チャット GD(アプリを使わずにチャットだけで模擬 GD)
Claude Code のチャットで「チャット GD を始めて。テーマは〇〇、人間はたろう、AI 2人、5分」のように頼むと、あなたの発言(「たろう: …」)に対して、進行役と AI 参加者が応答する様子がテキストで返ってきます。「(沈黙15秒)」で沈黙を続け、「終了」で制限時間まで進めて結果のまとめを表示します。終了時に、チャットで表示したのと同じテキスト版が `logs/report-日時-chat.txt` に保存されます(`exchange/transcript.txt` は次のセッションで上書きされます)。中身は `scripts/chat_gd.py`(アプリと同じロジック、仮想時計)。

## まとめログ
評価・シナリオ再生・シミュレーター・アプリのどれも、実行ごとに `logs/report-日時-名前.md` を1つ書き出します(終了時にパスを表示)。
- タイムライン: 発言ごとの **開始 – 終了** 時刻・話者・内容、進行役の判断(理由・発話・ルール層の見送り)、AI 参加者が話した理由
- AI に渡した入力と返事(折りたたみ)、評価のチェックポイント結果、発言量のまとめ
- アプリでは、AI の発言の終了時刻は実際に読み上げが終わった時刻です。

## Step 1: エージェント単体の検証(音声なし)
```bash
.venv/bin/python -m pytest -q tests                                 # ルール層のテスト
.venv/bin/python scripts/evaluate.py --llm mock                     # API なしで配線確認
.venv/bin/python scripts/evaluate.py -v                             # 10シナリオを Claude で評価
.venv/bin/python scripts/evaluate.py -s 06 -s 10 --repeat 3         # 特定シナリオを繰り返して判断のブレを見る
.venv/bin/python scripts/run_scenario.py scenarios/02_stalled.yaml  # 実運用と同じトリガーで時系列再生
```
- 評価の詳細は `results/`、判断ログ(JSONL)は `logs/` に出力されます(どちらも git 管理外)。
- シナリオの書き方は `scenarios/README.md` を参照してください。

| # | シナリオ | 確認すること |
|---|---|---|
| 01 | 順調に進む議論 | 開始説明。順調なときは介入しない。発話中は話さない |
| 02 | 沈黙・停滞 | 停滞したら質問または整理。介入直後はクールダウン |
| 03 | 脱線 | 軽い雑談は見守る。続いたら軌道修正 |
| 04 | 2案の対立 | 平行線になったら比較基準を問う |
| 05 | 残り時間わずか | 時間告知(強制)、結論が出たら見守る、時間切れで終了 |
| 06 | 一部の人しか話さない | 盛り上がり中は割り込まない。区切りで個人指名 |
| 07 | 1人が話し続ける | 発話中は遮らない。区切りで他の人に振る |
| 08 | 1人が一度も発言していない | 個人指名 |
| 09 | 発言が遮られる | 遮られた人に話を戻す |
| 10 | 話者の確信度が低い | 個人指名せず、全体への呼びかけにフォールバックする |

## 技術選定メモ(2026-10 調査)
| 機能 | 採用 | 根拠・未確認事項 |
|---|---|---|
| LLM | Claude Opus 5.5(effort low、構造化出力) | `config.py` / `.env` で `LLM_MODEL` を変更可能 |
| STT + 話者分離 | AssemblyAI Universal-3.6 Pro Realtime(予定) | 日本語ストリーミングに対応するのはこのモデル。`speaker_labels`・`max_speakers`・SpeakerRevision あり。日本語での話者分離精度は未確認 |
| 比較候補 | Soniox stt-rt | トークンごとに話者が付く。$0.12/h。日本語精度は自社発表のみ |
| 不採用 | Google Chirp 3 | ストリーミングでは話者分離不可(Batch/Recognize のみ) |
| 不採用 | Azure ConversationTranscriber | ja-JP の対応は未確認。リアルタイムで不安定という利用者報告あり |
| 不採用 | diart + pyannote | 自前運用になり、2日でのセットアップには重い |
| TTS | macOS `say -v Kyoko` | キー不要。品質が不足する場合は後で検討 |

## 推奨マイクセッティング
Step 2 の実測後に記入します。

## プライバシー
- 録音は保存しません(検証用の保存オプションはデフォルト OFF で、git 管理外)。
- 話者ラベル・名前の対応はメモリ上だけで保持し、セッション終了時に破棄します。
