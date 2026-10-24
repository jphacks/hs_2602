# AI Facilitator Agent

就活グループディスカッションで、AI が進行役(ファシリテーター)を自律的に務めるプロトタイプ(JPHACKS 2026)。
議論を聞き取り、「介入するか/しないか」を自分で判断して、必要なときだけ質問・論点整理・時間管理などを音声で行います。

## セットアップ
```bash
cd ai-facilitator
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # ANTHROPIC_API_KEY / ASSEMBLYAI_API_KEY を記入
# 話者照合モデル(約28MB)
mkdir -p models && curl -L -o models/campplus_zh_en_advanced.onnx \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx
```
マイクを使うスクリプトは、macOS の「ターミナル」アプリで実行してください(初回はマイクの許可を求められます)。

## mock-test ブランチ(トークンを消費しないテスト用)
- AI の判断は既定で**モック**(ルールベース)になり、Claude API を呼びません。画面右上に「モックAI(トークン消費なし)」と表示されます。
- 画面の初期モードは「テキスト入力のみ」です。音声モードを選ぶと文字起こし(AssemblyAI)の従量課金が発生します。
- モックの判断は単純です(12秒以上の沈黙で質問、時間告知、終了のみ)。UI や進行の流れの確認に使い、判断の質は main ブランチの Claude で確認してください。
- 一時的に Claude を使う場合は `LLM_PROVIDER=anthropic` を付けて起動します。
## AI 参加者
設定画面で人間と AI 参加者の人数を選ぶと、AI 参加者にはテーマに対する異なる立場が自動で割り当てられ、議論に加わります。人数比に応じて「練習相手モード」(積極的に発言)か「控えめモード」(名指しや長い沈黙のときだけ発言)になります。AI 参加者の発言でも LLM を使うため、Claude を使う場合はその分のトークンを消費します。

## チャット経由モード(API を使わずに判断の流れを確かめる)
`--llm handoff`(または `LLM_PROVIDER=handoff`)で実行すると、AI に問い合わせる場面でプロンプトを `exchange/request_NNN.md` に書き出し、`exchange/response_NNN.json` に返事が書かれるまで待ちます。Claude Code のチャットで「シナリオ06をチャット経由で流して」のように頼むと、チャットの AI がリクエストを読んで判断を書き込みます。
```bash
.venv/bin/python scripts/evaluate.py --llm handoff -s 06 -v
.venv/bin/python scripts/run_scenario.py scenarios/02_stalled.yaml --llm handoff
```
- ルール層(発話中・クールダウン・短い沈黙など)で止まった場面ではリクエストは出ません。AI が判断する場面だけ、何が渡され何を返したかをファイルで確認できます。
- 返事を待つ間は処理が止まるため、仮想時計で進むシナリオ再生向けです(実時間のアプリでは使いません)。

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
- 発話本文は既定でログに残しません(`log_transcript_text`)。
