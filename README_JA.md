<p align="left">
        <a href="README_CN.md">中文</a>&nbsp ｜ &nbsp<a href="README.md">English</a>&nbsp&nbsp ｜ &nbsp日本語&nbsp&nbsp
</p>
<br><br>

<p align="center">
    <img src="assets/logo.png" width="400"/>
<p>
<br>

<p align="center">
        Qwen-Audio 🤖 | 🤗 ｜ Qwen-Audio-Chat🤖 | 🤗
<br>
&nbsp&nbsp<a href="https://qwen-audio.github.io/Qwen-Audio/">Demo</a>&nbsp ｜ &nbsp<a href="http://arxiv.org/abs/2311.07919">Paper</a>&nbsp&nbsp
</p>
<br><br>

<p align="left">
        日本語ドキュメントメンテナー: <a href="https://github.com/eltociear">Ikko Eltociear Ashimine</a>
</p>

---

**Qwen-Audio**（Qwen 大規模音声言語モデル）は、アリババクラウドが提唱する大型モデルシリーズ Qwen（Qianwen の略）のマルチモーダル版です。Qwen-Audio は、多様な音声（人間の音声、自然音、音楽、歌）とテキストを入力として受け付け、テキストを出力します。Qwen-Audio の貢献は以下の通りです:

- **基本的なオーディオモデル**: Qwen-Audio は、様々なタスク、言語、オーディオタイプをサポートする基本的なマルチタスクオーディオ言語モデルであり、普遍的なオーディオ理解モデルとして機能します。Qwen-Audio をベースに、インストラクションのファインチューニングを行いながら Qwen-Audio-Chat を発展させ、マルチターン対話を可能にし、多様なオーディオ指向のシナリオをサポートします。
- **あらゆる種類の音声に対応するマルチタスク学習フレームワーク**: 音声言語の事前学習をスケールアップするために、知識の共有を可能にし、一対多の干渉を回避するマルチタスク学習フレームワークを提案することで、異なるデータセットに関連するテキストラベルのばらつきという課題に対処します。我々のモデルには30以上のタスクが組み込まれており、広範な実験により、このモデルが強力な性能を達成していることが示されています。
- **強力なパフォーマンス**: 実験の結果、Qwen-Audio は、タスク固有のファインチューニングを必要とすることなく、多様なベンチマークタスクで優れた性能を達成し、同種の製品を凌駕しています。具体的には、Aishell1、cochlscene、ClothoAQA、VocalSound のテストセットにおいて Qwen-Audio は最先端の結果を達成しています。
- **音声とテキスト入力からの柔軟なマルチ実行チャット**: Qwen-Audio は、複数の音声分析、音の理解と推論、音楽鑑賞、音声編集ツールの使用をサポートします。


<br>
<p align="center">
    <img src="assets/framework.png" width="800"/>
<p>
<br>

Qwen-Audio シリーズは、近日中に2モデルをリリースする予定です:

- Qwen-Audio: 学習済みのマルチタスク音声理解モデルは、LLM の初期化に Qwen-7B を、音声エンコーダの初期化に [Whisper-large-v2](https://github.com/openai/whisper) を使用しています。
- Qwen-Audio-Chat: マルチモーダル LLM ベースの AI アシスタントで、アライメント技術によって学習されます。Qwen-Audio-Chat は、複数の音声入力、複数ラウンドの質問応答、クリエイティブな機能など、より柔軟なインタラクションをサポートします。
  <br>


## 評価

Qwen-Audio の能力を、標準的なベンチマークで以下のように評価しました:

<p align="center">
    <img src="assets/evaluation.png" width="800"/>
<p>

評価結果は以下の通りになります:
<p align="center">
    <img src="assets/radar.png" width="800"/>
<p>

## 採用情報

正社員またはインターンとして入社を希望される方は、qwen_audio@list.alibaba-inc.com までご連絡ください。

## ライセンス契約

研究者や開発者は、Qwen-Audio と Qwen-Audio-Chat のコードとモデルウェイトを自由に使用することができます。また、商用利用も可能です。詳細は [LICENSE](LICENSE.txt) でライセンスを確認してください。
<br>


## お問い合わせ

研究チームまたは製品チームへのメッセージは、qianwen_opensource@alibabacloud.com までお気軽にお送りください。

