<p align="left">
        中文</a>&nbsp ｜ &nbsp<a href="README.md">English</a>
</p>
<br><br>

<p align="center">
    <img src="assets/logo.png" width="400"/>
<p>
<br>

<p align="center">
        Qwen-Audio 🤖 | 🤗 ｜ Qwen-Audio-Chat🤖 | 🤗 
<br>
&nbsp&nbsp<a href="https://modelscope.cn/studios/qwen/Qwen-VL-Chat-Demo/summary">Demo</a>&nbsp ｜ &nbsp<a href="https://arxiv.org/abs/2308.12966">Paper</a>&nbsp&nbsp
</p>
<br><br>

**Qwen-Audio** 是阿里云研发的大规模音频语言模型（Large Audio Language Model）。Qwen-Audio 可以以多种音频 (包括说话人语音、自然音、音乐、歌声）和文本作为输入，并以文本作为输出。Qwen-Audio 系列模型的特点包括：

- **音频基石模型**：Qwen-Audio是一个性能卓越的通用的音频理解模型，支持各种任务、语言和音频类型。在Qwen-Audio的基础上，我们通过指令微调开发了Qwen-Audio-Chat，支持多轮、多语言、多语言对话。Qwen-Audio和Qwen-Audio-Chat模型均即将开源。
- **兼容多种复杂音频的多任务学习框架**：为了避免由于数据收集来源不同以及任务类型不同，带来的音频到文本的一对多的干扰问题，我们提出了一种多任务训练框架，实现相似任务的知识共享，并尽可能减少不同任务之间的干扰。通过提出的框架，Qwen-Audio可以容纳训练超过30多种不同的音频任务；
- **出色的性能**：Qwen-Audio在不需要任何任务特定的微调的情况下，在各种基准任务上取得了领先的结果。具体得，Qwen-Audio在Aishell1、cochlscene、ClothoAQA和VocalSound的测试集上都达到了SOTA；
- **支持多轮音频和文本对话，支持各种语音场景**：Qwen-Audio-Chat支持声音理解和推理、音乐欣赏、多音频分析、多轮音频-文本交错对话以及外部语音工具的使用(如语音编辑)。

<br>
<p align="center">
    <img src="assets/framework.png" width="800"/>
<p>
<br>


我们即将开源 Qwen-Audio 系列的两个模型：

- Qwen-Audio: Qwen-Audio 以 Qwen-7B 的预训练模型作为语言模型的初始化，并以 [Whisper-large-v2](https://github.com/openai/whisper) 作为音频编码器的初始化。
- Qwen-Audio-Chat: 在 Qwen-Audio 的基础上，我们使用对齐机制打造了基于大语言模型的语音AI助手Qwen-Audio-Chat，它支持更灵活的交互方式，包括多音频、多轮问答、创作等能力。
  <br>


## 评测
我们在标准的12个学术数据集上评测了模型的能力，细节如下:

<p align="center">
    <img src="assets/evaluation.png" width="800"/>
<p>

评测结果如下：
<p align="center">
    <img src="assets/radar.png" width="800"/>
<p>


## 使用协议

研究人员与开发者可使用Qwen-Audio和Qwen-Audio-Chat或进行二次开发。我们同样允许商业使用，具体细节请查看[LICENSE](LICENSE)。如需商用，请填写[问卷](https://dashscope.console.aliyun.com/openModelApply/qianwen)申请。
<br>

## 团队招聘

我们是通义千问语音多模态团队，致力于以通义千问为核心，拓展音频多模态理解和生成能力，实现自由灵活的音频交互。目前团队蓬勃发展中，如有意向实习或全职加入我们，请发送简历至 .

## 联系我们

如果你想给我们的研发团队和产品团队留言，请通过邮件（qianwen_opensource@alibabacloud.com）联系我们。

