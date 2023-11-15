<p align="left">
        <a href="README_CN.md">中文</a>&nbsp ｜ &nbspEnglish&nbsp&nbsp ｜ &nbsp<a href="README_JA.md">日本語</a>&nbsp&nbsp
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

---

**Qwen-Audio** (Qwen Large Audio Language Model) is the multimodal version of the large model series, Qwen (abbr. Tongyi Qianwen), proposed by Alibaba Cloud. Qwen-Audio accepts diverse audio (human speech, natural sound, music and song) and text as inputs, outputs text. The contribution of Qwen-Audio include:

- **Fundamental audio models**: Qwen-Audio is a fundamental multi-task audio-language model that supports various tasks, languages, and audio types, serving as a universal audio understanding model. Building upon Qwen-Audio, we develop Qwen-Audio-Chat through instruction fine-tuning, enabling multi-turn dialogues and supporting diverse audio-oriented scenarios.
- **Multi-task learning framework for all types of audios**: To scale up audio-language pre-training, we address the challenge of variation in textual labels associated with different datasets by proposing a multi-task training framework, enabling knowledge sharing and avoiding one-to-many interference. Our model incorporates more than 30 tasks and extensive experiments show the model achieves strong performance.
- **Strong Performance**: Experimental results show that Qwen-Audio achieves impressive performance across diverse benchmark tasks without requiring any task-specific fine-tuning, surpassing its counterparts. Specifically, Qwen-Audio achieves state-of-the-art results on the test set of Aishell1, cochlscene, ClothoAQA, and VocalSound.
- **Flexible multi-run chat from audio and text input**: Qwen-Audio supports multiple-audio analysis, sound understading and reasoning, music appreciation, and tool usage for speech editing.


<br>
<p align="center">
    <img src="assets/framework.png" width="800"/>
<p>
<br>

We are going to release two models of the Qwen-Audio series soon:

- Qwen-Audio: The pre-trained multi-task audio understanding model uses Qwen-7B as the initialization of the LLM, and [Whisper-large-v2](https://github.com/openai/whisper) as the initialization of the audio encoder.
- Qwen-Audio-Chat: A multimodal LLM-based AI assistant, which is trained with alignment techniques. Qwen-Audio-Chat supports more flexible interaction, such as multiple audio inputs, multi-round question answering, and creative capabilities.
  <br>


## Evaluation

We evaluated the Qwen-Audio's abilities on the standard benchmarks as follows:

<p align="center">
    <img src="assets/evaluation.png" width="800"/>
<p>

The results of the evaluation are as follows:
<p align="center">
    <img src="assets/radar.png" width="800"/>
<p>

## We Are Hiring

If you are interested in joining us as full-time or intern, please contact us at qwen_audio@list.alibaba-inc.com.

## License Agreement

Researchers and developers are free to use the codes and model weights of both Qwen-Audio and Qwen-Audio-Chat. We also allow their commercial use. Check our license at [LICENSE](LICENSE) for more details.
<br>


## Contact Us

If you are interested to leave a message to either our research team or product team, feel free to send an email to qianwen_opensource@alibabacloud.com.

