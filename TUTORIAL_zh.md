# Qwen-Audio-Chat使用教程
Qwen-Audio-Chat是通用语音多模态大规模语言模型，因此它可以完成多种音频理解任务。在本教程之中，我们会给出一些简明的例子，用以展示Qwen-Audio-Chat在**语音识别，语音翻译，环境音理解，多音频理解和语音定位**(根据指令截取语音中指定文本的片段)等多方面的能力。

## 初始化Qwen-Audio-Chat模型
在使用Qwen-Audio-Chat之前，您首先需要初始化Qwen-Audio-Chat的分词器（Tokenizer）和Qwen-Audio-Chat的模型：
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# 如果您希望结果可复现，可以设置随机数种子。
# torch.manual_seed(1234)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="cuda", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)
```
在执行完上述代码后，```tokenizer```将对应Qwen-Audio-Chat使用的分词器，而```model```将对应Qwen-Audio-Chat的模型。```tokenizer```用于对音文混排输入进行分词和预处理，而```model```则是Qwen-Audio-Chat模型本身。

## 使用Qwen-Audio-Chat
### **多轮音频理解问答**
#### **第一个问题**
首先我们来看一个最简单的例子，如下图所示，文件```assets/audio/1272-128104-0000.flac```是一段librispeech数据集(未加入训练)中的一段语音。

[1272-128104-0000.flac](http://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/1272-128104-0000.webm)

我们来问一问Qwen-Audio-Chat音频中是什么声音。首先，我们使用tokenizer.from_list_format可以对音文混排输入进行分词与处理：
```python
query = tokenizer.from_list_format([
    {'audio': 'assets/audio/1272-128104-0000.flac'},
    {'text': 'what is that sound?'},
])
```
接下来，我们可以使用```model.chat```向Qwen-Audio-Chat模型提问并获得回复。注意在第一次提问时，对话历史为空，因此我们使用```history=None```。
```python
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
```
您应该会得到类似下列的输出结果：

> The sound is of a man speaking, in a English, saying, "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel".

这说明模型正确的回答了问题！根据librispeech的标注结果，确实是一个男性的声音，说的内容是"MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL"。

#### **多轮问答**
我们还可以继续向模型发问，例如询问"middle classes"的具体发声时间。在后续提问时，对话历史并不为空，我们使用```history=history```向```model.chat```传递之前的对话历史：
```python
query = tokenizer.from_list_format([
    {'text': 'Find the start time and end time of the word "middle classes"'},
])
response, history = model.chat(tokenizer, query=query, history=history)
print(response)
```

您应该会得到类似下列的输出结果：

> The word "middle classes" starts at <|2.33|> seconds and ends at <|3.26|> seconds.

模型再次正确回答了问题！我们可以用ffmpeg工具对语音进行截取```ffmpeg -i assets/audio/1272-128104-0000.flac -ss 2.33 -to 3.26 assets/audio/1272-128104-0000-middle_classes.wav```，可以听到截取的片段就是"middle classes"

[1272-128104-0000-middle_classes.flac](http://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/1272-128104-0000-middle_classes.webm)

### **多语种语音理解**
Qwen-Audio-Chat支持中英日韩德西意的语音理解能力。如下所示，文件```assets/audio/es.mp3```是一段西班牙语音。

[es.mp3](http://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/es.webm)

我们可以像之前一样向模型询问语音识别的结果，对话历史为空，因此我们使用```history=None```。
```python
query = tokenizer.from_list_format([
    {'audio': 'assets/audio/es.mp3'},
    {'text': 'Recognize the speech'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
```

您应该会得到类似下列的输出结果：

> The speech is of a man speaking, in a Spanish, saying, "Bueno, también podemos considerar algunas actividades divertidas como los deportes acuáticos.".

您同样可以进一步提出后续问题，此时需要使用```history=history```向```model.chat```传递之前的对话历史。

```python
query = tokenizer.from_list_format([
    {'text': 'What is his emotion'},
])
response, history = model.chat(tokenizer, query=query, history=history)
print(response)
```

您应该会得到类似下列的输出结果：

> Based on the voice, it sounds like this person is neutral.

除了多语种，Qwen-Audio-Chat还可以支持中文方言和口音识别。如下所示，文件```assets/audio/example-重庆话.wav```是一段重庆方言。

[example-重庆话.wav](http://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/example-重庆话.webm)

我们可以像之前一样向模型询问语音识别的结果，对话历史为空，因此我们使用```history=None```。
```python
query = tokenizer.from_list_format([
    {'audio': 'assets/audio/example-重庆话.wav'},
    {'text': 'Recognize the speech'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
```

您应该会得到类似下列的输出结果：

> The speech is of a man speaking, in a Southwestern Mandarin, saying, "对了我还想提议我们可以租一些自行车骑行一下既锻炼身体又心情愉悦".

### **环境音理解和推理**
Qwen-Audio-Chat支持对环境音的识别，并能基于此做推理，比如场景推理，事件应对等。如下所示，文件```assets/audio/glass-breaking-151256.mp3```是一段玻璃打碎的声音。

[glass-breaking-151256.mp3](http://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/glass-breaking-151256.webm)

```python
query = tokenizer.from_list_format([
    {'audio': 'assets/audio/glass-breaking-151256.mp3'},
    {'text': 'What is it'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
```

您应该会得到类似下列的输出结果：

> This is a sound effect of breaking glass.

您同样可以直接对音频进行推理分析。

```python
query = tokenizer.from_list_format([
    {'audio': 'assets/audio/glass-breaking-151256.mp3'},
    {'text': 'Recognize the sound information and assume that the sound is happening around the user. Please provide the user with some handling suggestions.'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
```

您应该会得到类似下列的输出结果：

> The sound is of a glass shattering. To handle this situation, the user could follow these steps:
> 
> 1. Stay calm and assess the situation.
> 2. Determine if anyone is injured or in danger.
> 3. If possible, turn off any nearby lights or appliances to reduce the risk of further injury.
> 4. Use a broom or dustpan to sweep up any broken glass shards.
> 5. If necessary, call for professional help to clean up and dispose of the broken glass.
> 6. Take steps to prevent similar accidents from occurring in the future, such as securing fragile items or improving lighting in the area.

### **音乐分析**
Qwen-Audio-Chat支持对音乐的识别和赏析。如下所示，文件```assets/audio/music.wav```是一段钢琴独奏。

[music.wav](http://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/music.webm)

```python
query = tokenizer.from_list_format([
    {'audio': 'assets/audio/music.wav'},
    {'text': 'what is the instrument'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
```

您应该会得到类似下列的输出结果：

> The instrument is the piano.

之后，您还可以让它做出详细的赏析并推荐相关的音乐：

```python
response, history = model.chat(tokenizer, query="Describe the music in detail and give me some recommendations for similar music.", history=histroy)
print(response)
```

您应该会得到类似下列的输出结果：

> The music is a solo piano piece. It is playful and humorous, with a fast tempo and a light, bouncy feel. The melody is simple and catchy, with a repetitive theme that is varied and developed throughout the piece. The style is classical, with a strong emphasis on the solo piano as the primary instrument. 

> Similar music to this piece might include the solo piano works of Frederic Chopin, Johannes Brahms, and Ludwig van Beethoven. Other classical composers who wrote humorous and playful solo piano pieces include Franz Schubert and Robert Schumann. For contemporary solo piano music with a similar feel, composers such as Max Richter and Olafur Arnalds might be worth exploring.


### **多音频输入**
Qwen-Audio-Chat不仅可以针对一段音频进行多轮问答，还可以支持多音频输入！下面的例子中，文件```assets/audio/你没事吧-轻松.wav```和```assets/audio/你没事吧-消极.wav```这两段音频都讲了"你没事吧"这句话，但是却用的不同的语气，现在我们可以让模型去进行分辨。

[你没事吧-轻松.wav](http://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/你没事吧-轻松.webm)

[你没事吧-消极.wav](http://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/你没事吧-消极.webm)

```python
query = tokenizer.from_list_format([
    {'audio': 'assets/audio/你没事吧-轻松.wav'},
    {'audio': 'assets/audio/你没事吧-消极.wav'},
    {'text': 'Is there any difference in the emotions of these two audio?'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
```

您应该会得到类似下列的输出结果：

> Based on the voice, it sounds like this person is happy in the first audio, but sad in the second audio.

您也可以给模型不同类型的音频内容，然后模型结合这些音频进行内容创作，比如，文件```assets/audio/glass-breaking-151256.mp3```先给出一个玻璃破碎的声音，之后文件```assets/audio/你没事吧-轻松.wav```又给出一个人询问的声音，现在我们让模型基于这些音频信息编写一个故事。

[glass-breaking-151256.mp3](http://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/glass-breaking-151256.webm)

[你没事吧-轻松.wav](http://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/你没事吧-轻松.webm)

```python
query = tokenizer.from_list_format([
    {'audio': 'assets/audio/glass-breaking-151256.mp3'},
    {'audio': 'assets/audio/你没事吧-轻松.wav'},
    {'text': 'based on these two audio, write a story'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
```

您应该会得到类似下列的输出结果：

> The first audio is of breaking glass, and the second audio is of a woman speaking, in a Chinese, saying, "哈哈你没事吧". Based on these two audio, a story could be written as follows:
> The sound of breaking glass echoed through the empty room. The sound was so loud that it startled the woman who was sitting on the couch, staring blankly at the TV. She quickly got up and ran towards the source of the noise.
> As she entered the room, she saw a large piece of glass lying on the floor, shattered into many small pieces. She gasped and quickly stepped back, afraid that she might get hurt by the sharp pieces.
> But then she heard a voice, a familiar voice, saying, "哈哈你没事吧". She turned around and saw her friend standing behind her, a smile on her face.
> "Are you okay?" her friend asked, coming closer to inspect the damage.
> The woman nodded, still in shock. "I'm fine, but that was really loud. How did you know I was here?"
> Her friend chuckled. "I heard the glass breaking and came to see what happened. You know me, always the first to offer help."
> The woman smiled, grateful for her friend's presence. Together, they cleaned up the broken glass and made sure that the room was safe again.
> As they were finishing up, the woman's phone rang. It was her boss calling to tell her that she had been promoted to a new position.
> She couldn't help but laugh as she told her friend the good news. "See, everything is going to be okay. Even the glass didn't break me."
> Her friend clapped her on the back. "That's the spirit! Now let's go celebrate."
> And with that, the two friends left the room, ready to start a new chapter in their lives.

### **Grounding能力**
在最后，我们展示Qwen-Audio-Chat模型字级别定位时间戳的能力。让我们来看下面的例子。如下所示，文件```assets/mm_tutorial/1089_134686_000007_000004.wav```是一段英文的语音，这句话讲的内容是"The music came nearer and he recalled the words, the words of Shelley's fragment upon the moon wandering companionless, pale for weariness."。我们可以指定具体的词，让模型找到具体发音位置。

[1089_134686_000007_000004.wav](http://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/1089_134686_000007_000004.webm)

这句话讲的内容是"The music came nearer and he recalled the words, the words of Shelley's fragment upon the moon wandering companionless, pale for weariness."

```python
query = tokenizer.from_list_format([
    {'audio': 'assets/audio/1089_134686_000007_000004.wav'},
    {'text': 'Find the word "companionless"'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
```

您应该会得到类似下列的输出结果：

> The word "companionless" starts at <|6.28|> seconds and ends at <|7.15|> seconds.

您可以通过调用ffmpeg工具进行音频剪辑，验证结果：

```shell
ffmpeg -i assets/audio/1089_134686_000007_000004.wav -ss 6.28 -to 7.15 assets/audio/1089_134686_000007_000004_companionless.wav
```

可以得到以下音频结果：

[1089_134686_000007_000004_companionless.wav](http://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/1089_134686_000007_000004_companionless.webm)

除此之外，我们还可以让模型通过语义理解找到想要定位的词汇，比如现在我们让模型找到句子中人名的位置：
```python
query = tokenizer.from_list_format([
    {'text': 'find the person name'},
])
response, history = model.chat(tokenizer, query=query, history=history)
print(response)
```

您应该会得到类似下列的输出结果：

> The person name "shelley's" is mentioned. "shelley's" starts at <|3.79|> seconds and ends at <|4.33|> seconds.

您可以通过调用ffmpeg工具进行音频剪辑，可以得到以下音频结果：

[1089_134686_000007_000004_person_name.wav](http://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/1089_134686_000007_000004_person_name.webm)

更多例子请到[Demo](https://qwen-audio.github.io/Qwen-Audio)页面了解详情。