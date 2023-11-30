# Tutorials of Qwen-Audio-Chat 
Qwen-Audio-Chat is a universal multimodal large-scale language model, capable of performing various audio understanding tasks. In this tutorial, we will provide concise examples to demonstrate the capabilities of Qwen-Audio-Chat in **speech recognition, speech translation, environmental sound understanding, multimodal audio understanding, and speech grounding** (extracting specific text segments from speech based on instructions).

## Initialization of Qwen-Audio-Chat
Before using Qwen-Audio-Chat, you first need to initialize the tokenizer and model of Qwen-Audio-Chat:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# Set the following seed to reproduce our results.
# torch.manual_seed(1234)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="cuda", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)
```
After executing the above code, the ```tokenizer``` refers to the tokenizer used by Qwen-Audio-Chat, while the ```model``` refers to the model of Qwen-Audio-Chat. The tokenizer is used for tokenizing and preprocessing mixed audio-text inputs, while the model is the Qwen-Audio-Chat model itself.

## Usage of Qwen-Audio-Chat
### **Multi-turn Audio-central Understanding**
#### **First Question**
Let's begin with a simple case shown beloe. The file ```assets/audio/1272-128104-0000.flac``` an audio clip from dev/test set of librispeec.

[1272-128104-0000.flac](http://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/1272-128104-0000.webm)

Let's first test whether Qwen-Audio-Chat understands its content. To begin with，we should use ```tokenizer.from_list_format``` to preprocess audios and tokenize tokens：
```python
query = tokenizer.from_list_format([
    {'audio': 'assets/audio/1272-128104-0000.flac'},
    {'text': 'what is that sound?'},
])
```
Then，you can use ```model.chat``` to obtain the response from the model. Nota that ```history=None``` for the first interaction.
```python
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
```
You are expected to obtain the following response:

> The sound is of a man speaking, in a English, saying, "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel".

The model's response is correct. You can listen to the speech where a man is saying "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL".

#### **Multi-turn QA**
We can further ask other questions to the model, such as finding the time slots of the phrase 'middle classes'。Nota that ```history=history``` to inform the model with previous interations.
```python
query = tokenizer.from_list_format([
    {'text': 'Find the start time and end time of the word "middle classes"'},
])
response, history = model.chat(tokenizer, query=query, history=history)
print(response)
```

You are expected to obtain the following results:

> The word "middle classes" starts at <|2.33|> seconds and ends at <|3.26|> seconds.

The model's response is correct again！We can verify it by ffmpeg tool```ffmpeg -i assets/audio/1272-128104-0000.flac -ss 2.33 -to 3.26 assets/audio/1272-128104-0000-middle_classes.wav```, and can find that the selected audio clips is the voice saying "middle classes".

[1272-128104-0000-middle_classes.flac](http://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/1272-128104-0000-middle_classes.webm)

### **Multilingual Audio Understanding**
Qwen-Audio-Chat supports speech understanding in Chinese, English, Japanese, Korean, German, Spanish, Italian. As shown below, the file ```assets/audio/es.mp3``` is a piece of Spanish audio.

[es.mp3](http://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/es.webm)

We can ask the model for speech recognition results just like before, with an empty conversation history ```history=None```.
```python
query = tokenizer.from_list_format([
    {'audio': 'assets/audio/es.mp3'},
    {'text': 'Recognize the speech'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
```

The expected response is:

> The speech is of a man speaking, in a Spanish, saying, "Bueno, también podemos considerar algunas actividades divertidas como los deportes acuáticos.".

Let's further ask some other questions about emotion by setting ```history=history```.

```python
query = tokenizer.from_list_format([
    {'text': 'What is his emotion'},
])
response, history = model.chat(tokenizer, query=query, history=history)
print(response)
```

The expected results are as follows：

> Based on the voice, it sounds like this person is neutral.

In addition to supporting multiple languages, Qwen-Audio-Chat can also handle Chinese dialect and accents. As shown below, the file ```assets/audio/example-重庆话.wav``` is a piece of Chongqing dialect audio

[example-重庆话.wav](http://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/example-重庆话.webm)

We can ask the model for speech recognition results just like before, with an empty conversation history ```history=None```.
```python
query = tokenizer.from_list_format([
    {'audio': 'assets/audio/example-重庆话.wav'},
    {'text': 'Recognize the speech'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
```

The expected results are as follows：

> The speech is of a man speaking, in a Southwestern Mandarin, saying, "对了我还想提议我们可以租一些自行车骑行一下既锻炼身体又心情愉悦".

### **Sound Understanding and Reasoning**
Qwen-Audio-Chat supports recognition of natural sounds and can perform inference based on them, such as scene reasoning and event handling. As shown below, the file ```assets/audio/glass-breaking-151256.mp3``` is a sound of glass breaking

[glass-breaking-151256.mp3](http://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/glass-breaking-151256.webm)

```python
query = tokenizer.from_list_format([
    {'audio': 'assets/audio/glass-breaking-151256.mp3'},
    {'text': 'What is it'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
```

You are expected to get the following results:

> This is a sound effect of breaking glass.

Alternatively, you can ask the model to perform analysis and reasoning based on audio.

```python
query = tokenizer.from_list_format([
    {'audio': 'assets/audio/glass-breaking-151256.mp3'},
    {'text': 'Recognize the sound information and assume that the sound is happening around the user. Please provide the user with some handling suggestions.'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
```

The results are as follows:

> The sound is of a glass shattering. To handle this situation, the user could follow these steps:
> 
> 1. Stay calm and assess the situation.
> 2. Determine if anyone is injured or in danger.
> 3. If possible, turn off any nearby lights or appliances to reduce the risk of further injury.
> 4. Use a broom or dustpan to sweep up any broken glass shards.
> 5. If necessary, call for professional help to clean up and dispose of the broken glass.
> 6. Take steps to prevent similar accidents from occurring in the future, such as securing fragile items or improving lighting in the area.

### **Music Analysis**
Qwen-Audio-Chat supports analysis and appreciation of music. As shown below, the file ```assets/audio/music.wav``` is a piano solo.

[music.wav](http://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/music.webm)

```python
query = tokenizer.from_list_format([
    {'audio': 'assets/audio/music.wav'},
    {'text': 'what is the instrument'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
```
The below is the response of Qwen-Audio-Chat: 

> The instrument is the piano.

Besides, you can also ask it to provide detailed appreciation and recommend related music.

```python
response, history = model.chat(tokenizer, query="Describe the music in detail and give me some recommendations for similar music.", history=histroy)
print(response)
```

The below is the response of Qwen-Audio-Chat:

> The music is a solo piano piece. It is playful and humorous, with a fast tempo and a light, bouncy feel. The melody is simple and catchy, with a repetitive theme that is varied and developed throughout the piece. The style is classical, with a strong emphasis on the solo piano as the primary instrument. 

> Similar music to this piece might include the solo piano works of Frederic Chopin, Johannes Brahms, and Ludwig van Beethoven. Other classical composers who wrote humorous and playful solo piano pieces include Franz Schubert and Robert Schumann. For contemporary solo piano music with a similar feel, composers such as Max Richter and Olafur Arnalds might be worth exploring.


### **Multi-Audio Inputs**
Qwen-Audio-Chat can not only handle multi-turn question and answer sessions for a single audio input but also support multiple audio inputs! In the following example, the files ```assets/audio/你没事吧-轻松.wav``` and ```assets/audio/你没事吧-消极.wav``` both contain the phrase "你没事吧" (Are you okay?), but with different tones. Now we can ask the model to distinguish between them.

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

The below is the response of the model：

> Based on the voice, it sounds like this person is happy in the first audio, but sad in the second audio.
You can also provide the model with audio content of different types and let it generate creative content based on these audio inputs. For example, the file ``` assets/audio/glass-breaking-151256.mp3``` provides the sound of glass shattering, and then the file ```assets/audio/你没事吧-轻松.wav``` provides the sound of someone asking if you are okay. Now, we can ask the model to generate a story based on this audio information.

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

The below is the response of Qwen-Audio-Chat: 

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

### **Grounding**
We showcase the ability of the Qwen-Audio-Chat model to provide word-level timestamp localization. Let's take a look at the example below. The file ```assets/mm_tutorial/1089_134686_000007_000004.wav``` is a speech spoken with English, and the spoken content is "The music came nearer and he recalled the words, the words of Shelley's fragment upon the moon wandering companionless, pale for weariness.". We ask the model to locate some words/phrases in the speech."

[1089_134686_000007_000004.wav](http://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/1089_134686_000007_000004.webm)

The transcript is "The music came nearer and he recalled the words, the words of Shelley's fragment upon the moon wandering companionless, pale for weariness."

```python
query = tokenizer.from_list_format([
    {'audio': 'assets/audio/1089_134686_000007_000004.wav'},
    {'text': 'Find the word "companionless"'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
```

You are expected to get the following response from the model:

> The word "companionless" starts at <|6.28|> seconds and ends at <|7.15|> seconds.

We can verify the result via ffmpeg tool as follows:

```shell
ffmpeg -i assets/audio/1089_134686_000007_000004.wav -ss 6.28 -to 7.15 assets/audio/1089_134686_000007_000004_companionless.wav
```

The results are as follows:

[1089_134686_000007_000004_companionless.wav](http://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/1089_134686_000007_000004_companionless.webm)

Furthuermore, we can also ask the model to locate words via semantic understanding. For example, now we ask the model to find the positions of the names in the sentence.
```python
query = tokenizer.from_list_format([
    {'text': 'find the person name'},
])
response, history = model.chat(tokenizer, query=query, history=history)
print(response)
```

The results are as follows:

> The person name "shelley's" is mentioned. "shelley's" starts at <|3.79|> seconds and ends at <|4.33|> seconds.

We can verify the result via ffmpeg tool as follows:

[1089_134686_000007_000004_person_name.wav](http://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/1089_134686_000007_000004_person_name.webm)

More funny examples can be seen at [Demo](https://qwen-audio.github.io/Qwen-Audio).
