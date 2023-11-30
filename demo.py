# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# torch.manual_seed(1234)

tokenizer = AutoTokenizer.from_pretrained("./", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("./", device_map="auto", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained("./", trust_remote_code=True)

# inputs = tokenizer('<img>https://upload.wikimedia.org/wikipedia/commons/a/a1/Two_Apples_on_a_Table_by_Paul_C%C3%A9zanne%2C_Speed_Art_Museum.jpg</img>Generate the caption in English with grounding:', return_tensors='pt')
# inputs = inputs.to(model.device)
# pred = model.generate(**inputs)
# response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
# print(response)
# image = tokenizer.draw_bbox_on_latest_picture(response)
# image.save('2.jpg')
# exit(0)

# response, history = model.chat(tokenizer, query='<img>W020210726613375501257.jpg</img>介绍作品', history=None)
# print(response)

# response, history = model.chat(tokenizer, '输出"抬轿子的人"的检测框', history=history)
# print(response)
# image = tokenizer.draw_bbox_on_latest_picture(response, history)
# cv2.imwrite('1.jpg', image)
# exit(0)

# 第一轮对话 1st dialogue turn
query = tokenizer.from_list_format([
    {'image': 'assets/apple.jpeg'},
    {'text': 'Generate the caption in English with grounding'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
image = tokenizer.draw_bbox_on_latest_picture(response, history)
image.save('3.jpg')
exit(0)
# 图中是一名年轻女子在海滩上与一只拉布拉多犬玩耍，犬伸出爪子与女子击掌。

# 第二轮对话 2st dialogue turn
response, history = model.chat(tokenizer, '输出"击掌"的检测框', history=history)
print(response)
# <ref>击掌</ref><box>(529,504),(587,610)</box>
image = tokenizer.draw_bbox_on_latest_picture(response, history)
image.save('1.jpg')
