## Evaluation

### Dependencies

```bash
apt-get update
apt-get install openjdk-8-jdk
pip install evaluate sacrebleu==1.5.1
pip install pycocoevalcap
pip install edit_distance editdistance
mkdir -p eval_audio/caption_evaluation_tools
git clone https://github.com/audio-captioning/caption-evaluation-tools.git eval_audio/caption_evaluation_tools
cd eval_audio/caption_evaluation_toolscoco_caption/
./get_stanford_models.sh
cd ../..
pip install sacrebleu
pip install sacrebleu\[ja\]
pip install sed_eval
pip install dcase_util
```
### ASR

- Data

> LibriSpeech: https://www.openslr.org/12

> Aishell1: https://www.aishelltech.com/kysjcp

> Aishell2: https://www.aishelltech.com/aishell_2

```bash
mkdir -p data/asr && cd data/asr

# download audios from above links

# download converted files
wget https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/evaluation/librispeech_eval.jsonl
wget https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/evaluation/aishell1_eval.jsonl
wget https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/evaluation/aishell2_eval.jsonl


cd ../..
```

```bash
 for ds in "librispeech" "aishell1" "aishell2"
 do
     python -m torch.distributed.launch --use_env \
         --nproc_per_node ${NPROC_PER_NODE:-8} --nnodes 1 \
         evaluate_asr.py \
         --checkpoint $checkpoint \
         --dataset $ds \
         --batch-size 20 \
         --num-workers 2
 done
```
### S2TT

- Data

> CoVoST 2: https://github.com/facebookresearch/covost

```bash
mkdir -p data/st && cd data/st

# download audios from https://commonvoice.mozilla.org/en/datasets

# download converted files
wget https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/evaluation/covost2_eval.jsonl

cd ../..
```
- Evaluate
```bash
ds="covost2"
python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-8} --nnodes 1 \
    evaluate_st.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 8 \
    --num-workers 2 
```

### SRWT

- Data


> industrial SRWT data: 
> This is a closed-source dataset from https://arxiv.org/pdf/2305.11013.pdf

Please organize the evaluation data into a json lines format, where each json format is as follows:
```json
{"audio": "001001.wav", 
  "gt": "<|0.00|><|sil|><|0.54|>面<|0.86|>条<|1.17|>机<|1.46|><|sil|><|2.01|>", 
  "source": "aishell2"}
```

- Evaluate
```bash
ds="covost2"
python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-8} --nnodes 1 \
    evaluate_st.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 8 \
    --num-workers 2 
```

### AAC

- Data

> Clotho: https://github.com/audio-captioning/clotho-dataset

```bash
mkdir -p data/caption && cd data/caption

# download audios from https://zenodo.org/record/3490684

# download converted files
wget https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/evaluation/clotho_eval.jsonl

cd ../..
```

- Evaluate

```bash
ds="clotho"
python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-8} --nnodes 1 \
    evaluate_caption.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 8 \
    --num-workers 2 \

```
### ASC
- Data
> TUT2017: https://dcase.community/challenge2017/task-acoustic-scene-classification

> CochlScene: https://zenodo.org/records/7080122



```bash
mkdir -p data/asc && cd data/asc

# download TUT2017 datasets from https://zenodo.org/records/1040168
# download CochlScene datasets from https://zenodo.org/records/7080122


# download converted files
wget https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/evaluation/tut2017_eval.jsonl
wget https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/evaluation/cochlscene_eval.jsonl

cd ../..
```
- Evaluate

```bash
 for ds in "cochlscene" "tut2017"
 do
     python -m torch.distributed.launch --use_env \
         --nproc_per_node ${NPROC_PER_NODE:-8} --nnodes 1 \
         evaluate_scene.py \
         --checkpoint $checkpoint \
         --dataset $ds \
         --batch-size 20 \
         --num-workers 2
 done
```

### SER
- Data
> MELD: https://affective-meld.github.io/



```bash
mkdir -p data/ser && cd data/ser

# download MELD datasets from above link

# download converted files
wget https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/evaluation/meld_eval.jsonl


cd ../..
```

- Evaluate

```bash
ds="meld"
python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-8} --nnodes 1 \
    evaluate_emotion.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 8 \
    --num-workers 2 \
    --few-shot 0
```

### AQA
- Data
> Clotho-AQA: https://zenodo.org/records/6473207



```bash
mkdir -p data/aqa && cd data/aqa

# download Clotho-AQA dataset from https://zenodo.org/records/6473207
# download converted files
wget https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/evaluation/clothoaqa_eval.jsonl


cd ../..
```

- Evaluate

```bash
ds="clothoaqa"
python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-8} --nnodes 1 \
    evaluate_aqa.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 8 \
    --num-workers 2 
```


### VSC
- Data
> VocalSound: https://github.com/YuanGongND/vocalsound


```bash
mkdir -p data/vsc && cd data/vsc

# download dataset from the above link
# download converted files
wget https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/evaluation/vocalsound_eval.jsonl


cd ../..
```

- Evaluate

```bash
ds="clothoaqa"
python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-8} --nnodes 1 \
    evaluate_aqa.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 8 \
    --num-workers 2 
```


### MNA
- Data
> NSynth: https://magenta.tensorflow.org/datasets/nsynth#files



```bash
mkdir -p data/mna && cd data/mna

# download NSynth datasets from above link

# download converted files
wget https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/evaluation/nsynth_eval.jsonl

cd ../..
```

- Evaluate

```bash
ds="nsynth"
python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-8} --nnodes 1 \
    evaluate_note_analysis.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 8 \
    --num-workers 2
```


### Acknowledgement

Part of these codes are borrowed from [HEAR Benchmark](https://github.com/hearbenchmark/hear-eval-kit) , [OFA-SYS](https://github.com/OFA-Sys/OFASys) and [Audio captioning](https://github.com/audio-captioning), thanks for their wonderful work.
