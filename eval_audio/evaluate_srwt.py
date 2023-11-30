import argparse
import itertools
import json
import os
import random
import time
from functools import partial
import re
from evaluate_tokenizer import EvaluationTokenizer
import editdistance as ed
import torch
import edit_distance

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
PUNCS = '!,.?;:'


ds_collections = {
    'industrial_srwt': {'path': 'data/srwt/industrial_srwt_eval.jsonl', 'language': 'zh'}
}


class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, ds, prompt):
        path = ds['path']
        self.language = ds['language']
        self.datas = open(path).readlines()
        self.prompt = prompt

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = json.loads(self.datas[idx].strip())
        audio_path = data['audio']
        source = data['source']
        gt = data['gt']

        return {
            'input_text': self.prompt.format(audio_path, self.language, self.language),
            'audio_path': audio_path,
            'source': source,
            'gt': gt
        }


def collate_fn(inputs, tokenizer):

    input_texts = [_['input_text'] for _ in inputs]
    source = [_['source'] for _ in inputs]
    gt = [_['gt'] for _ in inputs]
    audio_path = [_['audio_path'] for _ in inputs]
    audio_info = [tokenizer.process_audio(_['input_text']) for _ in inputs ]
    input_tokens = tokenizer(input_texts,
                             return_tensors='pt',
                             padding='longest',
                             audio_info= audio_info)

    return input_tokens.input_ids, input_tokens.attention_mask, source, gt,audio_path,audio_info


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

def remove_sp(text, language):
    gt = re.sub(r"<\|.*?\|>", " ", text)
    gt = re.sub(rf"\s+", r" ", gt)  # 将文本中的连续空格替换为单个空格
    gt = re.sub(f" ?([{PUNCS}])", r"\1", gt)
    gt = gt.lstrip(" ")
    if language == "zh":
        gt = re.sub(rf"\s+", r"", gt)
    return gt

def compute_wer(refs, hyps, language):
    distance = 0
    ref_length = 0
    tokenizer = EvaluationTokenizer(
            tokenizer_type="none",
            lowercase=True,
            punctuation_removal=False,
            character_tokenization=False,
        )
    for i in range(len(refs)):
        ref = refs[i]
        pred = hyps[i]
        ref_items = tokenizer.tokenize(ref).split()
        pred_items = tokenizer.tokenize(pred).split()
        if language == "zh":
            ref_items = [x for x in "".join(ref_items)]
            pred_items = [x for x in "".join(pred_items)]
        if i==0:
            print(f"ref: {ref}")
            print(f"pred: {pred}")
            print(f"ref_items:\n{ref_items}\n{len(ref_items)}\n{ref_items[0]}")
            print(f"pred_items:\n{pred_items}\n{len(ref_items)}\n{ref_items[0]}")
        distance += ed.eval(ref_items, pred_items)
        ref_length += len(ref_items)
    return distance/ref_length

class AverageShiftCalculator():
    def __init__(self):
        print("Calculating average shift.")

    def __call__(self, refs, hyps):
        ts_list1 = self.read_timestamps(refs)
        ts_list2 = self.read_timestamps(hyps)
        res = self.as_cal(ts_list1, ts_list2)
        print("Average shift : {}.".format(str(res)[:8]))
        print("Following timestamp pair differs most: {}, detail:{}".format(self.max_shift, self.max_shift_uttid))
        return res

    def _intersection(self, list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        if set1 == set2:
            print("Uttid same checked.")
            return set1
        itsc = list(set1 & set2)
        print("Uttid differs: file1 {}, file2 {}, lines same {}.".format(len(list1), len(list2), len(itsc)))
        return itsc

    def read_timestamps(self, body_list):
        ts_list = []
        pattern_error = 0
        for body in body_list:
            body = body.replace("<|startoftranscript|>","").replace("<|transcribe|>","")
            ts_pattern = r"<\|\d{1,2}\.\d+\|>"
            if "<|en|>" in body:
                body = body.replace("<|en|>","")
                lan = "en"
            elif "<|zh|>" in body:
                body = body.replace("<|zh|>","")
                lan = "zh"
            all_time_stamps = re.findall(ts_pattern, body)
            all_time_stamps = [ float(t.replace("<|","").replace("|>","")) for t in all_time_stamps]
            all_word_list = [x for x in re.split(ts_pattern, body)][1:-1]

            if len(all_time_stamps) != len(all_word_list) + 1:
                pattern_error += 1
                continue
            text = "\t".join(all_word_list)
            ts = [all_time_stamps[i:i + 2] for i in range(len(all_time_stamps) - 1)]
            ts_list.append((text, ts))
            assert len(ts) == len(all_word_list), f"{body}"
        print(f"pattern_error_num: {pattern_error}")
        return ts_list

    def _shift(self, filtered_timestamp_list1, filtered_timestamp_list2):
        shift_time = 0
        for fts1, fts2 in zip(filtered_timestamp_list1, filtered_timestamp_list2):
            shift_time += abs(fts1[0] - fts2[0]) + abs(fts1[1] - fts2[1])
        num_tokens = len(filtered_timestamp_list1)
        return shift_time, num_tokens

    def as_cal(self, ts_list1, ts_list2):
        # calculate average shift between timestamp1 and timestamp2
        # when characters differ, use edit distance alignment
        # and calculate the error between the same characters
        assert len(ts_list1) == len(ts_list2), f"{len(ts_list1)}, {len(ts_list2)}"
        self._accumlated_shift = 0
        self._accumlated_tokens = 0
        self.max_shift = 0
        self.max_shift_uttid = None
        for uttid in range(len(ts_list1)):
            (t1, ts1) = ts_list1[uttid]
            (t2, ts2) = ts_list2[uttid]
            _align, _align2, _align3 = [], [], []
            fts1, fts2 = [], []
            _t1, _t2 = [], []
            sm = edit_distance.SequenceMatcher(t1.split('\t'), t2.split('\t'))
            s = sm.get_opcodes()
            for j in range(len(s)):
                if s[j][0] == "replace" or s[j][0] == "insert":
                    _align.append(0)
                if s[j][0] == "replace" or s[j][0] == "delete":
                    _align3.append(0)
                elif s[j][0] == "equal":
                    _align.append(1)
                    _align3.append(1)
                else:
                    continue
            # use s to index t2
            for a, ts , t in zip(_align, ts2, t2.split('\t')):
                if a:
                    fts2.append(ts)
                    _t2.append(t)
            sm2 = edit_distance.SequenceMatcher(t2.split('\t'), t1.split('\t'))
            s = sm2.get_opcodes()
            for j in range(len(s)):
                if s[j][0] == "replace" or s[j][0] == "insert":
                    _align2.append(0)
                elif s[j][0] == "equal":
                    _align2.append(1)
                else:
                    continue
            # use s2 tp index t1
            for a, ts, t in zip(_align3, ts1, t1.split('\t')):
                if a:
                    fts1.append(ts)
                    _t1.append(t)
            # import ipdb;ipdb.set_trace()
            if len(fts1) == len(fts2):
                shift_time, num_tokens = self._shift(fts1, fts2)
                if num_tokens == 0:
                    print(ts_list1[uttid], ts_list2[uttid])
                    continue
                self._accumlated_shift += shift_time
                self._accumlated_tokens += num_tokens
                if shift_time/num_tokens > self.max_shift:
                    self.max_shift = shift_time/num_tokens
                    self.max_shift_uttid = uttid
            else:
                print("length mismatch")
        return self._accumlated_shift / self._accumlated_tokens


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))


    prompt = '<audio>{}</audio><|startoftranscript|><|{}|><|transcribe|><|{}|><|timestamps|><|0.00|>'

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, device_map='cuda', trust_remote_code=True).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint,
                                              trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id

    random.seed(args.seed)
    dataset = AudioDataset(
        ds=ds_collections[args.dataset],
        prompt=prompt
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )

    gts = []
    sources = []
    rets = []
    audio_paths = []
    for _, (input_ids, attention_mask, source, gt, audio_path, audio_info) in tqdm(enumerate(data_loader)):
        output_ids = model.generate(
            input_ids=input_ids.cuda(),
            attention_mask=attention_mask.cuda(),
            do_sample=False,
            max_new_tokens=256,
            min_new_tokens=1,
            length_penalty=1.0,
            num_return_sequences=1,
            repetition_penalty=1.0,
            use_cache=True,
            pad_token_id=tokenizer.eod_id,
            eos_token_id=tokenizer.eod_id,
            audio_info=audio_info
        )
        output_ids = output_ids[:, input_ids.size(1):]
        eos_token_id_tensor = torch.tensor([tokenizer.eod_id]).to(output_ids.device).unsqueeze(0).repeat(
            output_ids.size(0),
            1)
        output_ids = torch.cat([output_ids, eos_token_id_tensor], dim=1)
        eos_token_pos = output_ids.eq(tokenizer.eod_id).float().argmax(-1)

        pred = [output_ids[_, :eos_token_pos[_]].tolist() for _ in range(output_ids.size(0))]
        gts.extend(gt)
        rets.extend([
            tokenizer.decode(_,skip_special_tokens=False).strip() for _ in pred
        ])
        sources.extend(source)
        audio_paths.extend(audio_path)

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_gts = [None for _ in range(world_size)]
    merged_sources = [None for _ in range(world_size)]
    merged_responses = [None for _ in range(world_size)]
    merged_audio_paths = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_gts, gts)
    torch.distributed.all_gather_object(merged_sources, sources)
    torch.distributed.all_gather_object(merged_responses, rets)
    torch.distributed.all_gather_object(merged_audio_paths, audio_paths)

    merged_gts = [_ for _ in itertools.chain.from_iterable(merged_gts)]
    merged_sources = [_ for _ in itertools.chain.from_iterable(merged_sources)]
    merged_audio_paths = [_ for _ in itertools.chain.from_iterable(merged_audio_paths)]
    merged_responses = [
        _ for _ in itertools.chain.from_iterable(merged_responses)
    ]

    if torch.distributed.get_rank() == 0:
        print(f"Evaluating {args.dataset} ...")

        results = []
        for gt, response, source, audio_path in zip(merged_gts, merged_responses, merged_sources, merged_audio_paths):
            results.append({
                'gt': gt,
                'response': response,
                'source': source,
                'audio_path': audio_path,
            })
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file = f'{args.dataset}_{time_prefix}.json'
        json.dump(results, open(results_file, 'w'))
        results_dict = {}
        for item in tqdm(results):
            source = item["source"]
            results_dict.setdefault(source, []).append(item)
        lan = ds_collections[args.dataset]['language']
        for source in results_dict:
            refs, hyps = [], []
            pure_refs, pure_hyps = [], []
            results_list = results_dict[source]
            for result in results_list:
                gt = result["gt"]
                pure_gt = remove_sp(gt, lan)
                response = result["response"]
                pure_response = remove_sp(response, lan)
                refs.append(gt)
                pure_refs.append(pure_gt)
                hyps.append(response)
                pure_hyps.append(pure_response)
            asc = AverageShiftCalculator()
            aas_score = asc(refs, hyps)
            wer = compute_wer(pure_refs, pure_hyps, lan)
            print(f"source: {source}  cnt: {len(refs)} "
                  f"WER: {wer:.4f} AAS: {aas_score:.4f}")

    torch.distributed.barrier()
