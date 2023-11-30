import argparse
import itertools
import json
import os
import random
import time
from functools import partial
import torch

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sklearn.metrics import accuracy_score



ds_collections = {
    'tut2017': {'path': 'data/asc/tut2017_eval.jsonl'},
    'cochlscene': {'path': 'data/asc/cochlscene_eval.jsonl'},
}


class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, ds, prompt):
        path = ds['path']
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
            'input_text': self.prompt.format(audio_path),
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


    prompt = '<audio>{}</audio><|startofanalysis|><|unknown|><|keyword|><|en|><|notimestamps|><|scene|>'

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
            max_new_tokens=10,
            min_new_tokens=1,
            length_penalty=1.0,
            num_return_sequences=1,
            repetition_penalty=1.0,
            use_cache=True,
            pad_token_id=tokenizer.eod_id,
            eos_token_id=tokenizer.eod_id,
            audio_info = audio_info
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

        for source in results_dict:
            refs, hyps = [], []
            bi_refs, bi_hyps = [], []
            results_list = results_dict[source]
            for result in results_list:
                gt = result["gt"]
                response = result["response"]
                refs.append(gt)
                hyps.append(response)
            score = accuracy_score(refs, hyps)
            print(f"{source} ACC_score:", score, len(hyps))

    torch.distributed.barrier()
