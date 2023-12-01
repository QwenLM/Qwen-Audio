from __future__ import annotations
from typing import Dict, Any, Union, List, Tuple
from pathlib import Path
import json

from caption_evaluation_tools.coco_caption.pycocotools.coco import COCO
from caption_evaluation_tools.coco_caption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer



def write_json(data: Union[List[Dict[str, Any]], Dict[str, Any]],
               path: Path) \
        -> None:
    """ Write a dict or a list of dicts into a JSON file
    :param data: Data to write
    :type data: list[dict[str, any]] | dict[str, any]
    :param path: Path to the output file
    :type path: Path
    """
    with path.open("w") as f:
        json.dump(data, f)


def reformat_to_coco(predictions: List[str],
                     ground_truths: List[List[str]],
                     ids: Union[List[int], None] = None) \
        -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """ Reformat annotations to the MSCOCO format
    :param predictions: List of predicted captions
    :type predictions: list[str]
    :param ground_truths: List of lists of reference captions
    :type ground_truths: list[list[str]]
    :param ids: List of file IDs. If not given, a running integer\
                is used
    :type ids: list[int] | None
    :return: Predictions and reference captions in the MSCOCO format
    :rtype: list[dict[str, any]]
    """
    # Running number as ids for files if not given
    if ids is None:
        ids = list(range(len(predictions)))

    # Captions need to be in format
    # [{
    #     "audio_id": : int,
    #     "caption"  : str
    # ]},
    # as per the COCO results format.
    pred = []
    ref = {
        'info': {'description': 'Clotho reference captions (2019)'},
        'audio samples': [],
        'licenses': [
            {'id': 1},
            {'id': 2},
            {'id': 3}
        ],
        'type': 'captions',
        'annotations': []
    }
    cap_id = 0
    for audio_id, p, gt in zip(ids, predictions, ground_truths):
        p = p[0] if isinstance(p, list) else p
        pred.append({
            'audio_id': audio_id,
            'caption': p
        })

        ref['audio samples'].append({
            'id': audio_id
        })

        for cap in gt:
            ref['annotations'].append({
                'audio_id': audio_id,
                'id': cap_id,
                'caption': cap
            })
            cap_id += 1

    return pred, ref




class CocoTokenizer:
    def __init__(self, preds_str, trues_str):
        self.evalAudios = []
        self.eval = {}
        self.audioToEval = {}

        pred, ref = reformat_to_coco(preds_str, trues_str)

        tmp_dir = Path('tmp')

        if not tmp_dir.is_dir():
            tmp_dir.mkdir()

        self.ref_file = tmp_dir.joinpath('ref.json')
        self.pred_file = tmp_dir.joinpath('pred.json')

        write_json(ref, self.ref_file)
        write_json(pred, self.pred_file)

        self.coco = COCO(str(self.ref_file))
        self.cocoRes = self.coco.loadRes(str(self.pred_file))

        self.params = {'audio_id': self.coco.getAudioIds()}

    def __del__(self):
        # Delete temporary files
        self.ref_file.unlink()
        self.pred_file.unlink()

    def tokenize(self):
        audioIds = self.params['audio_id']
        # audioIds = self.coco.getAudioIds()
        gts = {}
        res = {}
        for audioId in audioIds:
            gts[audioId] = self.coco.audioToAnns[audioId]
            res[audioId] = self.cocoRes.audioToAnns[audioId]

        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
        return res, gts

    def setEval(self, score, method):
        self.eval[method] = score

    def setAudioToEvalAudios(self, scores, audioIds, method):
        for audioId, score in zip(audioIds, scores):
            if not audioId in self.audioToEval:
                self.audioToEval[audioId] = {}
                self.audioToEval[audioId]["audio_id"] = audioId
            self.audioToEval[audioId][method] = score

    def setEvalAudios(self):
        self.evalAudios = [eval for audioId, eval in self.audioToEval.items()]

