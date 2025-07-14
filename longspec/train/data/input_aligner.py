import copy
import json
import os.path
import random
from glob import glob
from typing import Dict, List, Callable, Union

from omegaconf.listconfig import ListConfig
from tqdm import tqdm

from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


def _format_option_list(option_list: List[str], _rank2option: List[str]) -> str:
    res = []
    for op_id, op in enumerate(option_list):
        res.append(f"{_rank2option[op_id]}. {op}")
    return "\n".join(res)


def option_id2str_aligner():
    option_id2str = ["A", "B", "C", "D", "E"]

    def func(data: List[Dict]):
        for sample in data:
            sample["str_label"] = option_id2str[sample["label"]]
        return data

    return func


def field_extract_aligner(input_index_field: str, extract_index_field: str, extract_fields: List[str], extra_file: str, renamed_fields: Dict[str, str] = None):
    if os.path.exists(extra_file):
        extra_data = json.load(open(extra_file, encoding="utf-8"))
    else:
        extra_data = []
        for file in glob(extra_file):
            extra_data += json.load(open(file))
        if len(extra_data) == 0:
            raise ValueError(f"No data found in {extra_file}")
    id2extra_data = {item[extract_index_field]: item for item in extra_data}
    renaming = {}
    for _field in extract_fields:
        if renamed_fields and _field in renamed_fields:
            renaming[_field] = renamed_fields[_field]
        else:
            renaming[_field] = _field

    def func(data: List[Dict]):
        missing = 0
        missing_field = 0
        outputs = []
        for item in data:
            item_id = item[input_index_field]
            if item_id not in id2extra_data:
                missing += 1
                continue
            extra_item = id2extra_data[item_id]
            if any(x not in extra_item for x in extract_fields):
                missing_field += 1
                continue
            for field in extract_fields:
                item[renaming[field]] = extra_item[field]
            outputs.append(item)

        logger.info(f"Extracted {len(outputs)} items from {extra_file}")
        logger.info(f"Missing {missing} items in {extra_file}")
        logger.info(f"Missing {missing_field} fields in {extra_file}")

        return outputs

    return func


def flat_aligner(input_index_field: str, extract_field: Union[str, List[str]], mode: str = "single"):
    if isinstance(extract_field, str):
        extract_field = [extract_field]

    def func(data: List[Dict]):
        outputs = []
        for item in data:
            item_id = item[input_index_field]
            # if not all(item[_field] for _field in extract_field):
            #     continue
            if any(item[_field] in [None, "", []] for _field in extract_field):
                continue

            num = len(item[extract_field[0]])
            for _field in extract_field[1:]:
                assert len(item[_field]) == num, f"Length not match: {item[_field]}"

            for i in range(num):
                new_item = copy.deepcopy(item)
                # FIXME: This could introduce a bug when the field is of type `int` and the value is `0`.
                # if any(not item[_field][i] for _field in extract_field):
                #     continue
                if any(item[_field][i] in [None, "", []] for _field in extract_field):
                    continue
                for _field in extract_field:
                    new_item[_field] = item[_field][i]
                new_item[input_index_field] = f"{item_id}_{i}"
                outputs.append(new_item)
                if mode == "single":
                    break
        return outputs

    return func


def option_flatten_aligner():
    def func(data: List[Dict]):
        for sample in data:
            sample["option_list"] = _format_option_list(sample["options"], ["A", "B", "C", "D"])
        return data

    return func


def empty_aligner(data: List[Dict]):
    return data


def add_id_aligner(id_field: str = "id"):
    def func(data: List[Dict]):
        for i, item in enumerate(data):
            item[id_field] = i
        return data

    return func


def concat_aligner(aligners: List[Callable]):
    def func(data: List[Dict]):
        for aligner in aligners:
            data = aligner(data)
        return data

    return func


def sharegpt_aligner(id_field: str = "id"):
    def func(data: List[Dict]):
        conversations = []
        for i, source in enumerate(data):
            if source["conversations"] and source["conversations"][0]["from"] == "gpt":
                # Skip if GPT is first to talk
                source["conversations"] = source["conversations"][1:]
            new_source = []
            for item in source["conversations"]:
                role = "assistant" if item["from"] == "gpt" else "user"
                content = item["value"]
                new_source.append({"role": role, "content": content})
            if len(new_source) > 0:
                conversations.append({
                    id_field: i,
                    "conversations": new_source,
                })
        return conversations
    
    return func


def dpo_pair_aligner_cleaned(response_field: str = "response",
                             id_field: str = "id",
                             do_sample: bool = False, ):
    """
    This aligner only accepts the cleaned file, which has removing all empty responses and combined with original data.
    :return: Callable
    """

    def func(data: List[Dict]):
        outputs = []
        for item in data:
            pos_resp = []
            neg_resp = []
            for i, (resp, pred) in enumerate(zip(item[response_field], item["pred"])):
                assert resp
                # assert pred
                if isinstance(resp, list):
                    assert isinstance(resp[0], str)
                    # assert "The answer is" in resp[-1], resp
                    resp = "".join(resp)

                if isinstance(item["label"], str):
                    if pred == item["label"]:
                        pos_resp.append((i, resp))
                    else:
                        neg_resp.append((i, resp))
                elif isinstance(item["label"], int):
                    if pred and ord(pred) - ord("A") == item["label"]:
                        pos_resp.append((i, resp))
                    else:
                        neg_resp.append((i, resp))
                else:
                    raise ValueError(f"Unknown type of label: {type(item['label'])}")

            if not (len(pos_resp) and len(neg_resp)):
                continue

            if do_sample:
                pos = random.choice(pos_resp)
                neg = random.choice(neg_resp)
                pos_resp = [pos]
                neg_resp = [neg]

            for pos in pos_resp:
                for neg in neg_resp:
                    new_item = copy.deepcopy(item)
                    new_item["pos"] = pos[1]
                    new_item["neg"] = neg[1]
                    new_item["pos_id"] = f"{item[id_field]}_{pos[0]}"
                    new_item["neg_id"] = f"{item[id_field]}_{neg[0]}"
                    outputs.append(new_item)

        logger.info(f"Counted {len(outputs)} DPO contrastive pairs.")
        return outputs

    return func


def dpo_pair_aligner(pos_field: Union[str, ListConfig], neg_field: Union[str, ListConfig]):
    def func(data: List[Dict]):
        outputs = []
        if isinstance(pos_field, str):
            _pos_fields = [pos_field]
        else:
            _pos_fields = list(pos_field)
        if isinstance(neg_field, str):
            _neg_fields = [neg_field]
        else:
            _neg_fields = list(neg_field)

        for item in tqdm(data, desc="DPO pair aligner", total=len(data)):
            pos_resp = []
            neg_resp = []
            for _field in _pos_fields:
                pos_resp += item[_field]
            for _field in _neg_fields:
                neg_resp += item[_field]

            for pos in pos_resp:
                for neg in neg_resp:
                    new_item = copy.deepcopy(item)

                    # To save memory
                    for _field in _pos_fields:
                        new_item.pop(_field)
                    for _field in _neg_fields:
                        new_item.pop(_field)

                    new_item["pos"] = pos
                    new_item["neg"] = neg
                    outputs.append(new_item)

        logger.info(f"Counted {len(outputs)} DPO contrastive pairs.")
        return outputs

    return func


def eval_multiple_choice(item):
    if isinstance(item, dict):
        pred = item["prediction"]
        label = item["answer"]
    elif isinstance(item, tuple):
        pred, label = item
    else:
        raise ValueError(f"Unknown type of item: {type(item)}")

    if isinstance(label, str):
        if pred == label:
            return True
        return False
    if isinstance(label, int):
        if pred and ord(pred) - ord("A") == label:
            return True
        return False

    raise ValueError(f"Unknown type of label: {type(item['label'])}")


def prompt_fill_aligner(prompt_file: str, mapping: Dict[str, str], prompt_field: str = "prompt"):
    full_prompt = open(prompt_file).read()

    def func(data: List[Dict]):
        for item in data:
            prompt = copy.deepcopy(full_prompt)
            for k, v in mapping.items():
                prompt = prompt.replace(k, item[v])
            item[prompt_field] = prompt

        return data

    return func


def value2pair_aligner(field: str, pos_field: str, neg_field: str, value_field: str):
    def func(data: List[Dict]):
        for item in data:
            pair_data = item.pop(field)
            values = item.pop(value_field)

            pos = []
            neg = []
            for x, v in zip(pair_data, values):
                if v:
                    pos.append(x)
                else:
                    neg.append(x)
            item[pos_field] = pos
            item[neg_field] = neg

        return data

    return func


def dpo_random_choice_aligner(anchor_field: str, paired_field: str):
    def func(data: List[Dict]):
        outputs = []
        for item in tqdm(data, desc="DPO random choice aligner", total=len(data)):
            if len(item[anchor_field]) == 0:
                continue
            if len(item[paired_field]) == 0:
                continue
            for anchor in item[anchor_field]:
                new_item = copy.deepcopy(item)
                new_item[anchor_field] = anchor
                new_item[paired_field] = random.choice(item[paired_field])
                outputs.append(new_item)
        return outputs

    return func
