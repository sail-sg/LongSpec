import collections
import json
import os
import re
from typing import Dict, Any, List, Tuple, Union, Callable
import numpy as np
from omegaconf import ListConfig
import vllm
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from scripts.apps.utils_execute import check_correctness as apps_check_correctness
from tqdm import tqdm

from general_util.logger import get_child_logger

logger = get_child_logger(__name__)

eval_func: Callable = None


def _mp_init_(_eval_func: Callable):
    global eval_func
    eval_func = _eval_func


def _eval_worker(_input):
    i, test_cases, response = _input
    if response is None:
        return i, [[False] * len(test_cases["inputs"]) if test_cases else 1], False
    full_res = eval_func(test_cases, response)
    full_res = [bool(tmp) if (not isinstance(tmp, bool)) and (not isinstance(tmp, int)) else tmp for tmp in full_res]
    res = all(item is True for item in full_res) is True
    return i, full_res, res


class APPsEvaluator:
    def __init__(self, ):
        pass

    def __call__(self, predictions, num_workers: int = 16):
        success = 0
        success_at_k = 0
        if "difficulty" in predictions[0]:
            all_difficulties = list(set([item["difficulty"] for item in predictions]))
            successes_at_difficulty = {difficulty: 0 for difficulty in all_difficulties}
            successes_at_k_at_difficulty = {difficulty: 0 for difficulty in all_difficulties}
            all_difficulties = {difficulty: 0 for difficulty in all_difficulties}
        else:
            successes_at_difficulty = None
            successes_at_k_at_difficulty = None
            all_difficulties = None

        evaluator = partial(apps_check_correctness, timeout=10, debug=False)
        # Multiprocessing
        _mp_inputs = []
        for i, item in enumerate(predictions):
            if item["test_cases"]:
                if isinstance(item["pred"], list):
                    preds = item["pred"]
                else:
                    preds = [item["pred"]]

                item["full_res"] = [[] for _ in range(len(preds))]
                item["res"] = [False for _ in range(len(preds))]

                for j, pred in enumerate(preds):
                    _mp_inputs.append(((i, j), item["test_cases"], pred))

        pbar = tqdm(_mp_inputs, total=len(_mp_inputs), desc="Evaluating", dynamic_ncols=True)

        if len(_mp_inputs) > 0:
            # _cache_fw = open(self.output_file.replace(".json", ".cache.jsonl"), "w")

            outputs = collections.defaultdict(dict)
            with ThreadPoolExecutor(max_workers=num_workers, initializer=_mp_init_, initargs=(evaluator,)) as executor:
                futures = []
                for _input in pbar:
                    future = executor.submit(_eval_worker, _input)
                    futures.append(future)
                    pbar.update()

                for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting results"):
                    idx, full_res, res = future.result()
                    outputs[idx[0]][idx[1]] = {
                        "res": res,
                        "full_res": full_res
                    }

            for i, item in enumerate(predictions):
                if item["test_cases"]:
                    if isinstance(item["pred"], list):
                        preds = item["pred"]
                    else:
                        preds = [item["pred"]]
                    item["full_res"] = []
                    item["res"] = []

                    for j, pred in enumerate(preds):
                        item["full_res"].append(outputs[i][j]["full_res"])
                        item["res"].append(outputs[i][j]["res"])

                    if all_difficulties is not None:
                        all_difficulties[item["difficulty"]] += 1

                    if any(item["res"]):
                        success_at_k += 1
                        if all_difficulties is not None:
                            successes_at_k_at_difficulty[item["difficulty"]] += 1
                    if item["res"][0]:
                        success += 1
                        if all_difficulties is not None:
                            successes_at_difficulty[item["difficulty"]] += 1

                    if len(item["res"]) == 1:
                        item["res"] = item["res"][0]
                        item["full_res"] = item["full_res"][0]

                else:
                    item["res"] = []
                    item["full_res"] = []

                # _cache_fw.write(json.dumps(item, ensure_ascii=False) + "\n")
            # _cache_fw.close()
        if len(predictions) == 0:
            metrics = {"acc": 0, "pass@k": 0, "correct": 0, "total": 0}
        else:
            metrics = {"acc": success / len(predictions), "pass@k": success_at_k / len(predictions), "correct": success,
                       "total": len(predictions)}
            if all_difficulties is not None:
                for difficulty in all_difficulties:
                    if all_difficulties[difficulty] > 0:
                        metrics[f"acc_{difficulty}"] = successes_at_difficulty[difficulty] / all_difficulties[difficulty]
                        metrics[f"pass@k_{difficulty}"] = successes_at_k_at_difficulty[difficulty] / all_difficulties[difficulty]
                    else:
                        metrics[f"acc_{difficulty}"] = 0.
                        metrics[f"pass@k_{difficulty}"] = 0.
                    metrics[f"correct_{difficulty}"] = successes_at_difficulty[difficulty]
                    metrics[f"total_{difficulty}"] = all_difficulties[difficulty]
        return predictions, metrics


class CodeExtractor:
    def __init__(self, output_file: str, answer_clean: Callable, resume: bool = False,
                 index_field: str = "index", test_case_field: str = "input_output", evaluator: Callable = None, num_workers: int = 8,
                 saved_keys: List[str] = None, completion_separator: str = None):
        self.predictions = []
        self.output_file = output_file
        self.answer_clean = answer_clean
        self.index_field = index_field
        self.test_case_field = test_case_field
        self.evaluator = evaluator
        self.num_workers = num_workers
        self.saved_keys = saved_keys
        if isinstance(self.saved_keys, ListConfig):
            self.saved_keys = list(self.saved_keys)
        self.completion_separator = completion_separator

        logging_file = output_file.replace(".json", ".jsonl")
        save_dir = os.path.dirname(logging_file)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        if os.path.exists(logging_file):
            if resume:
                with open(logging_file, "r", encoding="utf-8") as f:
                    for line in f.readlines():
                        item = json.loads(line)
                        if isinstance(item["response"], str):
                            if item["response"].strip() == "":
                                continue
                        elif isinstance(item["response"], list):
                            if any([tmp.strip() == "" for tmp in item["response"]]):
                                continue
                        self.predictions.append(item)
                logger.info(f"Load {len(self.predictions)} from {logging_file}")
            self.fw = open(logging_file, "a", encoding="utf-8")
        else:
            self.fw = open(logging_file, "w", encoding="utf-8")

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], **kwargs):
        text = meta_data["text"]
        if self.test_case_field in meta_data:
            test_cases = meta_data[self.test_case_field]
        else:
            test_cases = None
        index = meta_data[self.index_field]

        response = batch_model_outputs["response"]
        if isinstance(response, vllm.RequestOutput):
            if response.finished:
                response = [o.text for o in response.outputs]
                if len(response) == 1:
                    response = response[0]
            else:
                response = ""
        if isinstance(response, str):
            if self.completion_separator:
                pred_clean = self.answer_clean((text + response).split(self.completion_separator)[1])
            else:
                pred_clean = self.answer_clean(response)
        elif isinstance(response, list):
            if self.completion_separator:
                pred_clean = [self.answer_clean((text + item).split(self.completion_separator)[1]) for item in response]
            else:
                pred_clean = [self.answer_clean(item) for item in response]
        else:
            raise ValueError(f"Unknown type of response: {type(response)}")

        out_item = {
            "text": text,
            "test_cases": test_cases,
            "response": response,
            "pred": pred_clean,
            "id": index,
        }
        if self.saved_keys is not None:
            for key in self.saved_keys:
                out_item[key] = meta_data[key]
        self.predictions.append(out_item)
        self.fw.write(json.dumps(self.predictions[-1], ensure_ascii=False) + "\n")
        self.fw.flush()

    def eval_single_response(self, response: str, test_cases):
        if response is None:
            return [[False] * len(test_cases["inputs"]) if test_cases else 1], False
        full_res = self.evaluator(test_cases, response)
        res = all(item is True for item in full_res) is True
        return full_res, res

    def get_results(self):
        save_dir = os.path.dirname(self.output_file)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        self.fw.close()

        # Remove duplicated ids to satisfy the submission requirements of ReClor.
        outputs = sorted(self.predictions, key=lambda x: x["id"])
        id_set = set()
        new_outputs = []
        for item in outputs:
            if item["id"] not in id_set:
                new_outputs.append(item)
                id_set.add(item["id"])
        self.predictions = new_outputs

        # success = 0
        # success_at_k = 0
        # for item in self.predictions:
        #     if item["test_cases"] and self.evaluator is not None:
        #         if isinstance(item["pred"], list):
        #             preds = item["pred"]
        #         else:
        #             preds = [item["pred"]]
        #
        #         full_res, res = [], []
        #         for pred in preds:
        #             _full_res, _res = self.eval_single_response(pred, item["test_cases"])
        #             full_res.append(_full_res)
        #             res.append(_res)
        #         item["full_res"] = full_res
        #         item["res"] = res
        #         if res[0]:
        #             success += 1
        #         if any(res):
        #             success_at_k += 1

        # Multiprocessing
        # _mp_inputs = []
        # for i, item in enumerate(self.predictions):
        #     if item["test_cases"] and self.evaluator is not None:
        #         if isinstance(item["pred"], list):
        #             preds = item["pred"]
        #         else:
        #             preds = [item["pred"]]
        #
        #         item["full_res"] = [[] for _ in range(len(preds))]
        #         item["res"] = [False for _ in range(len(preds))]
        #
        #         for j, pred in enumerate(preds):
        #             _mp_inputs.append(((i, j), item["test_cases"], pred))
        #
        # pbar = tqdm(_mp_inputs, total=len(_mp_inputs), desc="Evaluating", dynamic_ncols=True)
        #
        # if len(_mp_inputs) > 0:
        #     _cache_fw = open(self.output_file.replace(".json", ".cache.jsonl"), "w")
        #
        #     outputs = collections.defaultdict(dict)
        #     with ThreadPoolExecutor(max_workers=self.num_workers, initializer=_mp_init_, initargs=(self.evaluator,)) as executor:
        #         futures = []
        #         for _input in pbar:
        #             future = executor.submit(_eval_worker, _input)
        #             futures.append(future)
        #             pbar.update()
        #
        #         for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting results"):
        #             idx, full_res, res = future.result()
        #             # self.predictions[idx[0]]["full_res"][idx[1]] = full_res
        #             # self.predictions[idx[0]]["res"][idx[1]] = res
        #             outputs[idx[0]][idx[1]] = {
        #                 "res": res,
        #                 "full_res": full_res
        #             }
        #
        #     for i, item in enumerate(self.predictions):
        #         if item["test_cases"] and self.evaluator is not None:
        #             if isinstance(item["pred"], list):
        #                 preds = item["pred"]
        #             else:
        #                 preds = [item["pred"]]
        #             item["full_res"] = []
        #             item["res"] = []
        #
        #             for j, pred in enumerate(preds):
        #                 item["full_res"].append(outputs[i][j]["full_res"])
        #                 item["res"].append(outputs[i][j]["res"])
        #
        #             if any(item["res"]):
        #                 success_at_k += 1
        #             if item["res"][0]:
        #                 success += 1
        #
        #             if len(item["res"]) == 1:
        #                 item["res"] = item["res"][0]
        #                 item["full_res"] = item["full_res"][0]
        #         else:
        #             item["res"] = []
        #             item["full_res"] = []
        #
        #         _cache_fw.write(json.dumps(item, ensure_ascii=False) + "\n")
        #
        #     _cache_fw.close()
        #
        #
        # if len(self.predictions) == 0:
        #     metrics = {"acc": 0, "pass@k": 0, "correct": 0, "total": 0}
        # else:
        #     metrics = {"acc": success / len(self.predictions), "pass@k": success_at_k / len(self.predictions), "correct": success,
        #                "total": len(self.predictions)}
        self.predictions, metrics = self.evaluator(self.predictions, self.num_workers)
        json.dump(self.predictions, open(self.output_file, "w", encoding="utf-8"), ensure_ascii=False)
        json.dump(metrics, open(self.output_file.replace(".json", ".metrics.json"), "w"), indent=2)
        return metrics, []
