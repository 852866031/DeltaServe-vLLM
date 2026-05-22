from .sampling_params import SamplingParams
from typing import Dict, List, Optional, Tuple
import asyncio

class Req:
    def __init__(self, adapter_dir, request_id, prompt_ids, sample_params: SamplingParams,
                 is_finetuning=False, text=None,
                 needs_to_notify_detokenize=False):
        self.adapter_dir = adapter_dir
        self.request_id = request_id
        self.prompt_ids = prompt_ids
        self.input_len = len(prompt_ids)
        self.max_output_len = sample_params.max_new_tokens
        self.sample_params = sample_params
        self.output_ids = []
        self.output_metadata_list = []
        self.has_generate_finished = False
        self.aborted = False
        self.is_finetuning = is_finetuning
        self.ask_for_label = False
        self.needs_to_notify_detokenize = needs_to_notify_detokenize
        self.text = text
        self.arrival_time = None
        self.time_between_tokens = []
        self.last_token_time = None
        self.time_to_first_token = None
    
    def avg_tbt(self):
        if len(self.time_between_tokens) == 0:
            return 0.0
        return sum(self.time_between_tokens) / len(self.time_between_tokens)

    def worst_tbt(self):
        if len(self.time_between_tokens) == 0:
            return 0.0
        return max(self.time_between_tokens)
    
    def export_perf_metrics(self):
        return (self.time_to_first_token, self.avg_tbt(), self.worst_tbt())
    
    def avg_tbt_if_next_token(self, time):
        return (sum(self.time_between_tokens) + time) / (len(self.time_between_tokens) + 1)

    def to_rpc_obj(self):
        return {"adapter_dir": self.adapter_dir,
                "request_id": self.request_id,
                "input_id": self.prompt_ids,
                "output_len": self.max_output_len,
                "sampling_param": self.sample_params.to_dict(),
                "is_finetuning": self.is_finetuning,}

    def to_req_detokenization_state(self):
        out = ReqDetokenizationState(self.request_id, self.prompt_ids, self.max_output_len, self.sample_params.ignore_eos)
        if self.output_metadata_list:
            out.gen_metadata.update(self.output_metadata_list[-1])
        return out
    
    def stop_sequences_matched(self):
        if self.sample_params.stop_sequences == None: #TODO: remove this, just for model can run
            return True
        for stop_token_ids in self.sample_params.stop_sequences:
            stop_len = len(stop_token_ids)
            if stop_len > 0:
                if len(self.output_ids) >= stop_len:
                    if all(self.output_ids[-(stop_len - i)] == stop_token_ids[i] for i in range(stop_len)):
                        return True
        return False

    def __repr__(self):
        return (f"request_id(n={self.request_id}, "
                f"adapter_dir={self.adapter_dir}, ")
                # f"prompt_ids={self.prompt_ids}, ")
        

class ReqDetokenizationState:
    def __init__(
        self,
        request_id: str,
        prompt_ids: List[int],
        max_output_len: int,
        ignore_eos: bool,
    ) -> None:
        self.request_id = request_id
        self.prompt_ids = prompt_ids
        self.output_ids = []
        self.output_tokens = []
        self.output_str = ""
        self.sub_texts = []
        self.current_sub_text = []
        self.max_output_len = max_output_len
        self.ignore_eos = ignore_eos
        self.gen_metadata = {}


class Batch:
    def __init__(self, batch_id, reqs: List[Req]):
        self.batch_id = batch_id
        self.reqs = reqs
        self.id_to_reqs = {req.request_id: req for req in reqs}

        self.adapter_dirs = set()
        for req in reqs:
            self.adapter_dirs.add(req.adapter_dir)
        
    def record_token_time(self, decode_end_time):
        for req in self.reqs:
            if not req.is_finetuning:
                req.time_between_tokens.append(decode_end_time - req.last_token_time)
                req.last_token_time = decode_end_time

    def record_time_to_first_token(self, prefill_end_time):
        for req in self.reqs:
            if not req.is_finetuning and req.time_to_first_token is None:
                req.time_to_first_token = prefill_end_time - req.arrival_time
                if req.last_token_time is None:
                    req.last_token_time = prefill_end_time

    def get_req_with_worst_avg_tbt(self):
        worst_req = None
        worst_avg_tbt = -1.0
        for req in self.reqs:
            if not req.is_finetuning:
                avg_tbt = req.avg_tbt()
                if avg_tbt > worst_avg_tbt:
                    worst_avg_tbt = avg_tbt
                    worst_req = req
        return worst_req
  
    def export_batch_info(self):
        inference_tokens_list = []
        finetuning_tokens_list = []
        for req in self.reqs:
            if req.is_finetuning:
                finetuning_tokens_list.append(req.input_len)
            else:
                inference_tokens_list.append(req.input_len)
        return inference_tokens_list, finetuning_tokens_list

    def input_tokens(self):
        batch_input_tokens = 0
        for req in self.reqs:
            batch_input_tokens += req.input_len
        return batch_input_tokens

    def has_inference(self):
        for req in self.reqs:
            if not req.is_finetuning:
                return True
        return False

    def get_inference_token_num(self):
        infer_tokens = 0
        for req in self.reqs:
            if not req.is_finetuning:
                infer_tokens += req.input_len
        return infer_tokens

    def is_coserving_batch(self):
        for req in self.reqs:
            if req.is_finetuning:
                return True
        return False

    def get_earliest_arrival_time(self):
        earliest_time = None
        for req in self.reqs:
            if req.arrival_time is not None:
                if earliest_time is None or req.arrival_time < earliest_time:
                    earliest_time = req.arrival_time
        return earliest_time

    def calcu_max_tokens(self):
        tokens = 0
        for req in self.reqs:
            tokens += req.input_len + req.max_output_len
        return tokens
    
    def calcu_used_tokens(self):
        tokens = 0
        for req in self.reqs:
            tokens += req.input_len + len(req.output_ids)
        return tokens

    def mark_finished_req(self, eos_id):
        from .router.mixed_req_queue import rprint
        has_new_finish = False
        count = 0
        for req in self.reqs:
            if req.stop_sequences_matched():
                req.has_generate_finished = True
                has_new_finish = True
                count += 1
            elif req.is_finetuning:
                req.has_generate_finished = True
                has_new_finish = True
                count += 1
            elif len(req.output_ids) > 0 and req.output_ids[-1] == eos_id and req.sample_params.ignore_eos == False:
                req.has_generate_finished = True
                has_new_finish = True
                count += 1
            elif len(req.output_ids) >= req.max_output_len or req.aborted:
                req.has_generate_finished = True
                has_new_finish = True
                count += 1
         
        return has_new_finish

    def filter_finished(self):
        unfinished_req = []
        for req in self.reqs:
            if not req.has_generate_finished:
                unfinished_req.append(req)
        self.reqs = unfinished_req
        self.id_to_reqs = {req.request_id: req for req in self.reqs}

        self.adapter_dirs = set()
        for req in self.reqs:
            self.adapter_dirs.add(req.adapter_dir)

    def is_clear(self):
        return len(self.reqs) == 0

    def merge(self, mini_batch):
        for _req in mini_batch.reqs:
            self.reqs.append(_req)
            self.adapter_dirs.add(_req.adapter_dir)
        self.id_to_reqs = {req.request_id: req for req in self.reqs}
        return

    def __repr__(self):
        return (f"batch_id={self.batch_id}, "
                # f"reqs={self.reqs}, "
                f"req_ids={self.id_to_reqs.keys()}")
        
class BatchTokenIdOut:
    def __init__(self):
        self.reqs_infs: List[Tuple[str, int, Dict, bool, bool, Tuple[float, float, float]]] = []  
        # [req_id, new_token_id, gen_metadata, finished_state, abort_state, perf_metrics]

class BatchStrOut:
    def __init__(self):
        self.reqs_infs: List[Tuple[str, str, Dict, bool, bool, Tuple[float, float, float]]] = [] 
        # [req_id, token_str, gen_metadata, finished_state, abort_state, perf_metrics]
        
class AbortReq:
    def __init__(self, req_id):
        self.req_id = req_id

class FinetuneReq:
    def __init__(self, exit_finetuning=False):
        self.exit_finetuning = exit_finetuning

class FinetuneStatusReq:
    def __init__(self):
        self.finished = False


class BatchAbortReq:
    def __init__(self, req_ids):
        self.reqs: List[str] = req_ids
