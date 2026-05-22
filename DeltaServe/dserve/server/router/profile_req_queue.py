import copy
import random
import string
import time
import uuid
import numpy as np
from typing import List, Optional

from dserve.server.router.finetuning_store import FinetuningManager

# Example import if you have a local definition:
# from ..io_struct import Batch, Req
from ..io_struct import Batch, Req
from ..tokenizer import get_tokenizer
from ..input_params import FinetuneParams, SLOParams
from ..sampling_params import SamplingParams

# If using the original time calculation decorator
# from dserve.utils.infer_utils import calculate_time

max_new_tokens_default = 40
inference_requests_length_default = 50


# 50 clients, 3 reqs --> flood the receiver
# 1 client 150 --> smoother
# generate_requests
# 0.3s slower in the co-serving

def get_finetuning_sampling_params() -> SamplingParams:
    """
    Return a 'dummy' sampling params object suitable for fine-tuning requests.
    By default, no sampling or advanced penalties are used.
    """
    sp = SamplingParams(
        do_sample=False,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        ignore_eos=False,     # Typically handle EOS in training
        max_new_tokens=1,   # Arbitrary placeholder
        stop_sequences=[]
    )
    return sp


def get_inference_sampling_params(max_new_tokens) -> SamplingParams:
    """
    Return a 'dummy' sampling params object suitable for fine-tuning requests.
    By default, no sampling or advanced penalties are used.
    """
    sp = SamplingParams(
        do_sample=False,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        ignore_eos=True,     # Typically handle EOS in training
        max_new_tokens=max_new_tokens,   # Arbitrary placeholder
        stop_sequences=[]
    )
    return sp


def generate_random_sentence(length: int) -> str:
    """Generate a random sentence of `length` words."""
    words = []
    for _ in range(max(1, length)):
        word = "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
        words.append(word)
    return " ".join(words).capitalize() + "."

def generate_inference_req(adapter_dir: str, length: int, max_new_tokens: int, tokenizer):
    prompt_text = generate_random_sentence(length)
    prompt_ids = tokenizer(prompt_text).get('input_ids', [])
    req  = Req(
        adapter_dir=adapter_dir,                
        request_id=uuid.uuid4().hex,     
        prompt_ids=prompt_ids,
        sample_params=get_inference_sampling_params(max_new_tokens),
        is_finetuning=False,  
        needs_to_notify_detokenize = True,           
        text=prompt_text,  
    )
    return req

def generate_three_inference_req(adapter_dir: str, length: int, max_new_tokens: int, tokenizer):
    prompt_text = generate_random_sentence(length)
    prompt_ids = tokenizer(prompt_text).get('input_ids', [])
    req_1 = Req(
        adapter_dir=adapter_dir,                
        request_id=uuid.uuid4().hex,     
        prompt_ids=prompt_ids,
        sample_params=get_inference_sampling_params(max_new_tokens),
        is_finetuning=False,     
        needs_to_notify_detokenize = True,         
        text=prompt_text,  
    )
    req_2 = Req(
        adapter_dir=adapter_dir,                
        request_id=uuid.uuid4().hex,     
        prompt_ids=copy.deepcopy(prompt_ids),
        sample_params=get_inference_sampling_params(max_new_tokens),
        is_finetuning=False,              
        needs_to_notify_detokenize = True,
        text=copy.deepcopy(prompt_text),  
    )
    req_3 = Req(
        adapter_dir=adapter_dir,                
        request_id=uuid.uuid4().hex,     
        prompt_ids=copy.deepcopy(prompt_ids),
        sample_params=get_inference_sampling_params(max_new_tokens),
        is_finetuning=False,              
        needs_to_notify_detokenize = True,
        text=copy.deepcopy(prompt_text),  
    )
    return req_1, req_2, req_3


class Profile_ReqQueue:
    """
    A queue that handles both inference requests and fine-tuning requests.
    Key differences from ReqQueue:
    - We store inference requests in `waiting_req_list`.
    - We store fine-tuning requests in `finetuning_req_list`.
    - We only add fine-tuning requests if (and only if) there are no inference
      requests waiting.
    """

    def __init__(self,
                 max_total_tokens: int,
                 batch_max_tokens: int,
                 running_max_req_size: int,
                 finetune_params: FinetuneParams,
                 slo_params: SLOParams,
                 inference_adapter_dir: str):
        self.max_total_tokens = max_total_tokens #1024
        self.batch_max_tokens = batch_max_tokens
        self.running_max_req_size = running_max_req_size
        self.inference_adapter_dir = inference_adapter_dir
        # config parameters
        self.finetuning_data_path = finetune_params.finetuning_data_path
        self.finetuning_prepare_size = finetune_params.finetuning_prepare_size
        self.finetuning_lora_path = finetune_params.finetuning_lora_path  
        self.max_saved_finetuning_tokens = finetune_params.max_saved_finetuning_tokens  #max size of saved activations in memory
        self.total_epoch = finetune_params.num_epochs
        self.start_task= finetune_params.start_on_launch
        self.ttft_slo = slo_params.ttft_slo
        self.avg_tbt_slo = slo_params.avg_tbt_slo
        self.max_tbt_slo = slo_params.max_tbt_slo
        print(f"\033[34m[Forward Batch Constructor]: ttft_slo={self.ttft_slo}, avg_tbt_slo={self.avg_tbt_slo}, max_tbt_slo={self.max_tbt_slo}\033[0m")

        try: 
            self.tokenizer = get_tokenizer(finetune_params.model_weightdir, 
                                           finetune_params.tokenizor_mode, 
                                           trust_remote_code=finetune_params.trust_remote_code) 
        except:
            print("Could not load tokenizer. Using default.")
            self.tokenizer = get_tokenizer("huggyllama/llama-7b", finetune_params.tokenizor_mode) 
        
        self.finetuning_manager = FinetuningManager(
            data_path=self.finetuning_data_path,
            tokenizer=self.tokenizer,
            adapter_dir=self.finetuning_lora_path,
            total_epochs=self.total_epoch,
            max_prepare=self.finetuning_prepare_size,
            trust_remote_code=finetune_params.trust_remote_code,
            max_saved_finetuning_tokens=self.max_saved_finetuning_tokens
        )
        self.finetuning_manager.load()
        self.waiting_req_list: List[Req] = []
        self.cache_len_list = []
        self.adapters = set()
        self.adapter_size = 0
        #self.prepare_finetuning_requests()
        self.prefill_estimator = None
        self.decode_estimator = None
        self.warmup_request_list = self.prepare_warmup_requests()
        self.first_wave_inf, self.second_wave_inf, self.third_wave_inf = self.prepare_inference_requests()
        self.first_wave_decoding_times = []
        self.second_wave_decoding_times = []
        self.finished = False

    def append(self, req: Req):
        self.waiting_req_list.append(req)

    def _init_cache_list(self, current_batch: Optional[Batch], lora_ranks: dict[str, int]):
        if current_batch is not None:
            self.cache_len_list = []
            self.adapters = set()
            self.adapter_size = 0
            for req in current_batch.reqs:
                used_len = req.input_len + len(req.output_ids)
                left_len = req.max_output_len - len(req.output_ids) - 1
                self.cache_len_list.append((used_len, left_len))

                if req.adapter_dir not in self.adapters and req.adapter_dir is not None:
                    self.adapter_size += lora_ranks[req.adapter_dir] * 4
                    self.adapters.add(req.adapter_dir)
        else:
            self.cache_len_list = []
            self.adapters = set()
            self.adapter_size = 0
    
    def prepare_warmup_requests(self):
        warmup_request_list = []
        for _ in range(5):
            req = generate_inference_req(
                adapter_dir=self.inference_adapter_dir,
                length=10,
                max_new_tokens=10,
                tokenizer=self.tokenizer
            )
            warmup_request_list.append(req)
        return warmup_request_list
    
    def prepare_inference_requests(self):
        inference_only_requests_list = []
        mixed_requests_list = []
        bwd_requests_list = []
        for _ in range(3):
            req_1, req_2, req_3 = generate_three_inference_req(
                adapter_dir=self.inference_adapter_dir,
                length=inference_requests_length_default,
                max_new_tokens=max_new_tokens_default,
                tokenizer=self.tokenizer
            )
            inference_only_requests_list.append(req_1)
            mixed_requests_list.append(req_2)
            bwd_requests_list.append(req_3)
        return inference_only_requests_list, mixed_requests_list, bwd_requests_list
    
    async def check_will_starve(self, current_batch) -> bool:
        return False


    def _can_add_new_req(self, req: Req, lora_ranks: dict[str, int]) -> bool:
        self.cache_len_list.append((req.input_len + 1, req.max_output_len - 1))
        self.cache_len_list.sort(key=lambda x: -x[1])
    
        if req.adapter_dir not in self.adapters and req.adapter_dir is not None:
            self.adapter_size += lora_ranks[req.adapter_dir] * 4
            self.adapters.add(req.adapter_dir)
        
        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        has_run_len_array  = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array  = np.cumsum(has_run_len_array)
        size_array         = np.arange(1, len(self.cache_len_list) + 1)

        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()

        if (need_max_token_num < (self.max_total_tokens - self.adapter_size) and
            len(self.cache_len_list) <= self.running_max_req_size):
            return True
        else:
            return False
    

    def generate_new_batch(self, current_batch: Optional[Batch], lora_ranks: dict[str, int], is_backward_running: bool) -> Optional[Batch]:
        if current_batch is not None:
            return None
        
        self._init_cache_list(current_batch, lora_ranks)
        new_batch_total_tokens = 0
        can_run_list = []
        aborted_count = 0
        if len(self.warmup_request_list) > 0:
            for req in self.warmup_request_list:
                if req.aborted:
                    print("Request aborted")
                    aborted_count += 1
                    continue
                req.arrival_time = time.time()
                if (self._can_add_new_req(req, lora_ranks) and
                    (new_batch_total_tokens + req.input_len) <= self.batch_max_tokens):
                    can_run_list.append(req)
                    new_batch_total_tokens += req.input_len
                else:
                    break
            if len(can_run_list) > 0:
                self.warmup_request_list = self.warmup_request_list[len(can_run_list) + aborted_count:]
            self.print_new_batch_info(can_run_list)
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            return new_batch
        if len(self.first_wave_inf) > 0:
            for req in self.first_wave_inf:
                if req.aborted:
                    print("Request aborted")
                    aborted_count += 1
                    continue
                req.arrival_time = time.time()
                if (self._can_add_new_req(req, lora_ranks) and
                    (new_batch_total_tokens + req.input_len) <= self.batch_max_tokens):
                    can_run_list.append(req)
                    new_batch_total_tokens += req.input_len
                else:
                    break
            if len(can_run_list) > 0:
                self.first_wave_inf = self.first_wave_inf[len(can_run_list) + aborted_count:]
            self.add_finetuning_req(new_batch_total_tokens, can_run_list, lora_ranks)
            self.print_new_batch_info(can_run_list)
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            print(f"\033[32mFirst Wave Inference Batch Created with {len(can_run_list)} requests\033[0m")
            return new_batch
        elif len(self.second_wave_inf) > 0:
            for req in self.second_wave_inf:
                if req.aborted:
                    print("Request aborted")
                    aborted_count += 1
                    continue
                req.arrival_time = time.time()
                if (self._can_add_new_req(req, lora_ranks) and
                    (new_batch_total_tokens + req.input_len) <= self.batch_max_tokens):
                    can_run_list.append(req)
                    new_batch_total_tokens += req.input_len
                else:
                    break
            if len(can_run_list) > 0:
                self.second_wave_inf = self.second_wave_inf[len(can_run_list) + aborted_count:]
            #self.add_finetuning_req(new_batch_total_tokens, can_run_list, lora_ranks)

            self.print_new_batch_info(can_run_list)
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            print(f"\033[32mSecond Wave Mixed Batch Created with {len(can_run_list)} requests\033[0m")
            return new_batch
        else:
            if self.finished == False:
                self.finished = True
                first_wave_avg_decode_time = np.mean(self.first_wave_decoding_times) if len(self.first_wave_decoding_times) > 0 else 0.0
                second_wave_avg_decode_time = np.mean(self.second_wave_decoding_times) if len(self.second_wave_decoding_times) > 0 else 0.0
                print(f"\033[34m[Profiling Result] First Wave Avg Decode Time: {first_wave_avg_decode_time:.4f}s, Second Wave Avg Decode Time: {second_wave_avg_decode_time:.4f}s \033[0m")
            return None

    def add_finetuning_req(self, new_batch_total_tokens, can_run_list, lora_ranks):
        ft_list = []
        ft_tokens = 0
        while self.finetuning_manager.has_next():
            ft_req = self.finetuning_manager.pop_next()
            if ft_req is None:
                break
            elif not self._can_add_new_req(ft_req, lora_ranks):
                break
            elif new_batch_total_tokens + ft_req.input_len > self.batch_max_tokens:
                break
            elif self.finetuning_manager.pending_bwd_tokens + ft_tokens  + ft_req.input_len > self.max_saved_finetuning_tokens:
                break
            else:
                ft_list.append(ft_req)
                new_batch_total_tokens += ft_req.input_len
                ft_tokens += ft_req.input_len
        can_run_list.extend(ft_list)

    
    def print_new_batch_info(self, can_run_list):
        if len(can_run_list) > 0:
            infer_tokens = 0
            finetune_tokens = 0
            for req in can_run_list:
                if req.is_finetuning:
                    finetune_tokens += req.input_len
                else:
                    infer_tokens += req.input_len
            unused = self.batch_max_tokens - (infer_tokens + finetune_tokens)
            print(f"\033[34mForward Batch Token Layout: [{infer_tokens} infer tokens/ {finetune_tokens} finetune / {unused} unused] \033[0m")
            print(f"\033[34mPending BWD token count: {self.finetuning_manager.pending_bwd_tokens}\033[0m")

    def get_req_timestamps(self, can_run_list):
        out = []
        for req in can_run_list:
            out.append(req.arrival_time)
        return out
    
    def add_to_decode_time_queue(self, decode_time: float):
        if len(self.second_wave_inf) > 0:
            self.first_wave_decoding_times.append(decode_time)
        else:
            self.second_wave_decoding_times.append(decode_time)


    def get_earliest_req_time(self):
        if len(self.waiting_req_list) == 0:
            return time.time()
        return self.waiting_req_list[0].arrival_time
    
    def ready_for_bwd(self):
        return self.finetuning_manager.ready_for_bwd()

    def set_estimators(self, prefill_estimator, decode_estimator):
        self.prefill_estimator = prefill_estimator
        self.decode_estimator = decode_estimator

    def update_finetuning_status_after_fwd(self, batch: Batch):
        return self.finetuning_manager.update_finetuning_status_after_fwd(batch)

    def update_finetuning_status_after_bwd(self, loss_list, num_processed_tokens):
        return self.finetuning_manager.update_finetuning_status_after_bwd(loss_list, num_processed_tokens)

    def finetuning_is_finished(self):
        if self.start_task == False:
            return True
        return self.finetuning_manager.finetuning_is_finished()        

    def update_counter(self, req: Req):
        pass