# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Inference code """

import os
import logging
import time
import copy
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm

from metrics import glue_compute_metrics as compute_metrics
from finetune_utils import load_and_cache_examples, AverageMeter


logger = logging.getLogger(__name__)


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        # Time Config
        batch_time = AverageMeter('Batch_time', ':6.4f')
        dataload_time = AverageMeter('Dataload_time', ':6.4f')
        torch.cuda.synchronize()
        end = time.perf_counter()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):

            # measure data loading time
            torch.cuda.synchronize()
            curr_time_dataload = time.perf_counter()
            dataload_time.update(curr_time_dataload - end)

            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            
            # measure elapsed time
            torch.cuda.synchronize()
            curr_time_batch = time.perf_counter()
            batch_time.update(curr_time_batch - end)
            torch.cuda.synchronize()
            end = time.perf_counter()

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
            print(preds)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    logger.info(f"Total Time on Testing Dataset: {batch_time.sum} seconds")
    logger.info(f"Total Dataload Time on Testing Dataset: {dataload_time.sum} seconds")
    logger.info(f"Eval Avg Loss: {eval_loss}")
    return results


def make_PTSQ_model(args, backend, model, train_dataloader):
    """Post-training Static Quantization for efficient inferencing
    model: this input is trained
    """

    #backend = 'qnnpack'#"fbgemm"
    ptsq_model = copy.deepcopy(model)
    ptsq_model.eval()

    # prepare (inserting observer modules to record the data for quantizing activations)
    torch.backends.quantized.engine = backend
    ptsq_model.qconfig = torch.quantization.get_default_qconfig(backend)

    # Custom layers
    class CustomLayerNorm(nn.Module):
        def __init__(self, layernorm):
            super().__init__()
            self.quant = torch.ao.quantization.QuantStub()
            self.layernorm = layernorm
        
        def forward(self, input):
            x = self.quant(input)
            return self.layernorm(x)
    
    class CustomElectraSelfAttention(nn.Module):
        def __init__(self, electra_self_attention):
            super().__init__()
            self.quant = torch.ao.quantization.QuantStub()
            self.dequant = torch.ao.quantization.DeQuantStub()
            self.electra_self_attention = electra_self_attention
        
        def forward(self, 
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
        ):
            x = self.dequant(hidden_states)
            x = self.electra_self_attention(
                hidden_states=x, 
                attention_mask=attention_mask, 
                head_mask=head_mask, 
                encoder_hidden_states=encoder_hidden_states, 
                encoder_attention_mask=encoder_attention_mask, 
                past_key_value=past_key_value, 
                output_attentions=output_attentions, 
            )
            x = self.quant(x)
            return x

    ptsq_model.electra.embeddings.LayerNorm = CustomLayerNorm(ptsq_model.electra.embeddings.LayerNorm)
    for i in range(12):
        ptsq_model.electra.encoder.layer[i].attention.self = CustomElectraSelfAttention(ptsq_model.electra.encoder.layer[i].attention.self)
    
    # Specify where not to quantize
    attention_names = [f"electra.encoder.layer.{x}.attention.self" for x in range(12)]
    for name, mod in ptsq_model.named_modules():
        if isinstance(mod, torch.nn.Embedding):
            #mod.qconfig = torch.ao.quantization.float_qparams_weight_only_qconfig
            mod.qconfig = None
        if name in attention_names:
            mod.qconfig = None

    torch.quantization.prepare(ptsq_model, inplace=True)

    # calibrate
    with torch.inference_mode():
        for batch in train_dataloader:
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            inputs["token_type_ids"] = (
                batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
            )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = ptsq_model(**inputs)
            loss = outputs[0]

    # convert
    torch.quantization.convert(ptsq_model, inplace=True)

    # check
    print(ptsq_model)
    return ptsq_model
