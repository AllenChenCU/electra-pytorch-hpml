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
""" Utility functions for finetuning and inferencing """

import os
import random
import logging
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from examples.glue.processors import glue_convert_examples_to_features as convert_examples_to_features
from examples.glue.processors import glue_output_modes as output_modes
from examples.glue.processors import glue_processors as processors


logger = logging.getLogger(__name__)


##################################################
# adapters for Google-like GLUE code

class TokenizerAdapter:
    def __init__(self, tokenizer, pad_token, cls_token="[CLS]", sep_token="[SEP]"):
        self.tokenizer = tokenizer
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)


    def truncate_sequences(
        self,
        ids,
        pair_ids,
        num_tokens_to_remove,
        truncation_strategy,
        stride,
    ):

        assert len(ids) > num_tokens_to_remove
        window_len = min(len(ids), stride + num_tokens_to_remove)
        overflowing_tokens = ids[-window_len:]
        ids = ids[:-num_tokens_to_remove]

        return (ids, pair_ids, overflowing_tokens)

    def encode_plus(self, text, text_pair, add_special_tokens, max_length, return_token_type_ids):

        # Tokenization
        token_ids_0 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        len_ids = len(token_ids_0)
        if text_pair:
            token_ids_1 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text_pair))
            len_pair_ids = len(token_ids_1)
        else:
            token_ids_1 = None
            len_pair_ids = 0

 
        # Truncation
        assert add_special_tokens
        num_special_tokens_to_add = (2 if not text_pair else 3)
        total_len = len_ids + len_pair_ids + num_special_tokens_to_add
        if max_length and total_len > max_length:
            token_ids_0, token_ids_1, overflowing_tokens = self.truncate_sequences(
                token_ids_0,
                pair_ids=token_ids_1,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy='only_first', # TODO(nijkamp): is this the correct truncation strategy for all GLUE tasks?
                stride=0,
            )


        # Add special tokens
        cls = [self.tokenizer.vocab[self.cls_token]]
        sep = [self.tokenizer.vocab[self.sep_token]]

        if not text_pair:

            input_ids = cls + token_ids_0 + sep
            token_type_ids = len(cls + token_ids_0 + sep) * [0]

        else:

            input_ids = cls + token_ids_0 + sep + token_ids_1 + sep
            token_type_ids = len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

        assert len(input_ids) <= max_length

        return {"input_ids": input_ids, "token_type_ids": token_type_ids}

    def __len__(self):
        return len(self.tokenizer.vocab)

    def save_pretrained(self, outputdir):
        pass

def wrap_tokenizer(tokenizer, pad_token):
    return TokenizerAdapter(tokenizer, pad_token)


##################################################
# distilled Google-like/HF glue code

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        )
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=False,  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def log_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def log_gpu_memory():
    peak_memory = torch.cuda.max_memory_allocated(device=None)
    curr_memory = torch.cuda.memory_allocated(device=None)
    logger.info(f"Current GPU memory used: {curr_memory}")
    logger.info(f"Peak GPU memory used: {peak_memory}")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}: Batch - {val' + self.fmt + '} | Epoch - {avg' + self.fmt + '} ({sum}/{count}) \n'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']\n'
    
