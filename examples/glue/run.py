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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import argparse
import glob
import json
import logging
import os
import random
import ast

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import AutoConfig, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, prepare_model_for_kbit_training

from pretraining.openwebtext.dataset import new_tokenizer
from metrics import glue_compute_metrics as compute_metrics
from processors import glue_convert_examples_to_features as convert_examples_to_features
from processors import glue_output_modes as output_modes
from processors import glue_processors as processors
from processors import glue_tasks_num_labels as task_num_labels
from finetune_utils import set_seed, wrap_tokenizer, load_and_cache_examples, log_trainable_parameters, log_gpu_memory
from finetune import train
from inference import evaluate


logger = logging.getLogger(__name__)


def main(task='MRPC', seed=42, ckpt='google/electra-small-discriminator'):
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=f'data/glue_data/{task}',
        type=str,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default="bert",
        type=str,
    )
    parser.add_argument(
        "--model_name_or_path",
        default=ckpt,
        type=str,
    )
    parser.add_argument(
        "--vocab_path",
        default='data/vocab.txt',
        type=str,
    )
    parser.add_argument(
        "--task_name",
        default=task,
        type=str,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default='output/glue',
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", default=True, type=ast.literal_eval, help="Whether to run training.")
    parser.add_argument("--do_eval", default=True, type=ast.literal_eval, help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", default=True, help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", default=True, help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", default=True, help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=seed, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--finetune_method", type=str, default="original", help="Finetune methodologies: original, lora, or qlora")
    args = parser.parse_args()

    ###################################################################################################
    # Config
    ###################################################################################################
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)


    ###################################################################################################
    # Prepare
    ###################################################################################################
    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    if args.finetune_method.lower() == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
    else:
        bnb_config = None

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        #cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        #cache_dir=args.cache_dir if args.cache_dir else None,
        quantization_config=bnb_config, 
        #device_map="auto", #{"":0}, 
        torch_dtype=torch.bfloat16, 
    )
    #if args.finetune_method.lower() == "qlora":
        #model.gradient_checkpointing_enable()
        #model = prepare_model_for_kbit_training(model)
        # for name, param in model.named_parameters():
        #     if any(l in name.lower() for l in ["lora", "lokr", "ia3", "base_layer"]):
        #         param.data = param.data.to(torch.float32)
        #         param.requires_grad = True
    if args.finetune_method.lower() in ["lora", "qlora"]:
        for param in model.parameters():
            param.requires_grad = False
            # if param.ndim == 1:
            #     # cast the small parameters (e.g. layernorm) to fp32 for stability
            #     param.data = param.data.to(torch.float32)
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=64, 
            lora_alpha=64, 
            target_modules=["key", "query", "value"], 
            lora_dropout=0.1, 
            bias="none", 
            modules_to_save=["classifier"], 
        )
        model.gradient_checkpointing_enable() # reduce number of stored activations
        model.enable_input_require_grads()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)

    tokenizer = wrap_tokenizer(new_tokenizer(args.vocab_path), pad_token='[PAD]')

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Compute number of trainable parameters
    log_trainable_parameters(model)
    logger.info("Before training: only model is loaded into GPU")
    log_gpu_memory()

    ###################################################################################################
    # Finetune
    ###################################################################################################
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        logger.info("After finetuning: ")
        log_gpu_memory()
        # if args.finetune_method.lower() == "lora":
        #     model = model.merge_and_unload()

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_to_save
        # model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
        # tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model.to(args.device)

    ###################################################################################################
    # Inference
    ###################################################################################################
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        # tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            # model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
            # if args.finetune_method.lower() == "lora":
            #     pretrained_model = AutoModelForSequenceClassification.from_pretrained(
            #         args.model_name_or_path,
            #         from_tf=bool(".ckpt" in args.model_name_or_path),
            #         config=config,
            #         cache_dir=args.cache_dir if args.cache_dir else None,
            #     )
            #     model = PeftModel.from_pretrained(pretrained_model, "output/lora")
            #     lora_params = {n: p for n, p in model.named_parameters() if "lora_B" in n}
            #     for n, p in lora_params.items():
            #         print(n, p.any())
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)
            logger.info("After eval:")
            log_gpu_memory()

    return results


if __name__ == "__main__":
    main()