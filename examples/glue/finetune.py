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
""" Finetune code """

import logging
import os
import json

from tqdm import tqdm, trange
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.profiler import profile, record_function, ProfilerActivity

from finetune_utils import get_linear_schedule_with_warmup, set_seed
from inference import evaluate


logger = logging.getLogger(__name__)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to global_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
        record_shapes=True, 
        profile_memory=True,
    ) as prof: 
        with record_function("model_finetune"):
            tr_loss, logging_loss = 0.0, 0.0
            model.zero_grad()
            train_iterator = trange(
                epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
            )
            set_seed(args)  # Added here for reproductibility
            for _ in train_iterator:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
                for step, batch in enumerate(epoch_iterator):

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        continue

                    model.train()
                    batch = tuple(t.to(args.device) for t in batch)
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                    outputs = model(**inputs)
                    loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

                    if args.n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu parallel training
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    if step % 10 == 0:
                        print(step, loss.item())

                    tr_loss += loss.item()
                    if (step + 1) % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        len(epoch_iterator) <= args.gradient_accumulation_steps
                        and (step + 1) == len(epoch_iterator)
                    ):
                        if args.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                        optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                        model.zero_grad()
                        global_step += 1

                        if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                            logs = {}
                            if (
                                args.local_rank == -1 and args.evaluate_during_training
                            ):  # Only evaluate when single GPU otherwise metrics may not average well
                                results = evaluate(args, model, tokenizer)
                                for key, value in results.items():
                                    eval_key = "eval_{}".format(key)
                                    logs[eval_key] = value

                            loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                            learning_rate_scalar = scheduler.get_lr()[0]
                            logs["learning_rate"] = learning_rate_scalar
                            logs["loss"] = loss_scalar
                            logging_loss = tr_loss

                            print(json.dumps({**logs, **{"step": global_step}}))

                        if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                            # Save model checkpoint
                            output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = (
                                model.module if hasattr(model, "module") else model
                            )  # Take care of distributed/parallel training
                            model_to_save.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)

                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            logger.info("Saving model checkpoint to %s", output_dir)

                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            logger.info("Saving optimizer and scheduler states to %s", output_dir)

                    if args.max_steps > 0 and global_step > args.max_steps:
                        epoch_iterator.close()
                        break
                if args.max_steps > 0 and global_step > args.max_steps:
                    train_iterator.close()
                    break
    logger.info(prof.key_averages().table(sort_by=f"{args.device}_time_total", row_limit=15))

    return global_step, tr_loss / global_step
