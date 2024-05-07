# Efficient Fine-tuning and Inferencing of ELECTRA 

- Branch Author: Allen Chen
- UNI #: atc2160

## Project Description

The primary objective of this research project is to optimize the development and the serving of ELECTRA language model in practical user applications by exploring memory-saving approaches for fine-tuning and inferencing. 

Techniques used:
1. Low-rank adaptation (LoRA and QLoRA)
2. Quantization (PTSQ and QAT)
3. Pruning (Unstructured vs Structured and criteria variants)

In this study, we aim to apply the above three memory-efficient optimization techniques to reduce ELECTRA language model size, thereby decreasing its memory footprint and computation time.

This repo branch implements the above techniques on ELECTRA, fine-tunes on MRPC training dataset, and evaluates on MRPC testing dataset.

The code used in this study builds on top of the repo electra-pytorch by lucidrains.

## How to run
```
# Download GLUE data
python3 examples/glue/download.py --data_dir ./data --tasks MRPC

# Download vocab.txt
cd data
wget https://huggingface.co/google/electra-small-generator/raw/b3cb16eb009f9e7969e12b0d38be3aeb2c0a9fd4/vocab.txt

# Setup repo and download dependencies
pip install -e .

# Setup WandB
wandb login

# create ouput dir
mkdir output

# Run through all experiments
./examples/glue/run.sh
```

## Outline of the code repository

This repo builds on top of electra-pytorch repo, and the modified and new code can be found in the following files: 

- examples/glue/run.sh: This shell script contains all commands to run all experiments

- examples/glue/run.py: This Python script configures the runs correctly, trains and evaluates the MRPC sub-task

- examples/glue/finetune.py: This Python script fine-tunes ELECTRA with MRPC training dataset

- examples/glue/inference.py: This Python script evalutes on MPRC dev dataset

- examples/glue/finetune_utils.py: This Python script has useful functions for fine-tuning and evaluating.

- examples/glue/quantization.py: This Python script has the custom modules and functions to prepare for quantization.

- examples/glue/prune.py: This Python script contains the relevant code for pruning.

electra_pytorch and pretraining folder contains the code for the PyTorch implementation of ELECTRA and its training respectively from the author lucidrains.


## Example command
```
# create output dir
mkdir output

# To run all experiments
./examples/glue/run.sh

# Example command: an original run with fine-tuning and evaluating but without any optimization techniques applied
python3 examples/glue/run.py \
  --model_name_or_path google/electra-small-discriminator \
  --task_name MRPC \
  --do_train True \
  --do_eval True \
  --data_dir data/glue_data/MRPC \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 20.0 \
  --output_dir output/original \
  --overwrite_output_dir True \
  --cache_dir electra_small_cache \
  --finetune_method original \
  --quantization_method original 
```

## Results

### LoRA Results
![LoRA Results](images/lora_results.png?raw=true)

ELECTRA with LoRA maintains the original performance while using only 9.2% of total trainable parameters from the original model. However, the performance for the model with QLoRA drops significantly as the addition of 4-bit quantization introduces more errors. One interesting observation is that the fine-tune and inference time for ELECTRA with LoRA are slower than those of the original model. This is different from what we expect as reducing the number of trainable parameters does not lower or maintain the computation time.

![Tuning Rank and Alpha parameters](images/tuning_rank_and_alpha.png?raw=true)

The performance at rank 128 and alpha 256 is only slightly higher than the performance at rank 64 and alpha 64 or 128. We conclude that this slight improvement does not warrant an sizable increase in trainable parameters, so we choose rank 64 and alpha 64 for the reported ELECTRA with LoRA. We further reason that the sub-task and the pre-trained model are equally important as both tasks train to improve its general natural language understanding without focusing on a specific domain. We then select the value of 64 for Alpha. The rank is relatively high because ELECTRA is a relatively smaller model, still requiring a decent number of trainable parameters to perform well.

### Quantization Results
![Quantization Results](images/quantization_results.png?raw=true)

Both PTSQ and QAT have a smaller model size and achieve slightly faster inference time while losing just a bit of accuracies and f1 scores. However, it is surprising to see PTSQ and QAT achieve the same performance as we expect QAT to yield higher accuracies.

### Pruning Results
![Pruning Results](images/pruning_results.png?raw=true)

Overall, Prune method number 2 and 3 outperform the rest of the methods and maintain the same accuracies and f1-scores. These results show that Unstructured pruning with L1 pruning criteria is more effective for the linear layers in ELECTRA. On the other hand, we see poor performances across runs with structured pruning along both dimensions. 

![Tuning prune rate](images/tuning_prune_rate.png?raw=true)

Previously all models are pruned at a pruning amount of 25 percent. Now we take the best performing model, which uses unstructured prune-by-layer pruning strategy with L1 criterion, and tune on its pruning amount. After the model was pruned at 35 percent, the performance decreased rapidly. We conclude that the pruning amount of 25 percent remains optimal. For future experiments, the pruning rate tuning strategy can be more refined by tuning-by-layer and using iterative pruning. 
