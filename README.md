# Efficient Fine-tuning and Inferencing of Electra 

- Branch Author: Allen Chen
- UNI #: atc2160

## Project Description

The primary objective of this research project is to optimize the development and the serving of ELECTRA language model in practical user applications by exploring memory-saving approaches for fine-tuning and inferencing. 

Techniques:
1. Low-rank adaptation (LoRA and QLoRA)
2. Quantization (PTSQ and QAT)
3. Pruning (Unstructured vs Structured and criteria variants)

This repo branch implements the above techniques on ELECTRA, fine-tunes on MRPC training dataset, and evaluates on MRPC testing dataset.

The code used in this study builds on top of the repo electra-pytorch by lucidrains

## How to run
```
# Download GLUE data
python3 examples/glue/download.py --data_dir ./data --tasks MRPC

# Download vocav.txt
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

# Example command: an original run with fine-tuning and evalating but without any optimization techniques applied
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

### Quantization Results
![Quantization Results](images/quantization_results.png?raw=true)

![Tuning Rank and Alpha parameters](images/tuning_rank_and_alpha.png?raw=true)

### Pruning Results
![Pruning Results](images/pruning_results.png?raw=true)

![Tuning prune rate](images/tuning_prune_rate.png?raw=true)
