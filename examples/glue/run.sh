
# warmup
python3 examples/glue/run.py \
  --model_name_or_path google/electra-small-discriminator \
  --task_name MRPC \
  --do_train False \
  --do_eval True \
  --data_dir data/glue_data/MRPC \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 20.0 \
  --output_dir output/warmup \
  --overwrite_output_dir True \
  --cache_dir electra_small_cache \
  --finetune_method original \
  --quantization_method original

# # un-finetuned
# python3 examples/glue/run.py \
#   --model_name_or_path google/electra-small-discriminator \
#   --task_name MRPC \
#   --do_train False \
#   --do_eval True \
#   --data_dir data/glue_data/MRPC \
#   --max_seq_length 128 \
#   --per_gpu_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 15.0 \
#   --output_dir output/unfinetuned \
#   --overwrite_output_dir True \
#   --cache_dir electra_small_cache \
#   --finetune_method original \
#   --quantization_method original

# # Command for the original run
# python3 examples/glue/run.py \
#   --model_name_or_path google/electra-small-discriminator \
#   --task_name MRPC \
#   --do_train True \
#   --do_eval True \
#   --data_dir data/glue_data/MRPC \
#   --max_seq_length 128 \
#   --per_gpu_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 20.0 \
#   --output_dir output/original \
#   --overwrite_output_dir True \
#   --cache_dir electra_small_cache \
#   --finetune_method original \
#   --quantization_method original 

# # Command for the LoRA run
# python3 examples/glue/run.py \
#   --model_name_or_path google/electra-small-discriminator \
#   --task_name MRPC \
#   --do_train True \
#   --do_eval True \
#   --data_dir data/glue_data/MRPC \
#   --max_seq_length 128 \
#   --per_gpu_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 20.0 \
#   --output_dir output/lora \
#   --overwrite_output_dir True \
#   --cache_dir electra_small_cache \
#   --finetune_method lora \
#   --quantization_method original


# Command for the QLoRA run
#bert-base-uncased
# python3 examples/glue/run.py \
#   --model_name_or_path google/electra-small-discriminator \
#   --task_name MRPC \
#   --do_train True \
#   --do_eval True \
#   --data_dir data/glue_data/MRPC \
#   --max_seq_length 128 \
#   --per_gpu_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 20.0 \
#   --output_dir output/qlora \
#   --overwrite_output_dir True \
#   --cache_dir electra_small_cache \
#   --finetune_method qlora \
#   --quantization_method original


# Command for the inference run on CPU
# python3 examples/glue/run.py \
#   --model_name_or_path google/electra-small-discriminator \
#   --task_name MRPC \
#   --do_train True \
#   --do_eval True \
#   --data_dir data/glue_data/MRPC \
#   --max_seq_length 128 \
#   --per_gpu_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 20.0 \
#   --output_dir output/inference_on_cpu \
#   --overwrite_output_dir True \
#   --cache_dir electra_small_cache \
#   --finetune_method original \
#   --quantization_method original \
#   --inference_on_cpu True


# # Command for the PTSQ run
# python3 examples/glue/run.py \
#   --model_name_or_path google/electra-small-discriminator \
#   --task_name MRPC \
#   --do_train True \
#   --do_eval True \
#   --data_dir data/glue_data/MRPC \
#   --max_seq_length 128 \
#   --per_gpu_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 20.0 \
#   --output_dir output/ptsq \
#   --overwrite_output_dir True \
#   --cache_dir electra_small_cache \
#   --finetune_method original \
#   --quantization_method ptsq


# Command for the QAT run
# python3 examples/glue/run.py \
#   --model_name_or_path google/electra-small-discriminator \
#   --task_name MRPC \
#   --do_train True \
#   --do_eval True \
#   --data_dir data/glue_data/MRPC \
#   --max_seq_length 128 \
#   --per_gpu_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 20.0 \
#   --output_dir output/qat \
#   --overwrite_output_dir True \
#   --cache_dir electra_small_cache \
#   --finetune_method original \
#   --quantization_method qat

# Pruning
# Unstructured global random
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
  --output_dir output/unstructured_g_r_25 \
  --overwrite_output_dir True \
  --cache_dir electra_small_cache \
  --finetune_method original \
  --quantization_method original \
  --prune True \
  --prune_structure_type unstructured \
  --prune_global True \
  --prune_criterion random \
  --prune_amount 0.25 \
  --prune_dim 0

# Unstructured global l1
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
  --output_dir output/unstructured_g_l1_25 \
  --overwrite_output_dir True \
  --cache_dir electra_small_cache \
  --finetune_method original \
  --quantization_method original \
  --prune True \
  --prune_structure_type unstructured \
  --prune_global True \
  --prune_criterion l1 \
  --prune_amount 0.25 \
  --prune_dim 0

# Unstructured layer l1
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
  --output_dir output/unstructured_l_l1_25 \
  --overwrite_output_dir True \
  --cache_dir electra_small_cache \
  --finetune_method original \
  --quantization_method original \
  --prune True \
  --prune_structure_type unstructured \
  --prune_global False \
  --prune_criterion l1 \
  --prune_amount 0.25 \
  --prune_dim 0
