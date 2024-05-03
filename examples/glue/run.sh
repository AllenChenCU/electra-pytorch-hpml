
# warmup
python examples/glue/run.py \
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
# python examples/glue/run.py \
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
# python examples/glue/run.py \
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
# python examples/glue/run.py \
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
# python examples/glue/run.py \
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


# Command for the PTSQ run
python examples/glue/run.py \
  --model_name_or_path google/electra-small-discriminator \
  --task_name MRPC \
  --do_train True \
  --do_eval True \
  --data_dir data/glue_data/MRPC \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 2.0 \
  --output_dir output/ptsq \
  --overwrite_output_dir True \
  --cache_dir electra_small_cache \
  --finetune_method original \
  --quantization_method ptsq

