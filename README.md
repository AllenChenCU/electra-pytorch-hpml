## Optimizing the performance of Electra 

- Branch Author: Allen Chen
- UNI #: atc2160


1. Efficient fine-tuning with LoRA and QLoRA

2. Efficient inferencing with quantization

3. Efficient inferencing with pruning

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

# Run through all experiments
./examples/glue/run.sh
```
