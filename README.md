## Optimizing the performance of Electra 

- Branch Author: Allen Chen
- UNI #: atc2160


1. Efficient fine-tuning with LoRA and QLoRA

2. Efficient inferencing with quantization

3. Efficient inferencing with pruning

## How to run
'''
# Download GLUE data
python3 examples/glue/download.py --data_dir ./data --tasks MRPC

# Download vocav.txt
cd data
wget https://huggingface.co/google/electra-small-generator/raw/b3cb16eb009f9e7969e12b0d38be3aeb2c0a9fd4/vocab.txt

# Setup repo and download dependencies
pip install -e .


'''


## Project Plan
1. Understand Electra and this repository of its PyTorch implementation
2. Be able to run through the commands in the Training section
3. Add profiling tools
    a. Weights and Biases
    b. PyTorch Profiler
4. Start Task 1
    a. Run a vanilla fine-tuning run
    b. Add LoRA for fine-tuning
    c. Add QLoRA for fine-tuning
    d. Visualize results
5. Start Task 2 (with the best model from task 1)
    a. Run a vanilla inferencing run 
    b. Add dynamic quantization
    c. Add post-training static quantization
    d. Visualize results
6. Start Task 3
    a. Add structured pruning
    b. Add unstructured pruning
    c. Visualize results
7. Run one with the best results from task 1-3

