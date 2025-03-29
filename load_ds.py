import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct") #ai-forever/mGPT

args = parser.parse_args()
model = args.model

for lang in ["tr", "tk", "de"]: # en included already

    # Load Wikipedia dataset
    dataset = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train")[-5308:]
    print("DS loaded")
    # Initialize tokenizer (using GPT-2 tokenizer for example)
    tokenizer = AutoTokenizer.from_pretrained(f"{model}")
   # tokenizer.pad_token = tokenizer.eos_token
    max_length = tokenizer.model_max_length


    # Tokenize and concatenate texts
    def tokenize_and_concat(dataset, tokenizer):
        tokenized_texts = []
        
        for entry in dataset["text"]:
            tokens = tokenizer.encode(
                entry, 
                #padding="max_length",  # Pad to max_length
                #truncation=True,  # Truncate to max_length
                #max_length=max_length,  # Specify max_length
            )
            
            tokenized_texts.extend(tokens)
        return tokenized_texts

    long_token_list = tokenize_and_concat(dataset, tokenizer)
    print("Tokenized")
    # Convert to LongTensor
    tensor_data = torch.tensor(long_token_list, dtype=torch.long, device="cuda")
    print("Tensor created")
    # Save the tensor as a file
    torch.save(tensor_data, f"data/id.{lang}.train.gpt")
    print("Saved")
    # Load and verify saved tensor
    tensor_data_loaded = torch.load(f"data/id.{lang}.train.gpt")
    print("Tokenized data successfully saved and loaded. Shape:", tensor_data_loaded.shape)
    
    torch.cuda.empty_cache()