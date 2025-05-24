# Originally from https://github.com/RUCAIBox/Language-Specific-Neurons
# Modified by: Raziye Sari for Project: Probing Language-Specific-Neurons

import argparse
from types import MethodType
import torch
from vllm import LLM, SamplingParams
from transformers import AutoConfig


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="ai-forever/mGPT")
parser.add_argument("-l", "--lang", type=str, default="tr")
args = parser.parse_args()

is_llama = bool(args.model.lower().find('llama') >= 0)
is_gpt2 = bool(args.model.lower().find("gpt2") >= 0)
is_gpt3 = bool(args.model.lower().find("gpt") >= 0) #ai-forever/mGPT

config = AutoConfig.from_pretrained(args.model, torch_dtype=torch.bfloat16)
print(config)

model = LLM(model=args.model, tensor_parallel_size=1, enforce_eager=True, dtype="float16")

max_length = model.llm_engine.model_config.max_model_len
num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
intermediate_size = model.llm_engine.model_config.hf_config.intermediate_size if is_llama else model.llm_engine.model_config.hf_config.hidden_size * 4

over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')


def factory(idx):
    def llama_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)  # b, l, 2i
        i = gate_up.size(-1)
        gate_up[:, :, : i // 2] = torch.nn.SiLU()(gate_up[:, :, : i // 2])
        activation = gate_up[:, :, : i // 2].float() # b, l, i
        over_zero[idx, :] += (activation > 0).sum(dim=(0,1))
        x = gate_up[:, :, : i // 2] * gate_up[:, :, i // 2 :]
        x, _ = self.down_proj(x)
        return x

    def bloom_forward(self, x: torch.Tensor):
        x, _ = self.dense_h_to_4h(x)
        x = self.gelu_impl(x)
        activation = x.float()
        over_zero[idx, :] += (activation > 0).sum(dim=(0,1))
        x, _ = self.dense_4h_to_h(x)
        return x
    
    def gpt3_forward(self, x: torch.Tensor):
        """Self-implemented by figuring out the layer normalization function of GPT3"""
        x, _ = self.c_fc(x)
        x = self.act(x)
        activation = x.float()
        over_zero[idx, :] += (activation > 0).sum(dim=(0,1))
        x, _ = self.c_proj(x)
        return x

    if is_llama:
        return llama_forward
    elif is_gpt2:
        #return gpt2_forward
        pass
    elif is_gpt3:
        return gpt3_forward
    else:
        return bloom_forward

for i in range(num_layers):
    if is_llama:
        obj = model.llm_engine.driver_worker.model_runner.model.model.layers[i].mlp
    else:
        obj = model.llm_engine.driver_worker.model_runner.model.transformer.h[i].mlp
    obj.forward = MethodType(factory(i), obj)

lang = args.lang

#for lang in ["de", "tr", "tk", "en"]: 
if is_llama:
    ids = torch.load(f'data/id.{lang}.train.llama')
else:
    ids = torch.load(f'data/id.{lang}.train.gpt')
l = ids.size(0)
l = min(l, 99999744) // max_length * max_length
input_ids = ids[:l].reshape(-1, max_length)


output = model.generate(prompt_token_ids=input_ids.tolist(), sampling_params=SamplingParams(max_tokens=1))

output = dict(n=l, over_zero=over_zero.to('cpu'))

if is_llama:
    torch.save(output, f'activations/activation.{lang}.train.llama-7b')
else:
    torch.save(output, f'activations/activation.{lang}.train.gpt')
