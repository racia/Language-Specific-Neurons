import argparse
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from vllm import LLM, SamplingParams


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("-a", "--activation_mask", type=str, default="")

args = parser.parse_args()

is_llama = bool(args.model.lower().find('llama') >= 0)
is_gpt2 = bool(args.model.lower().find("gpt2") >= 0)
is_gpt3 = bool(args.model.lower().find("gpt") >= 0)
model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)

num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
max_length = model.llm_engine.model_config.max_model_len

if args.activation_mask:
    activation_masks = torch.load(args.activation_mask)
else:
    activation_masks = [None]

final_output = []
if is_llama:
    languages = ["en", "de", "tr", "tk"]
else:
    languages = ["en", "de", "tr", "tk"]# "vi", "id"]

for activation_mask, mask_lang in zip(activation_masks, languages):
    if activation_mask:
        def factory(mask):
            def llama_forward(self, x):
                gate_up, _ = self.gate_up_proj(x)  # b, l, 2i
                i = gate_up.size(-1)
                activation = F.silu(gate_up[:, :, : i // 2])
                activation.index_fill_(2, mask, 0)
                x = activation * gate_up[:, :, i // 2 :]
                x, _ = self.down_proj(x)
                return x

            def bloom_forward(self, x: torch.Tensor):
                x, _ = self.dense_h_to_4h(x)
                x = self.gelu_impl(x)
                x.index_fill_(2, mask, 0)
                x, _ = self.dense_4h_to_h(x)
                return x

            def gpt3_forward(self, x: torch.Tensor):
                # GPT-2 does not have a separate MLP, so we'll just process the hidden states as they are
                # GPT-2 typically applies layer normalization, attention, and then feed-forward operations

                # gate_up = self.attn(x)[0]  # Assuming the attention module outputs the hidden states
                # activation = gate_up.float()
                # over_zero[idx, :] += (activation > 0).sum(dim=(0, 1))  # Track positive activations
                # x = self.ln_1(gate_up)
                # x = self.mlp(x)  # GPT-2 MLP layers typically apply some feed-forward networks
                # return x
                
                x, _ = self.c_fc(x)
                x = self.act(x)
                activation = x.float()
                x.index_fill_(2, mask, 0)
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

        for i, layer_mask in enumerate(activation_mask):
            if is_llama:
                obj = model.llm_engine.driver_worker.model_runner.model.model.layers[i].mlp
            else:
                obj = model.llm_engine.driver_worker.model_runner.model.transformer.h[i].mlp
            obj.forward = MethodType(factory(layer_mask.to('cuda')), obj)

    ppls = []
    for lang in languages:
        if is_llama:
            ids = torch.load(f'data/id.{lang}.valid.llama')
        else:
            ids = torch.load(f'data/id.{lang}.valid.gpt') #bloom
        l = ids.size(0)
        l = min(l, 2**20) // max_length * max_length
        input_ids = ids[:l].reshape(-1, max_length)
        outputs = model.generate(prompt_token_ids=input_ids.tolist(), sampling_params=SamplingParams(max_tokens=1, prompt_logprobs=0))
        ppl = []
        for output in outputs:
            ppl.append(np.mean([next(iter(r.values())) for r in output.prompt_logprobs if r]))
        ppls.append(np.mean(ppl))
    final_output.append(ppls)

for ppls in final_output:
    print(' '.join([str(-ppl) for ppl in ppls]))
    
