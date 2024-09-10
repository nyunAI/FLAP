# %%
import glob

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoModelForCausalLM

# %%
model = AutoModelForCausalLM.from_pretrained(
    "/home/azureuser/arnav/FLAP/llm_weights/flap_p0.5_WIFV_AL-AM_llama3_70b",
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

# %%
model.dtype

# %%
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-70B", device_map="auto", torch_dtype=torch.float16
)

# %%
shapes = {}
for tensor in glob.glob(
    "/home/azureuser/arnav/FLAP/llm_weights/flap_p0.5_WIFV_AL-AM_20_llama3_70b/*.safetensors"
):
    with safe_open(tensor, framework="pt") as f:
        for k in f.keys():
            shapes[k] = f.get_tensor(k)

# %%
num_param = 0
for k in shapes:
    num_param += shapes[k].numel()

# %%
num_heads = []
num_key_value_heads = []
intermediate_size = []
for layer_idx in range(80):
    if layer_idx >= 50:
        for layer in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            curr_shape = (
                model.model.layers[layer_idx].self_attn._modules[layer].weight.shape
            )
            pruned_shape = shapes[
                f"model.layers.{layer_idx}.self_attn.{layer}.weight"
            ].shape
            if curr_shape != pruned_shape:
                device = (
                    model.model.layers[layer_idx]
                    .self_attn._modules[layer]
                    .weight.device
                )
                bias = False
                if f"model.layers.{layer_idx}.self_attn.{layer}.bias" in shapes:
                    bias = True
                model.model.layers[layer_idx].self_attn._modules[layer] = nn.Linear(
                    pruned_shape[1],
                    pruned_shape[0],
                    device=device,
                    bias=bias,
                )
                model.model.layers[layer_idx].self_attn._modules[layer].weight.data = (
                    shapes[f"model.layers.{layer_idx}.self_attn.{layer}.weight"].to(
                        device
                    )
                )
                if bias:
                    model.model.layers[layer_idx].self_attn._modules[
                        layer
                    ].bias.data = shapes[
                        f"model.layers.{layer_idx}.self_attn.{layer}.bias"
                    ].to(
                        device
                    )

        for layer in ["down_proj", "up_proj", "gate_proj"]:
            curr_shape = model.model.layers[layer_idx].mlp._modules[layer].weight.shape
            pruned_shape = shapes[f"model.layers.{layer_idx}.mlp.{layer}.weight"].shape
            if curr_shape != pruned_shape:
                device = model.model.layers[layer_idx].mlp._modules[layer].weight.device
                bias = False
                if f"model.layers.{layer_idx}.mlp.{layer}.bias" in shapes:
                    bias = True
                model.model.layers[layer_idx].mlp._modules[layer] = nn.Linear(
                    pruned_shape[1], pruned_shape[0], device=device, bias=bias
                )
                model.model.layers[layer_idx].mlp._modules[layer].weight.data = shapes[
                    f"model.layers.{layer_idx}.mlp.{layer}.weight"
                ].to(device)
                if bias:
                    model.model.layers[layer_idx].mlp._modules[layer].bias.data = (
                        shapes[f"model.layers.{layer_idx}.mlp.{layer}.bias"].to(device)
                    )
    model.model.layers[layer_idx].self_attn.num_heads = (
        model.model.layers[layer_idx].self_attn._modules["q_proj"].weight.data.shape[0]
        // 128
    )
    model.model.layers[layer_idx].self_attn.num_key_value_heads = (
        model.model.layers[layer_idx].self_attn._modules["k_proj"].weight.data.shape[0]
        // 128
    )
    # model.model.layers[layer_idx].self_attn.hidden_size = model.model.layers[layer_idx].self_attn._modules['q_proj'].weight.data.shape[0]
    model.model.layers[layer_idx].mlp.intermediate_size = (
        model.model.layers[layer_idx].mlp._modules["gate_proj"].weight.data.shape[0]
    )
    num_heads.append(model.model.layers[layer_idx].self_attn.num_heads)
    num_key_value_heads.append(
        model.model.layers[layer_idx].self_attn.num_key_value_heads
    )
    intermediate_size.append(model.model.layers[layer_idx].mlp.intermediate_size)

# %%
import json

with open(
    "/home/azureuser/arnav/FLAP/llm_weights/flap_p0.5_WIFV_AL-AM_llama3_70b/config.json",
    "r",
) as f:
    config = json.load(f)

# %%
config["intermediate_size"] = intermediate_size
config["num_attention_heads"] = num_heads
config["num_key_value_heads"] = num_key_value_heads
config["first_compressed_layer_idx"] = 50

# %%
config["auto_map"] = (
    {
        "AutoConfig": "configuration_llama.LlamaConfig",
        "AutoModelForCausalLM": "modeling_llama.LlamaForCausalLM",
    },
)

# %%
config

# %%
with open(
    "/home/azureuser/arnav/FLAP/llm_weights/flap_p0.5_WIFV_AL-AM_llama3_70b/config.json",
    "w",
) as outfile:
    json.dump(config, outfile, indent=2)

# %%
import lm_eval
from lm_eval.models import huggingface

lm_obj = huggingface.HFLM(pretrained=model)

task_manager = lm_eval.tasks.TaskManager()

results = lm_eval.simple_evaluate(  # call simple_evaluate
    model=lm_obj,
    tasks=["winogrande"],
    num_fewshot=5,
    task_manager=task_manager,
    batch_size="auto",
)

# %%
