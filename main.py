import argparse
import os
from importlib.metadata import version

import lm_eval
import numpy as np
import torch
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.eval import eval_ppl
from lib.prune import check_sparsity, prune_flap
from models.hf_llama.modeling_llama import LlamaForCausalLM


def get_llm(model, device):
    if device == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.float16, trust_remote_code=True
        ).to(torch.device(f"cuda:{device}"))

    for i in range(len(model.model.layers)):
        model.model.layers[i].self_attn.o_proj.bias = torch.nn.Parameter(
            torch.zeros(
                model.model.layers[i].self_attn.o_proj.weight.shape[0],
                device=model.model.layers[i].self_attn.o_proj.weight.device,
                dtype=torch.float16,
            )
        )  # 或 'cuda'
        model.model.layers[i].mlp.down_proj.bias = torch.nn.Parameter(
            torch.zeros(
                model.model.layers[i].mlp.down_proj.weight.shape[0],
                device=model.model.layers[i].mlp.down_proj.weight.device,
                dtype=torch.float16,
            )
        )  # 或 'cuda'
        torch.nn.init.zeros_(model.model.layers[i].self_attn.o_proj.bias)
        torch.nn.init.zeros_(model.model.layers[i].mlp.down_proj.bias)

    model.seqlen = 128
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="LLaMA model",
    )  # Huggingface model name
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for sampling the calibration data.",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=1024,
        help="Number of calibration samples.",
    )
    parser.add_argument(
        "--pruning_ratio", type=float, default=0.5, help="Pruning ratio."
    )
    parser.add_argument("--remove_heads", type=int, default=-1, help="Remove num_heads")
    parser.add_argument(
        "--metrics",
        type=str,
        default="WIFV",
        choices=["IFV", "WIFV", "WIFN", "N/A"],
    )
    parser.add_argument("--structure", type=str, default="AL-AM", choices=["AL-AM"])
    parser.add_argument("--prune_method", type=str, default="flap", choices=["flap"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "--save_model",
        type=str,
        default=None,
        help="Path to save the pruned model.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--group_size", type=int, default=7, help="Group size, 1 for no GQA."
    )
    parser.add_argument(
        "--num_heads", type=int, default=14, help="Number of Query Heads"
    )
    parser.add_argument(
        "--prune_kv_heads",
        type=bool,
        default=True,
        help="Retains KV Heads if set to false.",
    )
    parser.add_argument(
        "--start_pruning_layer_idx",
        type=int,
        default=22,
        help="Layer idx post which pruning starts",
    )
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=896)
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Build the model and tokenizer
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    if args.device == "auto":
        device = model.hf_device_map["lm_head"]
    else:
        device = torch.device(f"cuda:{args.device}")

    # Prune the model
    print("pruning starts")
    if args.prune_method == "flap":
        if args.metrics == "N/A":
            raise ValueError(
                "For FLAP pruning, the metrics parameter must be chosen from ['IFV', 'WIFV', 'WIFN']. 'N/A' is not a valid choice."
            )
        if args.structure == "N/A":
            raise ValueError(
                "For FLAP pruning, the compressed model structure parameter must be chosen from ['UL-UM', 'UL-MM', 'AL-MM', 'AL-AM']. 'N/A' is not a valid choice."
            )
        prune_flap(args, model, tokenizer, device)

    # Check the sparsity of the model
    print("*" * 30)
    print(
        f"model parameter {sum(p.numel() for p in model.parameters()) / 1000 ** 3:.2f}B"
    )
    print("*" * 30)

    # Save the model
    if args.save_model:
        if not os.path.exists(args.save_model):
            os.makedirs(args.save_model)
        # torch.save(model, f'{args.save_model}/pruned_model.pt')
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

    # Evaluate the model
    if args.eval:
        lm_obj = HFLM(pretrained=model)
        task_manager = lm_eval.tasks.TaskManager()
        results = []
        result = lm_eval.simple_evaluate(
            model=lm_obj,
            tasks=["winogrande"],
            num_fewshot=5,
            task_manager=task_manager,
            batch_size="auto",
        )
        results.append(result["results"])
        print(results)
        result = lm_eval.simple_evaluate(
            model=lm_obj,
            tasks=["gsm8k"],
            num_fewshot=5,
            task_manager=task_manager,
            batch_size="auto",
        )
        results.append(result["results"])
        print(results)
        result = lm_eval.simple_evaluate(
            model=lm_obj,
            tasks=["mmlu"],
            num_fewshot=5,
            task_manager=task_manager,
            batch_size="auto",
        )
        results.append(result["results"])
        print(results)
        result = lm_eval.simple_evaluate(
            model=lm_obj,
            tasks=["boolq"],
            num_fewshot=0,
            task_manager=task_manager,
            batch_size="auto",
        )
        results.append(result["results"])
        print(results)
        result = lm_eval.simple_evaluate(
            model=lm_obj,
            tasks=["arc_challenge"],
            num_fewshot=25,
            task_manager=task_manager,
            batch_size="auto",
        )
        results.append(result["results"])
        print(results)
        result = lm_eval.simple_evaluate(
            model=lm_obj,
            tasks=["hellaswag"],
            num_fewshot=10,
            task_manager=task_manager,
            batch_size="auto",
        )
        results.append(result["results"])
        print(results)


if __name__ == "__main__":
    main()
