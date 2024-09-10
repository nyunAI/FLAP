import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .data import get_loaders
from .layerwrapper import BiasGPT, WrappedGPT

# create a dictionary to map the method name to the function
"""
    'IFV': Input Feature Variance
    'WIFV': Weighted Input Feature Variance
    'WIFN': Weighted Input Feature Norm
"""
metrics = {
    "IFV": lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp,
    "WIFV": lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp
    * torch.sum(subset[name].weight.data.pow(2), dim=0),
    "WIFN": lambda wrapped_layers, subset, name: (
        torch.abs(subset[name].weight.data)
        * torch.sqrt(wrapped_layers[name].scaler_inp.reshape((1, -1)))
    ).mean(axis=0),
}


def find_layers(module, layers=[nn.Linear], name=""):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child,
                layers=layers,
                name=name + "." + name1 if name != "" else name1,
            )
        )
    return res


def prepare_calibration_input(model, dataloader, device):
    """
    Prepare inputs for model calibration.

    Args:
        model (nn.Module): The model to prepare inputs for.
        dataloader (DataLoader): DataLoader object to fetch input data.
        device (torch.device): Device on which the model is loaded.

    Returns:
        inps (torch.Tensor): Input tensor for calibration.
        outs (torch.Tensor): Output tensor for calibration.
        position_ids (torch.Tensor): Position IDs tensor.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in getattr(model, "hf_device_map", {}):
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (2048, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=device,
    )
    inps.requires_grad = False
    cache = {"i": 0, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 



def compress(
    layer,
    attn_mask,
    mlp_mask,
    attn_mean_inp,
    mlp_mean_inp,
    device,
    bias=True,
    args=None,
):
    """
    Compress a model layer by masking or pruning based on the given masks.

    Args:
        layer (nn.Module): The model layer to compress.
        attn_mask (torch.Tensor): The mask to apply to the attention weights.
        mlp_mask (torch.Tensor): The mask to apply to the MLP weights.
        attn_mean_inp (torch.Tensor): The mean attention input.
        mlp_mean_inp (torch.Tensor): The mean MLP input.
        device (torch.device): Device on which the model is loaded.
        bias (bool, optional): Whether to consider bias while compressing. Defaults to True.

    Returns:
        None: This function modifies the layer in-place and doesn't return anything.
    """
    # Real Pruning
    # Attention Weight Pruning
    if attn_mask is not None:
        if args.prune_kv_heads:
            # In this case the number of groups changes during pruning.
            retain_heads_kv = torch.count_nonzero(attn_mask)
            retain_heads_qo = torch.count_nonzero(attn_mask).repeat_interleave(
                args.group_size
            )
            attn_mask_kv = attn_mask.repeat_interleave(args.head_dim)
            attn_mask_qo = attn_mask.repeat_interleave(
                args.group_size
            ).repeat_interleave(args.head_dim)
            layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[
                torch.where(attn_mask_kv)[0]
            ]
            layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[
                torch.where(attn_mask_kv)[0]
            ]

            layer.self_attn.k_proj.out_features = attn_mask_kv.sum().item()
            layer.self_attn.v_proj.out_features = attn_mask_kv.sum().item()

            if layer.self_attn.v_proj.bias is not None:
                layer.self_attn.v_proj.bias.data = layer.self_attn.v_proj.bias.data[
                    torch.where(attn_mask_kv)[0]
                ]
            if layer.self_attn.k_proj.bias is not None:
                layer.self_attn.k_proj.bias.data = layer.self_attn.k_proj.bias.data[
                    torch.where(attn_mask_kv)[0]
                ]

        elif args.group_size > 1 and not args.prune_kv_heads:
            # In this case KV heads do not get pruned, instead we change the group size of GQA uniformly
            retain_heads_qo = torch.count_nonzero(attn_mask) * (
                args.num_heads // args.group_size
            )
            attn_mask_qo = attn_mask.repeat(
                args.num_heads // args.group_size
            ).repeat_interleave(args.head_dim)

        # Prune the query projection weights
        # We reduce the size of the weights based on the attention mask
        layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[
            torch.where(attn_mask_qo)[0]
        ]

        # Update output dimensions of q projections based on remaining heads
        layer.self_attn.q_proj.out_features = attn_mask_qo.sum().item()

        output_weight = layer.self_attn.o_proj.weight.data

        # Support for models with Query Bias

        if layer.self_attn.q_proj.bias is not None:
            layer.self_attn.q_proj.bias.data = layer.self_attn.q_proj.bias.data[
                torch.where(attn_mask_qo)[0]
            ]

        if bias:
            # Add the additional bias to compensate for the loss
            output_bias = (
                attn_mean_inp.to(device) * ~attn_mask_qo.to(device)
            ) @ output_weight.T

        # Prune the output projection weight
        output_weight = layer.self_attn.o_proj.weight.data[
            :, torch.where(attn_mask_qo)[0]
        ]
        # Update layer configurations for the new output shape after pruning
        layer.self_attn.num_heads = int(torch.sum(retain_heads_qo))
        layer.self_attn.hidden_size = int(torch.sum(retain_heads_qo * args.head_dim))
        if args.group_size >= 1:
            if args.prune_kv_heads:
                layer.self_attn.num_key_value_heads = retain_heads_kv
            else:
                layer.self_attn.num_key_value_groups = (
                    layer.self_attn.num_heads // layer.self_attn.num_key_value_heads
                )

        if bias:
            # Re-initialize the Linear layer with new shape and bias
            layer.self_attn.o_proj.in_features = attn_mask_qo.sum().item()
            # layer.self_attn.o_proj = torch.nn.Linear(in_features=output_weight.shape[1], out_features=output_weight.shape[0], bias=True).to(device)
            layer.self_attn.o_proj.bias.data = output_bias

        # Assign the pruned weights
        layer.self_attn.o_proj.weight.data = output_weight

    # MLP Weight Pruning
    if mlp_mask is not None:
        # Prune the up and gate projection weights
        layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[
            torch.where(mlp_mask)[0]
        ]
        layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[
            torch.where(mlp_mask)[0]
        ]

        # Update output dimensions of up and gate projections based on the mlp mask
        layer.mlp.up_proj.out_features = mlp_mask.sum().item()
        layer.mlp.gate_proj.out_features = mlp_mask.sum().item()

        output_weight = layer.mlp.down_proj.weight.data
        layer.mlp.intermediate_size = mlp_mask.sum().item()
        if bias:
            # Add the additional bias to compensate for the loss
            output_bias = (
                mlp_mean_inp.to(device) * ~mlp_mask.to(device)
            ) @ output_weight.T

        # Prune the down projection weight
        output_weight = layer.mlp.down_proj.weight.data[:, torch.where(mlp_mask)[0]]

        if bias:
            # Re-initialize the Linear layer with new shape and bias
            layer.mlp.down_proj.in_features = mlp_mask.sum().item()
            # layer.mlp.down_proj = torch.nn.Linear(in_features=output_weight.shape[1], out_features=output_weight.shape[0], bias=True).to(device)
            layer.mlp.down_proj.bias.data = output_bias

        # Assign the pruned weights
        layer.mlp.down_proj.weight.data = output_weight

    # Explicitly empty the CUDA cache to clean up some memory
    torch.cuda.empty_cache()


def prune_flap(args, model, tokenizer, device=torch.device("cuda:0")):
    """
    Our FLAP Pruning.

    Args:
        args (object): Command line arguments parsed via argparse.
        model (nn.Module): PyTorch model to prune.
        tokenizer (Tokenizer): Tokenizer associated with the model.
        device (torch.device, optional): Device to move tensors to. Defaults to CUDA device 0.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    dataloader, _ = get_loaders(
        "wikitext2",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
    layers = model.model.layers
    attn_metric_list, mlp_metric_list = [], []
    attn_baseline_inp_list, mlp_baseline_inp_list = [], []
    attn_mask, mlp_mask = [], []

    # Split into sub-problems, separate statistics for each module
    for i in tqdm(
        range( args.start_pruning_layer_idx, len(layers)-4),
        desc="Processing layers",
    ):
        layer = layers[i]
        subset = {}
        subset.update({"self_attn.o_proj": find_layers(layer)["self_attn.o_proj"]})
        subset.update({"mlp.down_proj": find_layers(layer)["mlp.down_proj"]})

        if f"model.layers.{i}" in getattr(
            model, "hf_device_map", {}
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = BiasGPT(subset[name], args.metrics)

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                 outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            if name == "self_attn.o_proj":
                W_metric = metrics[args.metrics](wrapped_layers, subset, name) ** 2
                attn_metric_list.append(W_metric.cpu())
                attn_baseline_inp_list.append(
                    wrapped_layers[name].baseline_inp.type(torch.half)
                )
            else:
                W_metric = metrics[args.metrics](wrapped_layers, subset, name)
                mlp_metric_list.append(W_metric.cpu())
                mlp_baseline_inp_list.append(
                    wrapped_layers[name].baseline_inp.type(torch.half)
                )
            wrapped_layers[name].free()

        inps, outs = (
            outs,
            inps,
        )  # Use the original output as input to the next layer
        torch.cuda.empty_cache()

    standarlization = lambda x: (x - torch.mean(x, axis=1, keepdim=True)) / torch.std(
        x, axis=1, keepdim=True
    )

    if args.structure in ["AL-AM"]:
        attn_metric = torch.stack(attn_metric_list)
        attn_metric = standarlization(attn_metric)

        if args.prune_kv_heads:
            attn_metric = attn_metric.reshape(
                len(layers)-args.start_pruning_layer_idx - 4,
                -1,
                args.head_dim*args.group_size,
            ).mean(dim=2)
        elif args.group_size > 1 and not args.prune_kv_heads:
            attn_metric = (
                attn_metric.reshape(
                    len(layers) - args.start_pruning_layer_idx - 4,
                    args.num_heads // args.group_size,
                    args.group_size,
                    args.head_dim,
                )
                .mean(dim=1)
                .mean(dim=-1)
            )

        mlp_metric = torch.stack(mlp_metric_list)
        mlp_metric = standarlization(mlp_metric)

        prune_metric = torch.cat([attn_metric.view(-1), mlp_metric.view(-1)])
        sorted_prune, indices = torch.sort(prune_metric, descending=True)
        compression_weight = torch.ones_like(indices)
        compression_weight[indices < attn_metric.numel()] = 512.0 / 3
        threshold = sorted_prune[
            torch.argmin(
                torch.abs(
                    torch.cumsum(compression_weight, 0)
                    - torch.sum(compression_weight) * (1 - args.pruning_ratio)
                )
            )
        ]
        attn_mask = attn_metric > threshold
        mlp_mask = mlp_metric > threshold

    for idx in range(len(layers) - args.start_pruning_layer_idx - 4):
        actual_idx =  idx + args.start_pruning_layer_idx
        # print(attn_mask.shape)
        # print(idx, actual_idx)
        if f"model.layers.{i}" in getattr(model, "hf_device_map", {}):
            compress(
                model.model.layers[actual_idx],
                attn_mask[idx],
                None,
                attn_baseline_inp_list[idx],
                None,
                model.hf_device_map[f"model.layers.{actual_idx}"],
                args=args,
            )
        else:
            compress(
                model.model.layers[actual_idx],
                attn_mask[idx],
                None,
                attn_baseline_inp_list[idx],
                None,
                device,
                args=args,
            )

        if f"model.layers.{i}" in getattr(model, "hf_device_map", {}):
            compress(
                model.model.layers[actual_idx],
                None,
                mlp_mask[idx],
                None,
                mlp_baseline_inp_list[idx],
                model.hf_device_map[f"model.layers.{actual_idx}"],
            )
        else:
            compress(
                model.model.layers[actual_idx],
                None,
                mlp_mask[idx],
                None,
                mlp_baseline_inp_list[idx],
                device,
            )

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def calculate_bi(
    model,
    dataloader,
    tokenizer,
    pruning_method="angular_distance",
    pruning_token="last",
):
    """
    Calculate Block Influence (BI) scores for each layer.

    Parameters:
    - dataloader (DataLoader): DataLoader for the dataset.
    - tokenizer (Tokenizer): Tokenizer for the model inputs.
    - pruning_method (str, optional): Pruning method to use. One of "angular_distance", "cosine_similarity.
    - pruning_token (str, optional): Pruning token to use. One of "all", "last".

    Returns:
    - list: List of BI scores for each block.
    """
    scores = []
    num_batches = 0
    model = model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = tokenizer(
                batch["text"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(model.device)
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            if not scores:
                scores = [0] * (len(hidden_states) - 1)

            for i in range(1, len(hidden_states)):
                input_hidden_state = hidden_states[i - 1]
                output_hidden_state = hidden_states[i]
                if pruning_token == "last":
                    input_hidden_state = input_hidden_state[:, -1, :]
                    output_hidden_state = output_hidden_state[:, -1, :]
                sim = F.cosine_similarity(input_hidden_state, output_hidden_state)
                if pruning_method == "angular_distance":
                    sim = torch.clamp(sim, -1.0, 1.0)
                    sim = (1 / math.pi) * torch.acos(sim)
                elif pruning_method == "cosine_similarity":
                    sim = 1 - sim
                scores[i - 1] += sim.mean().item()

            num_batches += 1

    scores = [
        score / num_batches for score in scores
    ]  # Average scores over all batches
    return scores


def prune_model_blocks(
    model,
    importance_scores: list,
    num_blocks_to_prune: int,
    skip_blocks: list = None,
):
    """
    Prunes blocks from the transformer model based on the importance scores.

    Parameters:
    - importance_scores (list): List of importance scores for each block.
    - num_blocks_to_prune (int): Number of blocks to prune from the model.
    - skip_blocks (list, optional): List of block indices to skip. Defaults to None.

    Returns:
    - PreTrainedModel: The pruned transformer model.
    """

    # Assign max score to skip blocks
    if skip_blocks:
        for block in skip_blocks:
            importance_scores[block] = max(importance_scores)

    # Sort blocks by importance score
    sorted_blocks = sorted(
        range(len(importance_scores)), key=lambda i: importance_scores[i]
    )

    # Identify blocks to prune
    blocks_to_prune = sorted_blocks[:num_blocks_to_prune]

    # Create a new model without the pruned blocks
    pruned_model = copy.deepcopy(model)
    # pruned_model.load_state_dict(self.model.state_dict())

    # Prune the blocks
    layers = []
    for i, layer in enumerate(model.model.layers):
        if i in blocks_to_prune:
            continue
        layer = model.model.layers[i]
        layer.self_attn.layer_idx = len(layers)
        layers.append(layer)

    pruned_model.model.layers = torch.nn.ModuleList(layers)
    pruned_model.config.num_hidden_layers = len(model.model.layers) - len(
        blocks_to_prune
    )

    return pruned_model
