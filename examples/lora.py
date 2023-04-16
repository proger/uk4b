
import warnings

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType
from peft.tuners.lora import Linear, Linear8bitLt, MergedLinear, LoraLayer
from transformers.pytorch_utils import Conv1D
import bitsandbytes as bnb

gpt2_peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=4, lora_alpha=32, lora_dropout=0.1,
    target_modules=["c_attn"]
)

def lora_find_and_replace(self, peft_config):
    is_target_modules_in_base_model = False
    kwargs = {
        "r": peft_config.r,
        "lora_alpha": peft_config.lora_alpha,
        "lora_dropout": peft_config.lora_dropout,
        "fan_in_fan_out": peft_config.fan_in_fan_out,
        "merge_weights": peft_config.merge_weights,
    }
    key_list = [key for key, _ in self.named_modules()]
    for key in key_list:
        if any(key.endswith(target_key) for target_key in peft_config.target_modules):
            if not is_target_modules_in_base_model:
                is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self, key)
            bias = target.bias is not None
            if isinstance(target, bnb.nn.Linear8bitLt) and peft_config.enable_lora is None:
                kwargs.update(
                    {
                        "has_fp16_weights": target.state.has_fp16_weights,
                        "memory_efficient_backward": target.state.memory_efficient_backward,
                        "threshold": target.state.threshold,
                        "index": target.index,
                    }
                )
                new_module = Linear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
            elif isinstance(target, torch.nn.Linear) and peft_config.enable_lora is None:
                new_module = Linear(target.in_features, target.out_features, bias=bias, **kwargs)
            elif peft_config.enable_lora is not None:
                kwargs.update({"enable_lora": peft_config.enable_lora})
                if isinstance(target, Conv1D):
                    in_features, out_features = target.weight.shape
                else:
                    in_features, out_features = target.in_features, target.out_features
                    if kwargs["fan_in_fan_out"]:
                        warnings.warn(
                            "fan_in_fan_out is set to True but the target module is not a Conv1D. "
                            "Setting fan_in_fan_out to False."
                        )
                        kwargs["fan_in_fan_out"] = False
                new_module = MergedLinear(in_features, out_features, bias=bias, **kwargs)
            _replace_module(parent, target_name, new_module, target)
    if not is_target_modules_in_base_model:
        raise ValueError(
            f"Target modules {peft_config.target_modules} not found in the base model. "
            f"Please check the target modules and try again."
        )


def _get_submodules(self, key):
    parent = self.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = self.get_submodule(key)
    return parent, target, target_name


def _replace_module(parent_module, child_name, new_module, old_module):
    setattr(parent_module, child_name, new_module)
    new_module.weight = old_module.weight
    if old_module.bias is not None:
        new_module.bias = old_module.bias
    if getattr(old_module, "state", None) is not None:
        new_module.state = old_module.state
        new_module.to(old_module.weight.device)



# had to adapt it for `lora_only` to work
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def print_trainable_parameters(self):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in self.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

