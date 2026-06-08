"""Definition of the low-rank adaptation (LoRA) for large models.

References:
    1. LoRA: Low-Rank Adaptation of Large Language Models
       (https://arxiv.org/pdf/2106.09685.pdf)
    2. https://github.com/microsoft/LoRA.git
    3. https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora.py

"""

from typing import List
import logging

import torch
from typeguard import check_argument_types
from espnet2.torch_utils.model_summary import model_summary


from espnet2.layers.create_adapter_fn import (
        create_houlsby_adapter, 
        create_lora_adapter, 
        create_ssvdo_adapter,
        create_milora_adapter,
        create_bilora_adapter,
        create_oplora_adapter,
)
from espnet2.train.class_choices import ClassChoices

create_adapter_fn_table = {
    "lora": create_lora_adapter,
    "houlsby": create_houlsby_adapter,
    'ssvdo': create_ssvdo_adapter,
    'milora': create_milora_adapter,
    'bilora': create_bilora_adapter,
    'oplora': create_oplora_adapter,
}

def get_args(args, create_adapter_fn):
    import inspect  # to get init args of cl method
    opt_args = {
            'init_param': args.init_param,
               }
    method_args = inspect.signature(create_adapter_fn).parameters
    return {arg: val for arg, val in opt_args.items()
            if arg in method_args}

def create_adapter(
    model: torch.nn.Module,
    adapter: str,
    adapter_conf: dict,
    args,
):
    """Create adapter for the base model.


    Args:
        model (torch.nn.Module): Base model to be adapted.
        adapter_type (str): Name of adapter
        adapter_conf (dict): Configuration for the adapter
            e.g.  {"rank": 8, "alpha": 8, ...} for lora

    """
    assert check_argument_types()
    assert adapter in create_adapter_fn_table, f"Adapter {adapter} is not supported."
    create_adapter_fn = create_adapter_fn_table[adapter]
    add_adapter_conf = get_args(args, create_adapter_fn)
    adapter_conf = {**adapter_conf, **add_adapter_conf}
    logging.info(f'adapter_conf = {adapter_conf}')
    create_adapter_fn(model=model, **adapter_conf)
    logging.info(model_summary(model))
