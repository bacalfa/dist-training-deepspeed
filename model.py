from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple


def load_model(model_name: str = "gpt2") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads LLM and tokenizer to use with DeepSpeed.

    :param model_name: Model name (default: `"gpt2"`)
    :return: Tuple of `AutoModelForCausalLM` and `AutoTokenizer`
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer
