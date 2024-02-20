import sys
# import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

"""
Tests running simple inference on custom prompts.

` python run_inference.py "prompt" "optional additional prompt" "etc." `
"""


def main():
    # prompts (by default run unconditional generation)
    prompts = [""]
    if len(sys.argv) > 1:
        prompts = [sys.argv[i] for i in range(1, len(sys.argv))]

    # # set up system to run model
    # assert torch

    # set up model
    model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # generate
    inputs = tokenizer(prompts, return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids)
    predictions = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    for pred in predictions:
        print(pred)


if __name__ == "__main__":
    main()
