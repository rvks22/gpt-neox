# source: https://huggingface.co/EleutherAI/pythia-6.9b

from transformers import GPTNeoXForCausalLM, AutoTokenizer
import argparse

parser = argparse.ArgumentParser(
    prog="ks_test_checkpoint",
    usage="%(prog)s [step] [size] [max_new_tokens]",
    description="""
        save a checkpoint in HF format;
        step: number corresponding to a checkpoint;
        size: pythia model size, e.g. 160m, 6.9b
        max_new_tokens: number indicating how many tokens to generate
        """
)

parser.add_argument("step")
parser.add_argument("size")
parser.add_argument("max_new_tokens")
args = parser.parse_args()
print(args)

if args.step is not None:
    whichStep = int(args.step)
if args.size is not None:
    whichModelSize = args.size
if args.max_new_tokens is not None:
  max_new_tokens = int(args.max_new_tokens)
# print(f"whichStep={whichStep}, whichModelSize={whichModelSize}")


model = GPTNeoXForCausalLM.from_pretrained(
  f"./output/model_{whichModelSize}/step_{whichStep}",
  revision=f"global_step{whichStep}",
  cache_dir="./checkpoints",
)

tokenizer = AutoTokenizer.from_pretrained(
  f"./output/model_{whichModelSize}/step_{whichStep}",
  revision=f"global_step{whichStep}",
  cache_dir="./checkpoints",
)

inputs = tokenizer("Hello, I am", return_tensors="pt")
inputs2 = inputs.copy()
del inputs2['token_type_ids']
inputs2['max_new_tokens'] = max_new_tokens
tokens = model.generate(**inputs2)
answer = tokenizer.decode(tokens[0])
print(answer)