# whichStep = 1000
# whichModelSize = '160m'

import os, sys
import argparse

parser = argparse.ArgumentParser(
    prog="save_hf_model_checkpoint",
    usage="%(prog)s [step] [size]",
    description="""
        save a checkpoint in HF format;
        step: number corresponding to a checkpoint;
        size: pythia model size, e.g. 160m, 6.9b
        """
)

parser.add_argument("step")
parser.add_argument("size")
args = parser.parse_args()
print(args)

if args.step is not None:
    whichStep = int(args.step)
else:
    print('')
if args.size is not None:
    whichModelSize = args.size
print(f"whichStep={whichStep}, whichModelSize={whichModelSize}")


# some checks should be here: checkpoints, output should exist already

# create subdirectory if it doesn't exist
modelOutSubdir = f"output/model_{whichModelSize}/step_{whichStep}"
if not os.path.isdir(modelOutSubdir):
    os.makedirs(modelOutSubdir)

import subprocess
from capture_subprocess_output import capture_subprocess_output

print('Got here!!!')

# prepare the command
cmd = ['python','tools/convert_to_hf.py',
        "--input_dir", f"checkpoints/model_{whichModelSize}/global_step{whichStep}/",
        "--config_file", f"checkpoints/model_{whichModelSize}/global_step{whichStep}/configs/pythia-{whichModelSize}-deduped.yml",
        "--output_dir", f"./output/model_{whichModelSize}/step_{whichStep}"
        ]
# cmd = ['python',
#         f'tools/convert_to_hf.py --input_dir checkpoints/global_step{whichStep}/ --config_file checkpoints/global_step{whichStep}/configs/pythia-{whichModelSize}-deduped.yml --output_dir ./output/model_{whichModelSize}/step_{whichStep}'
#         ]
cmd2 = [x.replace('\n',' ') for x in cmd]
print(cmd2)
capture_subprocess_output(cmd2)
# subprocess.Popen(cmd, shell = True)