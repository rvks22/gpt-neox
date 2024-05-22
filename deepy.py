#!/usr/bin/env python
# Copyright (c) 2021, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os, sys

import deepspeed
from deepspeed.launcher.runner import main

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

from megatron.neox_arguments import NeoXArgs
from megatron.utils import get_wandb_api_key


neox_args = NeoXArgs.consume_deepy_args()
#@@RV insert
# neox_args.DEBUG = False
# neox_args.SKIP_BATCHES = 0
# neox_args.SAVE_BATCH_TO_FILE = '/workspace/Output/data_batch_1000.txt'
# neox_args.SAVE_BATCH_TO_FILE = neox_args.save + '/' + neox_args.SAVE_BATCH_TO_FILE

print('===== in deepy.py ======')
if neox_args.DEBUG:
    print('===========================================================================')
    print('NEOX_ARGS')
    print('===========================================================================')
    for k in sorted(list(neox_args.keys())):
        print((k, neox_args[k]))
    print('===========================================================================')
# CREATE SOME FOLDERS IF THEY DON'T EXIST
if not os.path.isdir(neox_args.save):
    os.makedirs(neox_args.save)


deepspeed_main_args = neox_args.get_deepspeed_main_args()

# Extract wandb API key and inject into worker environments
wandb_token = get_wandb_api_key(neox_args=neox_args)
if wandb_token is not None:
    deepspeed.launcher.runner.EXPORT_ENVS.append("WANDB_API_KEY")
    os.environ["WANDB_API_KEY"] = wandb_token

print('===== in deepy.py ======')
if neox_args.DEBUG:
    print('===========================================================================')
    print("DEEPSPEED_MAIN_ARGS")
    print('===========================================================================')
    print(deepspeed_main_args)
    print('===========================================================================')


# sys.exit(5)

if __name__ == "__main__":
    main(deepspeed_main_args)
