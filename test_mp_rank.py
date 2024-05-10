
import os, sys
import torch
import numpy as np

model_path1 = "/workspace/checkpoints/model_161m/global_step8"
model_path2 = "/workspace/checkpoints/model_162m/global_step8"

model_path1 = "/workspace/checkpoints/model_16xm/global_step16"
model_path2 = "/workspace/checkpoints/model_16ym/global_step16"

model_path1 = "/workspace/checkpoints/model_16A/global_step8"
model_path2 = "/workspace/checkpoints/model_16G/global_step8"

xx1 = torch.load(os.path.join(model_path1, f"mp_rank_00_model_states.pt"))
print(xx1.keys())

xx2 = torch.load(os.path.join(model_path2, f"mp_rank_00_model_states.pt"))
# print(xx)

# def compare_things(lhs, rhs):
#     return all(lhs == rhs)


def compare_things(lhs = None, rhs = None, tkey = None, pos = None, infostring = '', DEBUG = False):
    
    if tkey is not None:
        if DEBUG: print(f"{infostring} got tkey = {tkey}")
        lhs = lhs[tkey]
        rhs = rhs[tkey]
        tracker = f"tkey={tkey}"
    elif pos is not None:
        if DEBUG: print(f"{infostring} got position pos={pos}")
        lhs = lhs[pos]
        rhs = rhs[pos]
        tracker = f"pos={pos}"

    if DEBUG: print(type(lhs))

    if isinstance(lhs, tuple):
        if DEBUG: print(f"{infostring} comparing tuples")
        # is this a homogeneous tuple?
        tcounter = len(set([type(x) for x in lhs]))
        if tcounter == 1:
            if lhs != rhs:
                print(f"differences in {infostring}!")
        else:
            for i in range(len(lhs)):
                newinfostring = f"{infostring} {tracker} tuple.pos={i}"
                if DEBUG: print(f'{infostring} going to compare tuple position {i}')
                compare_things(lhs = lhs, rhs = rhs, pos = i, infostring = newinfostring, DEBUG = DEBUG)
    elif isinstance(lhs, torch.Tensor):
        if DEBUG: print(f"{infostring} comparing torch Tensors")
        if not torch.equal(lhs,rhs):
            print(f"differences in Tensor: {infostring}")
    elif isinstance(lhs, dict):
        if DEBUG: print(f"{infostring} comparing dicts")
        for tk in lhs.keys():
            newinfostring = f"{infostring} {tracker} dict.key={tk}"
            compare_things(lhs = lhs, rhs = rhs, tkey = tk, infostring = newinfostring, DEBUG = DEBUG)
            if DEBUG: print(f'{infostring} going to compare some dicts for {tk}')
    elif isinstance(lhs, np.ndarray):
        if DEBUG: print("{infostring} lhs is an np.ndarray")
        if any(lhs != rhs):
            print(f"differences in np.ndarray in {infostring}")
    else: # assume singleton
        if DEBUG: print(f"{infostring} lhs is a singleton")
        if lhs != rhs:
            print(f"differences in singleton {infostring}")

for tkey in xx1.keys():
    # [
    #     'random_rng_state',
    #         'np_rng_state',
    #         'torch_rng_state',
    #         'cuda_rng_state',
    #         'rng_tracker_states',
    #         ]:
    print(f'============ {tkey} =============')
    if tkey == 'iteration':
        print(xx1[tkey])
        print(xx2[tkey])
    compare_things(xx1, xx2, tkey = tkey, DEBUG = True)

