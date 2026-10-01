import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# basic rule of func:
#   1st argument: process index
#   2nd argument: collection of args, including addr, port and num_gpu
def run_process(func, args):
    ''' Problem 1: Run process
    (./handler/DDP/utils.py)
    Implement run_process function.
    run_process function is used to run main_func in multiple processes, for DDP GPU group.
    It is a wrapper of mp.spawn function.
    You can use mp.spawn function as a reference.
    '''
    mp.spawn(func, args=(args,), nprocs=args.num_gpu)

def initialize_group(proc_id, host, port, num_gpu):
    ''' Problem 2: Setup GPU group
    (./handler/DDP/utils.py)
    DDP requires to setup GPU group, which can broadcast weights to all GPUs.
    This function set tcp connection between processes.
    Implement initialize_group function.

    you should use
    1. dist.init_process_group() for tcp connection
    2. torch.cuda.set_device() for setting device
    '''
    dist_url = f"tcp://{host}:{port}"
    torch.cuda.set_device(proc_id)
    dist.init_process_group(backend="nccl", init_method=dist_url, world_size=num_gpu, rank=proc_id)


def destroy_process():
    ''' Problem 3: Destroy GPU group
    (./handler/DDP/utils.py)
    Implement destroy_process function.
    Just call the torch.distributed's destroy function.
    '''
    if dist.is_initialized():
        dist.destroy_process_group()
    torch.cuda.empty_cache()

