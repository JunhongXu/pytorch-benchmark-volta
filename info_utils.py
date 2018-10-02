import torch


def print_num_gpus():
    print('Number of GPUs on current device %i' % torch.cuda.device_count())


def print_cuda_version():
    print('CUDA version = %s' % torch.version.cuda)


def print_cudnn_version():
    print('cudnn version=', torch.backends.cudnn.version())

def print_device_name():
    print('device_name=', torch.cuda.get_device_name(0))
    

def print_info():
    print_num_gpus()
    print_cuda_version()
    print_cudnn_version()
    print_device_name()
