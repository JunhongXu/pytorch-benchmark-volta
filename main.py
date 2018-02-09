import torch.nn as nn
from torch.autograd import Variable
from torchvision.models.resnet import resnet152
import torch
import time
import numpy as np

torch.backends.cudnn.benchmark = True

def main(num_iter=20, num_warmups=5):
    # resnet = resnet152()
    device_list = []
    for i in range(0, 10, 2):
        devices = []
        for ngpu in range(i):
            devices.append(ngpu)
        device_list.append(devices)

    for devices in device_list:
        model = resnet152()
        if len(devices) > 0:
            model = nn.DataParallel(model, device_ids=devices)
        model.cuda()
        print('Starting benchmarking %i gpus' %(0 if len(devices)==0 else len(devices)))
        durations = []
        fake_img = Variable(torch.randn(16, 3, 224, 224)).cuda()
        for i in range(num_iter + num_warmups):
            torch.cuda.synchronize()
            start = time.time()
            model(fake_img)
            torch.cuda.synchronize()
            end = time.time()
            if i >= num_warmups:
                durations.append(end - start)
                print('\rTime spent %.4fms' % (end - start)*1000, flush=True, end='')

        del model
main()
