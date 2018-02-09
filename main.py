import torch.nn as nn
from torch.autograd import Variable
from torchvision.models.resnet import resnet152
import torch
import time
import numpy as np

torch.backends.cudnn.benchmark = True
ngpus = torch.cuda.device_count()
print('Availible devices: %i' % ngpus)


def main(num_iter=20, num_warmups=5):
    device_list = []
    for i in range(0, ngpus+2, 2):
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
                print('\rTime spent %.4fms' % ((end - start)*1000), flush=True, end='')
        print('\nAverage time spent %.4fms' % (np.mean(durations)*1000))
        del model
main()
