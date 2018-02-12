import torch.nn as nn
from torch.autograd import Variable
from torchvision.models.resnet import resnet50
import torch
import time
import numpy as np

torch.backends.cudnn.benchmark = True

NUM_GPUS = torch.cuda.device_count()
BATCH_LIST = [2**x for x in range(4, 10)] # 16 to 1024
WARM_UP = 5
NUM_STEP = 20

def main():
    benchmark = {}
    for batch_size in BATCH_LIST:
        benchmark[batch_size] = []
        for gpu in range(1, NUM_GPUS + 1):
            print('Benchmarking ResNet50 on batch size %i with %i GPUs' % (batch_size, gpu))
            model = resnet50()
            if gpu > 1:
                model = nn.DataParallel(model)
            model.cuda()
            model.eval()

            img = Variable(torch.randn(batch_size, 3, 224, 224), volatile=True).cuda()
            durations = []
            for step in range(NUM_STEP + WARM_UP):
                # test
                torch.cuda.synchronize()
                start = time.time()
                model(img)
                torch.cuda.synchronize()
                end = time.time()
                if step >= WARM_UP:
                    duration = (end - start) * 1000
                    durations.append(duration)
            benchmark[batch_size].append(durations)
            del model
    return benchmark


if __name__ == '__main__':
    benchmark = main()
    for key in benchmark.keys():
        for gpu, duration in enumerate(benchmark[key]):
            print('Batch size %i, # of GPUs %i, time cost %.4fms' % (key, gpu + 1, np.mean(duration)))
