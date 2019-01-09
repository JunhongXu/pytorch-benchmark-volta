#! /usr/bin/python3
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models.resnet import resnet50
import torch
import time
import numpy as np
import argparse
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

# Inference settings
parser = argparse.ArgumentParser(description='PyTorch Benchmarking')
parser.add_argument('--WARM_UP','-w', type=int,default=10, required=False, help="Num of warm up")
parser.add_argument('--NUM_TEST','-n', type=int,default=200,required=False, help="Num of Test")
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
NUM_GPUS = torch.cuda.device_count()
BATCH_LIST = [2 ** x for x in range(4, 8)]  # 16 to 2048
precision=[torch.float16,torch.float32,torch.float64] # precision

def main(i):
    benchmark = {}

    for batch_size in BATCH_LIST:
        benchmark[batch_size] = []
        for gpu in range(1, NUM_GPUS + 1):
            print('Benchmarking  type %s ResNet50 on batch size %i with %i GPUs' % (str(i).split('.')[-1],batch_size, gpu))
            model = resnet50()
            if gpu > 1:
                model = nn.DataParallel(model,device_ids=range(0,gpu))
            model.cuda()
            model.eval()

            img = torch.randn(batch_size, 3, 224, 224, device='cuda', requires_grad=False,dtype=torch.float32)
            durations = []
            for step in range(args.NUM_TEST + args.WARM_UP):
                # test
                torch.cuda.synchronize()
                start = time.time()
                model(img)
                torch.cuda.synchronize()
                end = time.time()
                if step >= args.WARM_UP:
                    duration = (end - start) * 1000
                    durations.append(duration)
            benchmark[batch_size].append(durations)
            del model
    return benchmark


if __name__ == '__main__':
    result=[]
for i in precision:
    result.append(main(i))
temp=[]
for bench,dtype in zip(result,['half','single','double']):
    for key in bench.keys():
        for gpu, duration in enumerate(bench[key]):
            print('Data Type %s, Batch size %i, # of GPUs %i, time cost %.4fms' % (dtype,key, gpu + 1, np.mean(duration)))
            temp.append([dtype,key, gpu + 1, np.mean(duration)])

# save csv
temp=np.array(temp)
df=pd.DataFrame(temp[:,1:],index=temp[:,0],columns=['batchs','gpus','times'])
df.to_csv('results/dgx.csv')
df=df.astype(np.float16)


# save fig
fig, axes = plt.subplots(nrows=4, ncols=3)
for i in range(0,len(df),4):
    df[i:i+4].plot(ax=axes[i//12,i%12//4],figsize=(12,10),x='gpus',y='times',grid=True,kind='bar',title=str(df['batchs'][i])+"batchs "+df.index[i]+" type")
fig.tight_layout()
fig.savefig('fig/dgx.png',dpi=600)
