"""Compare speed of different models with batch size 16"""
import torch
from torchvision.models import resnet, densenet, vgg, squeezenet,inception
from torch.autograd import Variable
from info_utils import print_info
import torch.nn as nn
import time
import pandas
import argparse
import os
from plot import *

print_info()

MODEL_LIST = {
    resnet: resnet.__all__[1:],
    densenet: densenet.__all__[1:],
    squeezenet: squeezenet.__all__[1:],
    vgg: vgg.__all__[5:]
}

precision=["single","half",'double']
device_name=torch.cuda.get_device_name(0)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Benchmarking')
parser.add_argument('--WARM_UP','-w', type=int,default=5, required=False, help="Num of warm up")
parser.add_argument('--NUM_TEST','-n', type=int,default=50,required=False, help="Num of Test")
parser.add_argument('--BATCH_SIZE','-b', type=int, default=20, required=False, help='Num of batch size')
parser.add_argument('--NUM_CLASSES','-c', type=int, default=1000, required=False, help='Num of class')
parser.add_argument('--NUM_GPU','-g', type=int, default=1, required=False, help='Num of class')

args = parser.parse_args()
device_name+='_'+str(args.NUM_GPU)+'_gpus_'
args.BATCH_SIZE*=args.NUM_GPU
torch.backends.cudnn.benchmark = True
def train(type='single'):
    """use fake image for training speed test"""
    img = Variable(torch.randn(args.BATCH_SIZE, 3, 224, 224)).cuda()
    target = Variable(torch.LongTensor(args.BATCH_SIZE).random_(args.NUM_CLASSES)).cuda()
    criterion = nn.CrossEntropyLoss()
    benchmark = {}
    for model_type in MODEL_LIST.keys():
        for model_name in MODEL_LIST[model_type]:
            model = getattr(model_type, model_name)(pretrained=False)
            if args.NUM_GPU > 1:
                model = nn.DataParallel(model)
            if type is 'double':
                model=model.double()
                img=img.double()
            elif type is 'single':
                model=model.float()
                img=img.float()
            elif type is 'half':
                model=model.half()
                img=img.half()
            model.cuda()
            model.train()
            durations = []
            print('Benchmarking Training '+type+' precision type %s' % (model_name))
            for step in range(args.WARM_UP + args.NUM_TEST):
                torch.cuda.synchronize()
                start = time.time()
                model.zero_grad()
                prediction = model.forward(img)
                loss = criterion(prediction, target)
                loss.backward()
                torch.cuda.synchronize()
                end = time.time()
                if step >= args.WARM_UP:
                    durations.append((end - start)*1000)
            del model
            benchmark[model_name] = durations
    return benchmark

def inference(type='single'):
    benchmark = {}
    img = Variable(torch.randn(args.BATCH_SIZE, 3, 224, 224), requires_grad=True).cuda()
    with torch.no_grad():
        for model_type in MODEL_LIST.keys():
            for model_name in MODEL_LIST[model_type]:
                model = getattr(model_type, model_name)(pretrained=False)
                if args.NUM_GPU > 1:
                    model = nn.DataParallel(model)
                if type is 'double':
                    model=model.double()
                    img=img.double()
                elif type is 'single':
                    model=model.float()
                    img=img.float()
                elif type is 'half':
                    model=model.half()
                    img=img.half()
                model.cuda()
                model.eval()
                durations = []
                print('Benchmarking Inference '+type+' precision type %s ' % (model_name))
                for step in range(args.WARM_UP + args.NUM_TEST):
                    torch.cuda.synchronize()
                    start = time.time()
                    model.forward(img)
                    torch.cuda.synchronize()
                    end = time.time()
                    if step >= args.WARM_UP:
                        durations.append((end - start)*1000)
                del model
                benchmark[model_name] = durations
    return benchmark



if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    for i in precision:
        training_benchmark = pandas.DataFrame(train(i))
        training_benchmark.to_csv('results/'+device_name+"_"+i+'_model_training_benchmark.csv', index=False)
        inference_benchmark = pandas.DataFrame(inference(i))
        inference_benchmark.to_csv('results/'+device_name+"_"+i+'_model_inference_benchmark.csv', index=False)
    train=arr_train()
    inference=arr_inference()


    total_model(train,device_name)
    total_model(inference,device_name)
