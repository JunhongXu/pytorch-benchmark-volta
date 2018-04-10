"""Compare speed of different models with batch size 16"""
import torch
from torchvision.models import resnet, densenet, vgg, squeezenet
from torch.autograd import Variable
from info_utils import print_info
import torch.nn as nn
import time
import pandas
import argparse
import os
print_info()

MODEL_LIST = {
    resnet: resnet.__all__[1:],
    densenet: densenet.__all__[1:],
    squeezenet: squeezenet.__all__[1:],
    vgg: vgg.__all__[5:]
}

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Benchmarking')
parser.add_argument('--WARM_UP','-w', type=int,default=10, required=False, help="Num of warm up")
parser.add_argument('--NUM_TEST','-n', type=int,default=100,required=False, help="Num of Test")
parser.add_argument('--BATCH_SIZE','-b', type=int, default=16, required=False, help='Num of batch size')
parser.add_argument('--NUM_CLASSES','-c', type=int, default=100, required=False, help='Num of class')
parser.add_argument('--DATA_TYPE','-t', type=int, default=1, required=False, help='Floating data type Ex Double ,Float ,Half')
args = parser.parse_args()
torch.backends.cudnn.benchmark = True
def train():
    """use fake image for training speed test"""
    img = Variable(torch.randn(args.BATCH_SIZE, 3, 224, 224)).cuda()
    target = Variable(torch.LongTensor(args.BATCH_SIZE).random_(args.NUM_CLASSES)).cuda()
    criterion = nn.CrossEntropyLoss()
    benchmark = {}
    for model_type in MODEL_LIST.keys():
        for model_name in MODEL_LIST[model_type]:
            model = getattr(model_type, model_name)()
            if args.DATA_TYPE is 1:
                model=model.double()
                img=img.double()
            elif args.DATA_TYPE is 2:
                model=model.float()
                img=img.float()
            elif args.DATA_TYPE is 3:
                model=model.half()
                img=img.half()
            model.cuda()
            model.train()
            durations = []
            print('Benchmarking %s' % (model_name))
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

def inference():
    benchmark = {}
    img = Variable(torch.randn(args.BATCH_SIZE, 3, 224, 224), volatile=True).cuda()
    for model_type in MODEL_LIST.keys():
        for model_name in MODEL_LIST[model_type]:
            model = getattr(model_type, model_name)()
            if args.DATA_TYPE is 1:
                model=model.double()
                img=img.double()
            elif args.DATA_TYPE is 2:
                model=model.float()
                img=img.float()
            elif args.DATA_TYPE is 3:
                model=model.half()
                img=img.half()
            model.cuda()
            model.eval()
            durations = []
            print('Benchmarking %s' % (model_name))
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
    os.makedir('results/'+str(args.WARM_UP)+'/'+str(args.NUM_TEST)+'/'+str(args.BATCH_SIZE)+'/'+str(args.NUM_CLASSES)+'/'+str(args.DATA_TYPE))
    training_benchmark = pandas.DataFrame(train())
    training_benchmark.to_csv('results/'+str(args.WARM_UP)+'/'+str(args.NUM_TEST)+'/'+str(args.BATCH_SIZE)+'/'+str(args.NUM_CLASSES)+'/'+str(args.DATA_TYPE)+'/model_training_benchmark', index=False)
    training_benchmark.describe().to_csv('results/'+str(args.WARM_UP)+'/'+str(args.NUM_TEST)+'/'+str(args.BATCH_SIZE)+'/'+str(args.NUM_CLASSES)+'/'+str(args.DATA_TYPE)+'/model_training_benchmark_describe', index=False)

    inference_benchmark = pandas.DataFrame(inference())
    inference_benchmark.to_csv('results/'+str(args.WARM_UP)+'/'+str(args.NUM_TEST)+'/'+str(args.BATCH_SIZE)+'/'+str(args.NUM_CLASSES)+'/'+str(args.DATA_TYPE)+'/model_inference_benchmark', index=False)
    inference_benchmark.describe().to_csv('results/'+str(args.WARM_UP)+'/'+str(args.NUM_TEST)+'/'+str(args.BATCH_SIZE)+'/'+str(args.NUM_CLASSES)+'/'+str(args.DATA_TYPE)+'/model_inference_benchmark_describe', index=False)
