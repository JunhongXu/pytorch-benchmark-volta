"""Compare speed of different models with batch size 16"""
import torch
from torchvision.models import resnet, densenet, vgg, squeezenet
from torch.autograd import Variable
import matplotlib

# set the backend that does not use display, necessary for DGX-1
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch.nn as nn
import time

MODEL_LIST = {
    resnet: resnet.__all__[1:],
    densenet: densenet.__all__[1:],
    squeezenet: squeezenet.__all__[1:],
    vgg: vgg.__all__[4:]
}

# for model_type in MODEL_LIST.keys():
#     for model_name in MODEL_LIST[model_type]:
#         print(getattr(model_type, model_name)())

torch.backends.cudnn.benchmark = True

WARM_UP = 5
NUM_TEST = 20
BATCH_SIZE = 16
NUM_CLASSES = 100


def train():
    """use fake image for training speed test"""
    img = Variable(torch.randn(BATCH_SIZE, 3, 224, 224)).cuda()
    target = Variable(torch.LongTensor(BATCH_SIZE).random_(NUM_CLASSES)).cuda()
    criterion = nn.CrossEntropyLoss()

    benchmark = {}
    for model_type in MODEL_LIST.keys():
        # model = getattr()
        for model_name in MODEL_LIST[model_type]:
            model = getattr(model_type, model_name)()
            model.cuda()
            model.eval()
            durations = []

            for step in range(WARM_UP + NUM_TEST):
                torch.cuda.synchronize()
                start = time.time()
                model.zero_grad()
                prediction = model.forward(img)
                loss = criterion(prediction, target)
                loss.backward()
                torch.cuda.synchronize()
                end = time.time()
                if step >= WARM_UP:
                    durations.append(end - start)

            benchmark[model_name] = durations
    return benchmark


def inference():
    pass


if __name__ == '__main__':
    print(train())
