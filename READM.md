## Goal

1. Single GPU with batch size 16: compare training and inference speed of **AlexNet, VGG-16, VGG-19, ResNet18, ResNet34, ResNet50, ResNet101, 
ResNet152, DenseNet121, DenseNet169, DenseNet201, DenseNet161**

2. Multiple GPUs vs variant batch sizes: from the initial observation of using 8 GPUs to do inference using batch size with 16
gives me much lower speed. I assume this might be caused by communication between GPUs, but I am not sure what information is ported
from one GPU to others. This experiment wants to find how the computation speed changes 
when we change # of GPUs use and the batch size. ResNet-50 is used across this experiment.
    - 1 GPU with batch size [2**i for i in range(4, 10)]
    - 2 GPU with batch size [2**i for i in range(4, 10)]
    - 3 GPU with batch size [2**i for i in range(4, 10)]
    - 4 GPU with batch size [2**i for i in range(4, 10)]
    - 5 GPU with batch size [2**i for i in range(4, 10)]
    - 6 GPU with batch size [2**i for i in range(4, 10)]
    - 6 GPU with batch size [2**i for i in range(4, 10)]
    - 8 GPU with batch size [2**i for i in range(4, 10)]
    
## Results