## Goal

1. Single GPU with batch size 16: compare training and inference speed of **AlexNet, VGG-16, VGG-19, ResNet18, ResNet34, ResNet50, ResNet101, 
ResNet152, DenseNet121, DenseNet169, DenseNet201, DenseNet161**

2. Multiple GPUs vs variant batch sizes: from the initial observation of using 8 GPUs to do inference using batch size with 16
gives me much lower speed. I assume this might be caused by communication between GPUs, but I am not sure what information is ported
from one GPU to others. This experiment wants to find how the computation speed changes 
when we change # of GPUs use and the batch size. ResNet101 is used across this experiment.
    - 1 GPU with batch size [2**i for i in range(4, 10)]
    - 2 GPU with batch size [2**i for i in range(4, 10)]
    - 3 GPU with batch size [2**i for i in range(4, 10)]
    - 4 GPU with batch size [2**i for i in range(4, 10)]
    - 5 GPU with batch size [2**i for i in range(4, 10)]
    - 6 GPU with batch size [2**i for i in range(4, 10)]
    - 6 GPU with batch size [2**i for i in range(4, 10)]
    - 8 GPU with batch size [2**i for i in range(4, 10)]
    
## Results

### Compare between networks (single GPU, training)

|   Mode  |squeezenet1_1| resnet18|seqeezenet1_0| resnet34| resnet50  |densenet121| vgg16 |densenet169| vgg19 |resnet101|densenet201|resnet152|densenet161|
|:-------:| :----------:|:-------:|:-----------:|:-------:|:---------:| ---------:|:-----:|:---------:|:-----:|:------:|:---------:|:-------:|:---------:|
|Training | 17.16ms     |18.09ms  |   18.58ms   | 30.04ms | 55.07ms   |  66.56ms  |76.74ms|  85.95ms  |88.35ms| 93.59ms| 108.81ms  |131.27ms |  131.55ms |
|Inference| 3.32ms      |5.03ms   |   5.24ms    | 8.51ms  | 15.74ms   |20.289ms   |23.83ms|  27.73ms  |27.66ms| 26.65ms|  36.27    |38.01ms  |   41.19ms |
