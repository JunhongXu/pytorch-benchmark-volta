## Goal

Performance of popular deep learning models on DGX-1 GPUs. DGX-1 GPUs are Tesla V100 with the following specs:

1. 15.7 TeraFLOPS.
2. 125 TeraFLOPS for deep learning.
3. 900 GB/S bandwidth.
4. 640 TensorCores.
5. 5120 CUDA cores.

The following tasks are compared:

1. Single GPU with batch size 16: compare training and inference speed of **SequeezeNet, VGG-16, VGG-19, ResNet18, ResNet34, ResNet50, ResNet101, 
ResNet152, DenseNet121, DenseNet169, DenseNet201, DenseNet161**

2. Multiple GPUs vs variant batch sizes: from the initial observation of using 8 GPUs to do inference using batch size with 16
gives me much lower speed. I assume this might be caused by communication between GPUs, but I am not sure what information is ported
from one GPU to others. This experiment wants to find how the computation speed changes 
when we change # of GPUs and the batch size. ResNet18 is used across this experiment.
    - 1 GPU with batch size [2 ** x for x in range(4, 12)]
    - 2 GPU with batch size [2 ** x for x in range(4, 12)]
    - 3 GPU with batch size [2 ** x for x in range(4, 12)]
    - 4 GPU with batch size [2 ** x for x in range(4, 12)]
    - 5 GPU with batch size [2 ** x for x in range(4, 12)]
    - 6 GPU with batch size [2 ** x for x in range(4, 12)]
    - 6 GPU with batch size [2 ** x for x in range(4, 12)]
    - 8 GPU with batch size [2 ** x for x in range(4, 12)]
    
## Results

### Comparison between networks (single GPU, training)

Each network is fed with 16 images with 224x224x3 dimensions.
For training, time durations of 20 passes of forward and backward are averaged. For inference, time durations of 
20 passes of forward are averaged. 5 warm up steps are performed that do not calculate towards the final result.


|   Mode  |squeezenet1_1| resnet18|seqeezenet1_0| resnet34| resnet50  |densenet121| vgg16 |densenet169| vgg19 |resnet101|densenet201|resnet152|densenet161|
|:-------:| :----------:|:-------:|:-----------:|:-------:|:---------:| ---------:|:-----:|:---------:|:-----:|:------:|:---------:|:-------:|:---------:|
|Training | 17.16ms     |18.09ms  |   18.58ms   | 30.04ms | 55.07ms   |  66.56ms  |76.74ms|  85.95ms  |88.35ms| 93.59ms| 108.81ms  |131.27ms |  131.55ms |
|Inference| 3.32ms      |5.03ms   |   5.24ms    | 8.51ms  | 15.74ms   |20.289ms   |23.83ms|  27.73ms  |27.66ms| 26.65ms|  36.27    |38.01ms  |   41.19ms |


### Comparison between GPUs on variant batch sizes (inference only)

ResNet18 is used for comparison. 

