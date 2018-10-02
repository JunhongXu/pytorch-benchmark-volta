# About
Comparison of learning and inference speed of different gpu with various cnn models in __pytorch__

* 1080TI
* TITAN V
* 2080TI

# Specification

|Graphics Card Name|NVIDIA GeForce GTX 1080 Ti|NVIDIA GeForce RTX 2080 Ti|NVIDIA TITAN V
|:-------:|:-------:|:-------:|:-------:|
|Process|16nm|12nm|12nm|
|Die Size|471mm²|754mm²|815mm²|
|Transistors|12 Billion|18.6 Billion|21.1Billion|
|CUDA Cores|3584 Cores|4352 Cores|5120 Cores|
|Clock|1480 MHz|1350 MHz|1455 MHz|
|Compute(single precision)|11.5 TFLOPs|13.4 TFLOPs|13.8 TFLOPS|
|Memory|11GB GDDR5X|11 GB GDDR6|12 GB HBM2|
|Memory Speed|11Gbps|14.00 Gbps|1.7Gbps HBM2|
|Memory Interface|352-bit|352-bit|3072-bit|
|Memory Bandwidth|484 GB/s|616 GB/s|653GB/s
|Price|$699 US|$1,199 US|$2,999 US|





1. Single GPU with batch size 16: compare training and inference speed of **SequeezeNet, VGG-16, VGG-19, ResNet18, ResNet34, ResNet50, ResNet101,
ResNet152, DenseNet121, DenseNet169, DenseNet201, DenseNet161**

2. Experiments are performed on three types of datatype. single precision, double precision, half precision

3. making plot

## Usage

`./test.sh`

## Results

###  requirement
* python3-tk
* matplotlib
* pandas
* PyTorch
* torchvision

### Environment

* Pytorch version `1.0.0a0+2cbcaf4`
* Number of GPUs on current device `1`
* CUDA version = `10.0.130`
* CUDNN version= `7301`


### Comparison between networks (single GPU)

Each network is fed with 16 images with 224x224x3 dimensions.
For training, time durations of 20 passes of forward and backward are averaged. For inference, time durations of
20 passes of forward are averaged. 5 warm up steps are performed that do not calculate towards the final result.


|   Mode  |gpu|precision|squeezenet1_1| resnet18|seqeezenet1_0| resnet34| resnet50  |densenet121| vgg16 |densenet169| vgg19 |resnet101|densenet201|resnet152|densenet161|
|:-------:| :----:|:--:|:----------:|:-------:|:-----------:|:-------:|:---------:| ---------:|:-----:|:---------:|:-----:|:------:|:---------:|:-------:|:---------:|
|Training | TITAN V|single|17.16ms     |18.09ms  |   18.58ms   | 30.04ms | 55.07ms   |  66.56ms  |76.74ms|  85.95ms  |88.35ms| 93.59ms| 108.81ms  |131.27ms |  131.55ms |
|Inference| TITAN V|single|3.32ms      |5.03ms   |   5.24ms    | 8.51ms  | 15.74ms   |20.289ms   |23.83ms|  27.73ms  |27.66ms| 26.65ms|  36.27    |38.01ms  |   41.19ms |
