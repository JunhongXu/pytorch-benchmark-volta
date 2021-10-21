# About
Comparison of learning and inference speed of different GPU with various CNN models in __pytorch__

* 1080TI
* TITAN XP
* TITAN V
* 2080TI
* Titan RTX
* RTX 2060
* RTX 3090
* A100-PCIE
* A100-SXM4

# Specification
| Graphics Card Name |   GTX 1080 Ti  |    TITAN XP    |     TITAN V    |    RTX 2060    |   RTX 2080 Ti  |    TITAN RTX   |    A100-PCIE   |    RTX 3090    |
|:------------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|
|       Process      |      16nm      |      16nm      |      12nm      |      12nm      |      12nm      |      12nm      |       7nm      |      8 nm      |
|      Die Size      |     471mm²     |     471mm²     |     815mm²     |     445 mm²    |     754mm²     |     754mm²     |     826mm²     |     628 mm²    |
|     Transistors    | 11,800 million | 11,800 million | 21,100 million | 10,800 million | 18,600 million | 18,600 million | 54,200 million | 28,300 million |
|     CUDA Cores     |   3584 Cores   |   3840 Cores   |   5120 Cores   |   1920 Cores   |   4352 Cores   |   4608 Cores   |   6912 Cores   |   10496 Cores  |
|    Tensor Cores    |      None      |      None      |    640 Cores   |       240      |    544 Cores   |    576 Cores   |    432 Cores   |    328 Cores   |
|     Clock(base)    |    1481 MHz    |    1405 Mhz    |    1200 MHz    |    1365 MHz    |    1350 MHz    |    1350 MHz    |     765 MHz    |    1395 MHz    |
|     FP16 (half)    |  177.2 GFLOPS  |   189.8GFLOPS  |  29,798 GFLOPS |  12.90 TFLOPS  |  26,895 GFLOPS |  32.62 TFLOPS  |  77.97 TFLOPS  |  35.58 TFLOPS  |
|    FP32 (float)    |  11,340 GFLOPS |   12.15FLOPS   |  14,899 GFLOPS |  6.451 TFLOPS  |  13,448 GFLOPS |  16.31 TFLOPS  |  19.49 TFLOPS  |  35.58 TFLOPS  |
|    FP64 (double)   |  354.4 GFLOPS  |  379.7 GFLOPS  |  7,450 GFLOPS  |  201.6 GFLOPS  |  420.2 GFLOPS  |  509.8 GFLOPS  |  9.746 TFLOPS  |  556.0 GFLOPS  |
|       Memory       |   11GB GDDR5X  |     GDDR5X     |   12 GB HBM2   |   6GB GDDR6    |   11 GB GDDR6  |   24 GB GDDR6  |   40GB HBM2e   |   24GB GDDR6X  |
|  Memory Interface  |     352-bit    |     384bit     |    3072-bit    |     192 bit    |     352-bit    |     384 bit    |    5120 bit    |     384 bit    |
|  Memory Bandwidth  |    484 GB/s    |    547.6GB/s   |     653GB/s    |   336.0 GB/s   |    616 GB/s    |   672.0 GB/s   |   1,555 GB/s   |   936.2 GB/s   |
|        Price       |     $699 US    |    $1,199 US   |    $2,999 US   |    $ 349 US    |    $1,199 US   |    $2,499 US   |                |   $ 1,499 USD  |
|    Release Date    | Mar 10th, 2017 |  Apr 6th 2017  |  Dec 7th, 2017 |  Jan 7th, 2019 | Sep 20th, 2018 | Dec 18th, 2018 | Jun 22nd, 2020 |  Sep 1st, 2020 |

[_reference site_](https://www.techpowerup.com/gpu-specs/)



1. Single & multi GPU with batch size 12: compare training and inference speed of **SequeezeNet, VGG-16, VGG-19, ResNet18, ResNet34, ResNet50, ResNet101,
ResNet152, DenseNet121, DenseNet169, DenseNet201, DenseNet161 mobilenet mnasnet ... **

2. Experiments are performed on three types of the datatype. single-precision, double-precision, half-precision

3. making plot(plotly)

## Usage

`./test.sh`

## Results

###  requirement
* python>=3.6(for f-formatting)
* torchvision
* torch>=1.0.0
* pandas
* psutil
* plotly(for plot)
* cufflinks(for plot)


### Environment

* Pytorch version `1.4`
* Number of GPUs on current device `4`
* CUDA version = `10.0`
* CUDNN version= `7601`
* `nvcr.io/nvidia/pytorch:20.10-py3` (docker container in A100 and 3090)



### Change Log
* 2021/02/27
  * Addition result in RTX3090
  * Addition result in RTX2060(thanks for gutama)
* 2021/01/07
  * Addition result in TITANXP
* 2021/01/05
  * Addition result in A100 A100-PCIE(PR#14)
* 2021/01/04
  * Addition result in A100 SXM4
  * Addition result in TitanRTX
  * Edit coding style benchmark_model
    * f-formatting
    * save option for json
  * Edit test.sh for bash shell
  * Edit README.md
* 2020/09/01
  * Addition result in windows10
  * Edit README.md
* 2020/01/17
  * Edit coding style and some bug
  * Change plot method 
  * Add results of various model experiments(only 2080ti)
* 2019/01/09
  * PR Update typo (thanks for johmathe)
  * Add requirements.txt
  * Add result figures
  * Add ('TkAgg') for cli
  * Addition Muilt GPUS (DGX-station)

### [RTX-3090](doc/3090.md)
- 2021/02/27 

### [RTX-2060](doc/2060.md)
- 2021/02/27 thanks for gutama [issue#16](https://github.com/ryujaehun/pytorch-gpu-benchmark/issues/16)
- 
### [TITANXP](doc/TITANXP.md)
- 2021/01/05 thanks for kirk86 [pr#14](https://github.com/ryujaehun/pytorch-gpu-benchmark/pull/14)
### [A100-PCIE(DGX A100)](doc/a100-pcie.md)
- 2021/01/05 Thanks for kirk86 [pr#14](https://github.com/ryujaehun/pytorch-gpu-benchmark/pull/14)

### [A100-SXM4(DGX A100)](doc/dgx-a100.md)
- 2021/01/04

### [TitanRTX](doc/TITANRTX.md)
- 2021/01/04

### [2080ti result on (new_results) windows10 system](doc/windows10.md)
* thanks for olixu

### [2080ti result on ubuntu (new_results)](doc/new_result.md)
* based on 2020/01/17 update

### Comparison between networks (single GPU)

Each network is fed with 12 images with 224x224x3 dimensions.
For training, time durations of 20 passes of forwarding and backward are averaged. For inference, time durations of
20 passes of forwarding are averaged. 5 warm-up steps are performed that do not calculate towards the final result.

_I conducted the experiment using two RTX 2080ti._


|   Mode  |gpu|precision|densenet121|densenet161|densenet169|densenet201|resnet101|resnet152|resnet18|resnet34|resnet50|squeezenet1_0|squeezenet1_1|vgg16|vgg16_bn|vgg19|vgg19_bn|
|:-------:| :----:|:--:|:----------:|:-------:|:-----------:|:-------:|:---------:| ---------:|:-----:|:---------:|:-----:|:------:|:---------:|:-------:|:---------:|:-------:|:---------:|
|Training | TITAN V|single|56.17 ms|120.7 ms|72.59 ms|93.35 ms|84.59 ms|119.5 ms|16.69 ms|28.27 ms|50.54 ms|15.30 ms|9.857 ms|72.85 ms|80.95 ms|85.55 ms|94.42 ms|
|Inference| TITAN V|single|17.49 ms|39.33 ms|23.63 ms|30.93 ms|23.96 ms|34.22 ms|4.827 ms|8.428 ms|14.27 ms|4.565 ms|2.765 ms|22.94 ms|25.41 ms|27.55 ms|30.28 ms|
|Training | TITAN V|double|139.8 ms|387.4 ms|175.9 ms|224.5 ms|509.9 ms|720.0 ms|94.21 ms|194.6 ms|271.7 ms|68.38 ms|31.18 ms|1463. ms|1484. ms|1993. ms|2016. ms|
|Inference| TITAN V|double|47.68 ms|170.5 ms|60.73 ms|78.43 ms|317.7 ms|448.6 ms|60.26 ms|129.9 ms|159.8 ms|42.37 ms|11.95 ms|1261. ms|1266. ms|1745. ms|1751. ms|
|Training | TITAN V|half|43.79 ms|75.16 ms|57.53 ms|70.88 ms|47.82 ms|67.43 ms|10.48 ms|17.19 ms|29.08 ms|13.15 ms|9.390 ms|36.03 ms|46.84 ms|41.16 ms|52.65 ms|
|Inference| TITAN V|half|11.87 ms|22.88 ms|16.04 ms|20.70 ms|12.80 ms|18.11 ms|3.085 ms|5.116 ms|7.608 ms|3.694 ms|2.329 ms|10.96 ms|13.26 ms|12.72 ms|15.17 ms|
|Training | 1080ti|single|77.18 ms|164.0 ms|99.66 ms|127.6 ms|112.8 ms|158.7 ms|22.48 ms|36.80 ms|68.87 ms|20.56 ms|13.29 ms|101.8 ms|114.1 ms|119.9 ms|133.2 ms|
|Inference| 1080ti|single|23.53 ms|51.53 ms|31.82 ms|41.73 ms|33.02 ms|47.02 ms|6.426 ms|10.97 ms|20.17 ms|7.174 ms|4.370 ms|33.73 ms|37.25 ms|39.95 ms|44.12 ms|
|Training | 1080ti|double|779.5 ms|2522. ms|940.4 ms|1196. ms|2410. ms|3546. ms|463.3 ms|969.9 ms|1216. ms|259.9 ms|131.5 ms|4227. ms|4271. ms|5475. ms|5522. ms|
|Inference| 1080ti|double|47.68 ms|275.2 ms|1157. ms|328.6 ms|414.9 ms|1080. ms|1589. ms|181.1 ms|390.8 ms|529.6 ms|110.9 ms|49.96 ms|2094. ms|2103. ms|2775. ms|2784. ms|
|Training | 1080ti|half|43.79 ms|70.00 ms|148.4 ms|89.43 ms|113.6 ms|151.0 ms|219.5 ms|21.00 ms|34.84 ms|76.24 ms|19.60 ms|13.18 ms|91.60 ms|105.9 ms|108.1 ms|123.6 ms|
|Inference| 1080ti|half|18.62 ms|42.26 ms|25.27 ms|33.01 ms|27.49 ms|38.88 ms|5.645 ms|9.765 ms|16.26 ms|5.869 ms|3.576 ms|30.69 ms|33.22 ms|36.71 ms|39.51 ms|

|   Mode  |gpu|precision|resnet18 |resnet34 |resnet50 |resnet101 |resnet152 |densenet121 |densenet169 |densenet201 |densenet161 |squeezenet1_0 |squeezenet1_1 |vgg16 |vgg16_bn |vgg19_bn |vgg19 |
|:-------:| :----:|:--:|:----------:|:-------:|:-----------:|:-------:|:---------:| ---------:|:-----:|:---------:|:-----:|:------:|:---------:|:-------:|:---------:|:-------:|:---------:|
|Training | RTX 2080ti(1)|single|16.36 ms|28.44 ms|49.63 ms|81.40 ms|115.1 ms|57.69 ms|75.18 ms|91.69 ms|112.7 ms|14.49 ms|9.108 ms|75.86 ms|85.42 ms|98.43 ms|88.05 ms|
|Inference| RTX 2080ti(1)|single|4.894 ms|8.624 ms|14.65 ms|24.57 ms|35.15 ms|16.70 ms|21.94 ms|28.89 ms|34.64 ms|4.704 ms|2.765 ms|23.70 ms|26.25 ms|30.82 ms|28.03 ms|
|Training | RTX 2080ti(1)|double|367.9 ms|755.4 ms|939.9 ms|1844. ms|2702. ms|593.5 ms|724.3 ms|921.3 ms|1916. ms|187.8 ms|94.99 ms|3251. ms|3277. ms|4265. ms|4238. ms|
|Inference| RTX 2080ti(1)|double|165.0 ms|328.5 ms|436.4 ms|831.0 ms|1196. ms|213.8 ms|266.0 ms|339.5 ms|910.7 ms|82.71 ms|35.79 ms|1702. ms|1708. ms|2280. ms|2274. ms|
|Training | RTX 2080ti(1)|half|13.17 ms|22.25 ms|35.46 ms|57.50 ms|81.38 ms|51.11 ms|66.88 ms|80.20 ms|88.37 ms|17.87 ms|35.75 ms|53.16 ms|63.06 ms|72.75 ms|61.95 ms|
|Inference| RTX 2080ti(1)|half|3.423 ms|5.662 ms|9.035 ms|14.51 ms|20.52 ms|13.47 ms|17.54 ms|22.51 ms|27.10 ms|4.280 ms|2.397 ms|16.14 ms|18.14 ms|19.76 ms|17.89 ms|
|Training | RTX 2080ti(2)|single|16.92 ms|29.51 ms|51.46 ms|84.90 ms|120.0 ms|58.13 ms|75.96 ms|92.47 ms|117.6 ms|14.95 ms|9.255 ms|78.95 ms|88.71 ms|102.3 ms|91.67 ms|
|Inference| RTX 2080ti(2)|single|5.107 ms|8.976 ms|15.18 ms|25.60 ms|36.60 ms|17.02 ms|22.40 ms|29.46 ms|36.72 ms|4.852 ms|2.786 ms|24.76 ms|27.25 ms|32.05 ms|29.27 ms|
|Training | RTX 2080ti(2)|double|381.9 ms|781.5 ms|971.6 ms|1900. ms|2777. ms|610.6 ms|744.7 ms|948.1 ms|1974. ms|191.9 ms|97.27 ms|3317. ms|3350. ms|4357. ms|4329. ms|
|Inference| RTX 2080ti(2)|double|171.8 ms|341.7 ms|449.5 ms|849.5 ms|1231. ms|221.1 ms|275.2 ms|352.5 ms|938.9 ms|83.66 ms|36.48 ms|1715. ms|1721. ms|2294. ms|2289. ms|
|Training | RTX 2080ti(2)|half|13.57 ms|22.97 ms|36.55 ms|59.10 ms|83.81 ms|51.74 ms|68.35 ms|81.21 ms|89.46 ms|15.75 ms|35.46 ms|55.28 ms|65.43 ms|75.75 ms|64.62 ms|
|Inference| RTX 2080ti(2)|half|3.520 ms|5.837 ms|9.272 ms|14.93 ms|21.13 ms|13.38 ms|18.71 ms|22.40 ms|26.82 ms|4.446 ms|2.406 ms|16.29 ms|17.91 ms|20.90 ms|19.14 ms|

### [TitanV ,1080ti , 2080ti result(old_results)](doc/old_result.md)
* Results using codes prior to 2020/01/17


### [DGX](doc/dgx.md)

### contribute
If you want to contribute to the experiment in an additional environment, please contribute to the result by subfolder in fig.
