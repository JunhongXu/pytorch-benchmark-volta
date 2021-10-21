

### DGX STATION SPEC

| Spec                        | NVIDIA DGX Station                                          |
|-----------------------------|-------------------------------------------------------------|
| GPUs                        | 4 x Tesla V100                                              |
| TFLOPS (GPU FP16)           |  480                                                        |
| GPU Memory                  | 64 GB total system                                          |
| CPU                         | 20-Core Intel Xeon E5-2698 v4 2.2 GHz                       |
| NVIDIA CUDA Cores           | 20,480                                                      |
| NVIDIA Tensor Cores         | 2,560                                                       |
| Maximum Power Requirements  | 1,500 W                                                     |
| System Memory               | 256 GB DDR4 LRDIMM                                          |
| Storage                     |  4 (data: 3 and OS: 1) x 1.92 TB SSD RAID 0                 |
| Network                     |  Dual 10 GbE, 4 IB EDR                                      |
| Display                     | 3X DisplayPort, 4K resolution                               |
| Acoustics                   | < 35 dB                                                     |
| Software                    |  Ubuntu Linux Host OSDGX Recommended GPU DriverCUDA Toolkit |
| System Weight               |  88 lbs / 40 kg                                             |
| System Dimensions           |  518 D x 256 W x 639 H (mm)                                 |
| Operating Temperature Range | 10 – 30 °C                                                  |


### result
![](/results/dgx.png)

|        | batchs | gpus | times            |
|--------|--------|------|------------------|
| half   | 16     | 1    | 15.6316900253296 |
| half   | 16     | 2    | 25.2950036525726 |
| half   | 16     | 3    | 32.5298488140106 |
| half   | 16     | 4    | 39.5952260494232 |
| half   | 32     | 1    | 28.9202857017517 |
| half   | 32     | 2    | 26.9314527511597 |
| half   | 32     | 3    | 32.6970362663269 |
| half   | 32     | 4    | 40.0277709960938 |
| half   | 64     | 1    | 54.6519541740418 |
| half   | 64     | 2    | 36.9417870044708 |
| half   | 64     | 3    | 35.1460886001587 |
| half   | 64     | 4    | 39.9034130573273 |
| half   | 128    | 1    | 105.689181089401 |
| half   | 128    | 2    | 62.5697267055512 |
| half   | 128    | 3    | 50.5970776081085 |
| half   | 128    | 4    | 45.686126947403  |
| single | 16     | 1    | 15.7001733779907 |
| single | 16     | 2    | 25.2602100372314 |
| single | 16     | 3    | 32.5334632396698 |
| single | 16     | 4    | 39.9562275409698 |
| single | 32     | 1    | 29.0114963054657 |
| single | 32     | 2    | 26.9594860076904 |
| single | 32     | 3    | 32.7185535430908 |
| single | 32     | 4    | 39.8312091827393 |
| single | 64     | 1    | 54.7226464748383 |
| single | 64     | 2    | 38.2881510257721 |
| single | 64     | 3    | 35.2633249759674 |
| single | 64     | 4    | 40.4890751838684 |
| single | 128    | 1    | 105.767976045609 |
| single | 128    | 2    | 62.6480567455292 |
| single | 128    | 3    | 50.3757321834564 |
| single | 128    | 4    | 45.5866599082947 |
| double | 16     | 1    | 15.703741312027  |
| double | 16     | 2    | 25.3219473361969 |
| double | 16     | 3    | 33.0831336975098 |
| double | 16     | 4    | 40.441951751709  |
| double | 32     | 1    | 29.0125107765198 |
| double | 32     | 2    | 27.3240101337433 |
| double | 32     | 3    | 33.0090951919556 |
| double | 32     | 4    | 40.2768909931183 |
| double | 64     | 1    | 54.7836709022522 |
| double | 64     | 2    | 36.7958390712738 |
| double | 64     | 3    | 35.0011682510376 |
| double | 64     | 4    | 39.9146497249603 |
| double | 128    | 1    | 105.872387886047 |
| double | 128    | 2    | 62.9272031784058 |
| double | 128    | 3    | 48.4100317955017 |
| double | 128    | 4    | 45.5989670753479 |
