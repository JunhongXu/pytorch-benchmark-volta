import pandas
import numpy as np
import re
from matplotlib import pyplot as plt


def read_train():
    df_train = pandas.read_csv('model_training_benchmark')
    print(df_train.mean(axis=0).sort_values())


def read_inference():
    df_inference = pandas.read_csv('model_inference_benchmark')
    print(df_inference.mean(axis=0).sort_values())


def read_gpus():
    batch_sizes = {2 ** i: [] for i in range(4, 12)}

    with open('gpu_batch_size.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            batch_size, num_gpu, duration = re.findall(r'\d+\.\d+|\d+', line)
            batch_sizes[int(batch_size)].append(float(duration))

    print(batch_sizes)
    for d in batch_sizes.keys():
        plt.plot(batch_sizes[d], label=d)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    read_gpus()


