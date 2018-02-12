import pandas
import numpy as np


df_train = pandas.read_csv('model_training_benchmark')
print(df_train.mean(axis=0).sort_values())

df_inference = pandas.read_csv('model_inference_benchmark')
print(df_inference.mean(axis=0).sort_values())