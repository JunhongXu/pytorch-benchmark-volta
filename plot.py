import numpy as np
import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple



def arr_train():
    result_path=os.path.join(os.getcwd(),'results')
    arr_train=[i for i in glob.glob(result_path+'/*training*.csv')]
    arr_train.sort()
    return arr_train

def arr_inference():
    result_path=os.path.join(os.getcwd(),'results')
    arr_inference=[i for i in glob.glob(result_path+'/*inference*.csv')]
    arr_inference.sort()
    return arr_inference


def total_model(arr):

    model_name=arr[0].split('/')[-1].split('_')[0]
    type=arr[0].split('/')[-1].split('_')[3]
    n_groups = 15

    double=pd.read_csv(arr[0])
    half=pd.read_csv(arr[1])
    single=pd.read_csv(arr[2])

    means_double =double.mean().values
    std_double =double.std().values

    means_half =half.mean().values
    std_half =half.std().values

    means_single =single.mean().values
    std_single =single.std().values

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.35

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(index, means_double, bar_width,
                    alpha=opacity, color='b',
                    yerr=std_double, error_kw=error_config,
                    label='double')

    rects2 = ax.bar(index + bar_width, means_half, bar_width,
                    alpha=opacity, color='r',
                    yerr=std_half, error_kw=error_config,
                    label='half')

    rects2 = ax.bar(index + bar_width*2, means_single, bar_width,
                    alpha=opacity, color='g',
                    yerr=std_single, error_kw=error_config,
                    label='single')

    ax.set_xlabel('models')
    ax.set_ylabel('times(ms)')
    ax.set_title("total_"+type+"_"+model_name)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(double.columns,rotation=60, fontsize=9)

    ax.legend()

    fig.tight_layout()
    plt.savefig('total.png',dpi=300)



def model_plot(arr,model):

    model_name=arr[0].split('/')[-1].split('_')[0]
    type=arr[0].split('/')[-1].split('_')[3]
    double=pd.read_csv(arr[0])
    half=pd.read_csv(arr[1])
    single=pd.read_csv(arr[2])
    if model.lower() =='densenet':
        n_groups = 4
        double=double[['densenet121', 'densenet161', 'densenet169', 'densenet201']]
        half=half[['densenet121', 'densenet161', 'densenet169', 'densenet201']]
        single=single[['densenet121', 'densenet161', 'densenet169', 'densenet201']]
    elif model.lower() =='resnet':
        n_groups = 5
        double=double[['resnet101','resnet152', 'resnet18', 'resnet34', 'resnet50']]
        half=half[['resnet101','resnet152', 'resnet18', 'resnet34', 'resnet50']]
        single=single[['resnet101','resnet152', 'resnet18', 'resnet34', 'resnet50']]
    elif model.lower() =='squeezenet':
        n_groups = 2
        double=double[['squeezenet1_0','squeezenet1_1']]
        half=half[['squeezenet1_0','squeezenet1_1']]
        single=single[['squeezenet1_0','squeezenet1_1']]
    elif model.lower() =='vgg':
        n_groups = 4
        double=double[[ 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']]
        half=half[[ 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']]
        single=single[[ 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']]
    else:
        raise NotImplementedError("To be implemented")


    means_double =double.mean().values
    std_double =double.std().values

    means_half =half.mean().values
    std_half =half.std().values

    means_single =single.mean().values
    std_single =single.std().values

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.25

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(index, means_double, bar_width,
                    alpha=opacity, color='b',
                    yerr=std_double, error_kw=error_config,
                    label='double')

    rects2 = ax.bar(index + bar_width, means_half, bar_width,
                    alpha=opacity, color='r',
                    yerr=std_half, error_kw=error_config,
                    label='half')

    rects2 = ax.bar(index + bar_width*2, means_single, bar_width,
                    alpha=opacity, color='g',
                    yerr=std_single, error_kw=error_config,
                    label='single')

    ax.set_xlabel('models')
    ax.set_ylabel('times(ms)')
    ax.set_title(model+'_'+type+"_"+model_name)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(double.columns,rotation=45, fontsize=9)

    ax.legend()
    fig.tight_layout()
    plt.savefig(model+'.png',dpi=300)


def arr_type(arr,type):
    arr=[x for x in arr if not '(2)' in x ]
    if type == 'single':
        temp=[x for x in arr if 'single' in x ]
    elif type == 'double':
        temp=[x for x in arr if 'double' in x ]
    elif type == 'half':
        temp=[x for x in arr if 'half' in x ]
    return temp

def model_plot2(arr,model):
    #model_name=arr[0].split('/')[-1].split('_')[0]
    type=arr[0].split('/')[-1].split('_')[3]
    ti1080=pd.read_csv(arr[0])
    ti2080=pd.read_csv(arr[1])
    titan=pd.read_csv(arr[2])
    if model.lower() =='densenet':
        n_groups = 4
        ti1080=ti1080[['densenet121', 'densenet161', 'densenet169', 'densenet201']]
        ti2080=ti2080[['densenet121', 'densenet161', 'densenet169', 'densenet201']]
        titan=titan[['densenet121', 'densenet161', 'densenet169', 'densenet201']]
    elif model.lower() =='resnet':
        n_groups = 5
        ti1080=ti1080[['resnet101','resnet152', 'resnet18', 'resnet34', 'resnet50']]
        ti2080=ti2080[['resnet101','resnet152', 'resnet18', 'resnet34', 'resnet50']]
        titan=titan[['resnet101','resnet152', 'resnet18', 'resnet34', 'resnet50']]
    elif model.lower() =='squeezenet':
        n_groups = 2
        ti1080=ti1080[['squeezenet1_0','squeezenet1_1']]
        ti2080=ti2080[['squeezenet1_0','squeezenet1_1']]
        titan=titan[['squeezenet1_0','squeezenet1_1']]
    elif model.lower() =='vgg':
        n_groups = 4
        ti1080=ti1080[[ 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']]
        ti2080=ti2080[[ 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']]
        titan=titan[[ 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']]
    else:
        raise NotImplementedError("To be implemented")


    means_double =ti1080.mean().values
    std_double =ti1080.std().values

    means_half =ti2080.mean().values
    std_half =ti2080.std().values

    means_single =titan.mean().values
    std_single =titan.std().values

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.25

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(index, means_double, bar_width,
                    alpha=opacity, color='b',
                    yerr=std_double, error_kw=error_config,
                    label='1080ti')

    rects2 = ax.bar(index + bar_width, means_half, bar_width,
                    alpha=opacity, color='r',
                    yerr=std_half, error_kw=error_config,
                    label='2080ti')

    rects2 = ax.bar(index + bar_width*2, means_single, bar_width,
                    alpha=opacity, color='g',
                    yerr=std_single, error_kw=error_config,
                    label='TitanV')

    ax.set_xlabel('models')
    ax.set_ylabel('times(ms)')
    ax.set_title(model+'_'+type)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(ti1080.columns,rotation=45, fontsize=9)

    ax.legend()
    fig.tight_layout()
    plt.savefig(model+'_'+type+'_'+model_name+'.png',dpi=300)
