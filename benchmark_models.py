"""Compare speed of different models with batch size 12"""
import torch
import torchvision.models as models
import platform,psutil
import torch.nn as nn
import time,os
import pandas as pd
import argparse
from torch.utils.data import Dataset, DataLoader

torch.backends.cudnn.benchmark = True



MODEL_LIST = {

    models.mnasnet:models.mnasnet.__all__[1:],
    models.resnet: models.resnet.__all__[1:],
    models.densenet: models.densenet.__all__[1:],
    models.squeezenet: models.squeezenet.__all__[1:],
    models.vgg: models.vgg.__all__[1:],
    models.mobilenet:models.mobilenet.__all__[1:],
    models.shufflenetv2:models.shufflenetv2.__all__[1:]
}

precisions=["float","half",'double']
device_name=str(torch.cuda.get_device_name(0))
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Benchmarking')
parser.add_argument('--WARM_UP','-w', type=int,default=5, required=False, help="Num of warm up")
parser.add_argument('--NUM_TEST','-n', type=int,default=50,required=False, help="Num of Test")
parser.add_argument('--BATCH_SIZE','-b', type=int, default=12, required=False, help='Num of batch size')
parser.add_argument('--NUM_CLASSES','-c', type=int, default=1000, required=False, help='Num of class')
parser.add_argument('--NUM_GPU','-g', type=int, default=1, required=False, help='Num of gpus')
args = parser.parse_args()
args.BATCH_SIZE*=args.NUM_GPU
class RandomDataset(Dataset):

    def __init__(self,  length):
        self.len = length
        self.data = torch.randn( 3, 224, 224,length)

    def __getitem__(self, index):
        return self.data[:,:,:,index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset( args.BATCH_SIZE*(args.WARM_UP + args.NUM_TEST)),
                         batch_size=args.BATCH_SIZE, shuffle=False,num_workers=8)
def train(type='single'):
    """use fake image for training speed test"""
    target = torch.LongTensor(args.BATCH_SIZE).random_(args.NUM_CLASSES).cuda()
    criterion = nn.CrossEntropyLoss()
    benchmark = {}
    for model_type in MODEL_LIST.keys():
        for model_name in MODEL_LIST[model_type]:
            model = getattr(model_type, model_name)(pretrained=False)
            if args.NUM_GPU > 1:
                model = nn.DataParallel(model,device_ids=range(args.NUM_GPU))
            model=getattr(model,type)()
            model=model.to('cuda')
            durations = []
            print('Benchmarking Training {} precision type {} '.format(type,model_name))
            for step,img in enumerate(rand_loader):
                img=getattr(img,type)()
                torch.cuda.synchronize()
                start = time.time()
                model.zero_grad()
                prediction = model(img.to('cuda'))
                loss = criterion(prediction, target)
                loss.backward()
                torch.cuda.synchronize()
                end = time.time()
                if step >= args.WARM_UP:
                    durations.append((end - start)*1000)
            print(model_name,' model average train time : ',sum(durations)/len(durations),'ms')
            del model
            benchmark[model_name] = durations
    return benchmark

def inference(type='float'):
    benchmark = {}
    with torch.no_grad():
        for model_type in MODEL_LIST.keys():
            for model_name in MODEL_LIST[model_type]:
                model = getattr(model_type, model_name)(pretrained=False)
                if args.NUM_GPU > 1:
                    model = nn.DataParallel(model,device_ids=range(args.NUM_GPU))
                model=getattr(model,type)()
                model=model.to('cuda')
                model.eval()
                durations = []
                print('Benchmarking Inference {} precision type {} '.format(type,model_name))
                for step,img in enumerate(rand_loader):
                    img=getattr(img,type)()
                    torch.cuda.synchronize()
                    start = time.time()
                    model(img.to('cuda'))
                    torch.cuda.synchronize()
                    end = time.time()
                    if step >= args.WARM_UP:
                        durations.append((end - start)*1000)
                print(model_name,' model average inference time : ',sum(durations)/len(durations),'ms')
                del model
                benchmark[model_name] = durations
    return benchmark




if __name__ == '__main__':
    folder_name='new_results'
    path=''
    device_name="".join((device_name, '_',str(args.NUM_GPU),'_gpus_'))
    system_configs=str(platform.uname())
    system_configs='\n'.join((system_configs,str(psutil.cpu_freq()),'cpu_count: '+str(psutil.cpu_count()),'memory_available: '+str(psutil.virtual_memory().available)))
    gpu_configs=[torch.cuda.device_count(),torch.version.cuda,torch.backends.cudnn.version(),torch.cuda.get_device_name(0)]
    gpu_configs=list(map(str,gpu_configs))
    temp=['Number of GPUs on current device : ','CUDA Version : ','Cudnn Version : ','Device Name : ']

    os.makedirs(folder_name, exist_ok=True)
    now = time.localtime()
    start_time=str("%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
    print('benchmark start : ',start_time)

    for idx,value in enumerate(zip(temp,gpu_configs)):
        gpu_configs[idx]=''.join(value)
        print(gpu_configs[idx])
    print(system_configs)

    with open(os.path.join(folder_name,"system_info.txt"), "w") as f:
        f.writelines('benchmark start : '+start_time+'\n')
        f.writelines('system_configs\n\n')
        f.writelines(system_configs)
        f.writelines('\ngpu_configs\n\n')
        f.writelines(s + '\n' for s in gpu_configs )

    
    for precision in precisions:
        train_result=train(precision)
        train_result_df = pd.DataFrame(train_result)
        path=''.join((folder_name,'/',device_name,"_",precision,'_model_train_benchmark.csv'))
        train_result_df.to_csv(path, index=False)

        inference_result=inference(precision)
        inference_result_df = pd.DataFrame(inference_result)
        path=''.join((folder_name,'/',device_name,"_",precision,'_model_inference_benchmark.csv'))
        inference_result_df.to_csv(path, index=False)

    now = time.localtime()
    end_time=str("%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
    print('benchmark end : ',end_time)
    with open(os.path.join(folder_name,"system_info.txt"), "a") as f:
        f.writelines('benchmark end : '+end_time+'\n')


