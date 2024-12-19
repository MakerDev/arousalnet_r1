#%%
"""
python prototype_01 --device 0 --epochs 9999 --lr 0.0001
"""
#/////////////////////////////////////////////////////////////////////////////
"[Library load]"
import warnings 
warnings.filterwarnings('ignore')

from os import mkdir, path
from datetime import datetime

"[Config Liabraries ]"
import platform
import argparse
import configparser
from pprint import pprint

"[Torch Liabraries]"
import torch
import torch.optim as optim
from torch.utils import data

from time import time
from models.model import ArousalWithStageApnea
from models.loader import DatasetForPrototype
from models.trainer import TrainerForPrototype
from models.tools import TensorBoard, ContextManager

#/////////////////////////////////////////////////////////////////////////////
"[Define Path]"
if platform.system()=='Windows':
    DISK = "V:"
    ROOT_TENSORBOARD = "V:/TENSORBOARD"
    CONFIG = "./gpu_server.conf"
else:
    DISK = "/mnt/AI_DATAS"
    ROOT_TENSORBOARD = "/mnt/AI_DATAS/TENSORBOARD"
    CONFIG = "./gpu_server.conf"
#/////////////////////////////////////////////////////////////////////////////

"Training script CONFIG"
N_TRAIN_SAMPLE = 300    # 전체 Train sample 중 N_TRAIN_SAMPLE만 sampling
N_VALID_SAMPLE = 64     # 전체 valid sample 중 N_VALID_SAMPLE sampling
RESAMPLING = True      # 매 에폭마다 각 sample을 다시 sampling (Random sampling)

# NAS에 배치된 데이터 PATH
# SFREQ = 100
# DATASET_ROOT = "/mnt/AI_DATAS/BATCHED_DATAS/somnum/prototype_100Hz/datas"
# DATASET_META = "/mnt/AI_DATAS/BATCHED_DATAS/somnum/prototype_100Hz/meta.csv"

# SFREQ = 50
# DATASET_ROOT = "/mnt/AI_DATAS/BATCHED_DATAS/somnum/prototype_50Hz/datas"
# DATASET_META = "/mnt/AI_DATAS/BATCHED_DATAS/somnum/prototype_50Hz/meta.csv"

# SSD에 배치된 데이터 PATH 
SFREQ = 50  
DATASET_ROOT = f"/mnt/CACHED_DATAS/prototype_50Hz/datas"
DATASET_META = f"/mnt/CACHED_DATAS/prototype_50Hz/meta.csv"
#/////////////////////////////////////////////////////////////////////////////

def init_training():
    """[summary]
    Parsing arg, config Return arguments dictionary.

    Arguments
    name            : Process name. (it will display at tensorboard)
    --device        : GPU code [0-3], if GPU is not avaliable, Run on CPU.{default:0}
    --epochs        : {default:10}
    --lr            : {default:0.0001}    
    """        

    "Initialize ArgumentParser"
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str,
                metavar='name',
                help="Process name. (it will display at tensorboard)")
    parser.add_argument('--device', type=int, default=0,
                help="GPU code [0-3], if GPU is not avaliable, Run on CPU.")
    parser.add_argument('--epochs', type=int, default=10,
                help="How many epochs your model train.")
    parser.add_argument('--lr', type=float, default=0.0001,
                help="Learning Rate")
    parser.add_argument('--n_job', type=int, default=1,
                help="number of child process")
    args = parser.parse_args()

    "Initialize ConfigParser"
    # config = configparser.ConfigParser()
    # config.read(CONFIG)
    # conf_Dataset = config['Dataset']
    # conf_DataLoader    = config['DataLoader']
    # conf_TENSORBOARD   = config['TENSORBOARD']
    # list(config.keys())

    
    params =  {
        'name'  : args.name,
        'device': torch.device(f'cuda:{args.device}') \
            if torch.cuda.is_available() and torch.cuda.device_count()>args.device else torch.device("cpu") ,
        'epochs': args.epochs,
        'lr'    : args.lr,
        'n_job' : args.n_job,
        'root4tools':ROOT_TENSORBOARD}

    return params


def main():

    #initialize training scripts
    params = init_training()
    print(f"[Display Config - {params['name']}]")
    pprint(params)

    print("[Init ETL Pipe - DatasetForPrototype]")
    train_set = DatasetForPrototype(
        # Directory config
        root = DATASET_ROOT, 
        meta = DATASET_META,
        # Dataset config
        set_key    = 'train', 
        sfreq      = SFREQ,
        batch_hour = 8,  
        weighted   = True, 
        # Sampling config
        n_sample_per_study = N_TRAIN_SAMPLE, 
        random_sampling    = RESAMPLING ) 
    
    train_generator = data.DataLoader(
        dataset=train_set, batch_size=2, shuffle=True, 
        num_workers=params['n_job'], drop_last=True)

    valid_set = DatasetForPrototype(
        # Directory config
        root = DATASET_ROOT, 
        meta = DATASET_META,
        # Dataset config
        set_key    = 'valid', 
        sfreq      = SFREQ,
        batch_hour = 8,  
        weighted   = True, 
        # Sampling config
        n_sample_per_study = N_VALID_SAMPLE, 
        random_sampling    = RESAMPLING ) 

    valid_generator = data.DataLoader(
        dataset=valid_set, batch_size=2, shuffle=True, 
        num_workers=params['n_job'], drop_last=True)

    print("[Model Define - ArousalWithStageApnea]")
    model = ArousalWithStageApnea(sfreq=SFREQ, num_signals=9,device=params['device'])

    print("[Utility Define - tensorboard]")
    dir_4_utils = path.join(params['root4tools'], params['name'])
    tensorboard = TensorBoard(root=dir_4_utils)

    print("[Utility Define - context_manager]")
    context_manager = ContextManager(root=dir_4_utils)

    print("[Trainer Define - StageNetTrainer]")
    trainer = TrainerForPrototype(
        model=model, 
        device=params['device'], 
        lr=1e-5, 
        tensorboard=tensorboard, 
        context_manager=context_manager)

    print("[Training Start]")
    trainer.fit_generator(
        gen_train = train_generator, 
        gen_valid = valid_generator, 
        epochs    = params['epochs'])

if __name__ == '__main__':

    main()