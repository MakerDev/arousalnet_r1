#%%
"[Library load]"
import argparse
import traceback
import warnings 
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd 
from time import time

import platform
from os import mkdir, path, listdir, system

"[Torch Liabraries]"
import torch
import torch.optim as optim
from torch.utils import data

from models.layer import Normalizer, SeperableDenseNetUnit, SkipLSTM
from models.model import ArousalWithStageApnea
from models.loader import DatasetForPrototype
from models.trainer import TrainerForPrototype
from models.tools import TensorBoard, ContextManager

from models.metrics import Challenge2018Score
from sklearn.metrics import cohen_kappa_score, accuracy_score

import numpy as np
from utils.io import EDF, xml_load, pk_save
from utils.configs import VIRTUAL_CH_KEY
from utils.preprocess import prep_edf
from datetime import datetime 

N_GPU = torch.cuda.device_count() if torch.cuda.is_available() else 0 
TARGET_CH = ['C4-M1', 'C3-M2', 'LOC', 'ROC', 'EMG', 
             'THERMISTOR', 'ABDOMEN', 'THORAX', 'SPO2'] 
PATH_MODEL = './saved_model/prototype_50hz_b_98.pt'
# HELPER FUNCTIONS ///////////////////////////////////////////

# Load functions
def is_valid_ch_set(ch_names):
    
    return all([ target in ch_names for target in TARGET_CH])

def load_edf(file_path, sfreq=50):

    edf = EDF(file_path)
    
    rename_map = { key:vkey
        for key in edf.ch_names
        for vkey, keys in VIRTUAL_CH_KEY.items()
        if key in keys }
    
    if not is_valid_ch_set(list(rename_map.values())):
        raise ValueError(f"This is not target EDF.\n- {file_path}")

    edf = edf.rename_ch(rename_map)
    edf = edf.pick_ch(picks=[
        edf.ch_names.index(ch)
        for ch in VIRTUAL_CH_KEY
        if ch in edf.ch_names])

    edf.load()

    edf = prep_edf(edf,sfreq=sfreq)

    edf.update()

    return edf

def load_hypnograms_sch(path_xml, meas_date):

    # ///////////////////////////////////////////////////////////////////////////////
    # 1. INIT CONSTANTS FOR THIS FUNCTION
    # ///////////////////////////////////////////////////////////////////////////////

    # MAP from sch tags to virtual tags
    event_maps = {
        'sleep_stage': {
            'SLEEP-S0':'W', 'SLEEP-REM':'R', 
            'SLEEP-S1':"N1", 'SLEEP-S2':"N2", 'SLEEP-S3':"N3",}, 
        'sleep_wake': {
            'SLEEP-S0':'W', 'SLEEP-REM':'S', 'SLEEP-S1':"S", 
            'SLEEP-S2':"S", 'SLEEP-S3':"S"},
        'arousal' : dict((tag, 'AROUSAL') 
            for tag in ['AROUSAL', 'AROUSAL-LM', 'AROUSAL-PLM', 'AROUSAL-RERA',
                        'AROUSAL-RESP', 'AROUSAL-APNEA', 'AROUSAL-SPONT']),
        'apnea'   : {
            'APNEA-OBSTRUCTIVE':'APNEA',
            'APNEA-CENTRAL'    :'APNEA_', 
            'APNEA-MIXED'      :'APNEA_'},
        'hypopnea': dict((tag, 'HYPOPNEA') 
            for tag in ['HYPOPNEA','HYPOPNEA-CENTRAL','HYPOPNEA-OBSTRUCTIVE',])}

    # MAP from virtual tags to numeric tags
    hypnogram_maps = {
        'sleep_stage':{'W':0,"R":1,"N1":2,"N2":3, "N3":4},
        'sleep_wake':{"W":0, "S":1},
        'arousal':{"AROUSAL":1},
        'apnea':{"APNEA":1, "APNEA_":-1},
        'hypopnea':{"HYPOPNEA":1}}

    # convert time from sch time string to datetime 
    conv_time = lambda str_time : datetime.strptime(str_time,"%Y-%m-%dT%H:%M:%S.%f")

    # calc start time fomr sch xml
    get_st = lambda events: min([event['st'] for event in events])

    # calc end time fomr sch xml
    get_et = lambda events: max([event['et'] for event in events])

    # parsing xml and extract events from xml 
    extract_events = lambda xml, event_map, piv=0: [{
            'annot': event_map[event['Type']['#text']],
            'st': (conv_time(event['StartTime']['#text'])-meas_date).seconds-piv,
            'et': (conv_time(event['StopTime']['#text'])-meas_date).seconds-piv
        } 
        for event in xml['EventExport']['Events']['Event']
        if event['Type']['#text'] in event_map.keys() ]
    
    # ///////////////////////////////////////////////////////////////////////////////
    # 2. Load XML & parse events 
    # ///////////////////////////////////////////////////////////////////////////////
    xml = xml_load(path_xml)

    dict_events = dict([
        (key, extract_events(xml, event_map))
        for key, event_map in event_maps.items()])

    st = min([get_st(events) for events in dict_events.values() if len(events)!=0])
    et = max([get_et(events) for events in dict_events.values() if len(events)!=0])

    # ///////////////////////////////////////////////////////////////////////////////
    # 3. Draw Hypnograms 
    # ///////////////////////////////////////////////////////////////////////////////
    hypnograms = np.zeros((et-st, len(dict_events.keys())))
    for i, (key, events) in enumerate(dict_events.items()):
        
        # if len(events) == 0: 
        #     hypnograms[:,i] = -1
        # else: 
        for e in events:
            hypnograms[e['st']-st:e['et']-st,i] = hypnogram_maps[key][e['annot']]

    return hypnograms, st, et

def load_hypnograms_shhs(path_xml, meas_date=None):
    # ///////////////////////////////////////////////////////////////////////////////
    # 1. INIT CONSTANTS FOR THIS FUNCTION
    # ///////////////////////////////////////////////////////////////////////////////
    # MAP from sch tags to virtual tags
    event_maps = {
        'sleep_stage': {
            'Wake|0':'W', 'REM sleep|5':'R', 'Stage 1 sleep|1':"N1", 
            'Stage 2 sleep|2':"N2", 'Stage 3 sleep|3':"N3",'Stage 4 sleep|4':"N3",
            'Movement|6':"N/A", 'Unscored|9':"N/A"}, 
        'sleep_wake': {
            'Wake|0':'W', 'REM sleep|5':'S', 'Stage 1 sleep|1':'S', 
            'Stage 2 sleep|2':'S', 'Stage 3 sleep|3':'S','Stage 4 sleep|4':'S',
            'Movement|6':"N/A", 'Unscored|9':"N/A"},
        'arousal' : dict((tag, 'AROUSAL') 
            for tag in ['Arousal|Arousal ()',
                'Arousal|Arousal (Standard)',
                'Arousal|Arousal (STANDARD)',
                'ASDA arousal|Arousal (ASDA)',
                'External arousal|Arousal (External Arousal)',
                'Arousal resulting from Chin EMG|Arousal (Cheshire)',
                'Arousal resulting from Chin EMG|Arousal (CHESHIRE)',
                'Arousal resulting from respiratory effort|Arousal (ARO RES)',]),
        'apnea'   : {
            'Obstructive apnea|Obstructive Apnea':'APNEA', 
            'Central apnea|Central Apnea':'APNEA_', 
            'Mixed apnea|Mixed Apnea':'APNEA_', 
            },
        'hypopnea': {'Hypopnea|Hypopnea':'HYPOPNEA'} }

    # MAP from virtual tags to numeric tags
    hypnogram_maps = {
        'sleep_stage':{'W':0,"R":1,"N1":2,"N2":3, "N3":4, "N/A":5},
        'sleep_wake':{"W":0, "S":1, "N/A": 2},
        'arousal':{"AROUSAL":1},
        'apnea':{"APNEA":1, "APNEA_":-1},
        'hypopnea':{"HYPOPNEA":1}}

    # convert time from sch time string to datetime 
    conv_time = lambda str_time : datetime.strptime(str_time,"%Y-%m-%dT%H:%M:%S.%f")

    # calc start time fomr sch xml
    get_st = lambda events: min([event['st'] for event in events])

    # calc end time fomr sch xml
    get_et = lambda events: max([event['et'] for event in events])

    # parsing xml and extract events from xml 
    extract_events = lambda xml, event_map, piv=0: [{
            'annot': event_map[event['EventConcept']],
            'st': int(np.ceil(float(event['Start'])))-piv,
            'et': int(np.ceil(float(event['Start'])+float(event['Duration'])))-piv
        } 
        for event in xml['PSGAnnotation']['ScoredEvents']['ScoredEvent']
        if event['EventConcept'] in event_map.keys() ]

    # ///////////////////////////////////////////////////////////////////////////////
    # 2. Load XML & parse events 
    # ///////////////////////////////////////////////////////////////////////////////
    xml = xml_load(path_xml)

    dict_events = dict([
        (key, extract_events(xml, event_map))
        for key, event_map in event_maps.items()])

    st = min([get_st(events) for events in dict_events.values() if len(events)!=0])
    et = max([get_et(events) for events in dict_events.values() if len(events)!=0])

    # ///////////////////////////////////////////////////////////////////////////////
    # 3. Draw Hypnograms 
    # ///////////////////////////////////////////////////////////////////////////////
    hypnograms = np.zeros((et-st, len(dict_events.keys())))
    for i, (key, events) in enumerate(dict_events.items()):
        
        # if len(events) == 0: 
        #     hypnograms[:,i] = -1
        # else: 
        for e in events:
            hypnograms[e['st']-st:e['et']-st,i] = hypnogram_maps[key][e['annot']]

    return hypnograms, st, et

# Prep fuctions
def load_signal_hypnograms(path_edf, path_xml, 
    study='SCH',sfreq=100, dtype=np.float32):
        
    # ///////////////////////////////////////////////////////////////////////////////
    # 1. Load EDF, XML
    # ///////////////////////////////////////////////////////////////////////////////
    edf = load_edf(path_edf, sfreq=sfreq)
    if   study.upper() in ['SCH']:
        hypnograms, st, et = load_hypnograms_sch(    
            path_xml=path_xml,meas_date=edf.meas_date)
    elif study.upper() in ['SHHS', 'SHHS1', 'SHHS2']:
        hypnograms, st, et = load_hypnograms_shhs(     
            path_xml=path_xml,meas_date=edf.meas_date)
    else:
        raise ValueError(f"{study} is not valid study.")
    
    # ///////////////////////////////////////////////////////////////////////////////
    # 2. Sysc signals and hypnograms
    # ///////////////////////////////////////////////////////////////////////////////
    for idx in range(len(edf)):
        fs = edf.signals[idx].sample_rate
        edf.signals[idx].signal = edf.signals[idx].signal[st*fs:et*fs]
    hypnograms = hypnograms[:len(edf.signals[0].signal)//edf.signals[0].sample_rate]

    # ///////////////////////////////////////////////////////////////////////////////
    # 3. Generate Labels ()
    # - [ Automated Detection of Sleep Arousals From Polysomnography Data 
    #   Using a Dense Convolutional Neural Network ]
    #   Arousal: 
    #      +1  < Target Arousal 
    #      -1  < Non Target Arousal (apnea/hypopnea or wake)
    #       0  < Normal
    #   Apnea-hypopnea/normal:
    #      +1  < Obstructive apnea/hypopnea
    #      -1  < Central/mixed apnea 
    #       0  < Normal
    #   Sleep/Wake:
    #      +1  < Sleep Stage (Rem, N1, N2, N3)
    #      -1  < Undefined Stage (Movement, Unscored)
    #       0  < Wake
    #    0'sleep_stage':{'W':0,"R":1,"N1":2,"N2":3, "N3":4, "N/A": 5},
    #    1'sleep_wake':{"W":0, "S":1, "N/A": 2},
    #    2'arousal':{"AROUSAL":1},          
    #    3'apnea':{"APNEA":1, "APNEA_":-1}, 
    #    4'hypopnea':{"HYPOPNEA":1}}        
    # ///////////////////////////////////////////////////////////////////////////////
    # Target Arousal & Normal
    arousal = hypnograms[:,2]
    # Non Target Arousal (apnea/hypopnea or wake)
    # Apnea나 Hypopnea event 가 존재하지 않으면... -1이 
    arousal[np.where(hypnograms[:,1]==0)[0]] = -1   # wake
    arousal[np.where(hypnograms[:,3]==1)[0]] = -1   # apnea
    arousal[np.where(hypnograms[:,4]==1)[0]] = -1   # hypopnea
    
    # Obstructive apnea/hypopnea
    aphypop = hypnograms[:,3]
    aphypop[np.where(hypnograms[:,4]==1)[0]] = 1
    # Sleep/Wake
    slpwake = hypnograms[:,1]
    # Undefined Stage (Movement, Unscored)
    slpwake[np.where(hypnograms[:,1]==2)[0]] = -1

    # ///////////////////////////////////////////////////////////////////////////////
    # 4. Concatenate All datas & Data type change
    # ///////////////////////////////////////////////////////////////////////////////
    signal = np.vstack([ edf[taget].signal for taget in TARGET_CH]).T.astype(dtype)
    hypnograms = np.vstack([arousal, aphypop, slpwake]).T.astype(dtype)

    return signal, hypnograms

def tranform_shape(signal, block_size=50*3600*8):

    n_block = len(signal)//block_size
    n_rest  = (len(signal)% block_size)
    n_pad   = block_size - n_rest

    block = [ signal[i*block_size:(i+1)*block_size] for i in range(n_block) ]
    if n_rest:   
        block.append(np.concatenate([
            signal[n_block*block_size:] , 
            np.zeros((n_pad, signal.shape[1]))]))

    return np.array(block).astype(signal.dtype)

# Model prediction functions
def clean_model(model):
    assert isinstance(model, ArousalWithStageApnea)
    model.eval()

    for module in model.modules():

        # Force padding to be integer valued
        if hasattr(module, 'padding'):
            module.padding = (int(module.padding[0]),)

        # Register necessary parameters for state_dict to work properly
        if isinstance(module, Normalizer):
            x = module.movingAverage
            delattr(module, 'movingAverage')
            module.register_buffer('movingAverage', x)

            x = module.movingVariance
            delattr(module, 'movingVariance')
            module.register_buffer('movingVariance', x)

            x = torch.nn.Parameter(module.BatchNormScale)
            delattr(module, 'BatchNormScale')
            module.register_parameter('BatchNormScale', x)

            x = torch.nn.Parameter(module.BatchNormBias)
            delattr(module, 'BatchNormBias')
            module.register_parameter('BatchNormBias', x)

        # Remove weight normalization
        if isinstance(module, SeperableDenseNetUnit):
            module.conv1 = torch.nn.utils.remove_weight_norm(module.conv1, 'weight')
            module.conv2 = torch.nn.utils.remove_weight_norm(module.conv2, 'weight')
            module.conv3 = torch.nn.utils.remove_weight_norm(module.conv3, 'weight')
            module.conv4 = torch.nn.utils.remove_weight_norm(module.conv4, 'weight')

        if isinstance(module, SkipLSTM):
            module.outputConv1 = torch.nn.utils.remove_weight_norm(module.outputConv1, name='weight')
            module.outputConv2 = torch.nn.utils.remove_weight_norm(module.outputConv2, name='weight')

            module.rnn = torch.nn.utils.remove_weight_norm(module.rnn, name='weight_ih_l0')
            module.rnn = torch.nn.utils.remove_weight_norm(module.rnn, name='weight_hh_l0')

    return model

# Post processing functions 
def to_numpy(x):

    x = x.detach().reshape(-1)
    if x.device.type =='cuda': x = x.cpu()

    return x.numpy()

def calc_prob_pred(x, length=False):

        if isinstance(x, torch.Tensor):
            x = to_numpy(x)

        if length:
            x = x[:length]
            
        prob = np.exp(x)-1
        pred = np.zeros_like(x)
        pred[prob>1]=1
        

        return {
            'probability': prob[:length], 
            'prediction': pred[:length] }

def postprocessing(x1, x2, x3, hypnograms=False):

    is_hypnogram = isinstance(hypnograms, np.ndarray)
    length = len(hypnograms) if is_hypnogram else 0

    results = {
        'arous': calc_prob_pred(x=x1[:,1,:],length=length),
        'apnea': calc_prob_pred(x=x2[:,1,:],length=length),
        'sleep': calc_prob_pred(x=x3[:,1,:],length=length) }

    if is_hypnogram:
        results['arous']['target'] = hypnograms[:,0]
        results['apnea']['target'] = hypnograms[:,1]
        results['sleep']['target'] = hypnograms[:,2]

    return results

# Profiling & Evaluation functions 
# Profiling execution time needs 120 seconds.
# So i don't use profiling process
def describe(s): 
    return { 
        'max'       : np.max(s), 
        'min'       : np.min(s), 
        'mean'      : np.mean(s), 
        'std'       : np.std(s),
        '95p'       : np.percentile(s,95),  
        '05p'       : np.percentile(s,5), 
        'median'    : np.median(s), 
        'abs_mean'  : np.mean(np.abs(s)),
        'abs_median': np.median(np.abs(s)),  
        'abs_95p'   : np.percentile(np.abs(s),95) }

def epoch_desc(x_):
    return dict([(f"{ch_name}_{key}", val)
        for idx, ch_name in enumerate(TARGET_CH)
        for key, val in describe(x_[:,idx]).items() ])

def profiling_singal(signal, sfreq):

    epochs = [signal[i:i+sfreq] 
        for i in range(0,len(signal)-sfreq,sfreq) 
        if len(signal[i:i+sfreq])==sfreq]

    profiles = list(map(epoch_desc, epochs))

    return profiles

def calc_metrics(target, prediction, printout=False):

    target_ = target
    prediction_ = prediction


    scorer = Challenge2018Score()
    scorer.score_record(truth=target_, predictions=prediction_)
    auroc, auprc = scorer._auc(scorer._pos_values, scorer._neg_values)
    kappa = cohen_kappa_score(target_, prediction_)
    accuracy = accuracy_score(target_, prediction_)

    if printout:    
        print(
            f"auroc: {auroc:.4f}, auprc: {auprc:.4f}, kappa: {kappa:.4f}, acc: {accuracy:.4f}")

    return auroc, auprc, kappa, accuracy

def gen_dict_metric(results):
    metric_key = ['auroc', 'auprc', 'kappa', 'acc']

    dict_metric = { f"{key}_{m_key}":val
        for key in ['arous', 'apnea', 'sleep']
        for m_key, val in zip(metric_key, calc_metrics(
            target=results[key]['target'], 
            prediction=results[key]['prediction']) )}

    return dict_metric

def to_dictionary(results, csv_path=''):
    df = pd.DataFrame(data=[
        results['arous']['target'],
        results['arous']['prediction'],
        results['arous']['probability'],
        results['apnea']['target'],
        results['apnea']['prediction'],
        results['apnea']['probability'],
        results['sleep']['target'],
        results['sleep']['prediction'],
        results['sleep']['probability'] ]).T

    df.columns = [
        'arous_true', 'arous_pred', 'arous_prob',
        'apnea_true', 'apnea_pred', 'apnea_prob',
        'sleep_true', 'sleep_pred', 'sleep_prob',]

    if csv_path:
        df.to_csv(csv_path)

    return df 

# //////////////////////

def init_demo():

    "Initialize Task with arguments"
    parser = argparse.ArgumentParser()
    parser.add_argument('--edf', type=str,
                metavar='EDF path', default="/disk2/Yujin/Honeynaps/SAMPLE/EDF/SCH-210726R1_M-20-NW-NO.edf",
                help="EDF file path.")
    parser.add_argument('--xml', type=str, default='/disk2/Yujin/Honeynaps/SAMPLE/EBX/SCH-210726R1_M-20-NO_AROUS.xml',
                help="XML file path.")
    parser.add_argument('--save', type=str, default='./',
                help="Directory path to save result xml file.")
    parser.add_argument('--f', type=str, default='SCH',
                choices=['SCH','SHHS'],
                help="Where the EDF file came from? SCH or SHHS")
    parser.add_argument('--verbose', type=bool, default=True,
                choices=[True,False],
                help="Will you print out process status?")
    parser.add_argument('--device', type=str, default='cpu',
                choices=['cpu'] + [f"cuda:{i}" for i in range(N_GPU)],
                help="Which device will you use?")
    import torch 

    "Parse task arguments"
    args = parser.parse_args()

    path_edf  = path.abspath(args.edf)
    path_xml  = path.abspath(args.xml) if args.xml != 'None' else None
    path_csv  = args.save
    verbose   = args.verbose
    study     = args.f 
    device    = args.device
    
    "Asserting arguments"
    assert path.isfile(path_edf), f'{path_edf} is not file.'
    assert (path_xml==None or path.isfile(path_xml)), f'{path_xml} is not file.'
    assert path.isdir(path_csv), f'{path_csv} is not directory.'

    
    
    path_csv = path.abspath(path.join(
        path_csv, path.basename(path_edf).replace('.edf','_ap.xml')))

    "Print out init results"
    if verbose:
        print(f"[ArousalNet R1 Baseline Demo version 0.0.1]\n")
        print(f"EDF file: {path_edf}")
        print(f"XML file: {path_xml}")
        print(f"EDF from: {study}\n")
        print(f"Results file will be saved at\n- {path_csv}\n")

    return {
        'path_edf'  :path_edf,
        'path_xml'  :path_xml,
        'path_csv'  :path_csv,
        'study'     :study,
        'verbose'   :verbose,
        'device'    :device  }

def demo(path_model, path_edf, 
    path_xml=None, path_csv=None, study='SCH', 
    sfreq=50, block_size=50*3600*8, device='cuda:0' ):

    device = torch.device(device)

    # loading data(edf, xml)
    if path_xml!= None:
        signal, hypnograms = load_signal_hypnograms(
            path_edf=path_edf, path_xml=path_xml, 
            study=study, sfreq=sfreq, dtype=np.float32)
    else:
        signal, hypnograms= load_edf(path_edf), False

    # loading model
    model = torch.load(path_model, map_location=device)
    model.train(False)

    # model prediction
    block = tranform_shape(signal, block_size=block_size)
    block = torch.Tensor(block).permute(0, 2, 1).to(device)
    x1, x2, x3 = model(block)

    # postprocessing
    results = postprocessing(x1, x2, x3, hypnograms)

    dict_metric = gen_dict_metric(results)
    df = to_dictionary(results, csv_path=path_csv)

    return dict_metric


if __name__ == '__main__':

    p = init_demo()

    demo(
        path_model =PATH_MODEL, 
        path_edf   =p['path_edf'], 
        path_xml   =p['path_xml'], 
        path_csv   =p['path_csv'], 
        study      =p['study'], 
        device     =p['device'],
        sfreq      =50, 
        block_size =50*3600*8, ) 