"""
Data Extracton from EDF, XML

For prototype  Arousal Net 

This is for demonstrating Physionet2018_Challenge_Submission
"""
#%% 
#////////////////////////////////////////////////////////////////////////////////////////////
# Constants
#////////////////////////////////////////////////////////////////////////////////////////////
import platform
DISK = "V:" if platform.system() == 'Windows' else "/mnt/AI_DATAS"
SRC_ROOTS = {
    "SCH":{
        'edf':f"{DISK}/RAWEDFXMLSET/sch/edf/",
        'xml':f"{DISK}/RAWEDFXMLSET/sch/xml/",
        'ext_x': '.edf', 'ext_y': '.xml'
    },
    "SHHS1":{
        'edf':f"{DISK}/RAWEDFXMLSET/shhs/polysomnography/edfs/shhs1",
        'xml':f"{DISK}/RAWEDFXMLSET/shhs/polysomnography/annotations-events-nsrr/shhs1",
        'ext_x': '.edf', 'ext_y': '-nsrr.xml'
    },
    "SHHS2":{
        'edf':f"{DISK}/RAWEDFXMLSET/shhs/polysomnography/edfs/shhs2",
        'xml':f"{DISK}/RAWEDFXMLSET/shhs/polysomnography/annotations-events-nsrr/shhs2",
        'ext_x': '.edf', 'ext_y': '-nsrr.xml' } }

import argparse
def init_prarms():

    parser = argparse.ArgumentParser()
    parser.add_argument('sfreq', type=int,
                metavar='sfreq', default=50,
                help="choose 50 or 100")
    parser.add_argument('--n_job', type=int, default=30,
                help="number of child process")
    args = parser.parse_args()

    return {'sfreq' : args.sfreq, 'n_job' : args.n_job}

prarms = init_prarms()

SFREQ  = prarms['sfreq']
DEST = {
    'root' : f"{DISK}/BATCHED_DATAS/somnum/prototype_{SFREQ}Hz",
    'dir_raw': 'datas', 'dir_raw_stat' : 'stats',
    'ext_x': '.npx', 'ext_y': '.npy', 'ext_s': '.nps' }

TASK_NAME = "prototype"
N_JOPS = prarms['n_job']

LOG_DIR      = './log'
ERR_LOG_TXT  = f"{TASK_NAME}_{SFREQ}hz_error_log.txt"
PROGRESS_TXT = f"{TASK_NAME}_{SFREQ}hz_progress.txt"

TARGET_CH = ['C4-M1', 'C3-M2', 'LOC', 'ROC', 'EMG', 'THERMISTOR', 'ABDOMEN', 'THORAX', 'SPO2'] 

#////////////////////////////////////////////////////////////////////////////////////////////
# Requirements
#////////////////////////////////////////////////////////////////////////////////////////////
import numpy as np
import multiprocessing as mp
from os import path, listdir, mkdir
from utils.io import EDF, xml_load, pk_save
from utils.configs import VIRTUAL_CH_KEY
from utils.preprocess import prep_edf
from datetime import datetime 

import warnings, traceback
warnings.filterwarnings('ignore')

load_keys = lambda study :list(set([ 
    file_.replace(study['ext_x'],'') 
    for file_ in listdir(study['edf'])])&set([
    file_.replace(study['ext_y'],'') 
    for file_ in listdir(study['xml'])]))

TOTAL_FILES = sum([len(load_keys(study))for study in SRC_ROOTS.values()])
#////////////////////////////////////////////////////////////////////////////////////////////
# Helper functions
#////////////////////////////////////////////////////////////////////////////////////////////

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
    desc = describe(edf)
    signal = np.vstack([ edf[taget].signal for taget in TARGET_CH]).T.astype(dtype)
    hypnograms = np.vstack([arousal, aphypop, slpwake]).T.astype(dtype)

    return signal, hypnograms, desc

def describe(edf):

    describe_s = lambda s: { 
        'ticks'     : len(s),
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

   
    return { taget:describe_s(edf[taget].signal) for taget in TARGET_CH }

def is_task_done(key, q):

    done_task = q.get()
    if key in done_task:
        q.put(done_task)
        return True
    else:
        done_task.append(key)

        with open(path.join(LOG_DIR, PROGRESS_TXT), "w") as f:
            for task in done_task: 
                print(task, file=f)
            print(f"{len(done_task)}/{TOTAL_FILES}", file=f)
        q.put(done_task)
        
        return False   
#////////////////////////////////////////////////////////////////////////////////////////////
# Main functions
#////////////////////////////////////////////////////////////////////////////////////////////

    

def main(queue=None):
    
    for study, path_ in SRC_ROOTS.items():
        for key in load_keys(path_):

            if queue!=None and is_task_done(key, queue): continue

            try:
                signal, hypnograms, desc = load_signal_hypnograms(
                    path_edf=path.join(path_['edf'], key+path_['ext_x']), 
                    path_xml=path.join(path_['xml'], key+path_['ext_y']), 
                    sfreq=SFREQ, study=study)

                

                pk_save(signal, path.join(DEST['root'], DEST['dir_raw'], key+DEST['ext_x']))
                pk_save(hypnograms , path.join(DEST['root'], DEST['dir_raw'], key+DEST['ext_y']))
                pk_save(desc , path.join(DEST['root'], DEST['dir_raw_stat'], key+DEST['ext_s']))
                
            except Exception as e:
                with open(path.join(LOG_DIR, ERR_LOG_TXT), 'a') as f :
                    print(f"[{key}]", file=f) 
                    print(f"{e}\n", file=f) 
                    print(traceback.format_exc(), file=f)
                    print('\n', file=f)

#%% 

if __name__ == "__main__": 
    
    assert path.isdir(DEST['root']), f"There is no working dir\n-{DEST['root']}"
    if not path.isdir(LOG_DIR): mkdir(LOG_DIR)
    if not path.isfile(path.join(LOG_DIR, PROGRESS_TXT)):
        with open(path.join(LOG_DIR, PROGRESS_TXT), "w") as f: print(file=f)
        
    with open(path.join(LOG_DIR, PROGRESS_TXT), "r") as f:
        a = [task.replace('\n','') for task in f.readlines() if task.replace('\n','') != '']

    queue:mp.Queue = mp.Queue()
    queue.put(a)

    processes = [mp.Process(target=main, args=(queue,)) for  _ in range(N_JOPS)]
    for process in processes : process.start()
    for process in processes : process.join() 
