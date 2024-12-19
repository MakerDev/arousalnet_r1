import numpy as np 
import pandas as pd 
from os import listdir, path
from utils.io import pk_load
from utils.io import pk_load
from torch.utils import data
from sklearn.utils.class_weight import compute_sample_weight

import platform
DISK = "V:" if platform.system() == 'Windows' else "/mnt/AI_DATAS"
# Define condition config. 
SIGNAL_REPLACE = {
    'F4-M1' : ['F3-M2', 'C4-M1'], 
    'F3-M2' : ['F4-M1', 'C3-M2'],
    'C4-M1' : ['C3-M2'], 
    'C3-M2' : ['C4-M1'], 
    'O2-M1' : ['O1-M2', 'C4-M1'], 
    'O1-M2' : ['O2-M1', 'C3-M2'] }

CONDITION_XMAP = {
    'EEG6'        : ['F4-M1', 'F3-M2', 'C4-M1', 'C3-M2', 'O2-M1', 'O1-M2'],
    'EEG2'        : ['C4-M1', 'C3-M2'],
    'EOG2'        : ['LOC', 'ROC'],
    'EMG'         : ['EMG'], 
    'THERMISTOR'  : ['THERMISTOR'],
    'AIRFLOW'     : ['AIRFLOW'], 
    'RESP_EFFORT' : ['ABDOMEN', 'THORAX'],
    'SPO2'        : ['SPO2'] }

# SleepStage:0, SleepWake:1, Arousal:2, Apnea:3, Hypopnea:4
CONDITION_YMAP = {
    'SleepStage'   : [0],
    'SleepWake'    : [1],  
    'Arousal'      : [2],
    'Apnea'        : [3],
    'ApneaHypopnea': [3,4]}

CONDITION_SET = {
    'SS_01' : {  # SS_01. EEG6, EOG2, EMG / SleepStage
        'x':['EEG6', 'EOG2', 'EMG'],
        'y':['SleepStage']},
    'SS_02' : { # SS_02. EEG2, EOG2, EMG / Sleep Stage
        'x':['EEG2', 'EOG2', 'EMG'],
        'y':['SleepStage']},
    'AR_01' : { # AR_01. EEG6, EMG / Arousal 
        'x':['EEG6', 'EMG'],
        'y':['Arousal']},
    'AR_02' : { # AR_02. EEG6, EMG / Arousal, SleepWake
        'x':['EEG6', 'EMG'],
        'y':['Arousal', 'SleepWake']},
    'AR_03' : { # AR_03. EEG6, EOG2, EMG / Arousal, SleepStage
        'x':['EEG6', 'EOG2', 'EMG'],
        'y':['Arousal', 'SleepStage']},
    'AR_04' : { # AR_04. EEG6, EOG2, EMG, THERMISTOR, SPO2 / Arousal, SleepWake, Apnea 
        'x':['EEG6', 'EOG2', 'EMG', 'THERMISTOR', 'SPO2'],
        'y':['Arousal', 'SleepWake', 'Apnea']},
    'AR_05' : { # AR_05. EEG6, EOG2, EMG, THERMISTOR, SPO2 / Arousal, SleepWake, ApneaHypopnea
        'x':['EEG6', 'EOG2', 'EMG', 'THERMISTOR', 'SPO2'],
        'y':['Arousal', 'SleepWake', 'ApneaHypopnea']},
    'AR_06' : { # AR_06. EEG2, EMG / Arousal 
        'x':['EEG2', 'EMG'],
        'y':['Arousal']},
    'AR_07' : { # AR_07. EEG2, EMG / Arousal, SleepWake
        'x':['EEG2', 'EMG'],
        'y':['Arousal', 'SleepWake']},
    'AR_08' : { # AR_08. EEG2, EOG2, EMG / Arousal, SleepStage
        'x':['EEG2', 'EOG2', 'EMG'],
        'y':['Arousal', 'SleepStage']},
    'AR_09' : { # AR_09. EEG2, EOG2, EMG, THERMISTOR, SPO2 / Arousal, SleepWake, Apnea 
        'x':['EEG2', 'EOG2', 'EMG', 'THERMISTOR', 'SPO2'],
        'y':['Arousal', 'SleepWake', 'Apnea']},
    'AR_10' : { # AR_10. EEG2, EOG2, EMG, THERMISTOR, SPO2 / Arousal, SleepWake, ApneaHypopnea
        'x':['EEG2', 'EOG2', 'EMG', 'THERMISTOR', 'SPO2'],
        'y':['Arousal', 'SleepWake', 'ApneaHypopnea']},
    'AP_01' : { # AP_01. THERMISTOR, SpO2 / Apnea 
        'x':['THERMISTOR', 'SPO2'],
        'y':['Apnea']},
    'AP_02' : { # AP_02. Thermistor, SpO2 / ApneaHypopnea
        'x':['THERMISTOR', 'SPO2'],
        'y':['ApneaHypopnea']}      }

MASK_REQIRED = ['AR_01', 'AR_06', 'AP_01', 'AP_02'] 

class DatasetFromRAW(data.Dataset):

    # Define meta config
    META_ROOT = f"{DISK}/BATCHED_DATAS/somnum/meta"
    META_VERSION = '01'

    # Define raw data config 
    RAW_ROOT = f"{DISK}/BATCHED_DATAS/somnum/raw_100Hz/datas"
    EXT_X = '.npx'
    EXT_Y = '.npy'
      
    def __init__(self, task_code='AR_01', limit_hour=2, sfreq=100, weighted=True):

        self.task_code    = task_code
        self.limit_hour   = limit_hour
        self.sfreq        = sfreq
        self.weighted     = weighted
        self.segment_size = self.limit_hour*self.sfreq*3600
        self.init_meta()
        
    def init_meta(self):

        self.meta = pd.read_csv(path.join(self.META_ROOT,f"META_{self.task_code}_{self.META_VERSION}.csv"))
        self.index_map = {}

        idx = 0 
        keys = self.meta.key.values
        nseg = (self.meta['seconds']//(self.segment_size)).values

        for key, seg_idxes in dict(zip(keys, nseg)).items():
            for seg_idx in range(seg_idxes):
                self.index_map[idx] = {'key':key, 'seg_idx':seg_idx }; idx +=1
                
    def __len__(self):

        return len(self.index_map)

    def __getitem__(self, index):
        
        # x: signal, y: labels, m: mask (sleep:1, wake:0)
        x, y, m = self.__load(**self.index_map[index])
        batch = (x, y, m,)
        if self.weighted:
            w = np.vstack([ self.__calc_weights(y[:,i]) for i in range(y.shape[1])]).T
            batch = batch + (w,)
        else:
            batch = batch + (None,)
        # x.shape -> (self.segment_size, n_signal)
        # y.shape -> (self.segment_size//sfreq, n_label)
        # m.shape -> (self.segment_size//sfreq, 1) or None
        # w.shape -> (self.segment_size//sfreq, n_label)

        return batch

    @staticmethod
    def __calc_weights(Y):

        return compute_sample_weight('balanced', Y.reshape(-1)).reshape(Y.shape)

    def __extract_y_with_conditions(self, y, y_conditions):
        # Ex) y_conditions = ['Arousal', 'SleepWake', 'ApneaHypopnea']

        def extract_labels_from_histogram(histogram, extract_idxes):
            label = np.zeros_like(histogram[:,0])

            for idx in extract_idxes:
                if -1 not in y[:, idx]:
                    label += y[:, idx]
                    
            return label

        return np.vstack([
            extract_labels_from_histogram(y, CONDITION_YMAP[condition])
            for condition in y_conditions]).T
            
    def __load(self, key, seg_idx):

        # Load X, Y 
        x = pk_load(path.join(self.RAW_ROOT, key+self.EXT_X))
        y = pk_load(path.join(self.RAW_ROOT, key+self.EXT_Y))

        # Impute Missing channel datas.
        # if there is no C3-M2 it will be replaced as C4-M1
        replace_log  = {}
        replace_data = {}
        miss_chs     = []
        
        for c_idx in CONDITION_SET[self.task_code]['x']:

            miss_chs.extend(list(set(CONDITION_XMAP[c_idx])-set(x.keys())))
            for miss_ch in miss_chs:

                for replace_ch in SIGNAL_REPLACE[miss_ch]:
                    if replace_ch in x.keys():   
                        replace_log[miss_ch] = replace_ch
                        replace_data[miss_ch] = x[replace_ch]
                        break

        x.update(replace_data)

        assert list(replace_log.keys()) == miss_chs
        assert all([ key in x.keys() 
            for c_idx in CONDITION_SET[self.task_code]['x'] 
            for key   in CONDITION_XMAP[c_idx] ])

        # To numpy
        x = np.vstack([x[key]
            for c_idx in CONDITION_SET[self.task_code]['x'] 
            for key   in CONDITION_XMAP[c_idx]]).T

        st = self.segment_size*seg_idx
        et = self.segment_size*(seg_idx+1)
        x = x[st:et]

        st = (self.segment_size//self.sfreq)*seg_idx
        et = (self.segment_size//self.sfreq)*(seg_idx+1)
        extracted_y = self.__extract_y_with_conditions(y[st:et], CONDITION_SET[self.task_code]['y'])

        if self.task_code in MASK_REQIRED:
            mask = y[st:et,1]
        else:
            mask = None

        return x, extracted_y, mask



class DatasetForPrototype(data.Dataset):

    def __init__(self, root, meta, ext_x = '.npx', ext_y='.npy',
        set_key='train', sfreq=100, batch_hour=8,  
        weighted=True, n_sample_per_study=None, random_sampling=False, 
        dtype='float32'):

        # Directory config 
        self.root  = root
        self.meta  = meta
        self.ext_x = ext_x
        self.ext_y = ext_y

        # Dataset config 
        self.set_key    = set_key
        self.sfreq      = sfreq
        self.batch_hour = batch_hour*sfreq*3600
        self.weighted   = weighted
        self.n_sample_per_study = n_sample_per_study
        self.random_sampling    = random_sampling

        
        self.keys = self.__init_keys_by_meta()
        self.dtype = dtype
        self.cnt = 0

    def __init_keys_by_meta(self):
        meta = pd.read_csv(self.meta)
        if self.n_sample_per_study == None:
            keys = meta[meta['set_key']==self.set_key]['key'].values
        else: 
            keys = np.concatenate([np.random.choice(
                    a=meta[(meta['study_key']==study_key)&(
                            meta['set_key']==self.set_key)]['key'].values,
                    size=self.n_sample_per_study, 
                    replace=True)
                for study_key in ['SCH', 'SHHS1', 'SHHS2'] ])
            
        return keys

    def __len__(self):
           
        return len(self.keys)

    def __getitem__(self, index):

        batch = self.__load(self.keys[index])
        self.__count_call_getitem()

        return batch

    def __count_call_getitem(self):

        if (self.cnt%len(self.keys)==0)and self.random_sampling:
            self.keys = self.__init_keys_by_meta()
        self.cnt+=1

    @staticmethod
    def __calc_weights(Y):

        return compute_sample_weight('balanced', Y.reshape(-1)).reshape(Y.shape)

    def __load(self, key):
        
        x = pk_load(path.join(self.root, key+self.ext_x))[:self.batch_hour]
        y = pk_load(path.join(self.root, key+self.ext_y))[:self.batch_hour//self.sfreq]
        w = np.zeros_like(y)+1
        
        if self.weighted:
            for i in range(3):
                w[:,i] = self.__calc_weights(y[:, i])

        if len(x)!=self.batch_hour:
            
            pad_len = self.batch_hour - len(x)
            x = np.concatenate([x, np.zeros(shape=(pad_len, x.shape[-1]))])
            y = np.concatenate([y, np.zeros(shape=(pad_len//self.sfreq, y.shape[-1]))])
            w = np.concatenate([w, np.zeros(shape=(pad_len//self.sfreq, w.shape[-1]))])

        x = np.transpose(x)
        y = np.transpose(y)
        w = np.transpose(w)

        batch = (x.astype(self.dtype), y.astype(self.dtype), w.astype(self.dtype))

        return batch