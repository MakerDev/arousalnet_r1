
import mne
import pyedflib
import xmltodict 
import pickle as pk 
from os import path

pk_load = lambda path: pk.load(open(path,'rb'))
pk_save = lambda obj, path: pk.dump(obj, open(path,'wb'))

class Signal:

    def __init__(self, 
        signal, label, dimension, sample_rate, 
        physical_max=None, physical_min=None, digital_max=None, digital_min=None, 
        prefilter=None, transducer=None):
                        
        self.signal       = signal
        self.label        = label    
        self.sample_rate  = sample_rate  
        self.transducer   = transducer 
        self.dimension    = dimension    
        self.prefilter    = prefilter 
        self.digital_max  = digital_max  
        self.digital_min  = digital_min  
        self.physical_max = physical_max     
        self.physical_min = physical_min     
        
    def apply(self, func):
        self.signal = func(self.signal)
        return self.signal

    def resample(self,sample_rate):
        
        def gcd(x, y): # Greatest Common Divisor
            while(y): x, y = y, x % y
            return x

        def lcm(x, y): # Least Common Multiple
            return (x*y)//gcd(x,y)

        # Calc upsample scale, downsample scale using lcm.
        if self.sample_rate % sample_rate !=0:
            up   = float(lcm(self.sample_rate, sample_rate)//self.sample_rate)
            down = float(lcm(self.sample_rate, sample_rate)//sample_rate)
        else:
            if self.sample_rate > sample_rate:
                up, down = 1.0, self.sample_rate/sample_rate
            else: 
                up, down = sample_rate/self.sample_rate, 1.0
            
        self.signal = mne.filter.resample(self.signal,up=up, down=down, npad='auto')
        self.sample_rate = sample_rate
        return self

    def seconds(self):
        return len(self.signal)/self.sample_rate

    def __str__(self):

        info =f"[Signal - {self.label}]\n"
        info +=f"sample_rate   : {self.sample_rate}\n"
        info +=f"signal.shape  : {self.signal.shape}\n"
        info +=f"signal.second : {self.seconds()}\n"

        info +=f"transducer    : {self.transducer}\n"
        info +=f"dimension     : {self.dimension}\n"
        info +=f"prefilter     : {self.prefilter}\n"

        info +=f"digital_max   : {self.digital_max}\n"
        info +=f"digital_min   : {self.digital_min}\n"
        info +=f"physical_max  : {self.physical_max}\n"
        info +=f"physical_min  : {self.physical_min}\n\n"

        return info

    def __repr__(self):
        return self.__str__()
    

class EDF:

    def __init__(self, file_path, preload=False):
        
        assert path.isfile(file_path), f"{file_path} is not exist!"
        self.file_path = file_path
        self.preload   = preload
        
        with pyedflib.EdfReader(self.file_path) as f:
            self.meas_date = f.getHeader()['startdate']
            self.ch_names = f.getSignalLabels()
            self.sfreqs = f.getSampleFrequencies()

        self.__sim = dict([(i,i) for i in range(len(self.ch_names))]) # singal index map

        if preload : self.load() 
        else       : self.signals = None
    
    def __getitem__(self, key):
        
        if not self.preload:
            raise RuntimeError("You should load siganl before get signal.\n- call EDF.load()")
        if isinstance(key, str) and key in self.ch_names:
            idx = self.ch_names.index(key)
        elif isinstance(key, int) and key < len(self.ch_names):
            idx = key
        else : raise IndexError(f"Invalid key or index : {key}")

        return self.signals[idx]

    def __str__(self):

        line = f"EDF file     : {path.basename(self.file_path)}]\n"
        line+= f"Measure date : {self.meas_date.strftime('%Y-%m-%d %H:%M:%S')}\n"
        line+= f"Recoring sec : {self.signals[0].seconds()}\n" if self.preload \
        else   f"Recoring sec : Signal wasn't loaded!\n"

        for i, (name, sfreq) in enumerate(zip(self.ch_names, self.sfreqs)): 
            line += f"{i:<3}{name} / {sfreq}\n"
        return line 

    def __repr__(self):
        return self.__str__()

    def __len__(self): 
        return len(self.signals)

    def load(self):
        
        self.preload = True 
        with pyedflib.EdfReader(self.file_path) as f:
            self.signals = []
            for i, (name, sfreq) in enumerate(zip(self.ch_names, self.sfreqs)):
                signal = Signal(
                    signal=f.readSignal(self.__sim[i]), **f.getSignalHeader(self.__sim[i]))
                signal.label = name
                self.signals.append(signal)

    def rename_ch(self, rename_map):

        assert all([ name in self.ch_names 
            for name in rename_map.keys()]), "Invalid name"
        
        for old, new in rename_map.items():
            self.ch_names[self.ch_names.index(old)] = new

        if self.preload:
            for new in rename_map.values():
                self.signals[self.ch_names.index(new)].label = new

        return self

    def pick_ch(self, picks):
     
        pick = lambda list_, picks: [list_[idx] for idx in picks ]

        self.__sim = dict([ (i, idx) for i, idx in enumerate(picks)])
        self.ch_names = pick(self.ch_names, picks)
        self.sfreqs  = pick(self.sfreqs,  picks)
        if self.preload:
            self.signals = pick(self.signals, picks)
        
        return self
    
    def update(self):

        if self.preload:
            self.ch_names = [ signal.label for signal in self.signals]
            self.sfreqs = [ signal.sample_rate for signal in self.signals]
        else:
            raise RuntimeError("Please call EDF.load() before calling EDF.update()")


def xml_load(path):

    with open(path, "rb") as f:
        doc = xmltodict.parse(f)
    
    return doc
