# ArousalNet R1

## 1. DEMO 

### 1) Quick start
```console
```

### 2) Argument usage
```console
usage: demo.py [-h] [--xml XML] [--save SAVE] [--f {SCH,SHHS}]
               [--verbose {True,False}] [--device {cpu,cuda:0}]
               EDF path

positional arguments:
  EDF path               EDF file path.

optional arguments:
  -h, --help             show this help message and exit
  --xml  XML             XML file path.
  --save SAVE            Directory path to save result xml file.
  --f       {SCH,SHHS}   Where the EDF file came from? SCH or SHHS
  --verbose {True,False} Will you print out process status?
  --device  {cpu,cuda:0} Which device will you use?
```

### 3) Example code
```console
python demo.py V:/RAWEDFXMLSET/sch/edf/SCH-PSG1612090.edf \
    --xml V:/RAWEDFXMLSET/sch/xml/SCH-PSG1612090.xml \
    --f SCH \
    --verbose True \
    --device cuda:0 
```

## 2. Training 





