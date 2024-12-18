
EEG  = ['F4-M1', 'F3-M2', 'C4-M1', 'C3-M2', 'O2-M1', 'O1-M2']
EOG  = ['LOC', 'ROC']
EMG  = ['EMG']
RESP = ['AIRFLOW', 'THERMISTOR', 'ABDOMEN', 'THORAX']
SPO2 = ['SPO2']

VIRTUAL_CH_KEY = {
    "F4-M1": [
        "F4-M1",
    ], 
    "F3-M2": [
        "F3-M2", 
    ], 
    "C4-M1": [
        "C4-M1","EEG"
    ], 
    "C3-M2": [
        "C3-M2","EEG(sec)", "EEG2", "EEG sec", "EEG 2", "EEG(SEC)"
    ], 
    "O2-M1": [
        "O2-M1", 
    ], 
    "O1-M2": [
        "O1-M2",
    ], 
    "LOC"  : [
        "LOC", "EOG(L)"
    ], 
    "ROC"  : [
        "ROC", "ROC-0", "ROC-1", "EOG(R)"
    ], 
    "EMG"  : [
        "Chin", "ChinL", "Lower.Left-Uppe", "EMG", "Lower.Left-Upper"
    ], 
    "AIRFLOW"    : [ # Nasal pressure tranducer - Hypopnea 
        "Flow_DR", "AIRFLOW-1", "AIRFLOW-0", "New AIR", "NEWAIR", 
        "New Air", "AUX", "New A/F", "new air", "AIRFLOW", "NEW AIR"
    ], 
    "THERMISTOR" : [ # Oronasal thermal airflow sensor - Apnea
        "Thermistor", "Thermistor2", "AIRFLOW-1", "AIRFLOW-0", "New AIR", 
        "NEWAIR", "New Air", "AUX", "New A/F", "new air", "AIRFLOW", "NEW AIR"
    ],
    "ABDOMEN": [
        "Abdomen", "ABDO RES"
    ], 
    "THORAX": [
        "Thorax", "THOR RES"], 
    "SPO2": [
        "SpO2", "SaO2"
    ], 
}