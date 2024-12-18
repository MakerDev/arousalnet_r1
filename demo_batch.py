from demo import *

# //demo//////////////////////////////////////////////////////
def init_batch_task(): 
    #Define path configs 
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0',
                choices=['cpu', 'cuda:0','cuda:1','cuda:2','cuda:3'],
                help="Which device will you use for this scripts.")

    args = parser.parse_args()
    device = args.device

    DISK = "V:" if platform.system() == 'Windows' else '/mnt/AI_DATAS'
    ROOT_EDF_SCH = f"{DISK}/RAWEDFXMLSET/sch/edf"
    ROOT_XML_SCH = f"{DISK}/RAWEDFXMLSET/sch/xml"

    ROOT_EDF_SHHS1 = f"{DISK}/RAWEDFXMLSET/shhs/polysomnography/edfs/shhs1"
    ROOT_XML_SHHS1 = f"{DISK}/RAWEDFXMLSET/shhs/polysomnography/annotations-events-nsrr/shhs1"

    ROOT_EDF_SHHS2 = f"{DISK}/RAWEDFXMLSET/shhs/polysomnography/edfs/shhs2"
    ROOT_XML_SHHS2 = f"{DISK}/RAWEDFXMLSET/shhs/polysomnography/annotations-events-nsrr/shhs2"

    path_dest = f"{DISK}/AROUSALNET_R1/results/prototype_50hz_b_98"

    EXT_X = '.edf'
    EXT_Y_SCH = '.xml'
    EXT_Y_SHHS = '-nsrr.xml'

    ROOTS = [
        (ROOT_EDF_SCH, ROOT_XML_SCH),
        (ROOT_EDF_SHHS1, ROOT_XML_SHHS1),
        (ROOT_EDF_SHHS2, ROOT_XML_SHHS2)]

    # Get keys 
    get_files = lambda root, ext: [ file_.replace(ext,'') for file_ in listdir(root)] 
    get_keys = lambda root_edf, root_xml, ext_x, ext_y: list(
        set(get_files(root_edf, ext_x))&set(get_files(root_xml, ext_y)))

    keys_sch   = get_keys(ROOT_EDF_SCH, ROOT_XML_SCH, EXT_X, EXT_Y_SCH)
    keys_shhs1 = get_keys(ROOT_EDF_SHHS1, ROOT_XML_SHHS1, EXT_X, EXT_Y_SHHS)
    keys_shhs2 = get_keys(ROOT_EDF_SHHS2, ROOT_XML_SHHS2, EXT_X, EXT_Y_SHHS)

    # Generate pathes
    pathes_sch = [{
        'study':'SCH',
        'key':key, 
        'edf':path.join(ROOT_EDF_SCH,key+EXT_X), 
        'xml':path.join(ROOT_XML_SCH,key+EXT_Y_SCH)}
        for key in keys_sch ]

    pathes_shhs1 = [{
        'study':'SHHS',
        'key':key, 
        'edf':path.join(ROOT_EDF_SHHS1,key+EXT_X), 
        'xml':path.join(ROOT_XML_SHHS1,key+EXT_Y_SHHS)}
        for key in keys_shhs1 ]

    pathes_shhs2 = [{
        'study':'SHHS',
        'key':key, 
        'edf':path.join(ROOT_EDF_SHHS2,key+EXT_X), 
        'xml':path.join(ROOT_XML_SHHS2,key+EXT_Y_SHHS)}
        for key in keys_shhs2 ]  

    # Unify pathes 
    pathes = pathes_sch+pathes_shhs1+pathes_shhs2

    return pathes, device, path_dest

def batch_task_main():

    pathes, device, path_dest = init_batch_task()
    dict_metrics = []    
    for i, path_ in enumerate(pathes):

        study    = path_['study']
        key      = path_['key']
        path_edf = path_['edf']
        path_xml = path_['xml']
        path_csv = path.join(path_dest, key+'.csv')

        if path.isfile(path_csv): continue
        else:
            # occupy
            with open(path_csv, 'w') as f: print(file=f)

        start = time()
        print(f"[{key}]")
        try:
            dict_metric = demo(PATH_MODEL, path_edf, path_xml, path_csv, study, 
                sfreq=50, block_size=50*3600*8, device=device )

            dict_metrics.append(dict_metric)
            average = pd.DataFrame(dict_metrics).mean()['arous_auprc']
            current = dict_metric['arous_auprc']
            exec_sec = time()-start

            print(f"Total Execute time  : {exec_sec:.2f} seconds [{i+1}/{len(pathes)}]")
            print(f"Current AUPRC       : {current:.4f}")
            print(f"Average AUPRC       : {average:.4f}\n")

        except Exception as e: 
            with open("./error_log.txt",'a') as f:
                print(f"[{key}]", file=f)
                print(traceback.format_exc(), file=f)
                print(file=f)

            print(traceback.format_exc())
            print()
            
# ////////////////////////////////////////////////////////////
if __name__ == '__main__':
    batch_task_main()