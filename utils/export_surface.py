import pandas as pd
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from datetime import datetime


from pathlib import Path


export_folder = str(Path.cwd())+"/../datasets/"
export_path = "dataset.csv"
memory_length = 7

datapaths = ["2016S1_NB_SURFACE.txt","2016S2_NB_SURFACE.txt","2017S1_NB_SURFACE.txt","2017_T3_NB_SURFACE.txt","2017_T4_NB_SURFACE.txt"]

if(len(sys.argv)==1):
    print("")
    print("No argument specified, here is the template : ")
    print("\tpy opening.py [<file1> <file2> ..... <fileN> = default_files] <memory_length=7> <export_file=dataset.csv>\n")
    continue_str = input("Press enter to continue, type in 0 to exit : ")
    if(continue_str=="0"):
        exit()
    print("")
elif(len(sys.argv)==2):
    export_path = sys.argv[-1]
elif(len(sys.argv)==3):
    export_path = sys.argv[-1]
    memory_length = int(sys.argv[-2])
else:
    datapaths = []
    for path_i in range(1,len(sys.argv)-2):
        datapaths.append(sys.argv[path_i])







def parse_prefix(line, fmt):
    try:
        t = datetime.strptime(line, fmt)
    except ValueError as v:
        if len(v.args) > 0 and v.args[0].startswith('unconverted data remains: '):
            line = line[:-(len(v.args[0]) - 26)]
            t = datetime.strptime(line, fmt)
        else:
            raise
    return t


def load_dataset(memory_length=7):
    
    dataList = []
    print("Reading data...")
    for datapath in datapaths:
        data = pd.read_csv(str(Path.cwd())+"/../raw_datasets/"+datapath, sep='\t', lineterminator='\r', encoding='latin-1')
        data = data[:len(data)-1]
        data = data.drop(data[data.LIBELLE_LIGNE=='NON DEFINI'].index)
        dataList.append(data)


    def removeEndl(x):
        spl = x.split('\n')
        if len(spl)==1:
            return x
        return spl[1]

    def parseNb(x):
        if x=="Moins de 5":
            return 0
        return int(x)
    
    def change_date_format(date):
        dateSpl = date.split('/')
        return dateSpl[1]+"/"+dateSpl[0]+"/"+dateSpl[2]

    data = pd.concat(dataList)
    dayCol = data['JOUR']
    dayCol = dayCol.apply(removeEndl)
    data['JOUR'] = dayCol
    dayCol = data['NB_VALD']
    dayCol = dayCol.apply(parseNb)
    data['NB_VALD'] = dayCol
    data['JOUR'] = data['JOUR'].apply(change_date_format)

    dataGroup = data.groupby(['JOUR','LIBELLE_LIGNE'])['NB_VALD'].sum()
    data = dataGroup.reset_index()

    print("Assembling...")

    dataGroup = data.groupby(['LIBELLE_LIGNE'])

    Ls = [[] for i in range(memory_length+1)]
    libelle_list = []
    weekday = []
    keys = dataGroup.groups.keys()
    for libelle in tqdm(keys):
        temp_df = dataGroup.get_group(libelle)
        c=0
        if(len(temp_df)<memory_length):
            continue
        for index, row in temp_df.iterrows():
            if(c<memory_length):
                Ls[memory_length-c-1].append(row['NB_VALD'])
            elif c==memory_length:
                Ls[memory_length-c-1].append(row['NB_VALD'])
                libelle_list.append(libelle)
                weekday.append(parse_prefix(row['JOUR'], '%m/%d/%y').weekday())
                
            else:
                libelle_list.append(libelle)
                for i in range(memory_length):
                    Ls[i].append(Ls[i+1][-1])
                Ls[-1].append(row['NB_VALD'])
                weekday.append(parse_prefix(row['JOUR'], '%m/%d/%y').weekday())
            c+=1
    
    d = {"LIBELLE_LIGNE": libelle_list, 'WEEKDAY': weekday}
    for i in range(memory_length):
        d["DAY_"+str(i+1)] = Ls[i]
    d["NBRE_VALIDATION"] = Ls[-1]
    final_data = pd.DataFrame(data=d)
    return final_data


data = load_dataset(memory_length)

print("Saving...")

data.to_csv(export_folder+export_path)

print("Done")
