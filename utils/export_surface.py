import pandas as pd
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from datetime import datetime


from pathlib import Path


export_folder = str(Path(__file__).parent.resolve())+"/../datasets/"
export_path = "dataset.csv"
memory_length = 7


datapaths_directory = "surface/"
datapaths = [
             "data-rs-2016/2016S1_NB_SURFACE.txt","data-rs-2016/2016S2_NB_SURFACE.txt",
             "data-rs-2017/2017S1_NB_SURFACE.txt","data-rs-2017/2017_T3_NB_SURFACE.txt","data-rs-2017/2017_T4_NB_SURFACE.txt",
             "data-rs-2018/2018_T1_NB_SURFACE.txt","data-rs-2018/2018_T2_NB_SURFACE.txt","data-rs-2018/2018_T3_NB_SURFACE.txt","data-rs-2018/2018_T4_NB_SURFACE.txt",
             "data-rs-2019/2019_T1_NB_SURFACE.txt","data-rs-2019/2019_T2_NB_SURFACE.txt","data-rs-2019/2019_T3_NB_SURFACE.txt","data-rs-2019/2019_T4_NB_SURFACE.txt",
             "data-rs-2020/2020_T1_NB_SURFACE.txt","data-rs-2020/2020_T2_NB_SURFACE.txt","data-rs-2020/2020_T3_NB_SURFACE.txt","data-rs-2020/2020_T4_NB_SURFACE.txt",
             "data-rs-2021/2021_T1_NB_SURFACE.txt","data-rs-2021/2021_T2_NB_SURFACE.txt","data-rs-2021/2021_T3_NB_SURFACE.txt","data-rs-2021/2021_T4_NB_SURFACE.txt",
             "data-rs-2022/2022_T1_NB_SURFACE.txt","data-rs-2022/2022_T2_NB_SURFACE.txt",
             ]

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




def is_school_holiday(date):
    """
        Return is a date is a school holiday, took thos dates from official resources
        Parameters
        ----------
        date : string
            The date of the day in format YYYY/MM/DD
    """
    dates = date.split("/")
    year = int(dates[0])
    month = int(dates[1])
    day = int(dates[2])
    if year==2016:
        if (month==2 and day>=21) or (month==3 and day<=6):
            return True
        elif (month==4 and day>=17) or (month==5 and day<=1):
            return True
        elif (month==7 and day>=6) or (month==8):
            return True
        elif (month==10 and day>=20) or (month==1 and day<=2):
            return True
        elif (month==12 and day>=17):
            return True
    elif year==2017:
        if month==1 and day<=2:
            return True
        elif month==2 and (day>=4 and day<=19):
            return True
        elif month==4 and (day>=1 and day<=17):
            return True
        elif (month==7 and day>=8) or (month==8 and day<=30):
            return True
        elif (month==10 and day>=21) or (month==11 and day<=5):
            return True
        elif month==12 and day>=23:
            return True
    return False



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

debug_date = True

def load_dataset(memory_length=7):
    
    dataList = []
    print("Loading data...")
    for datapath in datapaths:
        try:
            data = pd.read_csv(str(Path(__file__).parent.resolve())+"/../raw_datasets/"+datapaths_directory+datapath, sep='\t', lineterminator='\r', encoding='latin-1')
            data = data[:len(data)-1]
            data = data.drop(data[data.LIBELLE_LIGNE=='NON DEFINI'].index)
            dataList.append(data)
            print("Done reading \033[1m"+datapaths_directory+datapath+"\033[0m")
        except Exception:
            print("File\033[1m",datapaths_directory+datapath,"\033[0mnot found, make sure you have downloaded it with \033[1mdownload_datasets.py\033[0m")
        
    print("")

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
        return dateSpl[2]+"/"+dateSpl[1]+"/"+dateSpl[0]

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
    isSchoolHoliday = []
    dateList = []
    keys = dataGroup.groups.keys()
    for libelle in tqdm(keys):
        temp_df = dataGroup.get_group(libelle)
        c=0
        if(len(temp_df)<=memory_length):
            continue
        for index, row in temp_df.iterrows():
            if(c<memory_length):
                Ls[memory_length-c-1].append(row['NB_VALD'])
            elif c==memory_length:
                Ls[memory_length-c-1].append(row['NB_VALD'])
                libelle_list.append(libelle)
                weekday.append(parse_prefix(row['JOUR'], '%Y/%m/%d').weekday())
                isSchoolHoliday.append(1 if is_school_holiday(row['JOUR']) else 0)
                dateList.append(row['JOUR'])
                
            else:
                libelle_list.append(libelle)
                for i in range(memory_length):
                    Ls[i].append(Ls[i+1][-1])
                Ls[-1].append(row['NB_VALD'])
                weekday.append(parse_prefix(row['JOUR'], '%Y/%m/%d').weekday())
                isSchoolHoliday.append(1 if is_school_holiday(row['JOUR']) else 0)
                dateList.append(row['JOUR'])
            c+=1

    
    d = {"DATE":dateList,"LIBELLE_LIGNE": libelle_list, 'WEEKDAY': weekday, 'IS_SCHOOL_HOLIDAY':isSchoolHoliday}
    for i in range(memory_length):
        d["DAY_"+str(i+1)] = Ls[i]
    d["NBRE_VALIDATION"] = Ls[-1]
    for key in d.keys():
        print(key,len(d[key]))
    final_data = pd.DataFrame(data=d)

    print("Sorting...")

    final_data = final_data.sort_values(by=['DATE'])
    final_data['DATE'] = final_data['DATE'].apply(change_date_format)

    return final_data


data = load_dataset(memory_length)


print("Saving...")

data.to_csv(export_folder+export_path, index=False)

print("Done")

