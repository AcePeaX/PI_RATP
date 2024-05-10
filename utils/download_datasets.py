def download_datasets():
    try:
        import requests
    except Exception:
        print("To run this file, please install \033[1mrequests\033[0m package.")
        print("\tpip install requests\n")
        exit()

    import sys
    from tqdm import tqdm
    from pathlib import Path
    import os

    
    download_urls = {
        "surface": {
            "2016" : {"zip":True,"target":"surface/","url":"https://data.iledefrance-mobilites.fr/api/explore/v2.1/catalog/datasets/histo-validations-reseau-surface/files/c14c733e65e57a0f7aa918903f41055b","size":23529340},
            "2017" : {"zip":True,"target":"surface/","url":"https://data.iledefrance-mobilites.fr/api/explore/v2.1/catalog/datasets/histo-validations-reseau-surface/files/9c6b0ae07fb8476b58d76f9c5c5f149c","size":19697347},
            "2018" : {"zip":True,"target":"surface/","url":"https://data.iledefrance-mobilites.fr/api/explore/v2.1/catalog/datasets/histo-validations-reseau-surface/files/4c9abf782ec17616bcbcd68faa45ec99","size":19060057},
            "2019" : {"zip":True,"target":"surface/","url":"https://data.iledefrance-mobilites.fr/api/explore/v2.1/catalog/datasets/histo-validations-reseau-surface/files/6dd37a6a90a476a1090a751ea3e0d78c","size":20592620},
            "2020" : {"zip":True,"target":"surface/","url":"https://data.iledefrance-mobilites.fr/api/explore/v2.1/catalog/datasets/histo-validations-reseau-surface/files/41adcbd4216382c232ced4ccbf60187e","size":18279744},
            "2021" : {"zip":True,"target":"surface/","url":"https://data.iledefrance-mobilites.fr/api/explore/v2.1/catalog/datasets/histo-validations-reseau-surface/files/68cac32e8717f476905a60006a4dca26","size":24889411},
            "2022" : {"zip":True,"target":"surface/","url":"https://data.iledefrance-mobilites.fr/api/explore/v2.1/catalog/datasets/histo-validations-reseau-surface/files/fa7ac9a106a1ce78d4158101a17404bd","size":26959530},
        }
    }

    download_dataset_dict_raw = dict()
    for key in download_urls.keys():
        download_dataset_dict_raw[key] = download_urls[key].keys()
    download_dataset_dict = dict(download_dataset_dict_raw)


    if(len(sys.argv)==1):
        print("")
        print("No argument specified, here is the template : ")
        print("\tpy opening.py <what to download=download all>")
        print('\texample: py opening.py surface.2016 surface.2020')
        print("\nBy continuing, you will download all the datasets files!")
        continue_str = input("Press enter to continue, type in 0 to exit : ")
        if(continue_str=="0"):
            exit()
        print("")
    elif(len(sys.argv)>1):
        input_str = ""
        for key in download_dataset_dict.keys():
            download_dataset_dict[key] = []
        for str_arg in sys.argv[1:]:
            expl_arg = str_arg.split(".")
            if(len(expl_arg)==1):
                if expl_arg[0]!='':
                    if expl_arg[0] in download_dataset_dict_raw.keys():
                        download_dataset_dict[expl_arg[0]] = download_dataset_dict_raw[expl_arg[0]]
                    else:
                        print("No dataset :",expl_arg[0])
            if(len(expl_arg)==2):
                if expl_arg[0]!='' and expl_arg[1]!='':
                    if expl_arg[0] in download_dataset_dict_raw.keys():
                        if expl_arg[1] in download_dataset_dict_raw[expl_arg[0]]:
                            download_dataset_dict[expl_arg[0]].append(expl_arg[1])
                        else:
                            print("No dataset version :",expl_arg[1],"of",expl_arg[0])
                    else:
                        print("No dataset :",expl_arg[0])
                else:
                    print("Syntax error here :",str_arg)
            else:
                print("Syntax error here :",str_arg)





    for dataset in download_dataset_dict.keys():
        print("\nStarting to download dataset :",dataset.upper())
        url_dict = download_urls[dataset]
        print("\t"+str(len(download_dataset_dict[dataset]))+" file(s) available!\n")
        for file_key in url_dict.keys():
            if file_key not in download_dataset_dict[dataset]:
                continue
            file_info = url_dict[file_key]
            if(file_info["zip"]):
                try:
                    from zipfile import ZipFile
                except Exception:
                    print("To run this file, please install \033[1mzipfile\033[0m package.")
                    print("\tpip install zipfile\n")
                    exit()
                filedirectory = "raw_datasets/"+file_info["target"]
                filepath = "raw_datasets/"+file_info["target"]+file_key+".zip"
                Path(str(Path(__file__).parent.resolve())+"/../"+filedirectory).mkdir(parents=True, exist_ok=True)
            url = download_urls[dataset][file_key]["url"]
            size = download_urls[dataset][file_key]["size"]
            
            print("\nDownloading",file_key+"...")
            with requests.get(url, stream=True) as response:
                # Sizes in bytes.
                total_size = int(response.headers.get("content-length", size))
                block_size = 1024

                with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
                    with open(filepath, "wb") as file:
                        for data in response.iter_content(block_size):
                            progress_bar.update(len(data))
                            file.write(data)

                if total_size != 0 and progress_bar.n != total_size:
                    print(progress_bar.n)
                    raise RuntimeError("Could not download file")
            if(file_info["zip"]):
                print("Extracting ZIP...")
                with ZipFile(filepath, 'r') as zObject:
                    zObject.extractall(filedirectory) 
                os.remove(filepath)


    print("\n\nDone!")


    

    

if __name__ == "__main__":
    download_datasets()