import os
import re
from random import sample
from tqdm import tqdm
import numpy as np
import argparse
from omegaconf import OmegaConf
import glob
import zipfile
from sklearn.preprocessing import MinMaxScaler

# %% 
# Sorting files through its number in the name (0,1,...10)
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text) ]

######## ######## ######## ######## 

# %%

def compressData(data, scaler, dtype=np.int8):   
    data = scaler.fit_transform(np.array([data]).T).astype(dtype)
    
    # reshape to (320,303,228)
    #shape = (320,303,228)
    #_data = np.reshape(data, shape)
    
    return data[:,0]

def getDataFromZipPath(file):
    temp = os.path.join('temp', os.path.basename(file))
    with open(temp, 'wb') as f:
        f.write(zip.read(file))
    data = np.fromfile(temp)
    os.remove(temp)
    return data

def zipfolder(foldername, target_dir):            
    zipobj = zipfile.ZipFile(foldername + '.zip', 'w', zipfile.ZIP_DEFLATED)
    rootlen = len(target_dir)
    for base, dirs, files in os.walk(target_dir):
        for file in files:
            fn = os.path.join(base, file)
            zipobj.write(fn, fn[rootlen:])


# %%

if __name__=='__main__': 
    # Before starting the script, the paths to the raw data and the 'Atrial_geometry' file has to be passed

    parser = argparse.ArgumentParser(description='Unzipping file and proccessing to smaller numpy-arrays')

    parser.add_argument('-source_folder', '-source', type=str, help='', default='../../data/raw/')
    parser.add_argument('-target_folder', '-target', type=str, help='', default='../../data/processed/')
    parser.add_argument('-geometry_file', '-geometry', type=str, help='', default='../../data/processed/Atrial_geometry')

    args = parser.parse_args() 
    
    args = OmegaConf.create(vars(args))
    print(args)
    
#25.35750314224484
#-126.3851830903377

# %%

    shape = (320,303,228)
    geometry = np.reshape(np.fromfile(args.geometry_file, dtype=np.int32), shape)
    inds_heart = np.array(np.where(geometry!=0)).astype(np.int16) # shape: [3,n], transposed list of 3D-vectors
    np.save(args.target_folder + 'inds_heart', inds_heart)

    archives = glob.glob(args.source_folder + '*')
    
    for archive in archives:    
        archive_name = os.path.splitext(os.path.basename(archive))[0]
        
        with zipfile.ZipFile(archive, mode='r') as zip:
            namelist = zip.namelist()
            
            inds = np.where(["V_snap" in name for name in namelist])
            files = list(np.array(namelist)[inds[0]])
            files.sort(key=natural_keys)
            
            #################### find min, max ####################
    
            min_val, max_val = 10000000, -10000000
            for file in tqdm(sample(files, 128)):
                temp = os.path.join('temp', os.path.basename(file))
                with open(temp, 'wb') as f:
                    f.write(zip.read(file))
                data = np.fromfile(temp)
                os.remove(temp)
    
                try:
                    temp_min = data.min()
                    if temp_min < min_val:
                        #print(f"New min: {min_val}")
                        min_val = temp_min
                    
                    temp_max = data.max()
                    if temp_max > max_val:
                        #print(f"New max: {max_val}")
                        max_val = temp_max
                except:
                    print(f"Did not work here: {file}")
                    
            scaler = MinMaxScaler(feature_range=(np.iinfo(np.int8).min, np.iinfo(np.int8).max))
            scaler.data_min_ = min_val
            scaler.data_max_ = max_val
    
            #######################################################
            
            for file in tqdm(files[:]):
                temp = os.path.join('temp', os.path.basename(file))
                with open(temp, 'wb') as f:
                    f.write(zip.read(file))
                data = np.fromfile(temp)
                os.remove(temp)
    
                try:
                    data = compressData(data, scaler, dtype=np.int8)
                    data = np.reshape(data, shape)
                    
                	# return flat vector with values for V, position is stored in the array inds_heart    
                    data = data[inds_heart[0],inds_heart[1],inds_heart[2]] 
                    
                    target_folder = os.path.join(args.target_folder, archive_name)
                    os.makedirs(target_folder, exist_ok=True)
    
                    target_name = os.path.join(target_folder, os.path.basename(file))
                    np.save(target_name, data)
                except:
                    print(f"Did not work here: {file}")
                    
    target_folder_zip = os.path.abspath(os.path.join(args.target_folder, os.pardir))
    target_zip_name = os.path.join(target_folder_zip, 'processed')

#zipfolder(target_zip_name, args.target_folder)

