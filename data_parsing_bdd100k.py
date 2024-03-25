import json
import pandas as pd
import numpy as np
import os
import shutil
import argparse
from pathlib import Path


parser = argparse.ArgumentParser(description='Get arguments, data paths in this case.')
parser.add_argument('--link', type=str,
                    help='link to json file')
parser.add_argument('--data', type=str,
                    help='path to data path')
parser.add_argument('--dest', type=str,
                    help='destination folder')




args = parser.parse_args()

link = Path(args.link)
data_path = Path(args.data)
dst_folder = Path(args.dest)

def get_dataframe(link):    
    f = open(link)
    data = json.load(f)
    night_files = []
    weather = []
    scene = []
    for i in range(len(data)):
        if data[i]['attributes']['timeofday'] =='night':
            night_files.append(data[i]['name'])
            weather.append(data[i]['attributes']['weather'])
            scene.append(data[i]['attributes']['scene'])
    stacked_arr = np.hstack((np.array(night_files).reshape(-1,1),np.array(weather).reshape(-1,1),np.array(scene).reshape(-1,1)))
    cols = ['file_name','weather','scene']
    df = pd.DataFrame(data =stacked_arr,columns=cols )
    return df



rng = np.random.default_rng(seed=7)
train_ind = rng.integers(0,len(train),700)
out_train = train[train.index.isin(train_ind)]

#val_ind = rng.integers(0,len(val_df),700)
#out_val = val_df[val.index.isin(val_ind)]

def copy_paste(df,url_src, url_dst):
    
    for i in range(len(df)):
        file_name = df.iloc[i,2]+'_'+df.iloc[i,1]+'_'+df.iloc[i,0]
        new_path = os.path.join(url_dst, file_name)
        src_path = os.path.join(url_src, df.iloc[i,0])
        try:
            shutil.copy(src_path, new_path)
            #print('done')
        except:
            #print("not exist")
            continue
        

if __name__ == "__main__":
    train = get_dataframe(link)
    copy_paste(out_train,data_path, dst_folder)
    