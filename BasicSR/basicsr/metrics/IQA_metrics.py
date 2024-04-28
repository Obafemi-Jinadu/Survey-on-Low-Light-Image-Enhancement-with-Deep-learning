import json
import pandas as pd
import numpy as np
import os
import shutil
import argparse
from pathlib import Path
import cv2
import math
import numpy as np
import os
from scipy.ndimage import convolve
from scipy.special import gamma


import sys
from brisque import BRISQUE



parser = argparse.ArgumentParser(description='Get arguments, data paths in this case.')
parser.add_argument('--niqe_path', type=str,
                    help='path to basicSR/basicsr')
parser.add_argument('--data', type=str,
                    help='path/dir to data')
parser.add_argument('--metric', type=str,
                    help='enter either argument: NIQE, BRISQUE or both')

                    

args = parser.parse_args()

data_path = Path(args.data)
metric = args.metric
niqe_path = Path(args.niqe_path)


import sys
sys.path.insert(0, niqe_path)

from basicsr.metrics.niqe import estimate_aggd_param, compute_feature,niqe,calculate_niqe
from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.matlab_functions import imresize
from basicsr.utils.registry import METRIC_REGISTRY


def brisque_comp(dir_path):
    result = 0
    path_list = os.listdir(dir_path)
    obj = BRISQUE(url=False)
    for i in range(len(path_list)):
        img_path = os.path.join(dir_path,  path_list[i])
        im = cv2.imread(img_path)
        
        result+=obj.score(im)
        
        #print(result)
    return result/len(path_list)
    
def niqe_comp(dir_path):
    result = 0
    path_list = os.listdir(dir_path)
    
    for i in range(len(path_list)):
        img_path = os.path.join(dir_path,  path_list[i])
        im = cv2.imread(img_path)
        result+=calculate_niqe(im,0)
        #print(result)
    return result/len(path_list)
    
    
if __name__ == "__main__":
    if metric == 'BRISQUE':
    	brisque_score = brisque_comp(data_path)
    	print(f'BRISQUE score: {brisque_score:.3f}')
    elif metric == 'NIQE':
    	niqe_score =  niqe_comp(data_path)
    	print(f'NIQE score: {niqe_score:.3f}')
    elif metric == 'both':
    	brisque_score = brisque_comp(data_path)
    	niqe_score = niqe_comp(data_path)
    	print(f'BRISQUE score: {brisque_score:.3f} | NIQE score: {niqe_score:.3f}')
    else:
    	print("invalide metric args")
