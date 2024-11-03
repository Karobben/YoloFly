#!/usr/bin/env python3

'''
This script is for help you determine how to trunk the video.
It would plot the relative activities of flies.
For study the social behavior, it is important to make sure that flies are social rather sleep/resting.
'''

import sys
sys.path.sort()


import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('-p','-P','--path', default = "Y", type=str)     #输入文件

##获取参数
args = parser.parse_args()
PATH_N = args.path

import os, json
import numpy as np
import pandas as pd
import seaborn as sns
import multiprocessing as mp
import matplotlib.pyplot as plt
from pygam import LinearGAM
from scipy.signal import find_peaks


QC_path = 'QC/'
if not os.path.exists(QC_path):
  # Create a new directory because it does not exist
  os.makedirs(QC_path)

RAW_file = "csv/"
Video_ls = [i.replace(".csv", "") for i in os.listdir(RAW_file) if ".csv" == i[-4:]]
Raw_list = os.listdir(RAW_file)

def Postion_plot(Video):
    print(Video)
    CSV  = [RAW_file + i for i in Raw_list if  ".csv" == i[-4:]][0]
    JSON = [RAW_file + i for i in Raw_list if ".json" == i[-5:]][0]
    ## frame → fly → body and head
    Fly_dic = {}
    [Fly_dic.update(json.loads(i)) for i in open(JSON,"r").read().split(";")[:-1]]
    ## fly → frame → body and head
    Fly_pos={}
    [Fly_pos.update({fly:[Fly_dic[i][fly] for i in Fly_dic.keys()]})
        for fly in Fly_dic[list(Fly_dic.keys())[0]]]
    Fly_En = []
    for fly_id in Fly_pos.keys():
        X = [Fly_pos[fly_id][i]['body'][0] for i in range(len(Fly_pos[fly_id]))]
        Y = [Fly_pos[fly_id][i]['body'][1] for i in range(len(Fly_pos[fly_id]))]

        if PATH_N != "N":
            fig, ax = plt.subplots(figsize = (192/30,108/30))
            sns.scatterplot(x=X,y=Y,linewidth=0, alpha = .01)
            ax.set_xlabel(Video + "_"+fly_id)
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            fig.savefig(QC_path + Video + "_"+fly_id +"_path.png")

        X_d = np.array(X[1:]) - np.array(X[:-1])
        Y_d = np.array(Y[1:]) - np.array(Y[:-1])
        E_d = np.sqrt(X_d**2 + Y_d**2)

        # 30 fps, 60 second for 1 min
        Per = 30*60

        E_d_bar = [E_d[i*Per:(i+1)*Per].sum() for i in range(int(len(E_d)/Per))]
        Fly_En += [[fly_id] + E_d_bar]
        fig, ax = plt.subplots(figsize = (10,4))
        sns.lineplot(x=range(len(E_d_bar)), y = E_d_bar)
        ax.set_xlabel("Time: second")
        ax.set_ylabel("Relative motion: " + fly_id)
        ax.set_title(Video + "_"+fly_id)
        fig.savefig(QC_path + Video + "_"+fly_id +"_Energy.png")
    TB = pd.DataFrame(Fly_En)
    TB = TB.melt(id_vars=0)

    y = TB.value
    X = np.array([[i] for i in TB.variable])
    gam = LinearGAM(n_splines=25).gridsearch(X, y)
    XX = gam.generate_X_grid(term=0, n=500)
    peaks, properties =  find_peaks(gam.predict(XX), prominence=.1, width=20)

    fig, ax = plt.subplots(figsize = (10,4))
    plt.plot(XX[peaks], gam.predict(XX)[peaks], 'ro')
    plt.plot(XX, gam.predict(XX), '--', color="salmon")
    plt.plot(XX, gam.prediction_intervals(XX, width=.95), color='steelblue', ls='--')
    plt.scatter(X, y, facecolor='gray', edgecolors='none', alpha = .5)
    plt.title(Video+' Energetic Index')
    for x_, y_ in zip(XX[peaks], gam.predict(XX)[peaks]):
        plt.text(x_, y_,str(round(x_[0], 2)))
    fig.savefig(QC_path + Video +"_Energy.png")

    Peak_TB = pd.DataFrame([[i[0] for i in XX[peaks]], gam.predict(XX)[peaks]]).T
    Peak_TB.columns = ["Min", "Move_Index"]
    Peak_TB.to_csv(QC_path + Video +"_E_peak.csv")

def multicore(Pool=10):
  pool = mp.Pool(processes=Pool)
  for i in Video_ls:
    # Working function "echo" and the arg 'i'
    multi_res = [pool.apply_async(Postion_plot,(i,))]
  pool.close()
  pool.join()

multicore()
