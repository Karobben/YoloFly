#!/usr/bin/env python3
import sys
sys.path.sort()

import os, argparse
import pandas as pd
import seaborn as sns
from pathlib import Path
from shutil import copyfile
import matplotlib.pyplot as plt

SELF = str(Path(__file__).resolve()).replace("Plot_1.py", "")
print(SELF)

parser = argparse.ArgumentParser()
parser.add_argument('-p','-P','--process', default = 10, type=int)

args = parser.parse_args()
Process = args.process

import multiprocessing as mp

List = [i for i in os.listdir("Video_post") if "Correct_" in i]

def Plot_general(ID):
    BT = pd.read_csv("Video_post/" + ID, index_col=0)
    ALPHA = .1
    ## Nearst Distance
    PATH = "Video_post/Nearst_dist/"
    if not os.path.exists(PATH):
      # Create a new directory because it does not exist
      os.makedirs(PATH)

    fig, ax = plt.subplots(figsize=(10,6))
    sns.boxplot(data = BT, x="Fly_s", y = "Nst_dist", palette="Paired")
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
    ax.set_title(ID)
    fig.savefig(PATH+ID+".png")
    F = open(SELF+"/R/Nst_dist.R").read()
    F= F.replace("##DATA##", ID)
    F= F.replace("##TYPE##", "NstD")
    f = open(PATH+ID+".R", 'w')
    f.write(F)
    f.close()
    F = open(SELF+"/R/Nst_dist_all.R").read()
    F= F.replace("##TYPE##", "NstD")
    f = open(PATH+"Nst_dist_all"+".R", 'w')
    f.write(F)
    f.close()
    # Change the workind directory
    os.chdir(PATH)
    os.system("Rscript "+ID+".R")
    # back to origin directory
    os.chdir("../../")
    ## Nearst Number
    PATH = "Video_post/Nearst_num/"
    if not os.path.exists(PATH):
      # Create a new directory because it does not exist
      os.makedirs(PATH)

    fig, ax = plt.subplots(figsize=(10,6))
    sns.boxplot(data = BT, x="Fly_s", y = "Nst_num", palette="Paired")
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
    ax.set_title(ID)
    fig.savefig(PATH+ID+".png")
    F = open(SELF+"/R/Nst_dist.R").read()
    F= F.replace("##DATA##", ID)
    F= F.replace("##TYPE##", "NstN")
    f = open(PATH+ID+".R", 'w')
    f.write(F)
    f.close()
    F = open(SELF+"/R/Nst_dist_all.R").read()
    F= F.replace("##TYPE##", "NstN")
    f = open(PATH+"Nst_num_all"+".R", 'w')
    f.write(F)
    f.close()
    # Change the workind directory
    os.chdir(PATH)
    os.system("Rscript "+ID+".R")
    os.chdir("../../")
    ## Speed
    PATH = "Video_post/mm_S/"

    if not os.path.exists(PATH):
      # Create a new directory because it does not exist
      os.makedirs(PATH)

    fig, ax = plt.subplots(figsize=(10,6))
    sns.boxplot(data = BT, x="Fly_s", y = "mm/s", palette="Paired")
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
    ax.set_title(ID)
    fig.savefig(PATH+ID+".png")

    F = open(SELF+"/R/Nst_dist.R").read()
    F= F.replace("##DATA##", ID)
    F= F.replace("##TYPE##", "Speed")
    f = open(PATH+ID+".R", 'w')
    f.write(F)
    f.close()
    F = open(SELF+"/R/Nst_dist_all.R").read()
    F= F.replace("##TYPE##", "Speed")
    f = open(PATH+"Speed_all"+".R", 'w')
    f.write(F)
    f.close()

    # Change the workind directory
    os.chdir(PATH)
    os.system("Rscript "+ID+".R")
    os.chdir("../../")

    ### Behaviro
    PATH = "Video_post/Bhav/"

    if not os.path.exists(PATH):
      # Create a new directory because it does not exist
      os.makedirs(PATH)

    N_row = len(BT.Fly_s.unique())
    fig, ax = plt.subplots(figsize=(15,  N_row))
    Num = 0
    for Fly_id in BT.Fly_s.unique():
        Num += 1
        plt.subplot(N_row,1,Num)
        TMP = BT[BT.Fly_s==Fly_id]
        TMP1 = TMP[TMP.Grooming!=0]
        sns.rugplot(data=TMP1, x="Frame", y= 0,  height=1, alpha=ALPHA, color ="black")
        TMP1 = TMP[TMP.Chasing!=0]
        sns.rugplot(data=TMP1, x="Frame", y= 0,  height=1, alpha=ALPHA, color ="steelblue")
        TMP1 = TMP[TMP.Sing!=0]
        sns.rugplot(data=TMP1, x="Frame", y= 0,  height=1, alpha=ALPHA, color ="salmon").set(xlim=(min(BT.Frame),max(BT.Frame)))
        TMP1 = TMP[TMP.Hold!=0]
        sns.rugplot(data=TMP1, x="Frame", y= 0,  height=1, alpha=ALPHA, color ="yellow").set_ylabel(Fly_id)

    fig.savefig(PATH+ID+".png")
    F = open(SELF+"/R/Behav_count.R").read()
    F= F.replace("##DATA##", ID)
    F= F.replace("##TYPE##", "Behavior")
    f = open(PATH+ID+".R", 'w')
    f.write(F)
    f.close()
    # Between Groups
    F = open(SELF+"/R/Behav_count_all.R").read()
    F= F.replace("##TYPE##", "Behavior")
    f = open(PATH+"Behav_count_all.R", 'w')
    f.write(F)
    f.close()

    # Change the workind directory
    os.chdir(PATH)
    os.system("Rscript "+ID+".R")
    os.chdir("../../")


    ### Move
    PATH = "Video_post/Move/"

    if not os.path.exists(PATH):
      # Create a new directory because it does not exist
      os.makedirs(PATH)

    N_row = len(BT.Fly_s.unique())
    fig, ax = plt.subplots(figsize=(15,  N_row))
    Num = 0
    for Fly_id in BT.Fly_s.unique():
        Num += 1
        plt.subplot(N_row,1,Num)
        TMP = BT[BT.Fly_s==Fly_id]
        sns.rugplot(data=TMP, x="Frame", y= 0, hue="Move", height=1, alpha=ALPHA, palette="tab10").set(xlim=(min(BT.Frame),max(BT.Frame)))
        TMP1 = TMP[TMP.Hold!=0]
        sns.rugplot().set_ylabel(Fly_id)

    fig.savefig(PATH+ID+".png")

    F = open(SELF+"/R/Behav_count.R").read()
    F= F.replace("##DATA##", ID)
    F= F.replace("##TYPE##", "Move")
    f = open(PATH+ID+".R", 'w')
    f.write(F)
    f.close()
    # Between Groups
    F = open(SELF+"/R/Behav_count_all.R").read()
    F= F.replace("##TYPE##", "Move")
    f = open(PATH+"Move_count_all.R", 'w')
    f.write(F)
    f.close()

    # Change the workind directory
    os.chdir(PATH)
    os.system("Rscript "+ID+".R")
    os.chdir("../../")

    ### Motion
    PATH = "Video_post/Motion/"

    if not os.path.exists(PATH):
      # Create a new directory because it does not exist
      os.makedirs(PATH)

    N_row = len(BT.Fly_s.unique())
    fig, ax = plt.subplots(figsize=(15,  N_row))
    Num = 0
    for Fly_id in BT.Fly_s.unique():
        Num += 1
        plt.subplot(N_row,1,Num)
        TMP = BT[BT.Fly_s==Fly_id]
        sns.rugplot(data=TMP, x="Frame", y= 0, hue="Motion", height=1, alpha=ALPHA, palette="tab10").set(xlim=(min(BT.Frame),max(BT.Frame)))
        TMP1 = TMP[TMP.Hold!=0]
        sns.rugplot().set_ylabel(Fly_id)

    fig.savefig(PATH+ID+".png")

    F = open(SELF+"/R/Behav_count.R").read()
    F= F.replace("##DATA##", ID)
    F= F.replace("##TYPE##", "Motion")
    f = open(PATH+ID+".R", 'w')
    f.write(F)
    f.close()
    # Between Groups
    F = open(SELF+"/R/Behav_count_all.R").read()
    F= F.replace("##TYPE##", "Motion")
    f = open(PATH+"Motion_count_all.R", 'w')
    f.write(F)
    f.close()

    # Change the workind directory
    os.chdir(PATH)
    os.system("Rscript "+ID+".R")
    os.chdir("../../")
def Plot_comp():
    ## Nearst Distance
    PATH = "Video_post/Nearst_dist/"
    os.chdir(PATH)
    os.system("Rscript Nst_dist_all.R")
    os.chdir("../../")

    ## Nearst Number
    PATH = "Video_post/Nearst_num/"
    os.chdir(PATH)
    os.system("Rscript Nst_num_all.R")
    os.chdir("../../")

    ## Speed
    PATH = "Video_post/mm_S/"
    os.chdir(PATH)
    os.system("Rscript Speed_all.R")
    os.chdir("../../")


    ### Behaviro
    PATH = "Video_post/Bhav/"
    os.chdir(PATH)
    os.system("Rscript Behav_count_all.R")
    os.chdir("../../")


    ### Move
    PATH = "Video_post/Move/"
    os.chdir(PATH)
    os.system("Rscript Move_count_all.R")
    os.chdir("../../")

    ### Motion
    PATH = "Video_post/Motion/"
    os.chdir(PATH)
    os.system("Rscript Motion_count_all.R")
    os.chdir("../../")

def multicore(Pool=Process):
  pool = mp.Pool(processes=Pool)
  for i in List:
    # Working function "echo" and the arg 'i'
    multi_res = [pool.apply_async(Plot_general,(i,))]
  pool.close()
  pool.join()

multicore()
Plot_comp()
