#!/usr/bin/env python3
import sys
sys.path.sort()

import os, argparse
import subprocess
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
    sns.boxplot(data=BT, x="Fly_s", y="Nst_dist", hue="Fly_s", palette="Paired", legend=False)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right")
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
    ret = subprocess.run(["Rscript", ID + ".R"], cwd=PATH, capture_output=True, text=True)
    if ret.returncode != 0:
        print("(R warning) %s: %s" % (ID + ".R", ret.stderr[:500] if ret.stderr else ret.stdout[:500]))
    ## Nearst Number
    PATH = "Video_post/Nearst_num/"
    if not os.path.exists(PATH):
      # Create a new directory because it does not exist
      os.makedirs(PATH)

    fig, ax = plt.subplots(figsize=(10,6))
    sns.boxplot(data=BT, x="Fly_s", y="Nst_num", hue="Fly_s", palette="Paired", legend=False)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right")
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
    ret = subprocess.run(["Rscript", ID + ".R"], cwd=PATH, capture_output=True, text=True)
    if ret.returncode != 0:
        print("(R warning) %s: %s" % (ID + ".R", ret.stderr[:500] if ret.stderr else ret.stdout[:500]))
    ## Speed
    PATH = "Video_post/mm_S/"

    if not os.path.exists(PATH):
      # Create a new directory because it does not exist
      os.makedirs(PATH)

    fig, ax = plt.subplots(figsize=(10,6))
    sns.boxplot(data=BT, x="Fly_s", y="mm/s", hue="Fly_s", palette="Paired", legend=False)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right")
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

    ret = subprocess.run(["Rscript", ID + ".R"], cwd=PATH, capture_output=True, text=True)
    if ret.returncode != 0:
        print("(R warning) %s: %s" % (ID + ".R", ret.stderr[:500] if ret.stderr else ret.stdout[:500]))

    ### Behaviro
    PATH = "Video_post/Bhav/"

    if not os.path.exists(PATH):
      # Create a new directory because it does not exist
      os.makedirs(PATH)

    N_row = len(BT.Fly_s.unique())
    fig, ax = plt.subplots(figsize=(15, N_row))
    Num = 0
    for Fly_id in BT.Fly_s.unique():
        Num += 1
        ax_sub = fig.add_subplot(N_row, 1, Num)
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

    ret = subprocess.run(["Rscript", ID + ".R"], cwd=PATH, capture_output=True, text=True)
    if ret.returncode != 0:
        print("(R warning) %s: %s" % (ID + ".R", ret.stderr[:500] if ret.stderr else ret.stdout[:500]))

    ### Move
    PATH = "Video_post/Move/"

    if not os.path.exists(PATH):
      # Create a new directory because it does not exist
      os.makedirs(PATH)

    N_row = len(BT.Fly_s.unique())
    fig, ax = plt.subplots(figsize=(15, N_row))
    Num = 0
    for Fly_id in BT.Fly_s.unique():
        Num += 1
        ax_sub = fig.add_subplot(N_row, 1, Num)
        TMP = BT[BT.Fly_s==Fly_id]
        sns.rugplot(data=TMP, x="Frame", y=0, hue="Move", height=1, alpha=ALPHA, palette="tab10").set(xlim=(min(BT.Frame), max(BT.Frame)))
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

    ret = subprocess.run(["Rscript", ID + ".R"], cwd=PATH, capture_output=True, text=True)
    if ret.returncode != 0:
        print("(R warning) %s: %s" % (ID + ".R", ret.stderr[:500] if ret.stderr else ret.stdout[:500]))

    ### Motion
    PATH = "Video_post/Motion/"

    if not os.path.exists(PATH):
      # Create a new directory because it does not exist
      os.makedirs(PATH)

    N_row = len(BT.Fly_s.unique())
    fig, ax = plt.subplots(figsize=(15, N_row))
    Num = 0
    for Fly_id in BT.Fly_s.unique():
        Num += 1
        ax_sub = fig.add_subplot(N_row, 1, Num)
        TMP = BT[BT.Fly_s==Fly_id]
        sns.rugplot(data=TMP, x="Frame", y=0, hue="Motion", height=1, alpha=ALPHA, palette="tab10").set(xlim=(min(BT.Frame), max(BT.Frame)))
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

    ret = subprocess.run(["Rscript", ID + ".R"], cwd=PATH, capture_output=True, text=True)
    if ret.returncode != 0:
        print("(R warning) %s: %s" % (ID + ".R", ret.stderr[:500] if ret.stderr else ret.stdout[:500]))

def _run_r(cwd, script, label=None):
    ret = subprocess.run(["Rscript", script], cwd=cwd, capture_output=True, text=True)
    if ret.returncode != 0:
        print("(R warning) %s: %s" % (label or script, (ret.stderr or ret.stdout or "")[:500]))

def Plot_comp():
    ## Nearst Distance
    _run_r("Video_post/Nearst_dist/", "Nst_dist_all.R", "Nst_dist_all.R")
    ## Nearst Number
    _run_r("Video_post/Nearst_num/", "Nst_num_all.R", "Nst_num_all.R")
    ## Speed
    _run_r("Video_post/mm_S/", "Speed_all.R", "Speed_all.R")
    ### Behaviro
    _run_r("Video_post/Bhav/", "Behav_count_all.R", "Behav_count_all.R")
    ### Move
    _run_r("Video_post/Move/", "Move_count_all.R", "Move_count_all.R")
    ### Motion
    _run_r("Video_post/Motion/", "Motion_count_all.R", "Motion_count_all.R")

def multicore(Pool=Process):
  pool = mp.Pool(processes=Pool)
  for i in List:
    # Working function "echo" and the arg 'i'
    multi_res = [pool.apply_async(Plot_general,(i,))]
  pool.close()
  pool.join()

multicore()
Plot_comp()
