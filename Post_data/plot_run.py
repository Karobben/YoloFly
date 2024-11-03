#!/usr/bin/env python3
import argparse, os
import multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument('-p','-P','--process', type = int, default = 5)    #Video

args = parser.parse_args()
Pool = args.process

AUTO_PATH = os.path.realpath(__file__).replace("plot_run.py", "")



import pandas as pd

TB = pd.read_csv("Video_list.csv", sep ='\t', header= None)
MAX = os.popen("awk '{print $1}' Video_list| sort -n| tail -n 1").read().replace("\n", "")

print(MAX)
def Plat_plot(TMP, MAX):
    # plat_plot
    CMD ="python3 " + AUTO_PATH + "/Plate_plot.py -i " + TMP[0] + " -f " + str(TMP[3]) + " -e " +  str(TMP[4]) + " -c 0,2,3,4,5"
    print(CMD)
    #os.system(CMD)
    # Chain plot
    CMD ="python3 " + AUTO_PATH + "/Plot_chain.py -i " + TMP[0] + " -f " + str(TMP[3]) + " -e " +  str(TMP[4]) + " -m " + MAX
    print(CMD)
    os.system(CMD)




def multicore(Pool=Pool):
  pool = mp.Pool(processes=Pool)
  for i in range(len(TB)):
    TMP = TB.iloc[i, :]
    multi_res = [pool.apply_async(Plat_plot,(TMP,MAX,))]
  pool.close()
  pool.join()

if __name__ == '__main__':
  multicore(Pool)
