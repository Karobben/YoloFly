#!/usr/bin/env python3
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('-p','-P','--process', default = 10, type=int)     # input

## Aquire Args
args = parser.parse_args()
Process = args.process

## Aquire Path oof itself
script = os.path.realpath(__file__).replace("run.py","")
print("SCript path:", script)


import multiprocessing as mp


List = open("Video_list.csv").read().split("\n")[:-1]
Video_list  = [i.split("\t")[0] for i in List]

def Single_fly(Line):
    CMD = "python "+script+"/1_single_fly_run_arg.py -i " +Line[0]+" -pp "+Line[1]+" -pm "+Line[2]+" -fs "+Line[3]+" -fe "+Line[4]
    print(CMD)
    os.system(CMD)

def Interect_fly(Line):
    CMD = "python "+script+"/2_Chas_behavior_arg.py -i " +Line[0]+" -pp "+Line[1]+" -pm "+Line[2]+" -fs "+Line[3]+" -fe "+Line[4]
    print(CMD)
    os.system(CMD)

def SingTYpe_fly(Video_list):
    CMD = "python "+script+"/3_single_and_Chascls_arg.py -i " + Video_list
    print(CMD)
    os.system(CMD)

def Plot_1(Process):
    CMD = "python "+script+"/Plot_1.py -p " + str(Process)
    print(CMD)
    os.system(CMD)


def multicore(Pool=Process):
  pool = mp.Pool(processes=Pool)
  for i in List:
    # Working function "echo" and the arg 'i'
    multi_res = [pool.apply_async(Single_fly,(i.split("\t"),))]
  pool.close()
  pool.join()

def multicore2(Pool=Process):
  pool = mp.Pool(processes=Pool)
  for i in List:
    # Working function "echo" and the arg 'i'
    multi_res = [pool.apply_async(Interect_fly,(i.split("\t"),))]
  pool.close()
  pool.join()

def multicore3(Pool=Process):
  pool = mp.Pool(processes=Pool)
  for i in Video_list:
    # Working function "echo" and the arg 'i'
    multi_res = [pool.apply_async(SingTYpe_fly,(i,))]
  pool.close()
  pool.join()


if __name__ == '__main__':
  multicore()
  multicore2()
  multicore3()
  Plot_1(Process)
