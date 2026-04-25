#!/usr/bin/env python3
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('-p','-P','--process', default = 10, type=int)     # input
parser.add_argument('-r','--rerun', action='store_true', help='rerun all; if not set, skip lines whose output already exists')

## Aquire Args
args = parser.parse_args()
Process = args.process
Rerun = args.rerun

## Aquire Path oof itself
script = os.path.realpath(__file__).replace("run.py","")
print("SCript path:", script)
if not Rerun:
    print("Skip mode: will skip lines where output already exists (use --rerun to force run all)")

import multiprocessing as mp

Video_post = "Video_post"
List = open("Video_list.csv").read().split("\n")[:-1]
Video_list  = [i.split("\t")[0] for i in List]
# Create the output directory if it doesn't exist
if not os.path.exists(Video_post):
    os.makedirs(Video_post)

def _line_parts(i):
    return i.split("\t")

def _single_fly_done(Line):
    p = os.path.join(Video_post, f"{Line[0]}_{Line[3]}_{Line[4]}.csv")
    return os.path.exists(p)

def _interect_done(Line):
    p = os.path.join(Video_post, f"Interection_{Line[0]}_{Line[3]}_{Line[4]}.csv")
    return os.path.exists(p)

def _singtype_done(Line):
    p = os.path.join(Video_post, f"Correct_{Line[0]}_{Line[3]}_{Line[4]}.csv")
    return os.path.exists(p)

def Single_fly(Line):
    CMD = "python "+script+"/1_single_fly_run_arg.py -i " +Line[0]+" -pp "+Line[1]+" -pm "+Line[2]+" -fs "+Line[3]+" -fe "+Line[4]
    print(CMD)
    os.system(CMD)

def Interect_fly(Line):
    CMD = "python "+script+"/2_Chas_behavior_arg.py -i " +Line[0]+" -pp "+Line[1]+" -pm "+Line[2]+" -fs "+Line[3]+" -fe "+Line[4]
    print(CMD)
    os.system(CMD)

def SingTYpe_fly(Line):
    CMD = "python "+script+"/3_single_and_Chascls_arg.py -i " + Line[0] + " -fs " + Line[3] + " -fe " + Line[4]
    print(CMD)
    os.system(CMD)

def Plot_1(Process):
    CMD = "python "+script+"/Plot_1.py -p " + str(Process)
    print(CMD)
    os.system(CMD)


def multicore(Pool=Process):
  pool = mp.Pool(processes=Pool)
  for i in List:
    L = _line_parts(i)
    if Rerun or not _single_fly_done(L):
      pool.apply_async(Single_fly, (L,))
    else:
      print("(skip) single-fly already done:", L[0], L[3], L[4])
  pool.close()
  pool.join()

def multicore2(Pool=Process):
  pool = mp.Pool(processes=Pool)
  for i in List:
    L = _line_parts(i)
    if Rerun or not _interect_done(L):
      pool.apply_async(Interect_fly, (L,))
    else:
      print("(skip) interaction already done:", L[0], L[3], L[4])
  pool.close()
  pool.join()

def multicore3(Pool=Process):
  pool = mp.Pool(processes=Pool)
  for i in List:
    L = _line_parts(i)
    if Rerun or not _singtype_done(L):
      pool.apply_async(SingTYpe_fly, (L,))
    else:
      print("(skip) correct already done:", L[0], L[3], L[4])
  pool.close()
  pool.join()


if __name__ == '__main__':
  multicore()
  multicore2()
  multicore3()
  Plot_1(Process)
