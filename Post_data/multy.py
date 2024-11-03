import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument('-c','-C','--command')     #输入文件
parser.add_argument('-p','-P','--processes', type = int)     #输入文件

##获取参数
args = parser.parse_args()
INPUT = args.command
PROCESSES = args.processes

import multiprocessing as mp

F = open(INPUT,'r').readlines()

def run(CMD):
    print(CMD)
    os.system(CMD)

def multicore(Pool=PROCESSES):
  pool = mp.Pool(processes=Pool)
  for i in F:
    # Working function "echo" and the arg 'i'
    multi_res = [pool.apply_async(run,(i,))]
  pool.close()
  pool.join()

if __name__ == '__main__':
  multicore()
