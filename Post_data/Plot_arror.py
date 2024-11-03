#!/usr/bin/env python3
import numpy as np
import pandas as pd
import seaborn as sns
import argparse, os, math, json
import matplotlib.pyplot as plt

AUTO_PATH = os.path.realpath(__file__).replace("Plot_arror.py", "")

print(AUTO_PATH)

parser = argparse.ArgumentParser()
parser.add_argument('-i','-I','--input', type = str)    #Video
parser.add_argument('-f','-F','--fm', type = int, default = 1)     #Starts at the frame
parser.add_argument('-e','-E','--end', type = int, default = 1000)      #Ends at the frame
parser.add_argument('-t','-T','--target', nargs='+')      #Ends at the frame
parser.add_argument('-c','-C','--classes', nargs='+', type = int, default =  [0,2,3,4,5])    #Classies
parser.add_argument('-s','-S','--suffix', type =str)    #Classies
parser.add_argument('-o','-O','--output', type =str,  default ="")    #Classies

args = parser.parse_args()
video_id = args.input
FROM_fm = args.fm
END_fm = args.end
TARGETS = args.target
CLASS = args.classes
SUFFIX = args.suffix
OUTPUT = args.output

print(video_id)
print(TARGETS)
print(CLASS)

Mate = os.popen("grep " + video_id + " Video_list.csv").read().replace("\n", '').split("\t")
SCALE =  int(Mate[1])/int(Mate[2])

def Arrow_plt(Origin, Radian, Angle, COLOR = "steelblue"):
    End_x = math.cos(math.radians(Angle)) * Radian
    End_y = math.sin(math.radians(Angle)) * Radian

    return plt.arrow(Origin[0], Origin[1], End_x, End_y,
        head_width = 20, ec=  'white', fc = COLOR,
        width = 10, alpha = 0.2)

def Frame_judge(TB):
    if TB.Frame.min() > FROM_fm:
        print("Error, Frame not start from. Frame is started at " + str(TB.Frame.min()))
        raise "Error, Start Frame is too "
    if TB.Frame.max() < END_fm:
        print("Error, Frame out of boundary. The max frame is " + str(TB.Frame.max()))
        raise "Error"

_NAME_ = "Video_post/Correct_"+video_id + SUFFIX + ".csv"
TB = pd.read_csv(_NAME_, index_col=0)
Frame_judge(TB)
TB = TB[TB.Frame.isin(range(FROM_fm, END_fm))]
TB = TB[TB.Fly_s.isin(TARGETS)]
TB = TB[['Frame', 'X', 'Y', 'length', 'B_angle', 'Grooming', 'Fs_x', 'Sing', 'Hold']]

PALETTE = json.load(open(AUTO_PATH + "config.json"))['PALETTE']

TB["Class"] = "NA"
if 0 in CLASS:
    TB["Class"] = "Fly"
if 2 in CLASS:
    TB['Class'][TB['Grooming']!=0] = "Groom"
if 3 in CLASS:
    TB['Class'][TB['Fs_x']!="0"] = "Chase"
if 4 in CLASS:
    TB['Class'][TB['Sing']!=0] = "Sing"
if 5 in CLASS:
    TB['Class'][TB['Hold']!=0] = "Hold"


fig, ax = plt.subplots(figsize=(20, 10.9))
ax.set_xlim(0, 1920)
ax.set_ylim(0, 1080)


for i in range(len(TB)):
    TMP = TB.iloc[i,:]
    if TMP.Class != "NA":
        Arrow_plt( (TMP["X"], TMP["Y"]), TMP.length * SCALE /2, TMP.B_angle + 180, PALETTE[TMP.Class])

plt.gca().invert_yaxis()
if OUTPUT =="":
    plt.show()
else:
    plt.savefig(OUTPUT)
