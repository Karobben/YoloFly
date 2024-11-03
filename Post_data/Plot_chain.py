#!/usr/bin/env python3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse, os

os.system("mkdir img")
os.system("mkdir img/Chain_plot")

AUTO_PATH = os.path.realpath(__file__).replace("Plot_arror_multy.py", "")

print(AUTO_PATH)

parser = argparse.ArgumentParser()
parser.add_argument('-i','-I','--input', type = str)    #Video
parser.add_argument('-f','-F','--fm', type = int, default = 0)     #Starts at the frame
parser.add_argument('-e','-E','--end', type = int, default = 1000)      #Ends at the frame
parser.add_argument('-x','-x','--axisx', nargs='+', type = int, default =  [0,1920])    #Classies
parser.add_argument('-w','-W','--width',type = int, default = 20)    #Classies
parser.add_argument('-p','-P','--palette',nargs='?', default = "Paired")    #Classies
parser.add_argument('-m','-M','--max',nargs='?', type=int)    #Classies
parser.add_argument('-s','-S','--suffix', type =str)    #Classies



args = parser.parse_args()
video_id = args.input
FROM_fm = args.fm
END_fm = args.end
AXISX = args.axisx
WIDTH = args.width
PALETTE = args.palette
MAX = args.max
SUFFIX = args.suffix


print(video_id)


def Arrow_plt(Origin, Radian, Angle, WIDTH = 20 ,COLOR = "steelblue", COLOR2="white"):
    End_x = math.cos(math.radians(Angle)) * Radian
    End_y = math.sin(math.radians(Angle)) * Radian

    return plt.arrow(Origin[0],  Origin[1], End_x, End_y,
        head_width = WIDTH, ec=  COLOR2, fc = COLOR,
        width = WIDTH/2, alpha = .6, linewidth= WIDTH/2)

def Frame_judge(TB):
    if TB.Frame.min() > FROM_fm:
        print("Error, Frame not start from. Frame is started at " + str(TB.Frame.min()))
        raise "Error, Start Frame is too "
    if TB.Frame.max() < END_fm:
        print("Error, Frame out of boundary. The max frame is " + str(TB.Frame.max()))
        raise "Error"



_NAME_ = "Video_post/Correct_"+video_id + SUFFIX + ".csv"
TB = pd.read_csv(_NAME_, index_col=0)


if FROM_fm!=0:
    Frame_judge(TB)
    TB = TB[TB.Frame.isin([int(i) for i in range(FROM_fm, END_fm+1)])]
if FROM_fm==0:
    FROM_fm = min(TB.Frame)
    END_fm  = max(TB.Frame)


TB = TB[TB.Fs_x != "0"]
TB = TB[TB.Ft_x != "0"]
TB = TB[TB.Fs_y != "0"]
TB = TB[TB.Ft_y != "0"]

TB_plot = TB.Frame.value_counts().reset_index().reindex()

if MAX == None:
    MAX = TB_plot.Frame.max()
Cmap = sns.color_palette(PALETTE, MAX).as_hex()

if MAX >= 13 and len(set(sns.color_palette(PALETTE, MAX).as_hex())) != 13:
    print("Your Currunt color palette is:", PALETTE)
    print("The max possibility number is 13 which larger than the number of color in currunt pallete. Please indicate a continute color palette like 'rocket_r' by '-p rocket_r'. Here  we'll use rocket_r as default")
    PALETTE = 'rocket_r'
    Cmap = sns.color_palette(PALETTE, MAX).as_hex()



fig, ax = plt.subplots(figsize=(10, 1))
ax.set_xlim(FROM_fm, END_fm)
ax.set_ylim(0, 1)
sns.rugplot(data=TB_plot, x="index", y= 0,
    height =1 ,hue = "Frame", palette = Cmap[:TB_plot.Frame.max()],
    legend = None)
plt.xticks(range(FROM_fm, END_fm+1)[::1800])
plt.savefig("img/Chain_plot/" + video_id+'_chain.svg', bbox_inches='tight')

fig, ax = plt.subplots(figsize=(10, 0.4))
sns.heatmap([[i+1 for i in range(TB_plot.Frame.max())]], cmap = Cmap[:TB_plot.Frame.max()],  cbar=False ,annot=True, fmt='g',  yticklabels=False,  xticklabels=False).tick_params(left=False, bottom=False)
plt.savefig("img/Chain_plot/" + video_id+'_legend.svg')
