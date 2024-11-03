#!/usr/bin/env python3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse, os, math, json

AUTO_PATH = os.path.realpath(__file__).replace("Plot_arror_multy.py", "")

print(AUTO_PATH)

parser = argparse.ArgumentParser()
parser.add_argument('-i','-I','--input', type = str)    #Video
parser.add_argument('-f','-F','--fm', type = int, default = 1)     #Starts at the frame
parser.add_argument('-e','-E','--end', type = int, default = 1000)      #Ends at the frame
parser.add_argument('-t','-T','--target', nargs='+')      #Ends at the frame
parser.add_argument('-c','-C','--classes', nargs='+', type = int, default =  [0,2,3,4,5])    #Classies
parser.add_argument('-x','-x','--axisx', nargs='+', type = int, default =  [0,1920])    #Classies
parser.add_argument('-y','-Y','--axisy', nargs='+', type = int, default =  [0,1080])    #Classies
parser.add_argument('-w','-W','--width',type = int, default = 20)    #Classies
parser.add_argument('-p','-P','--photo',nargs='?', default = True)    #Classies
parser.add_argument('-g','-G','--gap',type = int, default = 1)    #Classies
parser.add_argument('-s','-S','--suffix', type =str)    #Classies


args = parser.parse_args()
video_id = args.input
FROM_fm = args.fm
END_fm = args.end
TARGETS = args.target
CLASS = args.classes
AXISX = args.axisx
AXISY = args.axisy
WIDTH = args.width
PHOTO = args.photo
GAP = args.gap
SUFFIX = args.suffix


print(video_id)
print(TARGETS)
print(CLASS)
print(PHOTO)


def Arrow_plt(Origin, Radian, Angle, WIDTH = 20 ,COLOR = "steelblue", COLOR2="white"):
    End_x = math.cos(math.radians(Angle)) * Radian
    End_y = math.sin(math.radians(Angle)) * Radian

    return plt.arrow(Origin[0],  Origin[1], End_x, End_y,
        head_width = WIDTH, ec=  COLOR2, fc = COLOR,
        width = WIDTH/2, alpha = .3, linewidth= WIDTH/2)

def Frame_judge(TB):
    if TB.Frame.min() > FROM_fm:
        print("Error, Frame not start from. Frame is started at " + str(TB.Frame.min()))
        raise "Error, Start Frame is too "
    if TB.Frame.max() < END_fm:
        print("Error, Frame out of boundary. The max frame is " + str(TB.Frame.max()))
        raise "Error"

Mate = os.popen("grep " + video_id + " Video_list.csv").read().replace("\n", '').split("\t")
SCALE =  int(Mate[1])/int(Mate[2])
PALETTE = json.load(open(AUTO_PATH + "config.json"))['PALETTE']

_NAME_ = "Video_post/Correct_"+video_id + SUFFIX + ".csv"
TB = pd.read_csv(_NAME_, index_col=0)

Frame_judge(TB)

TB = TB[TB.Frame.isin([int(i) for i in np.linspace(start=FROM_fm, stop=END_fm, num= int((END_fm- FROM_fm)/GAP))])]
TB = TB[TB.Fly_s.isin(TARGETS)]
TB = TB[['Frame', 'Fly_s', 'X', 'Y', 'length', 'B_angle', 'Grooming', 'Fs_x', 'Sing', 'Hold']]
TB["Class"] = TB.Fly_s+ "_" + TB.Frame.astype(str)

# color for each spots
Palette = json.load(open(AUTO_PATH + "config.json"))['Palette_sequential']

PALETTE = json.load(open(AUTO_PATH + "config.json"))['PALETTE']

TB['Color'] = 'grey'

fig, ax = plt.subplots(figsize=(20, 10.9))

Num = -1
for fly in TARGETS:
    Num += 1
    TB['Color'][TB.Fly_s==fly] = sns.color_palette(Palette[Num%len(Palette)], len(TB[TB.Fly_s==fly])).as_hex()
    plt.plot(TB['X'][TB.Fly_s==fly], TB['Y'][TB.Fly_s==fly], c = sns.color_palette(Palette[Num%len(Palette)],  len(TB[TB.Fly_s==fly])).as_hex()[-1], alpha = .5, linewidth = WIDTH/4 )

TB["Color2"] = "NA"
if 0 in CLASS:
    TB["Color2"] = PALETTE["Fly"]
if 2 in CLASS:
    TB['Color2'][TB['Grooming']!=0] = PALETTE["Groom"]
    print(PALETTE["Groom"])
if 3 in CLASS:
    TB['Color2'][TB['Fs_x']!="0"] = PALETTE["Chase"]
if 4 in CLASS:
    TB['Color2'][TB['Sing']!=0] = PALETTE["Sing"]
if 5 in CLASS:
    TB['Color2'][TB['Hold']!=0] = PALETTE["Hold"]


ax.set_xlim(AXISX[0], AXISX[1])
ax.set_ylim(AXISY[0], AXISY[1])

if PHOTO != True:
    import cv2
    print(PHOTO)
    print("import photos from video")
    if PHOTO == None:
        Video_loc = os.popen("grep " + video_id + " Video_list").read().replace("\n", '').split("\t")[-1]
    else:
        Video_loc = PHOTO + video_id
    cap=cv2.VideoCapture(Video_loc)
    cap.set(1,END_fm)
    ret,frame=cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame)


for i in range(len(TB)):
    TMP = TB.iloc[i,:]
    if TMP.Class != "NA":
        Arrow_plt( (TMP["X"], TMP["Y"]), TMP.length * SCALE /2, TMP.B_angle + 180, WIDTH, TMP.Color, TMP.Color2)

plt.gca().invert_yaxis()
plt.show()
