#!/usr/bin/env python3

import os, math
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance
from collections import namedtuple


if not os.path.exists("Video_post"):
  # Create a new directory because it does not exist
  os.makedirs("Video_post")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i','-I','--input')     #Video
parser.add_argument('-fs','--frame_start')     #Start
parser.add_argument('-fe','--frame_end')     #End
parser.add_argument('-pp','--petri_pixel')     #
parser.add_argument('-pm','--petri_mm')     #

## Aquire
args = parser.parse_args()
video_id = args.input
Frame_start = int(args.frame_start)
Frame_end = int(args.frame_end)
Len_plate_p = args.petri_pixel
Len_plate_m = args.petri_mm


_Scale = float(Len_plate_m)/float(Len_plate_p)


def Points4_angle(P1, P2, P3, P4):
    '''
    relative angle of 4 given points
    '''
    P1 = np.array(P1)
    P2 = np.array(P2)
    P3 = np.array(P3)
    P4 = np.array(P4)
    vector_1 = P2 - P1
    vector_2 = P4 - P3
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    mydegrees = math.degrees(angle)
    return mydegrees


Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
MATCH_result = None
def R_in_area(Rect1, Rect2):  # returns None if rectangles don't intersect
    a = Rectangle(Rect1[0]-Rect1[2]/2, Rect1[1]-Rect1[3]/2,
                  Rect1[0]+Rect1[2]/2, Rect1[1]+Rect1[3]/2)
    b = Rectangle(Rect2[0]-Rect2[2]/2, Rect2[1]-Rect2[3]/2,
                  Rect2[0]+Rect2[2]/2, Rect2[1]+Rect2[3]/2)
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return max([dx*dy/(Rect1[2]*Rect1[3]), dx*dy/(Rect2[2]*Rect2[3])])
    else:
        return 0

# Rectangle fit
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
MATCH_result = None
def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy


# source: https://www.kaggle.com/nroman/detecting-outliers-with-chauvenet-s-criterion
from scipy.special import erfc
def chauvenet(array):
    mean = array.mean()           # Mean of incoming array
    stdv = array.std()            # Standard deviation
    N = len(array)                # Lenght of incoming array
    criterion = 1.0/(2*N)         # Chauvenet's criterion
    d = abs(array-mean)/stdv      # Distance of a value to mean in stdv's
    prob = erfc(d)                # Area normal dist.
    return prob < criterion       # Use boolean array outside this function

def Find_nearst(frame):
    #Result = [np.array([math.dist(np.array(Fly_dic[frame][fly_id]["body"][:2]) * _pixel_frame ,
    #   np.array(Fly_dic[frame][fly]["body"][:2]) *_pixel_frame)
    #     for fly in Fly_dic[str(Frame_start)].keys() if fly != fly_id])]
    Result = [np.array([distance.cdist([Fly_dic[frame][fly_id]["body"][:2] * _pixel_frame], [Fly_dic[frame][fly]["body"][:2] *_pixel_frame])[0][0]    for fly in Fly_dic[str(Frame_start)].keys() if fly != fly_id])]


    return [Result[0].min()*_Scale, len(Result[0][Result[0]*_Scale<=5])]
#for fly_id in Fly_dic[str(Frame_start)].keys():

def video2tb(fly_id, Video_tb):
    # Neast Flies
    '''
    Stort the x and y loc in two tables (pixel)
    '''
    ### Nearst flies and numbers
    Nst_fly = [[fly for fly in Fly_dic[str(Frame_start)].keys() if fly != fly_id]
      [np.array([distance.cdist([Fly_dic[frame][fly_id]["body"][:2] * _pixel_frame], [Fly_dic[frame][fly]["body"][:2] *_pixel_frame])[0][0]
         for fly in Fly_dic[str(Frame_start)].keys() if fly != fly_id]).argmin()]
         for frame in Fly_dic.keys()]
    #
    Nst_dist = np.array([Find_nearst(frame)  for frame in Fly_dic.keys()])
    #
    Nearest_dist = pd.DataFrame([[int(frame) for frame in Fly_dic.keys()], Nst_fly, Nst_dist[:,0], Nst_dist[:,1]]).T
    Nearest_dist.columns = ["Frame", "Nst_fly", "Nst_dist", "Nst_num"]
    Nearest_dist['Frame'].astype(int)

    #Body length and angle

    def point2agl(P1, P2):
        myradians = math.atan2(P1[1]-P2[1], P1[0]-P2[0])
        mydegrees = math.degrees(myradians)
        return mydegrees

    B_loc_list = [[
      [Fly_dic[str(frame)][fly_id]['head'][0] * _pixel_X,
       Fly_dic[str(frame)][fly_id]['head'][1] * _pixel_Y,],
      [Fly_dic[str(frame)][fly_id]['body'][0] * _pixel_X,
       Fly_dic[str(frame)][fly_id]['body'][1] * _pixel_Y], frame,
      Fly_dic[str(frame)][fly_id]['body'][2] * _pixel_X,
      Fly_dic[str(frame)][fly_id]['body'][3] * _pixel_Y
       ]
     for frame in range(Frame_start, Frame_end)]
    Body_len_agl = [[i[2], distance.euclidean(i[0], i [1])*2* _Scale,
                    min(i[3], i[4])*2* _Scale, max(i[3], i[4])*2* _Scale,
                    min(i[3], i[4])/max(i[3], i[4]),
                     point2agl(i[1], i [0]),i[1][0], i[1][1]] for i in B_loc_list]
    Body_len_agl_TB = pd.DataFrame(Body_len_agl, columns=['Frame', "length",
    "B_w","B_l","B_r", "B_angle", "X", "Y"])
    #Body_len_agl_TB.to_csv("test_body.csv")

    # Moving angle
    def getAngle(a, b, c):
        ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
        if ang >= 180 :
            ang -= 360
        if ang <= -180:
            ang += 360
        return ang

    Moving_agl = np.array([[B_loc_list[i][2], getAngle(B_loc_list[i+1][1], B_loc_list[i][1], B_loc_list[i][0])] for i in range(len(B_loc_list)-1)])
    Moving_agl_TB = pd.DataFrame(Moving_agl, columns=['Frame', "M_angle"])

    # Moving define:
    Moving_agl_TB['Move'] = "Crab"
    TH1 = 75
    TH2 = 105
    Moving_agl_TB['Move'][abs(Moving_agl_TB['M_angle'])<=TH1] = "Ahead"
    Moving_agl_TB['Move'][abs(Moving_agl_TB['M_angle'])> TH2] = "Back"
    Moving_agl_TB.to_csv("test_Mo_angle.csv")

    # Motions (mm/s)
    '''
    inter_frame = 5
    i = 8900
    XY = (np.array(Fly_dic[str(i-inter_frame)][fly_id]['head'][:2]) - \
        np.array(Fly_dic[str(i+inter_frame)][fly_id]['head'][:2]))
    XY = (XY* np.array([ _pixel_X ,_pixel_Y]) *_Scale)**2
    math.sqrt(XY.sum())/ (30/inter_frame/2)
    '''
    inter_frame = 5
    Speed = [[float(i), math.sqrt((((np.array(Fly_dic[str(i-inter_frame)][fly_id]['head'][:2]) - \
        np.array(Fly_dic[str(i+inter_frame)][fly_id]['head'][:2])) * \
        np.array([ _pixel_X ,_pixel_Y]) *_Scale)**2).sum())/ (30/inter_frame/2)] \
             for i in range(Frame_start+inter_frame, Frame_end-inter_frame+1)]
    Speed = pd.DataFrame(Speed, columns=["Frame", 'mm/s'])
     # save and plot with cv

    # Classify the Walk
    Thread_1 = 0.25
    Thread_2 = 1
    Thread_3 = 2.5

    Speed["Motion"] = "Rest"
    TMP = Speed[Speed["mm/s"]>Thread_1]
    Walk_list = TMP["mm/s"][TMP["mm/s"]<= Thread_2]
    TMP = Speed[Speed["mm/s"]>Thread_2]
    Run_list = TMP["mm/s"][TMP["mm/s"]<= Thread_3]

    Speed["Motion"] [Speed["mm/s"].isin(Walk_list)] = "Walk"
    Speed["Motion"] [Speed["mm/s"].isin(Run_list)] = "Run"
    Speed["Motion"] [Speed["mm/s"]>Thread_3] = "Charge"

    # Previous Motions
    Speed["Pre_motion"]=None
    Speed["Pre_motion"][2:]=Speed["Motion"][:-2]
    #inheriate motions 2 frame ago

    #Speed.to_csv("test_speed.csv")

    # Fit the behaviors

    # Gromming box by nearst points

    # fly_box
    '''
    Gro_tmp = [i for i in CSV_matrix[["Frame","x", "y"]][CSV_matrix["class"]==2].to_numpy()][0]#* np.array([1,2])
    print(Gro_tmp)
    # all flys in this frame:
    fly_all = np.array([np.array(Fly_dic[str(int(Gro_tmp[0]))][ii]["body"][:2]) \
                              for  ii in  Fly_dic[str(int(Gro_tmp[0]))].keys()])
    Min_index = ((np.array(Gro_tmp[1:]) - fly_all)**2).sum(axis=1).argmin()
    list(Fly_dic[str(int(Gro_tmp[0]))].keys())[Min_index]
    '''
    Grooming_list = [list(Fly_dic[str(int(Gro_tmp[0]))].keys())[
        ((np.array(Gro_tmp[1:]) - \
              np.array([np.array(Fly_dic[str(int(Gro_tmp[0]))][ii]["body"][:2]) \
                        for  ii in  Fly_dic[str(int(Gro_tmp[0]))].keys()])
         )**2).sum(axis=1).argmin()] for Gro_tmp in CSV_matrix[["Frame","x", "y"]][CSV_matrix["class"]==2].to_numpy()]
    Grooming_TB = CSV_matrix[["Frame"]][CSV_matrix["class"]==2]
    Grooming_TB["fly"] = Grooming_list
    #Grooming_TB.to_csv("test_grom.csv")

    Singing_list = [list(Fly_dic[str(int(Sin_tmp[0]))].keys())[
        ((np.array(Sin_tmp[1:]) - \
              np.array([np.array(Fly_dic[str(int(Sin_tmp[0]))][ii]["body"][:2]) \
                        for  ii in  Fly_dic[str(int(Sin_tmp[0]))].keys()])
         )**2).sum(axis=1).argmin()] for Sin_tmp in CSV_matrix[["Frame","x", "y"]][CSV_matrix["class"]==4].to_numpy()]
    Singing_TB = CSV_matrix[["Frame"]][CSV_matrix["class"]==4]
    Singing_TB["fly"] = Singing_list
    #Singing_TB.to_csv("test_sing.csv")

    # Chasing and Hold
    # Chasing
    Chasing_TB = CSV_matrix[CSV_matrix['class']==3]
    # intersect with target

    Chasing_list = np.array([R_in_area(Chasing_TB.iloc[i,2:].to_numpy(),
                     np.array(Fly_dic[str(Chasing_TB.Frame.tolist()[i])][fly_id]['body']))
       for i in range(len(Chasing_TB))])
    Chasing_flyid = pd.DataFrame(Chasing_TB.Frame[Chasing_list >= .85].unique(), columns=["Frame"])
    Chasing_flyid["Chasing"]=1
    # Hold
    Hold_TB = CSV_matrix[CSV_matrix['class']==5]
    # intersect with target

    Hold_list = np.array([R_in_area(Hold_TB.iloc[i,2:].to_numpy(),
                     np.array(Fly_dic[str(Hold_TB.Frame.tolist()[i])][fly_id]['body']))
       for i in range(len(Hold_TB))])
    Hold_flyid = pd.DataFrame(Hold_TB.Frame[Hold_list >= .85].unique(), columns=["Frame"])
    Hold_flyid["Hold"]=1

    # Combine the results
    # Result
    Si_flyid = Singing_TB[Singing_TB["fly"]==fly_id]
    Si_flyid.columns= ['Frame', "Sing"]
    Si_flyid['Sing']= 1 #"Sing"
    Gr_flyid = Grooming_TB[Grooming_TB["fly"]==fly_id]
    Gr_flyid.columns= ['Frame', "Grooming"]
    Gr_flyid['Grooming']= 1 #"Grooming"

    TB_List = [Nearest_dist, Body_len_agl_TB, Moving_agl_TB, Speed, Si_flyid, Gr_flyid, Chasing_flyid, Hold_flyid]
    Fly_id_TB= Nearest_dist
    for TB in TB_List[1:]:
        Fly_id_TB = pd.merge(Fly_id_TB, TB, on="Frame", how='left')
    # add the sponser fly
    Fly_id_TB["Fly_s"]=fly_id
    # remove inter frame for speed calculating
    Fly_id_TB = Fly_id_TB[inter_frame:-inter_frame]

    # Move list
    M_list = ["Ahead", "Crab", "Back"]
    for i in range(len(M_list)):
        Fly_id_TB["Move"][Fly_id_TB["Move"]==M_list[i]]= i+1

    # Motion list
    M_list = ['Rest', 'Walk', 'Run', 'Charge']
    for i in range(len(M_list)):
        Fly_id_TB["Motion"][Fly_id_TB["Motion"]==M_list[i]]= i+1

    # fill the Na with 0
    Fly_id_TB = Fly_id_TB.fillna(0)
    Fly_id_TB = Fly_id_TB[2:]
    # concat previous result
    Video_tb = pd.concat([Video_tb,Fly_id_TB])
    return Video_tb


#########################################################################
### Start the job
#########################################################################

Raw_file = "csv"
_pixel_X = 1920
_pixel_Y = 1080
_pixel_frame = np.array([1920, 1080])
Raw_list = [i for i in os.listdir(Raw_file)]
print(Raw_list, video_id)
CSV_result = [i for i in Raw_list if video_id in i and ".csv" in i][0]
Json_result = [i for i in Raw_list if video_id in i and ".json" in i][0]
# mm per pixel
# read json files and row csv file
Json_list = open(Raw_file +"/"+Json_result, "r").read().split(";")[:-1]

Fly_dic = {}
[Fly_dic.update(json.loads(i)) for i in Json_list if int(list(json.loads(i).keys())[0])
    >= Frame_start and int(list(json.loads(i).keys())[0])<= Frame_end]
if len(Fly_dic)==0:
    Fly_dic = json.loads(open(Raw_file +"/"+Json_result, "r").read())
CSV_matrix = pd.read_csv(Raw_file +"/"+CSV_result, sep=" ", header=None)
CSV_matrix.columns= ["Frame", "class","x", "y","width", "hight"]
CSV_matrix = CSV_matrix[CSV_matrix['Frame'].isin(range(Frame_start, Frame_end))]
# Prepare a vacum data frame
Video_tb = pd.DataFrame()

for fly_id in Fly_dic[str(Frame_start)].keys():
    Video_tb = video2tb(fly_id, Video_tb)
Video_tb["Video"] = video_id
Video_tb.to_csv("Video_post/" + video_id + "_"+str(Frame_start)+ "_" + str(Frame_end) +".csv")
#All_videos = pd.concat([All_videos,Video_tb])
