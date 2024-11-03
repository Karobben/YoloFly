#!/usr/bin/env python3
'''
This script would output the result into Video_post as Interection_*.csv. All interaction events and corrosbonded response would be recorded
'''

import json, math, random, cv2, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg1458
from collections import namedtuple
from numpy.linalg import norm
import seaborn as sns
from itertools import chain, combinations
from scipy.spatial import distance


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i','-I','--input')     #Video
parser.add_argument('-fs','--frame_start')     #start frame
parser.add_argument('-fe','--frame_end')     #End Frame
parser.add_argument('-pp','--petri_pixel')     #
parser.add_argument('-pm','--petri_mm')     #

## aqure
args = parser.parse_args()
video_id = args.input
Frame_start = int(args.frame_start)
Frame_end = int(args.frame_end)
Len_plate_p = args.petri_pixel
Len_plate_m = args.petri_mm


_Scale = float(Len_plate_m)/float(Len_plate_p)


Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
MATCH_result = None

def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy

def Cdist(A, B):
    return distance.cdist([A],[B])[0][0]

def GetAngle(vector_1, vector_2):
    if vector_1[0] == 0 and vector_1[1] ==0 or vector_2[0] ==0 and vector_2[1] ==0:
        mydegrees = 0
    else:
        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = np.arccos(dot_product)
        mydegrees = math.degrees(angle)
        tmp = vector_2 - vector_1
        if tmp[0] * tmp[1] <= 0:
            mydegrees = mydegrees * (-1)
    return mydegrees

def Fly_relPosition(fly_t, fly_s):
    BeSp_h = np.array(Fly_dic[str(i)][fly_s]['head'][:2]) #p1
    BeSp_b = np.array(Fly_dic[str(i)][fly_s]['body'][:2]) #p2
    BeTg_h = np.array(Fly_dic[str(i)][fly_t]['head'][:2]) #p3.1
    BeTg_b = np.array(Fly_dic[str(i)][fly_t]['body'][:2]) #p3.2
    d_TH2SL = norm(np.cross(BeSp_b-BeSp_h, BeSp_h-BeTg_h))/norm(BeSp_b-BeSp_h)
    d_TB2SL = norm(np.cross(BeSp_b-BeSp_h, BeSp_h-BeTg_b))/norm(BeSp_b-BeSp_h)
    fly_s = BeSp_h- BeSp_b
    fly_t = BeTg_h - BeTg_b
    angle = GetAngle(fly_s,fly_t)
    return d_TH2SL, d_TB2SL , angle

def get_intersections(x0, y0, r0, x1, y1, r1):
    '''
    Get intersection of tow circles
    circle 1: (x0, y0), radius r0
    circle 2: (x1, y1), radius r1
    '''
    d=math.sqrt((x1-x0)**2 + (y1-y0)**2)
    # non intersecting
    if d > r0 + r1 :
        return None
    # One circle within other
    if d < abs(r0-r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a=(r0**2-r1**2+d**2)/(2*d)
        h=math.sqrt(r0**2-a**2)
        x2=x0+a*(x1-x0)/d
        y2=y0+a*(y1-y0)/d
        x3=x2+h*(y1-y0)/d
        y3=y2-h*(x1-x0)/d
        x4=x2-h*(y1-y0)/d
        y4=y2+h*(x1-x0)/d
        return (x3, y3, x4, y4)

def Re_pos(Fs, Ft,S_flyh, S_flyb, T_flyh, T_flyb, P1, P2, Force = False):
    S_flyh, S_flyb, T_flyh, T_flyb = np.array(S_flyh), np.array(S_flyb), np.array(T_flyh), np.array(T_flyb)
    h0 = Cdist(S_flyh, T_flyh)
    h1 = Cdist(P1, T_flyh)
    h2 = Cdist(P2, T_flyh)
    b0 = Cdist(S_flyh, T_flyb)
    b1 = Cdist(P1, T_flyb)
    b2 = Cdist(P2, T_flyb)
    # Distance from point to line
    hl = norm(np.cross(S_flyb-S_flyh, S_flyh-T_flyh))/norm(S_flyb-S_flyh)
    bl = norm(np.cross(S_flyb-S_flyh, S_flyh-T_flyb))/norm(S_flyb-S_flyh)
    # Thread of the scan area
    Th = (Cdist(S_flyh, P1)+Cdist(S_flyh,P2)+Cdist(P1,P2))/1.6
    # on the scan area or not:
    Scan_h = Cdist(T_flyh, P1) + Cdist(T_flyh, P2) + Cdist(T_flyh, S_flyh)
    Scan_b = Cdist(T_flyb, P1) + Cdist(T_flyb, P2) + Cdist(T_flyb, S_flyh)
    if Scan_h <= Th and Scan_b <= Th:
        #print("involved")
        Scan_pos = "Inner"
    elif Scan_h <= Th and Scan_b > Th:
        #print("Head from out")
        Scan_pos = "Gether"
        Head_to = "Head"
        Direction = "Against"
    elif Scan_h > Th and Scan_b <= Th:
        #print("Away from sight")
        Scan_pos = "Away"
        Head_to = "Butt"
        Direction = "Away"
    if Scan_h > Th and Scan_b > Th:
        #print("False Positive")
        Scan_pos = "Out of boundary"
        if Fs != Ft:
            if Force == False:
                #print("Out of boundary")
                return None
            if Force== True:
                Scan_pos = "Far"
                Head_to = "Far"
                Direction = "Far"
    ## Side of the target
    if h1/h2>1.2 and b1/b2>1.2:
        #print("Right")
        T_side = "Right"
    elif h1/h2<0.8 and b1/b2<0.8:
        #print("Left")
        T_side = "Left"
    else:
        #print("Middle")
        T_side = "Middle"
    ## head to the target or not
    ### body cross the middel line
    if Scan_pos == "Inner" or Fs ==Ft:
        #print("b0=", b0)
        if b0 == 0:
            Head_to ="Overlap"
            Direction = "Stick"
        else:
            if (h1-h2)*(b1-b2)<0:
                if h0/b0 >=1.05:
                    Head_to ="Butt"
                    Direction = "Away"
                elif h0/b0 <= 0.95:
                    Head_to ="Head"
                    Direction = "Against"
                else:
                    Head_to ="Body"
                    Direction = "Away"
            else:
                ## body on the left or right side
                if h0 > b0:
                    if hl/bl >= 1.1:
                        Head_to ="Butt"
                        Direction = "Away"
                    elif hl/bl <= 0.9:
                        Head_to ="Butt"
                        Direction = "Away"
                    else:
                        Head_to ="Butt"
                        Direction = "Away"
                elif h0 < b0:
                    if hl/bl >= 1.1:
                        Head_to ="Body"
                        Direction = "Away"
                    elif hl/bl <= 0.9:
                        Head_to ="Head"
                        Direction = "Against"
                    else:
                        Head_to ="Head"
                        Direction = "Against"
                else:
                    if hl/bl >= 1.1:
                        Head_to ="Butt"
                        Direction = "Away"
                    elif hl/bl <= 0.9:
                        Head_to ="Head"
                        Direction = "Against"
                    else:
                        Head_to ="Body"
                        Direction = "Away"
    Angle = GetAngle(T_flyh-T_flyb, S_flyh - S_flyb)
    #
    return Fs, Ft, Scan_pos, T_side, Head_to, Direction, Angle

## Calculate the ??
def pos_calcu(Fs, Ft, _FLY_L_pix, i, Force=False):
    fly_th = np.array([Fly_dic[str(i)][Ft]['head'][0]*_pixel_X,
                       Fly_dic[str(i)][Ft]['head'][1]*_pixel_Y])
    fly_tb = np.array([Fly_dic[str(i)][Ft]['body'][0]*_pixel_X,
                       Fly_dic[str(i)][Ft]['body'][1]*_pixel_Y])
    fly_t = fly_th - fly_tb
    fly_sh = np.array([Fly_dic[str(i)][Fs]['head'][0]*_pixel_X,
                       Fly_dic[str(i)][Fs]['head'][1]*_pixel_Y])
    fly_sb = np.array([Fly_dic[str(i)][Fs]['body'][0]*_pixel_X,
                       Fly_dic[str(i)][Fs]['body'][1]*_pixel_Y])
    fly_s = fly_sh - fly_sb
    ## Calculate the two detect-points of the head
    len_flys = Cdist([0,0],fly_s)
    if len_flys ==0:
        if Force==False:
            return None
        if Force == True:
            len_flys += _FLY_L_pix
    x0 = -(_FLY_L_pix/len_flys-1)* fly_s[0]
    y0 = -(_FLY_L_pix/len_flys-1)* fly_s[1]
    r0 = ( 2.5 * 2.5 /_Scale)  # Radium from body, 7.5 mm
    x1, y1 = fly_s
    r1 = (1.85* 2.5 /_Scale)    # Radium from head, 7.5 mm
    # intersecting with (x1, y1) but not with (x0, y0)
    circle1 = plt.Circle((x0, y0), r0, color='b', fill=False)
    circle2 = plt.Circle((x1, y1), r1, color='b', fill=False)
    intersections = get_intersections(x0, y0, r0, x1, y1, r1)
    if intersections is not None:
        i_x3, i_y3, i_x4, i_y4 = intersections
    else:
        return None, None, None, None, None, None, None
    #
    S_H = np.array(Fly_dic[str(i)][Fs]['head'][:2])
    S_B = np.array(Fly_dic[str(i)][Fs]['body'][:2])
    i_X1 =i_x3+S_H[0]*_pixel_X
    i_X2 =i_x4+S_H[0]*_pixel_X
    i_Y1 =i_y3+S_H[1]*_pixel_Y
    i_Y2 =i_y4+S_H[1]*_pixel_Y
    P1 = [i_X1,i_Y1]
    P2 = [i_X2,i_Y2]
    #
    S_flyh = [Fly_dic[str(i)][Fs]['head'][0]*_pixel_X,
              Fly_dic[str(i)][Fs]['head'][1]*_pixel_Y]
    S_flyb = [Fly_dic[str(i)][Fs]['body'][0]*_pixel_X,
              Fly_dic[str(i)][Fs]['body'][1]*_pixel_Y]
    T_flyh = [Fly_dic[str(i)][Ft]['head'][0]*_pixel_X,
              Fly_dic[str(i)][Ft]['head'][1]*_pixel_Y]
    T_flyb = [Fly_dic[str(i)][Ft]['body'][0]*_pixel_X,
              Fly_dic[str(i)][Ft]['body'][1]*_pixel_Y]
    Rep_ = Re_pos(Fs, Ft, S_flyh, S_flyb, T_flyh, T_flyb, P1, P2, Force)
    return Rep_

def pos_move(Fs, Ft, _FLY_L_pix,i, P_f = 5):
    fly_th = np.array([Fly_dic[str(i+P_f)][Ft]['head'][0]*_pixel_X,
                       Fly_dic[str(i+P_f)][Ft]['head'][1]*_pixel_Y])
    fly_tb = np.array([Fly_dic[str(i+P_f)][Ft]['body'][0]*_pixel_X,
                       Fly_dic[str(i+P_f)][Ft]['body'][1]*_pixel_Y])
    fly_t = fly_th - fly_tb
    #
    fly_sh = np.array([Fly_dic[str(i)][Fs]['head'][0]*_pixel_X,
                       Fly_dic[str(i)][Fs]['head'][1]*_pixel_Y])
    fly_sb = np.array([Fly_dic[str(i)][Fs]['body'][0]*_pixel_X,
                       Fly_dic[str(i)][Fs]['body'][1]*_pixel_Y])
    fly_s = fly_sh - fly_sb
    ## Calculate the two detect-points of the head
    len_flys = Cdist([0,0],fly_s)
    if len_flys == 0:
        return [None, None, None, None, None]
    x0 = -(_FLY_L_pix/len_flys-1)* fly_s[0]
    y0 = -(_FLY_L_pix/len_flys-1)* fly_s[1]
    r0 = ( 2.5 * 2.5 /_Scale)  # Radium from body, 7.5 mm
    x1, y1 = fly_s
    r1 = (1.8* 2.5 /_Scale)    # Radium from head, 7.5 mm
    # intersecting with (x1, y1) but not with (x0, y0)
    circle1 = plt.Circle((x0, y0), r0, color='b', fill=False)
    circle2 = plt.Circle((x1, y1), r1, color='b', fill=False)
    intersections = get_intersections(x0, y0, r0, x1, y1, r1)
    if intersections is not None:
        i_x3, i_y3, i_x4, i_y4 = intersections
    #
    S_H = np.array(Fly_dic[str(i)][Fs]['head'][:2])
    S_B = np.array(Fly_dic[str(i)][Fs]['body'][:2])
    i_X1 =i_x3+S_H[0]*_pixel_X
    i_X2 =i_x4+S_H[0]*_pixel_X
    i_Y1 =i_y3+S_H[1]*_pixel_Y
    i_Y2 =i_y4+S_H[1]*_pixel_Y
    P1 = [i_X1,i_Y1]
    P2 = [i_X2,i_Y2]
    #
    S_flyh = [Fly_dic[str(i)][Fs]['head'][0]*_pixel_X,
              Fly_dic[str(i)][Fs]['head'][1]*_pixel_Y]
    S_flyb = [Fly_dic[str(i)][Fs]['body'][0]*_pixel_X,
              Fly_dic[str(i)][Fs]['body'][1]*_pixel_Y]
    T_flyh = [Fly_dic[str(i+P_f)][Ft]['head'][0]*_pixel_X,
              Fly_dic[str(i+P_f)][Ft]['head'][1]*_pixel_Y]
    T_flyb = [Fly_dic[str(i+P_f)][Ft]['body'][0]*_pixel_X,
              Fly_dic[str(i+P_f)][Ft]['body'][1]*_pixel_Y]
    Rep_ = Re_pos(Fs, Ft,S_flyh, S_flyb, T_flyh, T_flyb, P1, P2)
    ## Moving distance
    Dist = Cdist(S_flyb, T_flyb)
    # forward or backward
    hp_extant = np.array(S_flyh)*2 - np.array(S_flyb)
    he1 = Cdist(T_flyh, hp_extant)
    he2 = Cdist(T_flyh, S_flyb)
    if he2 == 0:
        result_fd = "Stood"
    else:
        he_r = he1/he2
        if Dist*_Scale <= 1:
            if he_r >= 1.1:
                result_fd = "Back"
            elif he_r <= 0.9:
                result_fd = "Forward"
            else:
                result_fd = "Stood"
        else:
            result_fd = "Leap"
    #
    ## head direction:
    h1 = Cdist(P1, T_flyh)
    h2 = Cdist(P2, T_flyh)
    hh_r = h1/h2
    if hh_r > 1:
        result_hd = "Left"
    elif hh_r <1:
        result_hd = "Right"
    else:
        result_hd = "Middle"
    TMP = [i for i in Rep_]
    #print(TMP)
    return [result_fd, Dist*_Scale, result_hd] + [TMP[3]] + [TMP[6]]

def Chain_shrink(Chain_list, Pair_list):
    if len(Pair_list) ==1:
        return Pair_list
    while len(Pair_list)  > 0:
        if len(Chain_list) == 0:
            # add the first group
            Chain_list += [Pair_list[0]]
            Pair_list.remove(Pair_list[0])
        tmp = list(chain.from_iterable(Chain_list))
        for Chain_ind in range(len(Chain_list)):
            Chain = Chain_list[Chain_ind]
            for fly in Chain:
                tmp = [i for i in Pair_list if fly in i]
                [Pair_list.remove(i) for i in tmp]
                for i in tmp:
                    Chain_list[Chain_ind] += i
                Chain_list[Chain_ind] = list(set(Chain_list[Chain_ind]))
        # unique
        Chain_list = [[ii for ii in set(i)] for i in Chain_list]
        try:
            Chain_list += [Pair_list[0]]
            Pair_list.remove(Pair_list[0])
        except:
            break
    return Chain_list

def Chain_number(Chase_result):
    Ar = []
    for Frame in Chase_result[0].unique():
        print(Frame)
        TMP = Chase_result[Chase_result[0]==Frame]
        Pair_list = TMP[["Fs","Ft"]].to_numpy().tolist()
        Num_fly = len(set(TMP.Fs.tolist()+ TMP.Ft.tolist()))
        # down stream walk
        Num = 0
        Result_N = 0
        while Result_N != Num_fly:
            Chain_list = []
            Pair_list = Chain_shrink(Chain_list, Pair_list)
            Result_N = len(list(chain.from_iterable(Pair_list)))
        Chain_list = Pair_list
        List = [len(i) for i in Chain_list]
        List = list(chain.from_iterable([[i]*i for i in List]))
        Ar +=[[Frame,i,ii] for i, ii in zip(list(chain.from_iterable(Chain_list)),List)]
    Chain_TB = pd.DataFrame(Ar, columns = [0, "Fs", "Chain_number"])
    Chain_TB[0] = Chain_TB[0].astype(str)
    Chase_result[0] = Chase_result[0].astype(str)
    Chase_result = pd.merge(Chase_result, Chain_TB,  left_on= [0, "Fs"], right_on= [0, "Fs"], how="left")
    Chase_result[0] = Chase_result[0].astype(int)
    return Chase_result

def pos_C(Fs, Ft, _FLY_L_pix, Frame):
    TMP = pos_calcu(Fs, Ft, _FLY_L_pix, Frame)
    if TMP !=None:
        return [[Frame] + [i for i  in TMP]]

def Chase_get(Fly_dic, ch_frame):
    Result  = []
    CHase_list = (ch_frame.Frame.astype(str) + "_" + ch_frame.Fly_s).to_list()
    _FLY_L_pix = 2.5 / _Scale
    #AA = [[[ Result.append(pos_C(Fs,Ft,_FLY_L_pix,Frame)) for Ft in Fly_dic[Frame].keys() if Ft !=Fs and str(Frame) + "_" + Fs in CHase_list] for Fs in Fly_dic[Frame].keys()] for  Frame in Fly_dic.keys()]


    AA = [[[ Result.append(pos_C(Fs,Ft,_FLY_L_pix,Frame)) for Ft in Fly_dic[Frame].keys() if Ft !=Fs and str(Frame) + "_" + Fs in CHase_list] for Fs in Fly_dic[Frame].keys()] for  Frame in Fly_dic.keys()]

    TB = pd.DataFrame([i[0] for i in Result if i !=None], columns=[0, 'Fs', 'Ft', 'Scan_pos', 'T_side', 'Head_to', 'Direction', 'Angle'])
    return TB


Raw_file = "csv"
Raw_list = [i for i in os.listdir(Raw_file)]
# The first frame
# Arguments for moving states
## Combine the behaviors
_pixel_X = 1920
_pixel_Y = 1080
GFF_thre = 10
Chase_gap = 15

######################################
### Chasing
######################################

def Ch_table_descibe(video_id, cls=3):

    _FLY_L_pix = 2.5 / _Scale

    CSV_result = [i for i in Raw_list if video_id in i and ".csv" in i][0]
    Json_result = [i for i in Raw_list if video_id in i and ".json" in i][0]
    Json_list = open(Raw_file +"/"+Json_result, "r").read().split(";")[:-1]
    [Fly_dic.update(json.loads(i)) for i in Json_list if int(list(json.loads(i).keys())[0])
        >= Frame_start and int(list(json.loads(i).keys())[0])<= Frame_end]

    CSV_matrix = pd.read_csv(Raw_file +"/"+CSV_result, sep=" ", header=None)
    CSV_matrix.columns= ["Frame", "class","x", "y","width", "hight"]
    CSV_matrix = CSV_matrix[CSV_matrix['Frame'].isin(range(Frame_start, Frame_end))]
    CSV_matrix[["x","width"]]= CSV_matrix[["x","width"]]*_pixel_X
    CSV_matrix[["y","hight"]]= CSV_matrix[["y","hight"]]*_pixel_Y



    ####################################################
    # Starting to summary the Chasing Events
    ####################################################

    Fly_bhv = pd.read_csv("Video_post/"+  video_id + "_"+str(Frame_start)+ "_" + str(Frame_end) +".csv", index_col=0)

    ch_frame  = pd.concat([Fly_bhv[['Frame', 'Fly_s', 'Grooming']], Fly_bhv[["Sing", "Chasing", "Hold"]].sum(1)], axis=1)
    ch_frame = ch_frame[ch_frame[0]!=0]
    ch_frame = ch_frame[ch_frame.Grooming==0]


    Chase_result = Chase_get(Fly_dic, ch_frame)

    # Chasing events correction
    Chase_result[0] = Chase_result[0].astype(int)
    Chase_result = Chase_result.sort_values(0)
    Chase_result = Chase_result.drop_duplicates()
    Chase_result.index= range(len(Chase_result.index))
    Chase_result = pd.DataFrame(Chase_result)

    ## Duration of the frame
    Pframe = 3

    ##################################################
    ## Calculate the respons of the target and sponser
    ##################################################


    S_col = ["Sfly_act", "Sfly_dis", "Sfly_head", "Sfly_body", "Sfly_ang"]
    T_col = ["Tfly_act", "Tfly_dis", "Tfly_head", "Tfly_body", "Tfly_ang"]
    S_respons_TB = pd.DataFrame(columns=S_col)
    T_respons_TB = pd.DataFrame(columns=T_col)

    #pd.DataFrame([Result_T], columns=["Tfly_act", "Tfly_dis", "Tfly_head", "Tfly_body", "Tfly_ang"])
    for Num in Chase_result.index:
        FLy1 = Chase_result.iloc[Num,:]["Fs"]
        FLy2 = Chase_result.iloc[Num,:]["Ft"]
        frame = Chase_result.iloc[Num,:][0]
        if frame <= CSV_matrix['Frame'].tolist()[-1]- Pframe:
            Result_S = pos_move(FLy1, FLy1, _FLY_L_pix,frame, Pframe)
            Result_T = pos_move(FLy2, FLy2, _FLY_L_pix,frame, Pframe)
            S_tmp = pd.DataFrame([Result_S], columns = S_col)
            T_tmp = pd.DataFrame([Result_T], columns = T_col)
            S_respons_TB = pd.concat([S_respons_TB, S_tmp])
            T_respons_TB = pd.concat([T_respons_TB, T_tmp])

    S_respons_TB["Sfly_ang"] = S_respons_TB["Sfly_ang"].fillna(0)
    T_respons_TB["Tfly_ang"] = T_respons_TB["Tfly_ang"].fillna(0)
    S_respons_TB.index= range(len(S_respons_TB.index))
    T_respons_TB.index= range(len(T_respons_TB.index))



    ##################################################
    ## Detect the Chasing Evetns
    ##################################################

    #Gap fill Frame: 15
    #Chase_result = Chase_result.dropna()
    Chase_result["Chase_ID"] = -1
    Chase_id = 1
    for Fs in set(Chase_result["Fs"]):
        TMP =  Chase_result[Chase_result["Fs"] ==Fs]
        Chase_result["Chase_ID"].iloc[TMP.index[0]] = Chase_id
        for i in range(1,len(TMP)):
            if TMP.iloc[i-1,2] == TMP.iloc[i,2] and abs(TMP.iloc[i-1,0]- TMP.iloc[i,0]) <= Chase_gap:
                Chase_result["Chase_ID"].iloc[TMP.index[i]] = Chase_id
            else:
                Chase_id += 1
                Chase_result["Chase_ID"].iloc[TMP.index[i]] = Chase_id
        Chase_id += 1
    Chase_result = pd.concat([Chase_result.iloc[:len(S_respons_TB), :], S_respons_TB, T_respons_TB],axis=1)


    ##################################################
    ## Fill the Chasing Gaps
    ##################################################
    #Chase_result = Chase_result.dropna()

    for Chase_id in Chase_result.Chase_ID.unique():
        TMP = Chase_result[Chase_result.Chase_ID == Chase_id]
        flst = TMP[0]
        frame_lot = [i for i in range(flst.min(), flst.max()) if i not in TMP[0].to_list()]
        # Start the loop for each lost frame
        for losF in frame_lot:
            FLy1, FLy2 = TMP.iloc[0, 1:3].to_list()
            if losF <= CSV_matrix['Frame'].tolist()[-1]- Pframe:
                Result_S = pos_move(FLy1, FLy1, _FLY_L_pix,losF, Pframe)
                Result_T = pos_move(FLy2, FLy2, _FLY_L_pix,losF, Pframe)

            Rel_pos = pos_calcu(FLy1, FLy2, _FLY_L_pix, losF, Force=True)
            Fill_tmp =[[losF]+[i for i in Rel_pos] +[Chase_id]+Result_S + Result_T]
            Fill_tb = pd.DataFrame(Fill_tmp, columns= Chase_result.columns)
            Chase_result = pd.concat([Chase_result, Fill_tb])

    ##################################################
    ## Add Chaing Number
    ##################################################
    Chase_result = Chase_result[Chase_result.Fs.isnull()==False]
    Chase_result = Chain_number(Chase_result)

    #####################################
    # The result of the Chasing
    #####################################

    Chase_result['Fight_result'] = "Remove"
    for ID in set(Chase_result.Chase_ID):
        if len(Chase_result[Chase_result.Chase_ID==ID])>GFF_thre:
            TMP = Chase_result[Chase_result.Chase_ID==ID].tail(10)
            S_act = list(TMP.Sfly_act)
            T_act = list(TMP.Tfly_act)
            if "Leap" in list(TMP.Tfly_act):
                Chase_result['Fight_result'][Chase_result.Chase_ID==ID] = "win"
            elif "Leap" in list(TMP.Sfly_act):
                Chase_result['Fight_result'][Chase_result.Chase_ID==ID] = "lose"
            elif S_act.count("Forward") > T_act.count("Forward"):
                Chase_result['Fight_result'][Chase_result.Chase_ID==ID] = "win"
            elif S_act.count("Forward") < T_act.count("Forward"):
                Chase_result['Fight_result'][Chase_result.Chase_ID==ID] = "lose"
            else:
                Chase_result['Fight_result'][Chase_result.Chase_ID==ID] = "give up"

    Chase_result = Chase_result[Chase_result.Fight_result != "Remove"]
    Chase_result = Chase_result.sort_values(0)

    ######################################
    ## Add the Behavior Class           ##
    ######################################
    Chas_TB = Chase_result

    Chas_TB_index =  pd.DataFrame([i for i in Chas_TB[0].apply(str) + "_"+Chas_TB.Fs],  index = range(len(Chas_TB)), columns=["Fly"])
    Chas_TB_index['Chas_id'] = range(len(Chas_TB_index))
    Fly_bhv_index = pd.DataFrame([i for i in Fly_bhv.Frame.apply(str) +"_" + Fly_bhv.Fly_s], index= range(len(Fly_bhv)), columns=["Fly"])
    Fly_bhv_index['Flys_id'] = range(len(Fly_bhv_index))


    ID_TB = pd.merge(Chas_TB_index, Fly_bhv_index)
    ID_TB  = ID_TB.sort_values("Chas_id")


    Chas_TB = Chas_TB.iloc[ID_TB.Chas_id,:]
    Fly_bhv.iloc[ID_TB.Flys_id,:]
    Chas_TB["Class"] = "Touch"
    # Grooming
    G_TB =  Fly_bhv.iloc[ID_TB.Flys_id,:]["Grooming"]
    Chas_TB["Class"].iloc[[i for i in range(len(G_TB)) if G_TB.iloc[i] !=0]] = "Grooming"
    # Sing
    S_TB =  Fly_bhv.iloc[ID_TB.Flys_id,:]["Sing"]
    Chas_TB["Class"].iloc[[i for i in range(len(S_TB)) if S_TB.iloc[i] !=0]] = "Sing"
    # Mount
    M_TB =  Fly_bhv.iloc[ID_TB.Flys_id,:]["Hold"]
    Chas_TB["Class"].iloc[[i for i in range(len(M_TB)) if M_TB.iloc[i] !=0]] = "Hold"

    return Chas_TB


Fly_dic = {}

Chas_TB = Ch_table_descibe(video_id)
# remove accend duplications
Chas_TB = Chas_TB[Chas_TB[[0,"Fs"]].duplicated()==False]

Chas_TB.to_csv("Video_post/Interection_" + video_id + "_"+str(Frame_start)+"_" +str(Frame_end) +".csv")
