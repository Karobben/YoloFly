#!/usr/bin/env python3
'''
Cummary data based on the each chasing events
'''
import  os, sys
# Change the priority of the library
sys.path.sort()

import argparse
import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('-i','-I','--input')     #Video
parser.add_argument('-fs','--frame_start')     #start frame
parser.add_argument('-fe','--frame_end')     #End Frame


args = parser.parse_args()
video_id = args.input
Frame_start = int(args.frame_start)
Frame_end = int(args.frame_end)

'''
Arguments
'''

Nerst_dist_TH = 5   # Threads for fly has companies or not
Moving_TH = 0.25    # Threads for Resting or moving
Run_TH = 1          # Threads for Runing or walk
Leap_TH = 2         # Threads for Leapying or fying
Marge_angle = 90    # Angles for push
Stype_th  = 3       # Sing type threshold. The mean frame of sing for each time.
Window = 30         # The number of frame for each window in the sliding-window
Ran_exp  = 30       # Threshold for short time random-explore. When the number of the frame less than this threshold, the event would be termed as random-exploring.
Raw_dir = "Video_post/"


## read all tables to syncronize the colnames

Singl_list = [video_id + "_"+str(Frame_start)+ "_" + str(Frame_end) +".csv"]
Inter_list = ["Interection_" + video_id + "_"+str(Frame_start)+"_" +str(Frame_end) +".csv"]


All_TB  = pd.DataFrame()
for file_id in range(len(Inter_list)):
    Si = Singl_list[file_id]
    It = [i for i in Inter_list if Si in i][0]
    print(Si, It)
    # Single fly table, which has n_frame * n_fly rows
    Si_tb = pd.read_csv(Raw_dir + Si).iloc[:, 1:]
    #Si_tb = Si_tb[Si_tb['Frame']<=8000]
    # No Idea why there are some duplicate row in singel TB
    Si_tb = Si_tb[Si_tb.duplicated()==False]
    It_tb = pd.read_csv(Raw_dir + It).iloc[:, 1:]
    # Extract the length by Overlap frames
    Ran = range(min(It_tb['0'].min(),Si_tb['Frame'].min()),
        min(It_tb['0'].max(),Si_tb['Frame'].max())+1)

    Comd = pd.merge(Si_tb, It_tb.iloc[:,It_tb.columns!="Video"],  how='left', left_on=['Frame','Fly_s'], right_on = ['0','Fs'])
    Comd = Comd[Comd["Frame"].isin(Ran)]
    if len(Si_tb)== len(Comd):
        print("Yes")
    else:
        print(len(Si_tb), len(Comd))
    All_TB = pd.concat([All_TB, Comd])

# Add the Being chasing information
Chase_TB = All_TB[['Frame', 'Fs', 'Ft', 'Video', "Head_to"]][All_TB.Ft.isna() !=True]
Chase_TB = Chase_TB[Chase_TB.duplicated()==False]


All_TB = pd.merge(All_TB, Chase_TB, how="left", left_on=['Frame','Fly_s',
    "Video"], right_on = ['Frame','Ft', "Video"])
'''
After Merge, the Ft_x is the target Fly_s is chasing, the Ft_y is the sponser fly who is chasing Fly_s: Fs_y → Fs_x → Ft_x
    Only when this three are not empty, the result is chaining.
Notice: There could have duplicates because one fly could be chasing by two or even more flies at one time. So, we need to delete one of them for the rest of the calculation or we'll causing confusing results.
'''
All_TB = All_TB[All_TB[['Frame', 'Fly_s', "Video"]].duplicated()==False]

# Fill the NA
All_TB[[ "Fs_x", "Ft_x", "Fs_y", "Ft_y"]] = All_TB[[ "Fs_x", "Ft_x", "Fs_y", "Ft_y"]].fillna(0)

def Sing_Type(Fly_TB, Threshold = Stype_th):
    Duration = 0
    Duration_list= []
    Sing_TB=pd.DataFrame
    if Fly_TB.Sing.iloc[0] ==1:
        Duration += 1

    for i in range(1, len(Fly_TB)):
        if Fly_TB.Sing.iloc[i] == 1 and Fly_TB.Sing.iloc[i-1]==1:
            Duration += 1
        if Fly_TB.Sing.iloc[i] == 0 and Fly_TB.Sing.iloc[i-1]==1:
            Duration_list +=[Duration]
        if Fly_TB.Sing.iloc[i] == 1 and Fly_TB.Sing.iloc[i-1]!=1:
            Duration = 1
    # for the end:
    if Fly_TB.Sing.iloc[len(Fly_TB)-1] == 1:
        Duration_list +=[Duration]
    '''
    tb = pd.DataFrame([range(1, len(Duration_list)+1), Duration_list]).T
    tb.columns=['Times', 'Duration']
    tb['Fly'] = Fly_TB.Fly_s.unique()[0]
    tb['Video'] =  Fly_TB.Video.unique()[0]
    tb['Frame'] = str(min(Fly_TB.Frame)) + "_" + str(max(Fly_TB.Frame))
    tb['Mean'] = np.mean(Duration_list)
    tb['Sd'] = np.std(Duration_list)
    tb['Times'] = len(Duration_list)
    '''
    if np.mean(Duration_list) <= Threshold:
        return "Aggressive"
    else:
        return "Courtship"

def Behav_frame(TMP):
    # Not lonely
    if TMP.Nst_dist <= Nerst_dist_TH:
        # being Chase:
        if TMP.Fs_y != 0:
            # Target to head
            if TMP.Head_to_y=="Head":
                # Sing
                if TMP.Sing != 0:
                    return "ACST"
                else:
                    # It's Moving
                    if TMP['mm/s'] >= Moving_TH:
                        # Is Push?
                        # Chasing
                        if TMP.Fs_x != 0:
                            if TMP['mm/s'] == 0:
                                return "SCRT"
                            # Chasing each other
                            if len(TMP[["Fs_x","Ft_x", "Fs_y", "Ft_y"]].unique()) == 2:
                            #if abs(TMP.M_angle) <= Marge_angle:
                                return "PUSH"
                            else:
                                if TMP['mm/s'] <= Run_TH:
                                    return "DCW"
                                elif  TMP['mm/s'] <= Leap_TH:
                                    return "DCR"
                                elif  TMP['mm/s'] >= Leap_TH:
                                    return "DCF"
                        else:
                            if TMP['mm/s'] <= Run_TH:
                                return "DCW"
                            elif  TMP['mm/s'] <= Leap_TH:
                                return "DCR"
                            elif  TMP['mm/s'] >= Leap_TH:
                                return "DCF"
                    # It's not Moving
                    else:
                        return "ACFT"
            # Not head to head
            else:
                # Chase
                if TMP.Fs_x != 0:
                    return "CC"
                # Not Chase
                else:
                    if TMP.Sing !=0:
                        return "ACWT"
                    # Not Sing
                    else:
                        # It's Moving
                        if TMP['mm/s'] >= Moving_TH:
                            # moving ahead
                            if TMP['mm/s'] <= Run_TH:
                                return "DCW"
                            elif  TMP['mm/s'] <= Leap_TH:
                                return "DCR"
                            elif  TMP['mm/s'] >= Leap_TH:
                                return "DCF"
                        # It's not Moving
                        else:
                            return "DCDN"
        # Chasing
        # Chasing
        elif TMP.Fs_x != 0:
            # return Mount
            if TMP.Hold != 0:
                return "CM"
            # Moving?
            elif TMP['mm/s'] >= Moving_TH:
                # Not Crab?
                if TMP.Move==2:
                    # return Orientation
                    if TMP.Sing != 0:
                        return "ACO"
                    else:
                        return "ACOS"
                # Not crab walk
                else:
                    if TMP.Sing!=0:
                        return "ACS"
                    else:
                        return "ACC"
            # Not Moving
            elif TMP['mm/s'] < Moving_TH:
                if TMP.Sing!=0:
                    return "ACS"
                else:
                    return "ACT"
        else:
            # No Chasing at all
            # Moving
            if TMP['mm/s'] >= Moving_TH:
                # moving ahead
                if TMP['mm/s'] <= Run_TH:
                    return "SCW"
                elif  TMP['mm/s'] <= Leap_TH:
                    return "SCR"
                elif  TMP['mm/s'] >= Leap_TH:
                    return "SCL"
            # It's not Moving
            else:
                try:
                    if TMP.Feed != 0:
                        return "SCF"
                except:
                    return "SCRT"
                return "SCRT"
    else:
        # Moving
        if TMP['mm/s'] >= Moving_TH:
            # moving ahead
            if TMP['mm/s'] <= Run_TH:
                return "SOW"
            elif  TMP['mm/s'] <= Leap_TH:
                return "SOR"
            elif  TMP['mm/s'] >= Leap_TH:
                return "SOL"
        # It's not Moving
        else:
            try:
                if TMP.Feed != 0:
                    return "SOF"
            except:
                return "SORT"
            return "SORT"

## Type of Behavior 1
Be_sum = [Behav_frame(All_TB.iloc[i,:]) for i in range(len(All_TB))]
All_TB["Be_sum"] = Be_sum

#All_TB.to_csv("220602_All_TB.csv")

# Check the Sing in each event
tmp = All_TB[["Video","Chase_ID", "Sing"]].value_counts().rename_axis().reset_index()

N_Chase = len(All_TB[All_TB[["Video","Chase_ID"]].duplicated()==False])
tmp['ID'] = tmp.Chase_ID.astype(str) + "_" + tmp.Video
notSying_list = [i for i in tmp.ID.unique() if i not in tmp.ID[tmp.Sing!=0].tolist()]
N_notSing = len(notSying_list)


##################################
## 3rd Data
##################################

def Window_Mix(tmp, Cut_th=0.6):
    Rest_list = ['SORT', 'SCRT', "SCF", "SOF"]
    Walk_list = ['SOW', 'SOL', 'SOR',  'SCW', 'SCR', 'SCL']
    Chasing_list = ["ACC", "ACS", "ACT", "ACFT", "ACO", "ACOS", "CC", "ACWT", "ACST"]
    Defensive_list = ["ACFT", "DCDN"]
    Fleet_list = ["DCW", "DCR", "DCF", 'SCW', 'SCR', 'SCL']
    Aggr_list = ["PUSH", "ACST"]
    SUM = pd.DataFrame(tmp.value_counts())
    if SUM.iloc[0,0]/SUM.sum()[0] >= Cut_th:
        return SUM.index[0]
    elif SUM[SUM.index.isin(Rest_list +Walk_list+Fleet_list)].sum()[0]/SUM.sum()[0]>=Cut_th:
        return "Mix_Explore"
    elif SUM[SUM.index.isin(Chasing_list)].sum()[0]/SUM.sum()[0]>=Cut_th:
        return "Mix_Chasing"
    elif SUM[SUM.index.isin(Fleet_list + Chasing_list)].sum()[0]/SUM.sum()[0]>=Cut_th:
        return "Mix_Chase_fleet"
    elif SUM[SUM.index.isin(Aggr_list + Chasing_list)].sum()[0]/SUM.sum()[0]>=Cut_th:
        return "Mix_Ch_Arg"
    elif SUM[SUM.index.isin(Aggr_list)].sum()[0]/SUM.sum()[0]>=Cut_th:
        return "Mix_Aggr"
    elif SUM[SUM.index.isin(Fleet_list)].sum()[0]/SUM.sum()[0]>=Cut_th:
        return "Fleet"
    elif SUM[SUM.index.isin(Defensive_list + Fleet_list)].sum()[0]/SUM.sum()[0]>=Cut_th:
        return "Mix_Defence"
    elif SUM[SUM.index.isin(Rest_list + Defensive_list +Walk_list + Chasing_list)].sum()[0]/SUM.sum()[0]>=Cut_th:
        return "Mix_chase_Walk_Def"
    elif SUM[SUM.index.isin(Rest_list + Defensive_list +Fleet_list+Aggr_list)].sum()[0]/SUM.sum()[0]>=Cut_th:
        return "Mix_Rest_defensive"
    elif SUM[SUM.index.isin(Chasing_list + Defensive_list)].sum()[0]/SUM.sum()[0]>=Cut_th:
        return "Mix_Chase_defensive"
    elif SUM[SUM.index.isin(Fleet_list+Walk_list)].sum()[0]/SUM.sum()[0]>=Cut_th:
        return "Running"
    elif SUM[SUM.index.isin(Aggr_list +Fleet_list+ Defensive_list)].sum()[0]/SUM.sum()[0] >= Cut_th:
        return "Mix_Defence"
    else:
        return "NA"

def Window_sum(tmp, Cut_th=0.6):
    SUM = pd.DataFrame(tmp.value_counts())
    if len(SUM) ==1:
        return SUM.index[0]
    elif SUM.iloc[0,0]>SUM.iloc[1,0]:
        return SUM.index[0]
    elif (SUM.iloc[0,0] + SUM.iloc[1,0])/ SUM.Be_sum.sum()>=Cut_th:
        return "Mix_" + SUM.index[0]+"_"+SUM.index[1]
    else:
        return "NA"

Result = []
False_TMP = []
for Video in All_TB.Video.unique():
    Video_TB = All_TB[All_TB.Video==Video]
    for fly_id in Video_TB.Fly_s.unique():
        fly_TB = Video_TB[Video_TB.Fly_s==fly_id]
        for i in range(fly_TB.Frame.min(), fly_TB.Frame.max()-Window+1):
            TMP = fly_TB[fly_TB.Frame.isin(range(i,i+Window))]
            # Sing type
            if TMP.Sing.iloc[int(Window/2)] ==1:
                T_Sing = Sing_Type(TMP)
            else:
                T_Sing = "None"
            # Behavior Sliding
            tmp = TMP.Be_sum
            tmp_result = Window_sum(tmp)
            if tmp_result == "NA":
                tmp_result = Window_Mix(tmp)
                if tmp_result == "NA":
                    False_TMP += [TMP]
                else:
                    tmp_result == "None"
            Result += [[TMP.Frame.iloc[int(Window/2)], fly_id, tmp_result, T_Sing]]

'''
Some frames in SW_TB is missing
Fix it.
2698 3177 3181 3195 3640 3641 3966 4575 5556 5559 6496 6762 7315 7317
'''

SW_TB = pd.DataFrame(Result, columns=["Frame", "Fly_s", "Behavior", "Sing_Type"])

TB = pd.merge(All_TB, SW_TB, right_on = ["Frame", "Fly_s"], left_on=["Frame", "Fly_s"])

It_tb['Chase_type'] = "None"
for Chase_ID in It_tb.Chase_ID.unique():
    tmp = It_tb[It_tb.Chase_ID== Chase_ID]
    TMP = fly_TB[fly_TB.Frame.isin(range(min(tmp["0"]), max(tmp["0"])+1))]
    if sum(TMP.Sing)!=0:
        It_tb.Chase_type[It_tb.Chase_ID== Chase_ID] = Sing_Type(TMP)
    elif len(TMP) <= Ran_exp:
        It_tb.Chase_type[It_tb.Chase_ID== Chase_ID] = "Random Explore"
    else:
        It_tb.Chase_type[It_tb.Chase_ID== Chase_ID] = "None"

TB = pd.merge(TB, It_tb[["0", "Fs", "Chase_type"]],  left_on=["Frame", "Fly_s"], right_on = ["0", "Fs"], how = 'left')
# Remove duplicate columns
TB = TB[[i for i in TB.columns if i not in ['0_y', 'Fs']]]

# Mark the False Chasing
'''
In some case like two flies are too closing to each other and the image is very blurt could cause the false postive of chasing.
Filter:
long time chasing with minimal movement.
5s with
'''


Fly_tmp = TB[TB.Fly_s=='fly_3']
for Chase_ID in (Fly_tmp[Fly_tmp.Fs_x!=0].Chase_ID.unique()):
    Chase_TB = Fly_tmp[Fly_tmp.Chase_ID == Chase_ID]
    V_m = Chase_TB['mm/s'].sum()/len(Chase_TB)
    print(Chase_ID, len(Chase_TB), V_m)

TB.to_csv(Raw_dir + "Correct_"+video_id  + "_"+str(Frame_start)+"_" +str(Frame_end) + ".csv")
