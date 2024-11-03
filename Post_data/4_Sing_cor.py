#!/usr/bin/env python3
'''
Cummary data based on the each chasing events
'''
#import json, math, random, cv2, os
import os
import pandas as pd
#import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt

Raw_dir = "Video_post/"
#Singl_list = [i for i in os.listdir(Raw_dir) if "Interection_" not in i and "mp4.csv" in i]
Inter_list = [i for i in os.listdir(Raw_dir) if "Interection_" in i]



All_TB  = pd.DataFrame()
for file_id in range(len(Inter_list)):
    Si = Singl_list[file_id]
    It = [i for i in Inter_list if Si in i][0]
    print(Si, It)
    Si_tb = pd.read_csv(Raw_dir + Si).iloc[:, 1:]
    Si_tb = Si_tb[Si_tb['Frame']<=8000]
    It_tb = pd.read_csv(Raw_dir + It).iloc[:, 1:]
    Ran = range(min(It_tb['0'].min(),Si_tb['Frame'].min()),
        min(It_tb['0'].max(),Si_tb['Frame'].max()))

    Comd = pd.merge(Si_tb, It_tb.iloc[:,It_tb.columns!="Video"],  how='left', left_on=['Frame','Fly_s'], right_on = ['0','Fs'])
    Comd = Comd[Comd["Frame"].isin(Ran)]

    All_TB = pd.concat([All_TB, Comd])


Chase_TB = All_TB[['Frame', 'Ft', 'Video', 'Scan_pos', 'T_side', 'Head_to', 'Direction', 'Angle', 'Sfly_act', 'Sfly_dis', 'Sfly_head', 'Sfly_body', 'Sfly_ang']][All_TB.Ft.isna() !=True]
Chase_TB = Chase_TB[Chase_TB[["Frame", "Ft", "Video"]].duplicated()==False]

All_TB = pd.merge(All_TB, Chase_TB, how="left", left_on=['Frame','Fly_s', "Video"], right_on = ['Frame','Ft', "Video"])

# Hold is include in Touch and other column
All_TB = All_TB[All_TB[["Frame", "Fly_s", "Video"]].duplicated()!=True]

Group_tb = pd.DataFrame(All_TB.Video.unique(), columns=["Video"])
Group_tb['Group'] = "V33"
Group_tb.Group[Group_tb.Video.isin([i for i in Group_tb.Video if "Hnf4" in i])] = "HN4"
Group_tb.Group[Group_tb.Video.isin([i for i in Group_tb.Video if "GFP" in i])] = "GFP"

All_TB = All_TB.fillna(0)
All_TB = All_TB[All_TB.duplicated()==False]

## Group

All_TB = pd.merge(All_TB, Group_tb,  how='left', left_on=['Video'], right_on = ['Video'])

'''
for Video in All_TB.Video.unique():
    Video_TB = All_TB[All_TB.Video==Video]
    for fly_id in Video_TB.Fly_s.unique():
        fly_TB = Video_TB[Video_TB.Fly_s==fly_id]
        print(len(fly_TB) - (fly_TB.iloc[-1,0] - fly_TB.iloc[0,0]))
'''


##
Nerst_dist_TH = 5
Moving_TH = 0.25
Run_TH = 1
Leap_TH = 2
Marge_angle = 15


def Behav_frame(TMP):
    # Not lonely
    if TMP.Nst_dist <= Nerst_dist_TH:
        # being Chase:
        if TMP.Ft_y != 0:
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
                        if TMP.Fs != 0:
                            if abs(TMP.M_angle) <= Marge_angle:
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
                if TMP.Fs != 0:
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
        elif TMP.Fs != 0:
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

'''
TMP = All_TB.head(10000).tail(5000)
Be_sum = [Behav_frame(TMP.iloc[i,:]) for i in range(len(TMP))]
TMP['Be_sum'] = Be_sum
'''
Be_sum = [Behav_frame(All_TB.iloc[i,:]) for i in range(len(All_TB))]
All_TB["Be_sum"] = Be_sum
# All_TB.to_csv("All_TB.csv")



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
    elif SUM[SUM.index.isin(Rest_list +Walk_list)].sum()[0]/SUM.sum()[0]>=Cut_th:
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
    elif SUM[SUM.index.isin(Rest_list + Defensive_list)].sum()[0]/SUM.sum()[0]>=Cut_th:
        return "Mix_Rest_defensive"
    elif SUM[SUM.index.isin(Chasing_list + Defensive_list)].sum()[0]/SUM.sum()[0]>=Cut_th:
        return "Mix_Chase_defensive"
    elif SUM[SUM.index.isin(Fleet_list+Walk_list)].sum()[0]/SUM.sum()[0]>=Cut_th:
        return "Running"
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



Window = 30
Result = []
for Video in All_TB.Video.unique():
    Video_TB = All_TB[All_TB.Video==Video]
    for fly_id in Video_TB.Fly_s.unique():
        fly_TB = Video_TB[Video_TB.Fly_s==fly_id]
        for i in range(fly_TB.Frame.min(), fly_TB.Frame.max()-Window+1):
            TMP = fly_TB[fly_TB.Frame.isin(range(i,i+Window))]
            tmp = TMP.Be_sum
            tmp_result = Window_sum(tmp)
            if tmp_result == "NA":
                tmp_result = Window_Mix(tmp)
                if tmp_result == "NA":
                    raise "Stop"
            else:
                Result += [[TMP.iloc[0,:].Frame, TMP.iloc[0,:].Fly_s, TMP.iloc[0,:].Video, tmp_result]]


SW_TB = pd.DataFrame(Result, columns=["Frame", "Fly_s","Video", "Behavior"])
SW_TB.Behavior.iloc[[i for i in SW_TB.index if "Mix" in SW_TB.Behavior[i]]] = "Mix"
BE_TB = pd.DataFrame(SW_TB.value_counts(subset=['Fly_s',"Video",'Behavior']), columns = ["Counts"])

ID = pd.DataFrame([[ii for ii in i] for i in BE_TB.index], columns= BE_TB.index.names )
BE_TB.index = ID.index
Fly_Be_M = pd.concat([ID.Fly_s +"_" +ID.Video, ID.Behavior, BE_TB.Counts], axis=1, ignore_index=True)
Fly_Be_M.columns = ['ID', "Behavior", "Count"]
Fly_Be_TB = pd.DataFrame(Fly_Be_M.pivot(index="ID", columns="Behavior"))
Fly_Be_TB = Fly_Be_TB.fillna(0)
Fly_Be_TB[('Count',  'REST')] = Fly_Be_TB[('Count',  'SCRT')] + Fly_Be_TB[('Count',  'SORT')]

# Remove
R_list = [('Count',  'SCRT'), ('Count',  'SORT')]
Fly_Be_TB = Fly_Be_TB.iloc[:,Fly_Be_TB.columns.isin(R_list)==False]
#Fly_Be_TB = Fly_Be_TB.iloc[[0,2,4,6, 8],:]
#Fly_Be_TB = Fly_Be_TB.iloc[[1,3,5,7, 9],:]
##################
### hierarchy Clust
##################

from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.stats as stats
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist


Z = linkage(stats.zscore(Fly_Be_TB.iloc[:, np.array([i for i in Fly_Be_TB.sum()!=0])]) , 'ward')
c, coph_dists = cophenet(Z, pdist(Fly_Be_TB))


temp = {ii: Fly_Be_TB.index[ii].replace("_Trim.mp4", "") for ii in range(len(Fly_Be_TB.index))}
def llf(xx):
    return temp[xx]


plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_label_func=llf,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

##################
### T-test to fileter
##################
'''
from scipy import stats
# perform multiple pairwise comparison (Tukey's HSD)
# unequal sample size data, tukey_hsd uses Tukey-Kramer test
c_list = [ 'Nst_dist', 'Nst_num', 'length', 'B_w', 'B_l', 'B_r', 'B_angle',
   'M_angle', 'Move', 'mm/s', 'Motion', 'Sing', 'Grooming', 'Chasing',
   'Hold',  'Scan_pos', 'T_side',
   'Head_to', 'Direction', 'Angle', 'Sfly_act', 'Sfly_dis',
   'Sfly_head', 'Sfly_body', 'Sfly_ang', 'Tfly_act', 'Tfly_dis',
   'Tfly_head', 'Tfly_body', 'Tfly_ang',  'Chain_number', 'Fight_result', 'Class']

Sig_list = []
for i in c_list:
    try:
        S,P = stats.ttest_ind(All_TB[i][All_TB['Group']=="GFP"].to_numpy(), All_TB[i][All_TB['Group']=="V33"].to_numpy())
        if P<=0.05:
            Sig_list += [i]
        print(i, P)
    except:
        print("Pass", i ,P )

Sig_list2 = []
for i in Sig_list:
    try:
        S,P = stats.ttest_ind(All_TB[i][All_TB['Group']=="GFP"].to_numpy(), All_TB[i][All_TB['Group']=="HN4"].to_numpy())
        if P<=0.05:
            Sig_list2 += [i]
            print(i, P)
    except:
        print("Pass", i ,P )

Sig_list2 += ['Nst_num']

################################################
### Random Forest select features
#################################################

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


X = All_TB[c_list]
y = All_TB.Group

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)
sorted_idx = rf.feature_importances_.argsort()
plt.barh(boston.feature_names[sorted_idx], rf.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")

plt.show()
'''
####################################################
## Previous frame action, being chasing or some thing
####################################################


####################################################
## Prepare for the classification
####################################################

### Umape
from umap import UMAP
import plotly.express as px

features = np.array(Fly_Be_TB)

umap_2d = UMAP(n_components=2, init='random', random_state=0)
umap_2d = UMAP(
    n_neighbors=30,
    min_dist=0.0,
    n_components=2,
    random_state=42,
)

#umap_3d = UMAP(n_components=3, init='random', random_state=0)

%time proj_2d = umap_2d.fit_transform(features)
proj_tb = pd.concat([pd.DataFrame(proj_2d),pd.DataFrame(Fly_Be_TB.index) ], axis=1)

#proj_3d = umap_3d.fit_transform(features)



'''
fig = px.scatter(
    proj_2d, x=0, y=1,
    color=proj_tb.ID, labels={'color': 'species'}
)
fig.update_layout({"plot_bgcolor": 'rgba(0, 0, 0, 0)'})
fig.show()

fig_3d = px.scatter_3d(
    proj_3d, x=0, y=1, z=2,
    color=Video_TB.Group, labels={'color': 'species'}
)
fig_3d.show()


'''

'''
Num = 0
for m in ("euclidean", red_channel_dist, sl_dist, hue_dist, hsl_dist):
    Num +=1
    name = m if type(m) is str else m.__name__
    umap_2d = UMAP(n_components=2, init='random', random_state=0, metric=m)
    proj_2d = umap_2d.fit_transform(features)
    proj_tb = pd.concat([pd.DataFrame(proj_2d),Video_TB ], axis=1)
    plt.subplot(1,5,Num)
    sns.scatterplot(data=proj_tb, x=0, y=1, palette=CMAP, alpha=.1, hue="Group").set(title = m)


plt.show()

'''

# split density show
for i in range(3):
    plt.subplot(1,3,i+1)
    sns.scatterplot(data=proj_tb[proj_tb.Group==proj_tb.Group.unique()[i]], x=0, y=1, c=[CMAP[i]], alpha=.1).set(title = proj_tb.Group.unique()[i])
    sns.histplot(data=proj_tb[proj_tb.Group==proj_tb.Group.unique()[i]], x=0, y=1, bins=50, pthresh=.1, cmap="mako")
    sns.kdeplot(data=proj_tb[proj_tb.Group==proj_tb.Group.unique()[i]], x=0, y=1, levels=5, color="w", linewidths=1)

plt.show()

# total density show
sns.scatterplot(data=proj_tb, x=0, y=1, palette=CMAP, alpha=.1, hue="Group")
sns.histplot(data=proj_tb, x=0, y=1, bins=50, pthresh=.1, cmap="mako")
sns.kdeplot(data=proj_tb, x=0, y=1, levels=7, color="salmon", linewidths=1)
plt.show()


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10, random_state=0).fit(proj_2d)

proj_tb["K_mean"] = kmeans.labels_
# proj_tb.to_csv("proj_tb.csv", index=None)
for i in range(len(proj_tb["K_mean"].unique())):
    tmp = proj_tb[proj_tb.K_mean==i]
    plt.text(tmp.median()[0], tmp.median()[1], str(i), fontsize=20)

sns.scatterplot(data=proj_tb, x=0, y=1, hue =kmeans.labels_, palette="Paired", legend=None)
plt.show()


Num = 0
for i in range(10):
    tmp = pd.DataFrame(proj_tb.Group[proj_tb.K_mean==i].value_counts()/ proj_tb.Group.value_counts()*100)
    tmp = tmp.iloc[tmp.index.argsort(),:]
    plt.subplot(2,5, i+1)
    sns.barplot(data=tmp, x= tmp.index, y = "Group", palette="Paired").set(title="Group " + str(i))

plt.show()

'''
from sklearn.cluster import SpectralClustering

sc = SpectralClustering(n_clusters=10).fit(proj_2d)
labels = sc.labels_

proj_tb["K_mean"] = labels
ROW = len(proj_tb["K_mean"].unique())/5
if ROW > int(ROW):
    ROW = int(ROW) +1

for i in range(len(set(labels))):
    tmp = proj_tb[proj_tb.K_mean==i]
    plt.text(tmp.median()[0], tmp.median()[1], str(i), fontsize=20)

sns.scatterplot(data=proj_tb, x=0, y=1, hue =labels, palette="Paired", legend=None)
plt.show()

Num = 0
for i in range(len(proj_tb["K_mean"].unique() )):
    tmp = pd.DataFrame(proj_tb.Group[proj_tb.K_mean==i].value_counts()/ proj_tb.Group.value_counts()*100)
    tmp = tmp.iloc[tmp.index.argsort(),:]
    plt.subplot(ROW,5, i+1)
    sns.barplot(data=tmp, x= tmp.index, y = "Group", palette="Paired").set(title="Group " + str(i))

plt.show()

'''
