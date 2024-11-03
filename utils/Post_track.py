#!/usr/bin/env python3

import numpy as np
import pandas as pd
import cv2, json, warnings, os
from shapely.geometry import Polygon
from skimage.metrics import structural_similarity as ssim
from scipy.stats import median_abs_deviation as mad
from scipy.stats import zscore

from scipy.stats import norm
from scipy.spatial.distance import cdist
# create a polygon by following order:
# mute warning messages from pandas
from Head_bind import head_match
head_bind = head_match()


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i','-I','--input')     #输入文件
parser.add_argument('-o','-U','--output')     #输入文件
parser.add_argument('-v','-V','--video')     #输入文件

##获取参数
args = parser.parse_args()
INPUT = args.input
OUTPUT = args.output
Video = args.video


try:
    warnings.filterwarnings("ignore")
except:
    pd.options.mode.chained_assignment = None

#from Fly_Tra import fly_align 

## Functions

def Number_adjust(data, N = 13):
    Z_abs = abs(zscore(data))
    sorted_indices = np.argsort(Z_abs)#[::-1]
    tops = sorted_indices[:N]
    return tops

def Dots_Sort(points1, points2):
    # Generate two lists of 2D points
    #points1 = np.random.rand(10, 2)
    #points2 = np.random.rand(10, 2)
    # Calculate the pairwise distances between the points
    distances = cdist(points1, points2)
    # Sort the distances and get the indices of the sorted elements
    sorted_indices = np.argsort(distances, axis=None)
    # Keep track of which points have already been paired
    paired_points1 = set()
    paired_points2 = set()
    # Loop through the sorted distances and pair the closest points
    pairs = []
    for index in sorted_indices:
        i1, i2 = np.unravel_index(index, distances.shape)
        if i1 not in paired_points1 and i2 not in paired_points2:
            paired_points1.add(i1)
            paired_points2.add(i2)
            pairs.append((i1, i2))
    # Print the pairs
    Dots = pd.DataFrame(pairs)
    return(Dots)

def box_center(img_f, Type = 'R'):
    img_f[img_f>50] = 0
    nonzero_indices = np.nonzero(img_f)
    # Create a DataFrame with the non-zero indices and their corresponding values
    df = pd.DataFrame({
        'col': nonzero_indices[0],
        'row': nonzero_indices[1]
    })
    if Type == "R":
        return  ((df.row.mean()-(len(img_f[0])/2))/1920, (df.col.mean()-(len(img_f)/2))/1080 )
    else:
        return (df.row.mean(), df.col.mean())

def creat_polygon(arry):
    # arry = S_TMP_B[S_TMP_B.ID==ob_ls].to_numpy()[0][2:6]
    X1 = arry[0]
    Y1 = arry[1]
    X2 = arry[0] + arry[2]  
    Y2 = arry[1] + arry[3] 
    x = [X1, X2, X2, X1]
    y = [Y1, Y1, Y2, Y2]
    return Polygon([[i,j]for i,j in zip(x,y)])

def Overlap_test(ob_ls, TMP_B):
    rct_los = creat_polygon(TMP_B[TMP_B.ID==ob_ls].to_numpy()[0][2:6])
    Inter_dict1 = {}
    Inter_dict2 = {}
    for line in range(len(TMP_B)):
        if TMP_B.ID.iloc[line] != ob_ls:
            rct_tag = creat_polygon(TMP_B.iloc[line,2:6].to_numpy())
            Inter_dict1.update({ TMP_B.ID.iloc[line] : rct_los.intersection(rct_tag).area/ rct_los.area})
            Inter_dict2.update({ TMP_B.ID.iloc[line] : rct_los.intersection(rct_tag).area/ rct_tag.area})
    if max(Inter_dict1.values()) < max(Inter_dict2.values()):
        Inter_dict1 =  Inter_dict2
    if max(Inter_dict1.values()) >= Overlap_thres:
        ob_ov = max(Inter_dict1, key= Inter_dict1.get )
        TMP_cache = TB_cache[TB_cache.ID == ob_ov]
        TMP_cache['Area'] = TMP_cache[4] * TMP_cache[5]
        Ar_change = (TMP_B[TMP_B.ID == ob_ov][4] * TMP_B[TMP_B.ID == ob_ov][5]).to_list()[0] / TMP_cache.Area.mean()
        if Ar_change >= Box_size_check:
            bst_frame = TMP_cache[0][np.abs(TMP_cache.Area - TMP_cache.Area.mean())== min(np.abs(TMP_cache.Area - TMP_cache.Area.mean()))]
            # then, read images and drift by the center
            cap.set(1,frame)
            ret,img_t=cap.read()
            OB_TB = S_TMP_B[S_TMP_B.ID == ob_ov]
            OB_TB[2] *= 1920
            OB_TB[4] *= 1920
            OB_TB[3] *= 1080
            OB_TB[5] *= 1080
            img_t = img_t[int(OB_TB[3]- OB_TB[5]/2):int(OB_TB[3] + OB_TB[5]/2), int(OB_TB[2] - OB_TB[4]/2 ):int(OB_TB[2] + OB_TB[4]/2)]
            img_t = cv2.cvtColor(img_t,cv2.COLOR_RGB2GRAY)
            img_t = cv2.GaussianBlur(img_t, (5, 5), 10)
            return (bst_frame.to_list()[0], ob_ov, box_center(img_t))
    return False

def Obj_los_test(frame, ob_ls, cap):
    Result = None
    OB_TB = S_TMP_B[S_TMP_B.ID == ob_ls]
    #OB_TB = S_TMP_B[S_TMP_B.ID == 'fly_1']
    OB_TB[2] *= 1920
    OB_TB[4] *= 1920
    OB_TB[3] *= 1080
    OB_TB[5] *= 1080

    cap.set(1,frame)
    ret,img_t=cap.read()
    img_t = img_t[int(OB_TB[3]- OB_TB[5]/2):int(OB_TB[3] + OB_TB[5]/2), int(OB_TB[2] - OB_TB[4]/2 ):int(OB_TB[2] + OB_TB[4]/2)]
    img_t = cv2.cvtColor(img_t,cv2.COLOR_RGB2GRAY)
    img_t = cv2.GaussianBlur(img_t, (5, 5), 10)

    # ratio of fly-body to the blank background. A normal single fly ~= 14, >20 means over corroded.
    CoverR = len(img_t.ravel()[img_t.ravel()<50])/len(img_t.ravel())
    print("Mask ratio:", CoverR )
    if CoverR>=.19:
        print("Over Corroded caused object lost, test the overlap")
        return {'Type' : "CroLst", "drift" : box_center(img_t)}
    if CoverR==0:
        print("Object lost")
        return {'Type' : "CroLst", "drift" : (0,0)}

    cap.set(1,frame-1)
    ret,img_f=cap.read()
    img_f = img_f[int(OB_TB[3]- OB_TB[5]/2):int(OB_TB[3] + OB_TB[5]/2), int(OB_TB[2] - OB_TB[4]/2 ):int(OB_TB[2] + OB_TB[4]/2)]
    img_f = cv2.cvtColor(img_f, cv2.COLOR_RGB2GRAY)
    img_f = cv2.GaussianBlur(img_f, (5, 5), 10)
    # similarity clean vs single fly: 68.8%
    try:
        ssim_V = ssim(img_f, img_t)
        print(ssim_V )
        if ssim_V>=.85:
            print("Single fly lost")
            
            # ove lap check
            rct_los = creat_polygon(S_TMP_B[S_TMP_B.ID==ob_ls].to_numpy()[0][2:6])
            Inter_dict = {}
            for line in range(len(TMP_B)):
                Inter_dict.update({ TMP_B.ID.iloc[line] : rct_los.intersection(creat_polygon(TMP_B.iloc[line,2:6].to_numpy())).area/ rct_los.area})
            Over_p = list(np.where(np.array(list(Inter_dict.values())) > .2)[0])
            if len(Over_p) > 0:
                print("Overlap caused in frame:", frame)
                #max_value = max(Inter_dict, key=Inter_dict.get)
                cap.set(1,frame)
                ret,img_n=cap.read()

                Scores = {}
                fly_cache = TB_cache[TB_cache.ID == ob_ls] 
                for i in TB_cache[0].unique():
                    cap.set(1, i)
                    ret,img_p=cap.read()
                    Loc = TB_cache[TB_cache[0] == i]
                    OB_TB = Loc[Loc.ID== ob_ls]
                    OB_TB[2] *= 1920
                    OB_TB[4] *= 1920
                    OB_TB[3] *= 1080
                    OB_TB[5] *= 1080
                    img_old = img_p[int(OB_TB[3]- OB_TB[5]/2):int(OB_TB[3] + OB_TB[5]/2), int(OB_TB[2] - OB_TB[4]/2 ):int(OB_TB[2] + OB_TB[4]/2)]
                    img_old = cv2.cvtColor(img_old, cv2.COLOR_RGB2GRAY)
                    img_old[img_old>50] =0
                    img_now = img_n[int(OB_TB[3]- OB_TB[5]/2):int(OB_TB[3] + OB_TB[5]/2), int(OB_TB[2] - OB_TB[4]/2 ):int(OB_TB[2] + OB_TB[4]/2)]
                    img_now = cv2.cvtColor(img_now, cv2.COLOR_RGB2GRAY)
                    img_now[img_now>50] =0
                    Scores.update({ i: ssim(img_now, img_old)})
                best_frame = max(Scores, key=Scores.get)
                return {'Type' : "Overlap", "frame": best_frame, "drift" : box_center(img_t)}
            else:
                print(' No overlap')
                return {'Type' : "CroLst", "drift" : box_center(img_t)}
    
    except:
        print('Small box size')
    print('Less similarities, fastmoving')
    return {'Type' : "CroLst", "drift" :(0,0)}

## Functions down

# argumetns


CSV_f = INPUT#"Mix7.MP4.csv"
#Video = "/home/ken/Videos/Mix7.MP4"
Num = 12
#OUTPUT = 'test.csv'

Box_size_check_d = 1.6
Box_size_check = 1.3
Overlap_thres  = .45

#TB = pd.read_csv(CSV_f, sep = ' ', header = None)
TB_np = np.load(INPUT)
TB = pd.DataFrame(TB_np)
TB[0]= TB[0].astype(int)#.astype(str)
TB[1]= TB[1].astype(int)#.astype(str)

cap=cv2.VideoCapture(Video)

Start = 1
End = 18000
# Define the Start 
S_TMP = TB[TB[0]==Start]
S_TMP_B = S_TMP[S_TMP[1]==0]
S_TMP_B = S_TMP_B.iloc[Number_adjust((S_TMP_B[4] * S_TMP_B[5]).to_numpy(), 12)]
S_TMP_B['ID'] = ['fly_' +str(i) for i in range(len(S_TMP_B))]
Dots_from = S_TMP_B.iloc[:,2:4].to_numpy()
S_TMP_B['find'] = True
# Save the first frame
S_TMP_B.to_csv(OUTPUT, header=False, index=False)

TB_cache = S_TMP_B

FLY_matrix = {}
TMP_Dict = {}
for fly in S_TMP_B.ID:
    TMP_Dict.update({fly : {'body': S_TMP_B[S_TMP_B.ID==fly].to_numpy().tolist()[0][2:6]}})
FLY_matrix.update({Start: TMP_Dict})

# head bind
TB_head = S_TMP[S_TMP[1]==1].iloc[:,1:]
head_bind.main(FLY_matrix, Start, TB_head)
#print("Head Match:", head_bind.MATCH_result)
for fly in FLY_matrix[Start].keys():
    try:
        FLY_matrix[Start][fly].update({"head":list(TB_head.iloc[int(head_bind.MATCH_result[fly]),1:])})
    except:
        FLY_matrix[Start][fly].update({"head":FLY_matrix[Start][fly]['body']})


# write results
dic_ID = list(FLY_matrix.keys())[-1]
tmp = {dic_ID:FLY_matrix[dic_ID]}
FLY_matrix_tmp_str = json.dumps(tmp) +";"
os.system("rm  " + OUTPUT+"_"+str(Start)+"_.json")
Trac_out = open(OUTPUT+"_"+str(Start)+"_.json", "a")
Trac_out.write(FLY_matrix_tmp_str)


for frame in TB[0].unique()[1:int(TB_np[-1][0]) + 1]: #range(Start +1, End+1):
    # Define the Start 
    TMP = TB[TB[0]==frame]
    TMP_B = TMP[TMP[1]==0]
    TMP_B['ID'] = None
    TMP_B['find'] = True
    # Check the over-sized box and update it if it caused by two/more flies
    TMP_B['Areas'] = (TMP_B[4]*TMP_B[5]*1920*1080).to_numpy()

    Clear_lst = []
    for line in np.where( TMP_B.Areas/ TMP_B.Areas.mean() > Box_size_check_d)[0]:
        TMP_BL = TMP_B.iloc[line, :]
        # Check the overlap
        rct_los = creat_polygon(TMP_BL.to_numpy()[2:6])
        Inter_dict1 = [ rct_los.intersection(creat_polygon(TMP_B.iloc[line,2:6].to_numpy())).area / rct_los.area  for line in range(len(TMP_B)) ] 
        Inter_dict2 = [  rct_los.intersection(creat_polygon(TMP_B.iloc[line,2:6].to_numpy())).area / creat_polygon(TMP_B.iloc[line,2:6].to_numpy()).area for line in range(len(TMP_B))]
        Over_p = list(np.where(np.array(Inter_dict1) > Overlap_thres)[0]) + list(np.where(np.array(Inter_dict2) > Overlap_thres)[0])
        while line in Over_p:
            Over_p.remove(line)
        if len(Over_p) > 0:
            Clear_lst += [line]
    TMP_B = TMP_B.drop(TMP_B.index[Clear_lst])


    # remove the Areas column
    TMP_B = TMP_B.drop('Areas', axis = 1)
    Dots_to = TMP_B.iloc[:,2:4].to_numpy()

    # two steps sort
    ## Step 1
    Dots = Dots_Sort(Dots_from[S_TMP_B.find], Dots_to)
    # inherit IDs from previous based on sorting distance 
    for i in range(len(Dots)):
        TMP_B.ID.iloc[Dots[1][i]] = S_TMP_B.ID[S_TMP_B.find].iloc[Dots[0][i]]
    ## Step 2
    Dots = Dots_Sort(Dots_from[~S_TMP_B.find],Dots_to[TMP_B.ID.isna()])
    mask = TMP_B[TMP_B.ID.isna()]
    for i in range(len(Dots)):
        TMP_B.ID.iloc[ TMP_B.index == mask.iloc[Dots[1][i]].name] = S_TMP_B.ID[~S_TMP_B.find].iloc[Dots[0][i]]

    # Adjust the size of boxs
    TMP_B['Areas'] = (TMP_B[4]*TMP_B[5]*1920*1080).to_numpy()
    for ob_ov in TMP_B.ID[~TMP_B.ID.isna()]:
        if TMP_B.Areas[TMP_B.ID==ob_ov].iloc[0]/ TMP_B.Areas.mean() > Box_size_check_d:
            Over_adjust = Overlap_test(ob_ov, TMP_B[~TMP_B.ID.isna()])
            if Over_adjust != False:
                ob_ov = Over_adjust[1]
                #print("Adjust the size of ", frame, ob_ov)
                TMP_chage = TB_cache[ TB_cache[0]== Over_adjust[0]]
                TMP_chage = TMP_chage[ TMP_chage.ID == Over_adjust[1]]
                TMP_B.iloc[np.where(TMP_B.ID==ob_ov)[0],4] = TMP_chage[4] * .9
                TMP_B.iloc[np.where(TMP_B.ID==ob_ov)[0],5] = TMP_chage[5] * .9
    # remove the Areas column
    TMP_B = TMP_B.drop('Areas', axis = 1)
    Dots_to = TMP_B.iloc[:,2:4].to_numpy()


    # Missing Check
    if len(TMP_B[TMP_B.ID != None]) < len(S_TMP_B):
        #print("object lost, Number:", len(S_TMP_B) - len(TMP_B[TMP_B.ID != None]))
        for losN in range(len(S_TMP_B) - len(TMP_B[TMP_B.ID != None])):
            ob_ls = S_TMP_B.ID[S_TMP_B.ID.isin(TMP_B.ID)==False].iloc[0]
            #print("lost object:", ob_ls, "in frame:", frame)
            Lost = Obj_los_test(frame, ob_ls, cap)
            if Lost['Type'] == "CroLst":
                # update the frame from the object lost
                Obl_TB = S_TMP_B[S_TMP_B.ID==ob_ls]
                Obl_TB[0] = frame
                Obl_TB[2] += Lost['drift'][0]
                Obl_TB[3] += Lost['drift'][1]
                Obl_TB.find = False
                TMP_B = pd.concat([TMP_B, Obl_TB])
            elif Lost['Type'] == "Overlap":
                Obl_TB = TB_cache[TB_cache[0]== Lost['frame']]
                Obl_TB = Obl_TB[Obl_TB.ID==ob_ls]
                Obl_TB[0] = frame
                Obl_TB[2] += Lost['drift'][0]
                Obl_TB[3] += Lost['drift'][1]
                Obl_TB.find = False
                TMP_B = pd.concat([TMP_B, Obl_TB])
            else:
                print("\n\nERROR!!!:", frame, ob_ls, "\n\n")
            Over_adjust = Overlap_test(ob_ls, TMP_B)
            if Over_adjust != False:
                ob_ov = Over_adjust[1]
                #print("Adjust the size of ", frame, ob_ov)
                #print(Over_adjust)
                TMP_chage = TB_cache[ TB_cache[0]== Over_adjust[0]]
                TMP_chage = TMP_chage[ TMP_chage.ID == Over_adjust[1]]
                #TMP_B.iloc[np.where(TMP_B.ID==ob_ov)[0],2] += Over_adjust[2][0]
                #TMP_B.iloc[np.where(TMP_B.ID==ob_ov)[0],3] += Over_adjust[2][1]
                TMP_B.iloc[np.where(TMP_B.ID==ob_ov)[0],4] = TMP_chage[4] * .9
                TMP_B.iloc[np.where(TMP_B.ID==ob_ov)[0],5] = TMP_chage[5] * .9

    # remove false positive results
    TMP_B = TMP_B[TMP_B.ID.isna()==False]
    Dots_to = TMP_B.iloc[:,2:4].to_numpy()
    # give the ID to new frame
    # save the results
    TMP_B.to_csv(OUTPUT, header=False, index=False, mode='a')

    # Update FLY_matrix
    TMP_Dict = {}
    for fly in TMP_B.ID:
        TMP_Dict.update({fly : {'body': TMP_B[TMP_B.ID==fly].to_numpy().tolist()[0][2:6]}})
    FLY_matrix.update({frame: TMP_Dict})
    if len(FLY_matrix) >10:
        FLY_matrix.pop(list(FLY_matrix.keys())[0])
    #print("FLY_matrix:", len(FLY_matrix))
   
    # head bind
    TB_head = TMP[TMP[1]==1].iloc[:,1:]
    head_bind.main(FLY_matrix, frame, TB_head)
    #print("Head Match:", head_bind.MATCH_result)
    for fly in FLY_matrix[frame].keys():
        try:
            FLY_matrix[frame][fly].update({"head":list(TB_head.iloc[int(head_bind.MATCH_result[fly]),1:])})                        #print(FLY_matrix)
        except:
            # Inherate the head from previous frame based on relative position
            last_body = FLY_matrix[frame-1][fly]['body']
            last_head = FLY_matrix[frame-1][fly]['head']
            new_body  = FLY_matrix[frame][fly]['body']
            rel_pos = [last_head[0] - last_body[0], last_head[1] - last_body[1]]
            rel_pos_new = [rel_pos[0]+ new_body[0], rel_pos[1]+ new_body[1]]
            FLY_matrix[frame][fly].update({"head": rel_pos_new + last_head[2:4]})

    # write results
    dic_ID = int(list(FLY_matrix.keys())[-1])
    tmp = {dic_ID:FLY_matrix[dic_ID]}
    FLY_matrix_tmp_str = json.dumps(tmp) +";"
    Trac_out = open(OUTPUT+"_"+str(Start)+"_.json", "a")
    Trac_out.write(FLY_matrix_tmp_str)

    # update catch table
    TB_cache = pd.concat([TB_cache, TMP_B])
    TB_cache = TB_cache[TB_cache[0].isin(TB_cache[0].unique()[-10:])]

    #S_TMP_B = None
    S_TMP_B = TMP_B
    Dots_from = Dots_to

Trac_out.close()



'''
head_bind.main(FLY_matrix, Num_frame, TB_head)
print("Head Match:", head_bind.MATCH_result)
for fly in FLY_matrix[Num_frame].keys():
    FLY_matrix[Num_frame][fly].update({"head":list(TB_head.iloc[int(head_bind.MATCH_result[fly]),1:])})                        #print(FLY_matrix)
dic_ID = list(FLY_matrix.keys())[-1]
tmp = {dic_ID:FLY_matrix[dic_ID]}
FLY_matrix_tmp_str = json.dumps(tmp) +";"
print(FLY_matrix_tmp_str)
# update the tar_tr_start
os.system("rm  csv/" + Video+"_"+str(tar_tr_start)+"_.json")
Trac_out = open("csv/" + Video+"_"+str(tar_tr_start)+"_.json", "a")
Trac_out.write(FLY_matrix_tmp_str)
'''


'''
ptLeftTop = (int(OB_TB[2] - OB_TB[4]/2 ), int(OB_TB[3]- OB_TB[5]/2))
ptRightBottom = (int(OB_TB[2] + OB_TB[4]/2), int(OB_TB[3] + OB_TB[5]/2))
point_color = (0, 0, 255) # BGR
thickness = 2
lineType = 8
cap.set(1,frame-1)
ret,img_f=cap.read()
img_f = img_f[int(OB_TB[3]- OB_TB[5]/2):int(OB_TB[3] + OB_TB[5]/2), int(OB_TB[2] - OB_TB[4]/2 ):int(OB_TB[2] + OB_TB[4]/2)]
img_f = cv2.cvtColor(img_f,cv2.COLOR_RGB2GRAY)
img_f = cv2.GaussianBlur(img_f, (5, 5), 100)
#img_f[img_f >= 50]=0
'''


'''    
# codes visualize the result and output as video

TBR = pd.read_csv(OUTPUT, header= None)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output2.avi',fourcc, 20.0, (1920,1080))

cap=cv2.VideoCapture(Video)
Num = Start -1
cap.set(1, Num+1)

while Num <= End:
    Num += 1 
    TMP = TBR[TBR[0]==Num]
    ret,img = cap.read()
    cv2.putText(img, str(Num) ,(100, 100), cv2.FONT_HERSHEY_COMPLEX, .5, (100, 200, 200), 2)

    for fly in range(len(TMP)):
        ptLeftTop = (int( 1920 * (TMP.iloc[fly, 2]- TMP.iloc[fly, 4]/2)), int( 1080 * (TMP.iloc[fly, 3]- TMP.iloc[fly, 5]/2)))
        ptRightBottom = (int( 1920 * (TMP.iloc[fly, 2]+ TMP.iloc[fly, 4]/2)), int( 1080 * (TMP.iloc[fly, 3] + TMP.iloc[fly, 5]/2)))
        point_color = (0, 0, 255) # BGR
        thickness = 1
        lineType = 8
        cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
        cv2.putText(img, TMP.iloc[fly, 6] ,(int( 1920 * TMP.iloc[fly, 2]), int( 1080 * TMP.iloc[fly, 3])), cv2.FONT_HERSHEY_COMPLEX, .5, (100, 200, 200), 2)
    cv2.imshow("video", img)
    out.write(img)
    if cv2.waitKey(30)&0xFF==ord('q'):
        cv2.destroyAllWindows()
        break
out.release()
cv2.destroyAllWindows()



TBR = pd.read_csv('Mix7.MP4.csv', header= None, sep = ' ')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Standard.avi',fourcc, 20.0, (1920,1080))

cap=cv2.VideoCapture(Video)
Num = Start -1
cap.set(1, Num+1)

while Num <= End:
    Num += 1 
    TMP = TBR[TBR[0]==Num]
    TMP = TMP[TMP[1]==0]
    ret,img = cap.read()
    cv2.putText(img, str(Num) ,(100, 100), cv2.FONT_HERSHEY_COMPLEX, .5, (100, 200, 200), 2)

    for fly in range(len(TMP)):
        ptLeftTop = (int( 1920 * (TMP.iloc[fly, 2]- TMP.iloc[fly, 4]/2)), int( 1080 * (TMP.iloc[fly, 3]- TMP.iloc[fly, 5]/2)))
        ptRightBottom = (int( 1920 * (TMP.iloc[fly, 2]+ TMP.iloc[fly, 4]/2)), int( 1080 * (TMP.iloc[fly, 3] + TMP.iloc[fly, 5]/2)))
        point_color = (0, 0, 255) # BGR
        thickness = 1
        lineType = 8
        cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
    cv2.imshow("video", img)
    out.write(img)
    if cv2.waitKey(30)&0xFF==ord('q'):
        cv2.destroyAllWindows()
        break
out.release()
cv2.destroyAllWindows()




'''

'''
# code for show specific fly in a frame

frame = 661 
fly = 'fly_3'

TBR = pd.read_csv(OUTPUT, header= None)
TMP = TBR[TBR[0]==frame]
fly_TB = TMP[TMP[6] == fly]
fly_TB[2] *= 1920
fly_TB[4] *= 1920
fly_TB[3] *= 1080
fly_TB[5] *= 1080

cap=cv2.VideoCapture(Video)
cap.set(1,frame)
ret,img_f=cap.read()
img_f = img_f[int(fly_TB[3]- fly_TB[5]/2):int(fly_TB[3] + fly_TB[5]/2), int(fly_TB[2] - fly_TB[4]/2 ):int(fly_TB[2] + fly_TB[4]/2)]
img_f = cv2.cvtColor(img_f,cv2.COLOR_RGB2GRAY)
img_f = cv2.GaussianBlur(img_f, (5, 5), 100)
img_f[img_f >= 50]=0

nonzero_indices = np.nonzero(img_f)

# Create a DataFrame with the non-zero indices and their corresponding values
df = pd.DataFrame({
    'row': nonzero_indices[0],
    'col': nonzero_indices[1],
    'value': img_f[nonzero_indices]
})

# weight the points
df.value = df.value/df.value.max()
df['srow'] = df.row * df.value
df['scol'] = df.col * df.value



plt.imshow(img_f)
plt.show()
plt.scatter( df.col.median(), df.row.median(), c= 'r')
plt.scatter(df.col.mean(), df.row.mean(),  c= 'black')


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(cv2.cvtColor(img_f, cv2.COLOR_BGR2RGB))
ax1.plot(int(box_center(img_f, '')[0]), int(box_center(img_f, '')[1]), 'ro')
ax2.imshow(cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB))
ax2.plot(int(box_center(img_t, '')[0]), int(box_center(img_t, '')[1]), 'ro')
plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(cv2.cvtColor(img_old, cv2.COLOR_BGR2RGB))
ax2.imshow(cv2.cvtColor(img_now, cv2.COLOR_BGR2RGB))
plt.show()


cv2.destroyAllWindows()
cv2.imshow("video",img_f)
if cv2.waitKey(0)&0xFF==ord('q'):
    cv2.destroyAllWindows()
'''
