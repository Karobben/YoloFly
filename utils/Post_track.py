#!/usr/bin/env python3

import cv2
import warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


# mute warning messages from pandas

warnings.filterwarnings("ignore", category=pd.core.common.SettingWithCopyWarning)

#from Fly_Tra import fly_align 

## Functions

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

def Obj_los_test(frame, ob_ls, cap):
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
    print("Cover ratio:", CoverR )
    if CoverR>=.19:
        print("Over Corroded caused object lost")
        return "CroLst"

    cap.set(1,frame-1)
    ret,img_f=cap.read()
    img_f = img_f[int(OB_TB[3]- OB_TB[5]/2):int(OB_TB[3] + OB_TB[5]/2), int(OB_TB[2] - OB_TB[4]/2 ):int(OB_TB[2] + OB_TB[4]/2)]
    img_f = cv2.cvtColor(img_f, cv2.COLOR_RGB2GRAY)
    img_f = cv2.GaussianBlur(img_f, (5, 5), 10)
    # similarity clean vs single fly: 68.8%
    ssim_V = ssim(img_f, img_t)
    if ssim_V>=.19:
        print("Single fly lost")
        return "CroLst"


## Functions down

CSV_f = "~/tmp/test.csv"
Video = "/home/ken/Videos/Mix7.MP4"
Num = 12
OUTPUT = '123.csv'

TB = pd.read_csv(CSV_f, sep = ' ', header = None)
cap=cv2.VideoCapture(Video)

# Define the Start 
S_TMP = TB[TB[0]==1]
S_TMP_B = S_TMP[S_TMP[1]==0]
S_TMP_B['ID'] = ['fly_' +str(i) for i in range(len(S_TMP_B))]
Dots_from = S_TMP_B.iloc[:,2:4].to_numpy()
# Save the first frame
S_TMP_B.to_csv(OUTPUT, header=False, index=False)


for frame in range(2, TB[0].max() +1):
    TMP = TB[TB[0]==frame]
    TMP_B = TMP[TMP[1]==0]
    Dots_to = TMP_B.iloc[:,2:4].to_numpy()
    Dots = Dots_Sort(Dots_from, Dots_to)
    TMP_B['ID'] = None
    for i in range(len(Dots)):
        TMP_B.ID.iloc[Dots[1][i]] = S_TMP_B.ID.iloc[Dots[0][i]]
    if len(Dots) < len(S_TMP_B):
        print("object lost, Number:", len(S_TMP_B) - len(Dots))
        for losN in range(len(S_TMP_B) - len(Dots)):
            ob_ls = S_TMP_B.ID[S_TMP_B.ID.isin(TMP_B.ID)==False].iloc[losN]
            print("lost object:", ob_ls)
            Lost = Obj_los_test(frame, ob_ls, cap)
            if Lost == "CroLst":
                TMP_B = pd.concat([TMP_B, S_TMP_B[S_TMP_B.ID==ob_ls]])
            else:
                print("\n\nERROR!!!:", frame, ob_ls, "\n\n")
    # remove false positive results
    TMP_B = TMP_B[TMP_B.ID.isna()==False]
    # give the ID to new frame
    # save the results
    TMP_B.to_csv(OUTPUT, header=False, index=False, mode='a')
    # substitute old frame
    S_TMP_B = TMP_B
    Dots_from = Dots_to
    




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
cv2.imshow("video",img_f)
if cv2.waitKey(0)&0xFF==ord('q'):
    cv2.destroyAllWindows()
'''