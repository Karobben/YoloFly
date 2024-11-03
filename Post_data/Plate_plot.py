import argparse, os, json

parser = argparse.ArgumentParser()
parser.add_argument('-i','-I','--input', type = str)    #Video
parser.add_argument('-f','-F','--fm', type = int, default = 1)     #Starts at the frame
parser.add_argument('-e','-E','--end', type = int, default = 1000)      #Ends at the frame
parser.add_argument('-c','-C','--classes', nargs='+', type = str, default =  [0,2,3,4,5])    #Classies

args = parser.parse_args()
video_id = args.input
FROM_fm = args.fm
END_fm = args.end
CLASS = [int(i) for i in args.classes]

AUTO_PATH = os.path.realpath(__file__).replace("Plate_plot.py", "")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

os.system("mkdir img")
os.system("mkdir img/plat_row_plot")
os.system("mkdir img/plat_row_plot/histbar")
os.system("mkdir img/plat_row_plot/hist2d")
os.system("mkdir img/plat_row_plot/hist2d")
os.system("mkdir img/plat_row_plot/DotsPlateS")
os.system("mkdir img/plat_row_plot/DotsPlateP")
os.system("mkdir img/plat_row_plot/GaussianDens")

def hist_bar():
    sns.set_style("whitegrid")
    f, ax = plt.subplots(figsize=(7, 5))
    sns.despine(f)
    sns.histplot(
        AA[AA[1]>=2],
        x=0, hue='class',
        multiple="stack",
        palette=PALETTE,
        linewidth=.1,
        bins= 30
    )
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    f.savefig("img/plat_row_plot/histbar/" + ID + "_hist.svg")

def hist2d():
    fig, ax = plt.subplots(figsize=(10, 5))
    h =plt.hist2d(AA[2], AA[3],
        bins= 30,  cmap="coolwarm")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.colorbar(h[3])
    plt.gca().invert_yaxis()
    fig.savefig("img/plat_row_plot/hist2d/" + ID + "_2dhist.svg")

def DotsPlate():
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(7, 7))

    if 0 in CLASS:
        sns.scatterplot(x=2, y=3,
            color ="grey",
            alpha = .1,
            data=AA[AA[1]==0])
    if 2 in CLASS:
        sns.scatterplot(x=2, y=3,
            color ="green",
            alpha = .1,
            data=AA[AA[1]==2])
    if 3 in CLASS:
        sns.scatterplot(x=2, y=3,
            color ="steelblue",
            alpha = .7,
            data=AA[AA[1]==3])
    if 4 in CLASS:
        sns.scatterplot(x=2, y=3,
            color ="salmon",
            alpha = .7,
            data=AA[AA[1]==4])
    if 5 in CLASS:
        sns.scatterplot(x=2, y=3,
            color ="yellow",
            alpha = .7,
            data=AA[AA[1]==5])
    plt.gca().invert_yaxis()

    fig.savefig("img/plat_row_plot/DotsPlateS/" + ID + "_dots.svg")
    fig.savefig("img/plat_row_plot/DotsPlateP/" + ID + "_dots.png")

def Gaussian_density(data):
    x = data[:, 0]
    y = data[:, 1]# Define the borders
    deltaX = (max(x) - min(x))/10
    deltaY = (max(y) - min(y))/10
    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    fig = plt.figure(figsize=(20, 10.9))
    ax = fig.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
    #ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
    cset = ax.contour(xx, yy, f, colors='k')
    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.title('2D Gaussian Kernel density estimation')
    plt.gca().invert_yaxis()
    plt.savefig("img/plat_row_plot/GaussianDens/" + ID + "_Gaussin.svg")


TB = pd.read_csv("csv/"+ video_id + ".csv", sep=" ", header = None)
TB['class'] = "NA"
TB['class'][TB[1]==2]="Groom"
TB['class'][TB[1]==3]="Chase"
TB['class'][TB[1]==4]="Sing"
TB['class'][TB[1]==5]="Hold"
PALETTE = json.load(open(AUTO_PATH + "config.json"))['PALETTE']

AA = TB[TB[0].isin(range(FROM_fm, END_fm +1))]

ID = video_id + "_" +  str(FROM_fm) + "_" + str(END_fm)


hist_bar()

DotsPlate()

hist2d()

Gaussian_density(AA[[2,3]].to_numpy())
