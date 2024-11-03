library(readxl)
library(ggplot2)
library(stringr)
library(networkD3)
library(chorddiag)
library(RColorBrewer)

args <- commandArgs(trailingOnly = TRUE)

Raw_dir = "Video_post/"
File_list = paste("Correct_", args[1], ".csv", sep= "")

Net_plot <- function(TB_Times, Type, Time, Time2=1){
    Nodes_list <- sort(unique(c(as.character(unique(TB_Times$Fs)), as.character(unique(TB_Times$Ft)))))
    Nodes = data.frame(name = Nodes_list, group = 1, size=1, ID = c(0: (length(Nodes_list)-1)))
    tmp = as.data.frame.array(table(TB_Times[c("Fs", Type)]))
    tmp_sum  = as.data.frame(apply(as.numeric(colnames(tmp)) * tmp,1, sum))
    Nodes$size = tmp_sum[match(Nodes$name, row.names(tmp_sum)),]* Time
    Nodes$color = head(c(brewer.pal(12, "Paired"), brewer.pal(12, "Set1"),  brewer.pal(12, "Set2")), nrow(Nodes))

    Links = TB_Times
    Links$Fs = Nodes$ID[match(Links$Fs, Nodes$name)]
    Links$Ft = Nodes$ID[match(Links$Ft, Nodes$name)]
    colnames(Links) = c("source", "target", "value")
    Links$value = Links$value * Time2
    Links$color = Nodes$color[match(Links$source, Nodes$ID)]

    colors <- paste(Nodes$color, collapse = '", "')
    colorJS <- paste('d3.scaleOrdinal(["', colors, '"])')

    p<- forceNetwork(Links=Links, Nodes=Nodes, linkDistance = 400,
        Source="source", Target="target",
        Value="value", NodeID="name",
        fontSize=40, Group="ID", Nodesize = "size",
        arrows=TRUE, zoom=TRUE, legend=TRUE,
        colourScale = colorJS, linkColour = Links$color,
        opacityNoHover = 1)
    return(p)
}


File_list = list.files(Raw_dir)[grep("Correct_", list.files(Raw_dir))]
TB_Times_all = data.frame()
for (File in File_list){
    TB = read.csv(paste(Raw_dir, File, sep=""), row.names= 1)
    TB$ID = paste(TB$Fly_s, TB$Video, sep=":")

    TB_Ac = data.frame()
    for (fly in unique(TB$Fly_s)){
        TMP = TB[TB$Fly_s==fly,]
        TMP <- TMP[TMP$Fs_x !=0,]
        TMP2 <- TMP[c("Frame", "Fs_x", "Ft_x", "Chase_ID")]
        TMP_ChaseID = TMP2[!duplicated(TMP2$Chase_ID),]
        Duration_tb <- as.data.frame(table(TMP2$Chase_ID))
        TMP_ChaseID$Duration = Duration_tb$Freq[match(TMP_ChaseID$Chase_ID, Duration_tb$Var1)]
        colnames(TMP_ChaseID)[2:3] <- c("Fs", "Ft")
        TB_Ac = rbind(TB_Ac, TMP_ChaseID)
    }

    TB_Times = as.data.frame(table(TB_Ac[c("Fs", "Ft")]))
    TB_Times <- TB_Times[TB_Times$Freq!=0,]

    TB_Ac$ID <- paste(TB_Ac$Fs,TB_Ac$Ft)
    TB_duration = as.data.frame.array(table(TB_Ac[c("ID", "Duration")]))
    TB_duration <- as.numeric(colnames(TB_duration))*TB_duration
    TB_duration = as.data.frame(apply(TB_duration, 1, sum))

    TB_Times$Duration[match(row.names(TB_duration), paste(TB_Times$Fs, TB_Times$Ft))] <- TB_duration[[1]]
    TB_Times$Video = as.character(TB$Video[1])
    TB_Times_all = rbind(TB_Times_all, TB_Times )
}

#  Network plot

name= str_replace(str_replace(File, "Correct_", ""),".csv","")

for(name in unique(TB_Times_all$Video)){
    TB_Times <- TB_Times_all[TB_Times_all$Video == name,]
    p <- Net_plot(TB_Times, "Freq", 1, 2)
    saveNetwork(p, "sn2.html")
    webshot::webshot("sn2.html",paste("img/NetWork/Times_Behavior_", name, ".png",sep=""), vwidth = 1000, vheight = 1000)
    p <- Net_plot(TB_Times, "Duration", 0.01, 2)
    saveNetwork(p, "sn2.html")
    webshot::webshot("sn2.html",paste("img/NetWork/Duration_Behavior_", name, ".png",sep=""), vwidth = 1000, vheight = 1000)

}


## Statistics

Group_T <- read.csv("Video_mate.csv")

TB_Times_all$ID = paste(TB_Times_all$Fly_s, TB_Times_all$Video, sep=":")
TB_Times_all$Cha = paste(TB_Times_all$Fs, TB_Times_all$Ft, sep=":")
TB_Times_all$Group <- Group_T$Group[match(TB_Times_all$Video, Group_T$Video_Name)]



TB_array <- as.data.frame.array(reshape(TB_Times_all[c("ID", "Cha", "Duration")], timevar = "Cha", idvar = "ID", direction = 'wide'))
row.names(TB_array) <- TB_array$ID
