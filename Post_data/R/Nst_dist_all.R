library(BSDA)
library(readxl)
library(ggplot2)
library(stringr)
library(reshape2)
library(RColorBrewer)

Wei_Mean <- function(TB){
  W_mean_t1 = 0
  W_mean_t2 = 0
  W_sd = 0
  for(i in c(1: nrow(TB))){
    Mean = TB[i,1]
    Sd = TB[i,2]
    if(Sd==0){
        Sd=0.001
    }
    W_mean_t1 = W_mean_t1 + Mean/(Sd^2)
    W_mean_t2 = W_mean_t2 + 1/(Sd^2)
    W_sd =  1/(Sd^2)
  }
  WM = W_mean_t1/W_mean_t2
  WS = sqrt(1/W_mean_t2)
  DF = sum(TB[3]) - nrow(TB)
  SE = WS/sqrt(sum(TB[3]))
  #print(paste("Weighted Mean:", WM))
  #print(paste("Weighted Sd:", WS))
  #print(paste("df:", DF))
  #print(paste("SE:", SE))
  return(c(WM, WS))
}
P2S <- function(P){
    S = "ns"
    if (P <= 0.05){
        S= "*"
    }
    if (P <= 0.01){
        S= "**"
    }
    if (P <= 0.001){
        S= "***"
    }
    if (P <= 0.0001){
        S= "****"
    }
    return(S)
}
gen_data <- function(means, sds, samplesizes){
  n.grp <- length(means)
  grps <- factor(rep(1:n.grp, samplesizes))
  dat <- lapply(1:n.grp, function(i) {scale(rnorm(samplesizes[i]))*sds[i] + means[i]})
  y <- do.call(rbind, dat)
  out <- data.frame(group = grps, y = y)
  out
}


TYPE="##TYPE##"

if(TYPE=="NstD"){
    TITLE = "Nearst Distance"
    TB_clist = c("Fly_s", "Nst_dist")
}else if(TYPE=="NstN"){
    TITLE = "Nearst Number"
    TB_clist = c("Fly_s", "Nst_num")
}else if(TYPE=="Speed"){
    TITLE = "Speed mm/s"
    TB_clist = c("Fly_s", "mm.s")
}

CMP = brewer.pal(n = 12, name = "Paired")
W=2.96
H=3.25

Wc=3.21
Hc=2.81


GROUP_exel = "../../Video_list.xlsx"
group_TB <- read_xlsx(GROUP_exel)
Group_lt <- unique(group_TB$Group)
Pair_TB <- as.data.frame(combn(Group_lt , 2))

if(length(unique(group_TB$Group))>12){
    CMP = rep(CMP, round(length(unique(group_TB$Group))/12)+1)
}
CMP = head(CMP, length(unique(group_TB$Group)))
CMP_TB = data.frame(Group = unique(group_TB$Group), Color= CMP)

## Pairwise Comparing and based on T test
for(Pair_id in c(1:ncol(Pair_TB))){
    Video_TB <- group_TB[c("Video_Name", "From_frame", "To_Frame", "Group")][group_TB$Group %in% Pair_TB[[Pair_id]],]
    Video_lt <- paste(Video_TB[[1]], Video_TB[[2]], Video_TB[[3]], sep ='_')
    Video_lt <- data.frame(Video_lt, Video_TB$Group)
    TB_SD <- data.frame()
    for (ROW in c(1:nrow(Video_lt))){
        Video = Video_lt[ROW, 1]
        Group = Video_lt[ROW, 2]
        TMP <- read.table(paste("Stat_Correct_", Video, ".csv", sep =""))
        TMP$N[is.na(TMP$N)] = nrow(TMP)-1
        TMP$Group = Group
        TMP$Video = Video
        TB_SD <- rbind(TB_SD, TMP[row.names(TMP)== "Weighted",])
    }
    Wted_TB <- data.frame()
    for(Group in unique(TB_SD$Group)){
        tmp_TB <- TB_SD[TB_SD$Group==Group,]
        Weighted_MS <- Wei_Mean(tmp_TB)
        tmp <- data.frame(Mean= Weighted_MS[1], SD = Weighted_MS[2], N = sum(tmp_TB$N), Group = Group)
        Wted_TB <- rbind(Wted_TB, tmp)
    }
    Wted_TB$Color<- CMP_TB$Color[match(Wted_TB$Group, CMP_TB$Group)]
    Wted_TB$Group_inf =  paste(paste(Wted_TB$Group, Wted_TB$N, sep ="\nN="))
    p.value <- tsum.test(Wted_TB[1,1], Wted_TB[1,2], Wted_TB[1,3],
              Wted_TB[2,1], Wted_TB[2,2], Wted_TB[2,3])$p.value
    Wted_TB$Group <- factor(Wted_TB$Group, levels = unique(Wted_TB$Group))
    ggplot(Wted_TB, aes(Group_inf, Mean)) +
        geom_bar(color="black", fill= "white",stat = 'identity', width = .4) +
        geom_errorbar(aes(ymin=Mean - SD, ymax=Mean+SD), width=.15) +
        theme_bw() + ggtitle(TITLE ) +
        theme(plot.title =  element_text(hjust = .5)) +
        geom_text(aes(x=2, y = 1.1* max(Wted_TB$Mean + Wted_TB$SD), label=P2S(p.value)), vjust=1)
    Name = paste("Ttest_weighted_", paste(Pair_TB[[Pair_id]], sep = "", collapse = "_VS_"), ".svg", sep='')
    ggsave(Name, w= W, h= H)
    ggplot(Wted_TB, aes(Group, Mean, fill= Group_inf)) +
        geom_bar(color="black", stat = 'identity', width = .4) +
        geom_errorbar(aes(ymin=Mean - SD, ymax=Mean+SD), width=.15) +
        theme_bw() + ggtitle(TITLE) +
        theme(plot.title =  element_text(hjust = .5)) +
        geom_text(aes(x=2, y = 1.1* max(Wted_TB$Mean + Wted_TB$SD), label=P2S(p.value)), vjust=1) +
        scale_fill_manual(values = as.character(Wted_TB$Color))
    Name = paste("Ttest_weighted_c_", paste(Pair_TB[[Pair_id]], sep = "", collapse = "_VS_"), ".svg", sep='')
    Name2 = paste("Ttest_weighted_c_", paste(Pair_TB[[Pair_id]], sep = "", collapse = "_VS_"), ".png", sep='')
    ggsave(Name, w= Wc, h= Hc)
    ggsave(Name2, w= Wc, h= Hc)
    Name = paste("Ttest_weighted_c_", paste(Pair_TB[[Pair_id]], sep = "", collapse = "_VS_"), ".csv", sep='')
    Wted_TB = Wted_TB[-ncol(Wted_TB)]
    Wted_TB$Pval = p.value
    write.table(Wted_TB,Name, sep ='\t', quote=F)
}

## Comparing all groups

Video_TB <- group_TB[c("Video_Name", "From_frame", "To_Frame", "Group")]
Video_lt <- paste(Video_TB[[1]], Video_TB[[2]], Video_TB[[3]], sep ='_')
Video_lt <- data.frame(Video_lt, Video_TB$Group)
TB_SD <- data.frame()
for (ROW in c(1:nrow(Video_lt))){
    Video = Video_lt[ROW, 1]
    Group = Video_lt[ROW, 2]
    TMP <- read.table(paste("Stat_Correct_", Video, ".csv", sep =""))
    TMP$N[is.na(TMP$N)] = nrow(TMP)-1
    TMP$Group = Group
    TMP$Video = Video
    TB_SD <- rbind(TB_SD, TMP[row.names(TMP)== "Weighted",])
}
Wted_TB <- data.frame()
for(Group in unique(TB_SD$Group)){
    tmp_TB <- TB_SD[TB_SD$Group==Group,]
    Weighted_MS <- Wei_Mean(tmp_TB)
    tmp <- data.frame(Mean= Weighted_MS[1], SD = Weighted_MS[2], N = sum(tmp_TB$N), Group = Group)
    Wted_TB <- rbind(Wted_TB, tmp)
}
Wted_TB$Color<- CMP_TB$Color[match(Wted_TB$Group, CMP_TB$Group)]
Wted_TB$Group <- factor(Wted_TB$Group, levels = unique(Wted_TB$Group))
Wted_TB$Group_inf =  paste(paste(Wted_TB$Group, Wted_TB$N, sep ="\nN="))
Wted_TB$Group_inf <- factor(Wted_TB$Group_inf, levels = unique(Wted_TB$Group_inf))

simulated_data <- gen_data(Wted_TB$Mean, Wted_TB$SD,Wted_TB$N)
av <- aov(y ~ group, data = simulated_data)
Turky_TB <- as.data.frame(TukeyHSD(av)$group)

# Line

line_TB <- data.frame(matrix(as.numeric(str_split_fixed(row.names(Turky_TB), "-", 2)), ncol = 2))
Turky_TB <- cbind(Turky_TB, line_TB)
Turky_TB$Y = NA
Y = max(Wted_TB$Mean + Wted_TB$SD)*1.05
Turky_TB$star = "ns"
for(i in c(1:nrow(Turky_TB))) {
    Turky_TB[i,'star'] <- P2S(Turky_TB[i,'p adj'])
    if(Turky_TB[i,'star'] != 'ns'){
        Turky_TB[i,'Y'] = Y
        Y = Y * 1.05
    }
}

ggplot(Wted_TB, aes(Group, Mean)) +
    geom_bar(aes(, fill= Group_inf),color="black", stat = 'identity', width = .6) +
    geom_errorbar(aes(ymin=Mean - SD, ymax=Mean+SD), width=.15) +
    theme_bw() + ggtitle(paste(TITLE, "(Turky)")) +
    theme(plot.title =  element_text(hjust = .5)) +
    scale_fill_manual(values = as.character(Wted_TB$Color))+
    geom_segment(data= Turky_TB[Turky_TB$star!="ns",], aes(x=X1, xend= X2, y = Y, yend= Y)) +
    geom_text(data= Turky_TB[Turky_TB$star!="ns",], aes(x=X1, y= Y, label=star), vjust=.4)

ggsave('Anova_Turky.svg', w=Wc + 0.2 *(length(Wted_TB$Group)-2), h=Hc )
ggsave('Anova_Turky.png', w=Wc + 0.2 *(length(Wted_TB$Group)-2), h=Hc )

row.names(Turky_TB) <- paste(Wted_TB$Group[Turky_TB$X1], Wted_TB$Group[Turky_TB$X2], sep =" Vs ")
write.table(Turky_TB[-c(5:7)], 'Anova_Turky.csv', sep = '\t', quote = F)



library(DescTools)
Dunnet_TB <- DunnettTest(y ~ group, data = simulated_data)[['1']]

line_TB <- data.frame(matrix(as.numeric(str_split_fixed(row.names(Dunnet_TB), "-", 2)), ncol = 2))
Dunnet_TB <- cbind(Dunnet_TB, line_TB)
Dunnet_TB$Y = NA
Y = max(Wted_TB$Mean + Wted_TB$SD)*1.05
Dunnet_TB$star = "ns"
for(i in c(1:nrow(Dunnet_TB))) {
    Dunnet_TB[i,'star'] <- P2S(Dunnet_TB[i,'pval'])
    if(Dunnet_TB[i,'star'] != 'ns'){
        Dunnet_TB[i,'Y'] = Y
        Y = Y * 1.05
    }
}

ggplot(Wted_TB, aes(Group, Mean)) +
    geom_bar(aes(, fill= Group_inf),color="black", stat = 'identity', width = .6) +
    geom_errorbar(aes(ymin=Mean - SD, ymax=Mean+SD), width=.15) +
    theme_bw() + ggtitle(paste(TITLE, "(Dunnett)")) +
    theme(plot.title =  element_text(hjust = .5)) +
    scale_fill_manual(values = as.character(Wted_TB$Color))+
    geom_segment(data= Dunnet_TB[Dunnet_TB$star!="ns",], aes(x=X1, xend= X2, y = Y, yend= Y)) +
    geom_text(data= Dunnet_TB[Dunnet_TB$star!="ns",], aes(x=X1, y= Y, label=star), vjust=.4)
ggsave('Anova_Dunnett.svg', w=Wc + 0.2 *(length(Wted_TB$Group)-2), h=Hc )
ggsave('Anova_Dunnett.png', w=Wc + 0.2 *(length(Wted_TB$Group)-2), h=Hc )

row.names(Dunnet_TB) <- paste(Wted_TB$Group[Dunnet_TB$X1], Wted_TB$Group[Dunnet_TB$X2], sep =" Vs ")
write.table(Dunnet_TB[-c(5:7)], 'Anova_Dunnett.csv', sep = '\t', quote = F)
