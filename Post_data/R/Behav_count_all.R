library(ggdendro)
library(readxl)
library(ggkaboom)


TYPE="##TYPE##"

if(TYPE=="Behavior"){
    TB_clean <- function(TB, Behavior){
        TB <- TB[c("Fly_s", "Sing", "Grooming", "Chasing", "Hold", "Motion")]
        TB$Behavior = "Rest"
        TB$Behavior[TB$Motion==2] = "Walk"
        TB$Behavior[TB$Motion==3] = "Run"
        TB$Behavior[TB$Motion==4] = "Leap"
        TB$Behavior[TB$Chasing==1] = "Chasing"
        TB$Behavior[TB$Grooming==1] = "Grooming"
        TB$Behavior[TB$Sing==1] = "Sing"
        TB$Behavior[TB$Hold==1] = "Hold"
        TB$Behavior <- factor(TB$Behavior, levels = Behavior)
        TB$Fly_s<- factor(TB$Fly_s, levels =  unique(TB$Fly_s))
        return(TB)
    }
    Behavior <- c("Rest", "Walk", "Run", "Leap",
    "Chasing", "Grooming", "Sing", "Hold")
    TIITLE = "Flies Behavior Counts"
}else if(TYPE=="Move"){
    TB_clean <- function(TB, Behavior){
        TB <- TB[c("Fly_s", "Move", "Motion")]
        TB$Behavior = "Ahead"
        TB$Behavior[TB$Move==2] = "Crab"
        TB$Behavior[TB$Move==3] = "Back"
        TB$Behavior[TB$Motion==1] = "Rest"
        TB$Behavior <- factor(TB$Behavior, levels = Behavior)
        TB$Fly_s<- factor(TB$Fly_s, levels =  unique(TB$Fly_s))
        return(TB)
    }
    Behavior <- c("Rest", "Ahead", "Crab", "Back")
    TIITLE = "Flies Move Counts"
}else if(TYPE=="Motion"){
    TB_clean <- function(TB, Behavior){
        TB <- TB[c("Fly_s", "Motion")]
        TB$Behavior = "Rest"
        TB$Behavior[TB$Motion==2] = "Walk"
        TB$Behavior[TB$Motion==3] = "Run"
        TB$Behavior[TB$Motion==4] = "Leap"
        TB$Behavior <- factor(TB$Behavior, levels = Behavior)
        TB$Fly_s<- factor(TB$Fly_s, levels =  unique(TB$Fly_s))
        return(TB)
    }
    Behavior <- c("Rest", "Walk", "Run", "Leap")
    TIITLE = "Flies Motion Counts"
}


if(TYPE=="Behavior"){
    TB_clean <- function(TB, Behavior){
        TB <- TB[c("Fly_s", "Sing", "Grooming", "Chasing", "Hold", "Motion")]
        TB$Behavior = "Rest"
        TB$Behavior[TB$Motion==2] = "Walk"
        TB$Behavior[TB$Motion==3] = "Run"
        TB$Behavior[TB$Motion==4] = "Leap"
        TB$Behavior[TB$Chasing==1] = "Chasing"
        TB$Behavior[TB$Grooming==1] = "Grooming"
        TB$Behavior[TB$Sing==1] = "Sing"
        TB$Behavior[TB$Hold==1] = "Hold"
        TB$Behavior <- factor(TB$Behavior, levels = Behavior)
        TB$Fly_s<- factor(TB$Fly_s, levels =  unique(TB$Fly_s))
        return(TB)
    }
    Behavior <- c("Rest", "Walk", "Run", "Leap",
    "Chasing", "Grooming", "Sing", "Hold")
    Color <- c("#A6CEE3", "#B2DF8A", "#33A02C", "#E31A1C", "#1F78B4", "#6A3D9A", "salmon", "#FFFF99")
    TIITLE = "Flies Behavior Counts"
}else if(TYPE=="Move"){
    TB_clean <- function(TB, Behavior){
        TB <- TB[c("Fly_s", "Move", "Motion")]
        TB$Behavior = "Ahead"
        TB$Behavior[TB$Move==2] = "Crab"
        TB$Behavior[TB$Move==3] = "Back"
        TB$Behavior[TB$Motion==1] = "Rest"
        TB$Behavior <- factor(TB$Behavior, levels = Behavior)
        TB$Fly_s<- factor(TB$Fly_s, levels =  unique(TB$Fly_s))
        return(TB)
    }
    Behavior <- c("Rest", "Ahead", "Crab", "Back")
    Color <- c("#A6CEE3", "#FF7F00", "#CAB2D6", "#B15928")
    TIITLE = "Flies Move Counts"
}else if(TYPE=="Motion"){
    TB_clean <- function(TB, Behavior){
        TB <- TB[c("Fly_s", "Motion")]
        TB$Behavior = "Rest"
        TB$Behavior[TB$Motion==2] = "Walk"
        TB$Behavior[TB$Motion==3] = "Run"
        TB$Behavior[TB$Motion==4] = "Leap"
        TB$Behavior <- factor(TB$Behavior, levels = Behavior)
        TB$Fly_s<- factor(TB$Fly_s, levels =  unique(TB$Fly_s))
        return(TB)
    }
    Behavior <- c("Rest", "Walk", "Run", "Leap")
    Color <- c("#A6CEE3", "#B2DF8A", "#33A02C", "#E31A1C")
    TIITLE = "Flies Motion Counts"
}

CMP_TB <- data.frame(Color, Behavior)


GROUP_exel = "../../Video_list.xlsx"
group_TB <- read_xlsx(GROUP_exel)
Group_lt <- unique(group_TB$Group)
Pair_TB <- as.data.frame(combn(Group_lt , 2))


W= 6.75
H= 4.94
Ratio = 1

CMP = head(brewer.pal(n = 12, name = "Paired"),8)
if(length(unique(group_TB$Group))>12){
    CMP = rep(CMP, round(length(unique(group_TB$Group))/12)+2)
}
CMP = head(CMP, length(unique(group_TB$Group)))
CMP_TB_G = data.frame(Group = unique(group_TB$Group), Color= CMP)


for(Pair_id in c(1:ncol(Pair_TB))){
    Video_TB <- group_TB[c("Video_Name", "From_frame", "To_Frame", "Group")][group_TB$Group %in% Pair_TB[[Pair_id]],]
    Video_lt <- paste(Video_TB[[1]], Video_TB[[2]], Video_TB[[3]], sep ='_')
    Video_lt <- data.frame(Video_lt, Video_TB$Group)
    TB <- data.frame()
    for (ROW in c(1:nrow(Video_lt))){
        Video = Video_lt[ROW, 1]
        Group = Video_lt[ROW, 2]
        TMP <- read.table(paste("../Correct_", Video, ".csv", sep =""), sep = ",", header = T)
        TMP <- TB_clean(TMP, Behavior)
        TMP <- data.frame(table(TMP[c("Fly_s", "Behavior")]))
        TMP$Group = Group
        TMP$Video = Video
        TB <- rbind(TB, TMP)
    }
    TB$Group<- factor(TB$Group, levels =  unique(TB$Group))

    P <- Kaboom_bar(TB[2:4], 'Behavior', 'Group',Facet = F, fill = "Group",  P_test="wilcox", Var = "SEM")
    P[[1]] +
      scale_fill_manual(values= as.character(CMP_TB_G$Color[match( unique(TB$Group), CMP_TB_G$Group)])) +
      ggtitle(TIITLE) +
      theme(plot.title = element_text(hjust= .5) )
    TMP <-  P[[2]]
    TMP$Group <- as.data.frame(str_split_fixed(TMP$Group, "\n", 2))[[1]]
    ggsave(paste(paste(Pair_TB[[Pair_id]],  collapse = "_VS_"),'.png', sep=""), w=W, h=H)
    ggsave(paste(paste(Pair_TB[[Pair_id]],  collapse = "_VS_"),'.svg', sep=""), w=W, h=H)
    write.table(TMP,paste(paste(Pair_TB[[Pair_id]],  collapse = "_VS_"),'.csv'), sep = '\t', quote = F)


    TB$ID <- paste(TB$Video, TB$Fly_s,sep="__")
    TB <- TB[c("ID","Group", "Behavior", "Freq")]
    TB_Wi <- reshape(TB, idvar = c("ID", "Group"), timevar = c("Behavior"), direction = 'wide')


    hc       <- hclust(dist(TB_Wi[-c(1:2)]))           # heirarchal clustering
    dendr    <- dendro_data(hc, type="rectangle") # convert for ggplot
    clust    <- cutree(hc,k=5)                    # find 2 clusters
    clust.df <- data.frame(label=names(clust), cluster=factor(clust))
    ## dendr[["labels"]] has the labels, merge with clust.df based on label column
    dendr[["labels"]] <- merge(dendr[["labels"]],clust.df, by="label")
    ## plot the dendrogram; note use of color=cluster in geom_text(...)

    Pending = max(apply(TB_Wi[-c(1:2)], 1, sum))
    Ratio   = max(apply(TB_Wi[-c(1:2)], 1, sum))/ max(segment(dendr)[c("y", "yend")])/3
    TB$ID <- factor(TB$ID, levels= TB$ID[as.numeric(as.character(dendr$labels$label))][order(dendr$labels$x)])
    ggplot() + geom_bar(data=TB, aes(ID, Freq, fill= Behavior), stat= 'identity') +
        scale_fill_manual(values=as.character(CMP_TB$Color[match(sort(unique(TB$Behavior)), CMP_TB$Behavior)])) +
        geom_segment(data=segment(dendr), aes(x=x, y=y*Ratio + Pending*1.02, xend=xend, yend=yend*Ratio+Pending*1.02))+
        coord_flip() + theme_bw() +  labs(y= "Counts", x = "Fly", title = TIITLE)+
        theme(plot.title=element_text(hjust = .5)) + geom_point(data=TB, aes(x=ID, y=Pending*1.01, color=Group)) +
        scale_color_manual(values= as.character(CMP_TB_G$Color[match( unique(TB$Group), CMP_TB_G$Group)]))
    ggsave(paste(paste(Pair_TB[[Pair_id]],  collapse = "_VS_"),'_hclust.png', sep=""), w= 10, h =10)

}


Video_TB <- group_TB[c("Video_Name", "From_frame", "To_Frame", "Group")]
Video_lt <- paste(Video_TB[[1]], Video_TB[[2]], Video_TB[[3]], sep ='_')
Video_lt <- data.frame(Video_lt, Video_TB$Group)
TB <- data.frame()
for (ROW in c(1:nrow(Video_lt))){
    Video = Video_lt[ROW, 1]
    Group = Video_lt[ROW, 2]
    TMP <- read.table(paste("../Correct_", Video, ".csv", sep =""), sep = ",", header = T)
    TMP <- TB_clean(TMP, Behavior)
    TMP <- data.frame(table(TMP[c("Fly_s", "Behavior")]))
    TMP$Group = Group
    TMP$Video = Video
    TB <- rbind(TB, TMP)
}
TB$Group<- factor(TB$Group, levels =  unique(TB$Group))

### Anov-DunTest
P <- Kaboom_bar(TB[2:4], 'Behavior', 'Group',Facet = F, fill = "Group",  P_test="DunTest", Var = "SEM")

P[[1]]+  scale_fill_manual(values= as.character(CMP_TB_G$Color[match( unique(TB$Group), CMP_TB_G$Group)])) +
  ggtitle(paste(TIITLE, "(DunnettTest)")) +
  theme(plot.title = element_text(hjust= .5) )
TMP <-  P[[2]]
TMP$Group <- as.data.frame(str_split_fixed(TMP$Group, "\n", 2))[[1]]
W_adust = (length(unique(TB$Group))-2) * Ratio
W_adust = (length(unique(TB$Group))-2) * Ratio

ggsave("All_Groups_Dunnet.png", w=W + W_adust, h=H)
ggsave("All_Groups_Dunnet.svg", w=W + W_adust, h=H)

write.table(TMP, 'All_Groups_Dunnet.csv', sep = '\t', quote = F)

### Anova-Tukey

P <- Kaboom_bar(TB[2:4], 'Behavior', 'Group',Facet = F, fill = "Group",  P_test="Tukey", Var = "SEM")

P[[1]]+  scale_fill_manual(values= as.character(CMP_TB_G$Color[match( unique(TB$Group), CMP_TB_G$Group)])) +
  ggtitle(paste(TIITLE, "(Tukey Test)")) +
  theme(plot.title = element_text(hjust= .5) )
TMP <-  P[[2]]
TMP$Group <- as.data.frame(str_split_fixed(TMP$Group, "\n", 2))[[1]]
W_adust = (length(unique(TB$Group))-2) * Ratio
W_adust = (length(unique(TB$Group))-2) * Ratio

ggsave("All_Groups_Tukey.png", w=W + W_adust, h=H)
ggsave("All_Groups_Tukey.svg", w=W + W_adust, h=H)

write.table(TMP, 'All_Groups_Stat.csv', sep = '\t', quote = F)
write.table(P[[3]], 'All_Groups_Tukey.csv', sep = '\t', quote = F)




TB$ID <- paste(TB$Video, TB$Fly_s,sep="__")
TB <- TB[c("ID","Group", "Behavior", "Freq")]
TB_Wi <- reshape(TB, idvar = c("ID", "Group"), timevar = c("Behavior"), direction = 'wide')


hc       <- hclust(dist(TB_Wi[-c(1:2)]))           # heirarchal clustering
dendr    <- dendro_data(hc, type="rectangle") # convert for ggplot
clust    <- cutree(hc,k=5)                    # find 2 clusters
clust.df <- data.frame(label=names(clust), cluster=factor(clust))
## dendr[["labels"]] has the labels, merge with clust.df based on label column
dendr[["labels"]] <- merge(dendr[["labels"]],clust.df, by="label")
## plot the dendrogram; note use of color=cluster in geom_text(...)

Pending = max(apply(TB_Wi[-c(1:2)], 1, sum))
Ratio   = max(apply(TB_Wi[-c(1:2)], 1, sum))/ max(segment(dendr)[c("y", "yend")])/3
TB$ID <- factor(TB$ID, levels= TB$ID[as.numeric(as.character(dendr$labels$label))][order(dendr$labels$x)])
ggplot() + geom_bar(data=TB, aes(ID, Freq, fill= Behavior), stat= 'identity') +
    scale_fill_manual(values=as.character(CMP_TB$Color[match(sort(unique(TB$Behavior)), CMP_TB$Behavior)])) +
    geom_segment(data=segment(dendr), aes(x=x, y=y*Ratio + Pending*1.02, xend=xend, yend=yend*Ratio+Pending*1.02))+
    coord_flip() + theme_bw() +  labs(y= "Counts", x = "Fly", title = TIITLE)+
    theme(plot.title=element_text(hjust = .5)) + geom_point(data=TB, aes(x=ID, y=Pending*1.01, color=Group)) +
    scale_color_manual(values= as.character(CMP_TB_G$Color[match( unique(TB$Group), CMP_TB_G$Group)]))
ggsave("All_clust.png", w= 10, h =10)
