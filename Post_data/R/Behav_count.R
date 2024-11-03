library(RColorBrewer)
library(ggplot2)
library(ggdendro)

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

File = "##DATA##"
TB <- read.table(paste("../../Video_post", File, sep="/"), sep=",", head = T)
TB <- TB_clean(TB)
TB_matrix <- as.data.frame.array(table(TB[c("Fly_s", "Behavior")]))
TB_hclust <- hclust(dist(t(scale(t(TB_matrix)))))
TB$Fly_s<- factor(TB$Fly_s, levels =  TB_hclust$labels[TB_hclust$order])



hc       <- hclust(dist(t(scale(t(TB_matrix)))))           # heirarchal clustering
dendr    <- dendro_data(hc, type="rectangle") # convert for ggplot
clust    <- cutree(hc,k=5)                    # find 2 clusters
clust.df <- data.frame(label=names(clust), cluster=factor(clust))
## dendr[["labels"]] has the labels, merge with clust.df based on label column
dendr[["labels"]] <- merge(dendr[["labels"]],clust.df, by="label")
## plot the dendrogram; note use of color=cluster in geom_text(...)

Pending = max(table(TB$Fly_s))
Ratio   = max(table(TB$Fly_s))/ max(segment(dendr)[c("y", "yend")])/3
ggplot() + geom_bar(data=TB, aes(Fly_s, fill= Behavior)) +
    scale_fill_manual(values=as.character(CMP_TB$Color[match(sort(unique(TB$Behavior)), CMP_TB$Behavior)])) +
    geom_segment(data=segment(dendr), aes(x=x, y=y*Ratio + Pending, xend=xend, yend=yend*Ratio+Pending))+
    coord_flip() + theme_bw() +  labs(y= "Counts", x = "Fly", title = TIITLE)+
    theme(plot.title=element_text(hjust = .5))


W= 3.03
H= 2.55
if (length(unique(TB$Fly_s)) > 10){
    W= W + ((5.39-3.03)/30) * (length(unique(TB$Fly_s)) - 10)
    H= H + ((6.92-2.55)/30) * (length(unique(TB$Fly_s)) - 10)
}
ggsave(paste(File,".svg", sep =""), w= W, h= H)
ggsave(paste(File,".png", sep =""), w= W, h= H)

Mean = apply(as.data.frame.array(table(TB[c("Fly_s", "Behavior")])), 2, mean)
Sd = apply(as.data.frame.array(table(TB[c("Fly_s", "Behavior")])), 2, sd)
N = apply(as.data.frame.array(table(TB[c("Fly_s", "Behavior")])), 2, length)
write.table(data.frame(Mean, Sd, N), File, sep = "\t", quote = F)
