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

FILE="##DATA##"
W=5
H=2.4
NAME = str_replace(FILE, "csv", "svg")
NAME2 = str_replace(FILE, "csv", "png")
TB <- read.csv(paste("..", FILE, sep= "/"))[TB_clist]

if(length(levels(TB[[1]]))>12){
    CMP = head(rep(CMP, round(length(levels(TB[[1]]))/12)+1), length(levels(TB[[1]])))
}

P <- ggplot(TB, aes(Fly_s, TB[[2]])) + geom_boxplot() + theme_bw() + ylab(label = TYPE)
ggsave(paste("Nnst",NAME,sep="_" ), w= W, h = H)
ggsave(paste("Nnst",NAME2,sep="_" ), w= W, h = H)

Pc<- ggplot(TB, aes(Fly_s, TB[[2]], fill=Fly_s)) + geom_boxplot() + theme_bw() + scale_fill_manual(values = CMP) + ylab(label = TYPE)
ggsave(paste("Nnst_c",NAME,sep="_" ), w= W, h = H)
ggsave(paste("Nnst_c",NAME2,sep="_" ), w= W, h = H)

TB$Frame<- rep(c(1:length(which(TB$Fly_s==TB$Fly_s[1]))), length(levels(TB$Fly_s)))
TB_W <- reshape(TB, idvar = "Fly_s", timevar =  "Frame", direction = "wide")
row.names(TB_W) = TB_W[[1]]
TB_W <- TB_W[-1]


TB_sd = data.frame(Mean = apply(TB_W, 1, mean), Sd = apply(TB_W, 1, sd), N = apply(TB_W, 1, length))

TB_Stat <- rbind(TB_sd, c(Wei_Mean(TB_sd), NA))
row.names(TB_Stat)[nrow(TB_Stat)]="Weighted"
write.table(TB_Stat, paste("Stat_",FILE, sep=""), sep="\t")
