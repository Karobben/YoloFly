library(ggplot2)
library(ggthemes)
library(RColorBrewer)

colorRampPalette(rev(brewer.pal(n = 7,name = "RdYlBu"))) -> cc

CSV = {CSVFILE}
Fms = {FRAMEFROME}
Fme = {FRAMEEND}
ALPHA = {ALPHA}
OUTPUT = {OUTPUT}
GAP= {GAP}

CMP = c("Grey", "green", "skyblue", "salmon", "Yellow")

# Read file
TB <- read.table(CSV)

TB_sl <- TB[TB[[1]] %in% seq(from = Fms, to = Fme, by = GAP),]
# Remove head point
TB_sl <- TB_sl[TB_sl$V2!=1,]
TB_sl$V2 <- factor(TB_sl$V2 , levels= c(0, 2, 3, 4, 5))
levels(TB_sl$V2) <- c("fly", "groom", "contact", "wing", "Hold")

ggplot(TB_sl, aes(V3, V4, color= V2)) +
    geom_point(aes(color= V2, fill = V2),  shape = 21,
    size = 2, color = "white", alpha = ALPHA)+
    coord_fixed(ratio =9/16) +
    expand_limits(y=c(0,1), x = c(0, 1)) +
    theme_map() + scale_fill_manual(values = CMP)+
    theme(legend.position = c(0.2,0.2), legend.text = element_text(size = 12))

ggsave(paste("Point", OUTPUT, sep = "_"), w = 20, h= 10.9)


ggplot(TB_sl, aes(V3, V4, color= V2))+
    coord_fixed(ratio =9/16) +
    geom_bin2d(bins = 80) + theme_map() +
    expand_limits(y=c(0,1), x = c(0, 1))+
    scale_fill_gradientn(colors=cc(100))+
    theme(legend.position = c(0.2,0.2), legend.text = element_text(size = 12))

ggsave(paste("Dens", OUTPUT, sep = "_"), w = 20, h= 10.9)
