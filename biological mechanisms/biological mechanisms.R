###########GESA
install.packages("BiocManager")

BiocManager::install("enrichplot")
setwd("F:/data/result/")
mata_mat<-read.table("F:/data/result/group.txt", header = T, row.names = 1)
library(DOSE)
library(clusterProfiler)
library(ReactomePA)
exp_mat<-(read.table('F:/data/result/GSE103584_R01_NSCLC_RNAseq.txt',header=T,row.names=1,sep='\t',fill=T))
exp_mat<-exp_mat[which(rowMeans(!is.na(exp_mat)) > 0.5),]

library(limma)
sel<-match(mata_mat$ID,colnames(exp_mat),nomatch = 0)
exp_mat_filt<-exp_mat[,sel]#
mata_mat_filt<-mata_mat[match(colnames(exp_mat_filt),mata_mat$ID,nomatch = 0),]
#EGFR-mutant EGFR-wild-type
class_dif<-ifelse(mata_mat_filt$condition=="EGFR-mutant","High",'Low')
pho_mat<-data.frame(ID=mata_mat_filt$ID,Class=class_dif)

design<-model.matrix(~Class,data = pho_mat)

fit<-lmFit(log2(exp_mat_filt),design)

fit2<-eBayes(fit)

diff_table<-topTable(fit2,coef=2,n=nrow(exp_mat_filt))
geneList<- 0-diff_table$logFC
names(geneList)<-rownames(diff_table)


library(org.Hs.eg.db)
library(enrichplot)
Sym2En<-na.omit(AnnotationDbi::select(org.Hs.eg.db,names(geneList),'ENTREZID','SYMBOL'))
(sel_dup<-which(duplicated(Sym2En$SYMBOL)))
Sym2En<-Sym2En[-sel_dup,]

geneList<-geneList[match(Sym2En$SYMBOL,names(geneList),nomatch=0)]
(all(names(geneList)==Sym2En$SYMBOL))
names(geneList)<-Sym2En$ENTREZID

geneList<-sort(geneList,decreasing = T)

write.table(geneList)

geneList<-2^(geneList)#

gsea_rea<-gsePathway(geneList, organism = "human", exponent = 1, nPerm = 10000,minGSSize = 15, 
                     maxGSSize = 200, pvalueCutoff = 1,pAdjustMethod = "BH", verbose = TRUE, seed = FALSE, by = "fgsea")
gsea_table<-gsea_rea@result
class <- as.numeric(gsea_rea$NES <  0)
up <- gsea_table[gsea_table$NES > 0,]
down <- gsea_table[gsea_table$NES < 0,]
#gsea_rea$dataframe

write.csv(gsea_table,'reactome.csv')
write.csv(up,'up.csv')
write.csv(down,'down.csv')
edo<-read.csv('reactome.csv')


library(ggplot2)
library(egg)
library(stringr)
reactome<-read.csv('/reactome.csv')
up<-read.csv('/up.csv')
down<-read.csv('/down.csv')
up$Description = factor(up$Description,ordered = T)#
up1<-up[1:15,]
write.csv(up1,'up1.csv')
up1<-read.csv('up1.csv')
p1<-ggplot(up1,aes(x = NES,y = Description), showCategory = 15)+
  #aes(y=Description)+
  geom_point(aes(color =pvalue,
                 size = setSize))+
  scale_size_continuous(range=c(4,8))+
  scale_color_gradient(low = "#DD4714", high = "#A61650")+ #####c("#598979", "#4EB043", "#E69D2A", "#DD4714", "#A61650")
  xlab("NES")+
  scale_x_continuous(limits = c(0,4))+
  ylab("Pathway types") +
  scale_y_discrete(labels = function(y)str_wrap(y, width = 40))+
  theme_bw()+
  theme(axis.title = element_text(
      face='bold',）
      size=12, 
      lineheight = 1),
      axis.text = element_text(
      face="bold", 
      size=12))
p1
ggsave("up.png",width = 13,height = 6.5, egg::set_panel_size(p1, width=unit(5, "in"), height=unit(5, "in")),units = "in", dpi = 900)
p1
#dev.off()
down$Description = factor(down$Description,ordered = T)#
down1<-down[1:15,]
write.csv(down1,'down1.csv')
down1<-read.csv('down1.csv')

options(repr.plot.width = 10, repr.plot.height =10)
p2<-ggplot(down1,aes(x = NES,y = Description), showCategory = 15)+
  geom_point(aes(color =pvalue,size = setSize))+#color =pvalue,
  scale_size_continuous(range=c(3,8))+
  scale_color_gradient(low = "#4EB043", high = "#598979")+ #####c("#598979", "#4EB043", "#E69D2A", "#DD4714", "#A61650")
  xlab("NES")+
  ylab("Pathway types") + 
  scale_x_continuous(limits = c(-4,0))+
  scale_y_discrete(position = "right",labels = function(y)str_wrap(y, width = 40))+
  theme_bw()+#主题
  theme(legend.position = "left") +
  theme(axis.title = element_text(
      face='bold',）
      size=12, 
      lineheight = 1),
      axis.text = element_text(
      face="bold", 
      size=12))
ggsave("down.png",width = 16,height = 6.5, egg::set_panel_size(p2, width=unit(5, "in"), height=unit(5, "in")),units = "in", dpi = 600) 
dev.off()
############################################
label_data = data.frame(A=c(0,100,200,300,400,500,600,700,800,900,1000,1100),B=c(0,1,2,3,4,5,0,1,2,3,4,5))
pdf('result.pdf',width = 13,height = 6.5)
p_result = ggplot(reactome,aes(x=gene.sets,y=pvalue,fill=NES))+
  geom_bar(stat = 'identity', position = 'dodge', 
           width = 1)+  
  scale_fill_gradientn(colours =c("#598979", "#4EB043", "#E69D2A", "#DD4714", "#A61650"),limits = c(-3, 3))+ 
  xlab("Gene sets(ranked in ordered list)")+ 
  ylab("Ranked metric P value")+
  scale_x_continuous(breaks=label_data$A, labels = label_data$B*100,guide = guide_axis(position = "top"))+
  theme_bw()+
  theme(axis.title = element_text(
    face='bold', 
    size=12, 
    lineheight = 1),
    axis.text = element_text(
      face="bold", 
      size=12))
p_result
dev.off()


############################### ssGSEA ###############################
library(dplyr)
library(tibble)
library(GSVA)
library(Hmisc)
library(pheatmap)

geneSet <- read.csv("CellReports.csv",header = F,sep = ",") # 用EXCEL打开删除NA列
class(geneSet)

geneSet <- geneSet %>%
  column_to_rownames("V1")%>%t()
a <- geneSet
a <- a[1:nrow(a),]
set <- colnames(a)
print(set)
l <- list()
#i <- "Activated CD8 T cell"
for (i in set) {
  x <- as.character(a[,i])
  x <- x[nchar(x)!=0]
  x <- as.character(x)
  l[[i]] <- x
}
print(l)
save(l,file = "gene_set.Rdata")

rm(list=ls())
library(GSVA)
load(file = "gene_set.Rdata")
exp <- read.csv("GSE103584_R01_NSCLC_RNAseq_selected.csv",row.names = 1)
# exp <- read.table("GSE103584_R01_NSCLC_RNAseq.txt",sep = "\t",check.names = F,stringsAsFactors = F,header = T,row.names = 1)   
exp <- as.matrix(exp)   
exp <- na.omit(exp)
write.csv(exp, "RNA_seq_dropna.csv")
                   
# start ssGSEA
gsva_data <- gsva(exp, l, method='ssgsea')
a <- gsva_data %>% t() %>% as.data.frame()
write.csv(a,"ssGSEA.csv")
mygroup1 <- read.csv("group.csv")
row.names(mygroup1) <- mygroup1[,1]
mygroup <- mygroup1[,-1]   

### New
# identical(rownames(a),rownames(mygroup))
# a$group <- mygroup$group
# a <- a %>% rownames_to_column("sample")
#write.table(a,"ssGSEA.txt",sep = "\t",row.names = T,col.names = NA,quote = F)
# b <- gather(a,key=ssGSEA,value = Expression,-c(group,sample))
                   
# mygene <- c("R01.027", "R01.034")  
# nc = t(rbind(a[mygene,]))  
# m = rcorr(nc)$r[1:nrow(gsva_data),(ncol(nc)-length(mygene)+1):ncol(nc)]
# p = rcorr(nc)$P[1:nrow(gsva_data),(ncol(nc)-length(mygene)+1):ncol(nc)]
library(tinyarray)
draw_boxplot(gsva_data, mygroup, xlab='Cell', pvalue_cutoff=0.05, color=c("#e5171a", "#1d4a9b"))

library(ggplot2)
library(ggpubr)
library(limma)
library(ggsignif)

gsva_data11 <- t(gsva_data)
data <- data.frame(gsva_data11, group = mygroup1[,2])
write.csv(data,"ggplot_data_cc.csv")
data <- read.csv("ggplot_data.csv")
p_test <- compare_means(Score ~ group1, data = data, group.by="Cell", method="t.test")
tiff("./tiff1/ssGSEA.tiff", bg = "green", width = 50, height = 18, units = "cm", pointsize = 18, res = 300, family = "serif")
ggplot(data, aes(x = Cell, y = Score)) +
  geom_boxplot(aes(fill = group1),position=position_dodge(0.5),width=0.6) +
  labs(x = 'Cell', y = 'Expression') + 
  scale_fill_manual(values = c("#e5171a", "#1d4a9b"))+
  theme(axis.text.x = element_text(angle = 60, hjust = 1, vjust = 1))  +
  geom_signif(y_position=c(0.4), xmin=c(4.85,3.85, 1.85, 6.85, 9.85, 15.85), xmax=c(5.15, 4.15, 2.15, 7.15, 10.15, 16.15), annotation=c("***", "**","*","*","*","*"), tip_length=0.04)
  # stat_compare_means(label = "p.signif", method = "t.test")
dev.off()


############################### GO/KEGG ###############################     
library(tidyverse)
library("BiocManager")
library(clusterProfiler)
library(readxl)
library(DESeq2)

#In this file, We take intersection and union of DEGs.
# exp <- read.csv("GSE103584_R01_NSCLC_RNAseq_selected.csv",row.names = 1)
mata_mat <- read.csv("model_score.csv")
library(DOSE)
library(clusterProfiler)
library(ReactomePA)
exp_mat<-(read.table('GSE103584_R01_NSCLC_RNAseq.txt',header=T,row.names=1,sep='\t',fill=T))
exp_mat<-exp_mat[which(rowMeans(!is.na(exp_mat)) > 0.5),]

#colname_exp<-paste0('R',substr(colnames(exp_mat),5,7))
#colnames(exp_mat)<-colname_exp
library(limma)
sel<-match(mata_mat$name,colnames(exp_mat),nomatch = 0)
exp_mat_filt<-exp_mat[,sel]#
mata_mat_filt<-mata_mat[match(colnames(exp_mat_filt),mata_mat$name,nomatch = 0),]
#EGFR-mutant EGFR-wild-type
class_dif<-ifelse(mata_mat_filt$status=="Mutant","High",'Low')
pho_mat<-data.frame(ID=mata_mat_filt$name,Class=class_dif)
design<-model.matrix(~Class,data = pho_mat)
fit<-lmFit(log2(exp_mat_filt),design)
fit2<-eBayes(fit)
diff_table<-topTable(fit2,coef=2,n=nrow(exp_mat_filt))

diff_table_cc <- diff_table[order(abs(diff_table$logFC)), ]
write.csv(diff_table_cc,'diff_table_cc.csv')
condition <- abs(diff_table$logFC) > 0.5 & diff_table$P.Value < 0.05
filtered_data <- diff_table[condition, ]
# geneList<- 0-diff_table$logFC
# names(geneList)<-rownames(diff_table)
geneList<- 0-filtered_data$logFC
names(geneList)<-rownames(filtered_data)
# geneList <- na.omit(exp)
library("BiocManager")
library(org.Hs.eg.db)
library(enrichplot)
library(ReactomePA)
keys(org.Hs.eg.db, keytype = "SYMBOL")
Sym2En<-na.omit(AnnotationDbi::select(org.Hs.eg.db,names(geneList),'ENTREZID',keytype = "SYMBOL"))
(sel_dup<-which(duplicated(Sym2En$SYMBOL)))
Sym2En<-Sym2En[-sel_dup,]
geneList<-geneList[match(Sym2En$SYMBOL,names(geneList),nomatch=0)]
(all(names(geneList)==Sym2En$SYMBOL))
names(geneList)<-Sym2En$ENTREZID
#############################
write.table(geneList)
geneList<-2^(geneList)#
cc <- list(Sym2En$SYMBOL)
write.csv(cc,'cc.csv')

gsea_rea<-gsePathway(geneList, organism = "human", exponent = 1, nPerm = 10000,minGSSize = 15, 
                     maxGSSize = 200, pvalueCutoff = 1,pAdjustMethod = "BH", verbose = TRUE, seed = FALSE, by = "fgsea")
gsea_table<-gsea_rea@result
class <- as.numeric(gsea_rea$NES <  0)
up <- gsea_table[gsea_table$NES > 0,]
down <- gsea_table[gsea_table$NES < 0,]
#gsea_rea$dataframe

write.csv(gsea_table,'reactome.csv')
write.csv(up,'up.csv')
write.csv(down,'down.csv')
edo<-read.csv('reactome.csv')

p<-gseaplot2(gsea_rea,geneSetID = 2,title = gsea_rea$Description[2])
p2<-dotplot(down,showCategory=10)
p3<-barplot(as.matrix(gsea_table),showCategory=10)

data(geneList)
de <- names(geneList)[abs(geneList) > 2]
df <- data.frame(ENTREZID = c(de))
SYMBOL <- AnnotationDbi::select(org.Hs.eg.db, keys = df$ENTREZID, columns = "SYMBOL", keytype = "ENTREZID")
result_df <- cbind(df, SYMBOL)
write.csv(result_df,'diff_gene.csv')
################################################### start GO
ego <- enrichGO(gene = de,
                OrgDb = org.Hs.eg.db, 
                ont = "all",
                pAdjustMethod = "BH",
                minGSSize = 1,
                pvalueCutoff =0.05, 
                readable = TRUE)
ego_res <- ego@result
save(ego,ego_res,file = "GO_radiomics_DEG.Rdata")
dotplot(ego, showCategory = 10)
load('GO_radiomics_DEG.Rdata')
# ego_res <- ego_res[order(-ego_res[, 10]), ]
ego_res$GeneRatio <- sapply(strsplit(as.character(ego_res$GeneRatio), "/"), function(x) as.numeric(x[1])/as.numeric(x[2]))
data_res <- ego_res
data_go <- data_res[order(data_res[, 7]), ][1:100,]
write.csv(data_go,"data_go.csv")
ggplot(data_go,aes(x = GeneRatio,fct_reorder(factor(Description), GeneRatio)))+
  geom_point(aes(color = p.adjust, size = Count))+
  scale_color_gradient(low = "red", high = "blue")+
  # labs(color=expression(-log[10](p_value)),size="Count") +
  xlab("GeneRatio") + ylab("") + 
  theme_bw()  
  # scale_x_discrete(limits = c("0.01", "0.02", "0.03", "0.04", "0.05"))

tiff("./tiff_1/GOcc.tiff", bg = "green", width = 20, height = 14, units = "cm", pointsize = 18, res = 300, family = "serif")
data_go <- read.csv("data_go.csv")
ggplot(data_go,aes(x = GeneRatio,fct_reorder(factor(Description), GeneRatio)))+
  geom_point(aes(color = p.adjust, size = Count))+
  scale_color_gradient(low = "red", high = "blue")+
  # labs(color=expression(-log[10](p_value)),size="Count") +
  xlab("GeneRatio") + ylab("") + scale_y_discrete(labels = function(Description) str_wrap(Description, width = 35)) +
  theme_bw() + theme(text = element_text(family = "Arial", size = 18))
dev.off()
                            
############################################################ start KEGG 
kk <- enrichKEGG(gene         = de,
                 organism     = 'hsa',
                 pvalueCutoff = 0.05)
kk2<-setReadable(kk, OrgDb = org.Hs.eg.db, keyType="ENTREZID")
kk_res <- kk2@result

#save(kk,kk_res,file = "KEGG_radiomics_DEG.Rdata")
dotplot(kk, showCategory = 10)
kk_res$GeneRatio <- sapply(strsplit(as.character(kk_res$GeneRatio), "/"), function(x) as.numeric(x[1])/as.numeric(x[2]))
data_kk <- kk_res[order(kk_res[, 7]), ][1:100,]
write.csv(data_kk,"data_kk.csv")
ggplot(data_kk,aes(x = GeneRatio,fct_reorder(factor(Description), GeneRatio)))+
  geom_point(aes(color = p.adjust, size = Count))+
  scale_color_gradient(low = "red", high = "blue")+
  # labs(color=expression(-log[10](p_value)),size="Count") +
  xlab("GeneRatio")+ ylab("") + scale_y_discrete(labels = function(Description) str_wrap(Description, width = 30)) +
  theme_bw()  

tiff("./tiff_1/KEGG.tiff", bg = "green", width = 20, height = 14, units = "cm", pointsize = 18, res = 300, family = "serif")
data_kk <- read.csv("data_kk.csv")
ggplot(data_kk,aes(x = GeneRatio,fct_reorder(factor(Description), GeneRatio)))+
  geom_point(aes(color = p.adjust, size = Count))+
  scale_color_gradient(low = "red", high = "blue")+
  # labs(color=expression(-log[10](p_value)),size="Count") +
  xlab("GeneRatio")+ ylab("") + scale_y_discrete(labels = function(Description) str_wrap(Description, width = 30)) +
  theme_bw() + theme(text = element_text(family = "Arial", size = 18))
dev.off()
                                                 

                                                 
############################### GlueGO ###############################     
gluego <- read.csv("ClueGOResultTable.csv")
# gluego1 <- gluego[order(gluego[, 3]), ][1:55,]
gluego1 <- gluego[1:180,]
(gluego1_dup<-which(duplicated(gluego1$Term)))
gluego1<-gluego1[-gluego1_dup,]
# write.csv(gluego1, 'GlueGo.csv')
library(RColorBrewer)
library(ggpubr)
gluegocc <- read.csv("GlueGo1.csv")
tiff("./tiff_1/GlueGOcc.tiff", bg = "green", width = 35, height = 18, units = "cm", pointsize = 18, res = 300, family = "serif")
colourCount = length(unique(gluegocc$GOGroups))
getPalette = colorRampPalette(brewer.pal(9,'Paired'))
ggbarplot(gluegocc,
          x="Term", y="Count", fill = "GOGroups", color = "white", 
          orientation = "horiz",   
           #palette = "nejm",        #npg，aaas，jama，jco,nejm
          legend = "none",        
          sort.val = "none",       
          sort.by.groups=TRUE)  + # geom_histogram(fill=getPalette(colourCount)) + 
          geom_text(aes(label = '*'), x = 1, y = 48.5, size = 6.5, color = 'red') + 
          geom_text(aes(label = '*'), x = 2, y = 38.5, size = 6.5, color = 'red') + 
          geom_text(aes(label = '*'), x = 10, y = 40.5, size = 6.5, color = 'red') + 
          geom_text(aes(label = '*'), x = 11, y = 39.5, size = 6.5, color = 'red') + labs(x = 'Term', y = 'Count', fill=NULL) + 
          # geom_signif(y_position=c(48), x_position=c(5.0,6.0, 7.0, 8.0, 9.0,10.0), annotation=c("*", "*","*","*","*","*"),color='red') + 
        scale_fill_manual(values = getPalette(colourCount)) + 
          theme(legend.position="none") + geom_text(aes(label = gluegocc$Count),size = 6.5, hjust = -0.5) + 
         scale_y_continuous(limits=c(0, 51), expand=c(0,0)) + theme_bw() + theme(text = element_text(family = "Arial", size = 18), legend.text = element_blank(), legend.position = "none")#  + ylim(0, 50)
dev.off()                                             
