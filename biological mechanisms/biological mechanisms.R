###########GESA
install.packages("BiocManager")

BiocManager::install("enrichplot")
setwd("F:/data/img实验结果/生物信息分析")
mata_mat<-read.table("F:/data/img实验结果/生物信息分析/分组2.txt", header = T, row.names = 1)#分组文件
library(DOSE)
library(clusterProfiler)
library(ReactomePA)
exp_mat<-(read.table('F:/data/img实验结果/生物信息分析/GSE103584_R01_NSCLC_RNAseq.txt',header=T,row.names=1,sep='\t',fill=T))#基因文件，剔除空
exp_mat<-exp_mat[which(rowMeans(!is.na(exp_mat)) > 0.5),]#选择删除空白行大于50%的行

library(limma)
sel<-match(mata_mat$ID,colnames(exp_mat),nomatch = 0)#从基因值中选出对应
exp_mat_filt<-exp_mat[,sel]#
mata_mat_filt<-mata_mat[match(colnames(exp_mat_filt),mata_mat$ID,nomatch = 0),]
#EGFR-mutant EGFR-wild-type
class_dif<-ifelse(mata_mat_filt$condition=="EGFR-mutant","High",'Low')
pho_mat<-data.frame(ID=mata_mat_filt$ID,Class=class_dif)
# 1.构建比较矩阵
design<-model.matrix(~Class,data = pho_mat)
##2.线性模型拟合
fit<-lmFit(log2(exp_mat_filt),design)
#3.贝叶斯检验
fit2<-eBayes(fit)
#差异表达基因计算完成
diff_table<-topTable(fit2,coef=2,n=nrow(exp_mat_filt))#topTable为给定的分类或回归过程提取前n个最重要的特征
geneList<- 0-diff_table$logFC
names(geneList)<-rownames(diff_table)

#富集
library(org.Hs.eg.db)
library(enrichplot)
Sym2En<-na.omit(AnnotationDbi::select(org.Hs.eg.db,names(geneList),'ENTREZID','SYMBOL'))#ID和基因对应
(sel_dup<-which(duplicated(Sym2En$SYMBOL)))#看是否有重复的
Sym2En<-Sym2En[-sel_dup,]

geneList<-geneList[match(Sym2En$SYMBOL,names(geneList),nomatch=0)]
(all(names(geneList)==Sym2En$SYMBOL))
names(geneList)<-Sym2En$ENTREZID

geneList<-sort(geneList,decreasing = T)#排序

write.table(geneList)

geneList<-2^(geneList)#

gsea_rea<-gsePathway(geneList, organism = "human", exponent = 1, nPerm = 10000,minGSSize = 15, 
                     maxGSSize = 200, pvalueCutoff = 1,pAdjustMethod = "BH", verbose = TRUE, seed = FALSE, by = "fgsea")
gsea_table<-gsea_rea@result
class <- as.numeric(gsea_rea$NES <  0)#将因子转换为数字
up <- gsea_table[gsea_table$NES > 0,]
down <- gsea_table[gsea_table$NES < 0,]
#gsea_rea$dataframe

write.csv(gsea_table,'reactome.csv')
write.csv(up,'up.csv')
write.csv(down,'down.csv')
edo<-read.csv('reactome.csv')

p<-gseaplot2(gsea_rea,geneSetID = 2,title = gsea_rea$Description[2])
p
p2<-dotplot(down,showCategory=10)
p2

p3<-barplot(as.matrix(gsea_table),showCategory=10)
p3


data(geneList)
de <- names(geneList)[abs(geneList) > 2]
edo <- enrichDGN(de)
write.csv(geneList,'geneList.csv')
library(enrichplot)
barplot(edo, showCategory=20)

library(ggplot2)
library(egg)
library(stringr)
reactome<-read.csv('富集分析/reactome.csv')
up<-read.csv('富集分析/up.csv')
down<-read.csv('富集分析/down.csv')
up$Description = factor(up$Description,ordered = T)#
up1<-up[1:15,]#选择前15个
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
  theme_bw()+#主题
  theme(axis.title = element_text(
      face='bold', ##字体外形（粗斜体等）
      size=12, ##字体大小
      lineheight = 1),##标签行间距的倍数
      axis.text = element_text(
      face="bold", ##字体外形（粗斜体等）
      size=12))
p1
ggsave("up.png",width = 13,height = 6.5, egg::set_panel_size(p1, width=unit(5, "in"), height=unit(5, "in")),units = "in", dpi = 900)
p1
#dev.off()
#
down$Description = factor(down$Description,ordered = T)#
down1<-down[1:15,]
write.csv(down1,'down1.csv')
down1<-read.csv('down1.csv')
#下调中将名字放到右边，图例放到左边
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
    face='bold', ##字体外形（粗斜体等）
    size=12, ##字体大小
    lineheight = 1),##标签行间距的倍数
    axis.text = element_text(
      face="bold", ##字体外形（粗斜体等）
      size=12))
ggsave("down.png",width = 16,height = 6.5, egg::set_panel_size(p2, width=unit(5, "in"), height=unit(5, "in")),units = "in", dpi = 600)  ##ggplot 中直接保存
p2
dev.off()
############################################
#实验结果图表1：
label_data = data.frame(A=c(0,100,200,300,400,500,600,700,800,900,1000,1100),B=c(0,1,2,3,4,5,0,1,2,3,4,5))
pdf('result.pdf',width = 13,height = 6.5)
p_result = ggplot(reactome,aes(x=gene.sets,y=pvalue,fill=NES))+
  geom_bar(stat = 'identity', position = 'dodge', 
           width = 1)+  
  scale_fill_gradientn(colours =c("#598979", "#4EB043", "#E69D2A", "#DD4714", "#A61650"),limits = c(-3, 3))+ ##多颜色
  xlab("Gene sets(ranked in ordered list)")+ 
  ylab("Ranked metric P value")+
  scale_x_continuous(breaks=label_data$A, labels = label_data$B*100,guide = guide_axis(position = "top"))+#坐标轴在上面
  theme_bw()+
  theme(axis.title = element_text(
    face='bold', ##字体外形（粗斜体等）
    size=12, ##字体大小
    lineheight = 1),##标签行间距的倍数
    axis.text = element_text(
      face="bold", ##字体外形（粗斜体等）
      size=12))
p_result
dev.off()
