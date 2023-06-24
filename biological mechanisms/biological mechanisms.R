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

#colname_exp<-paste0('R',substr(colnames(exp_mat),5,7))#提取第一行名
#colnames(exp_mat)<-colname_exp#列名字替换
#差异基因表达
library(limma)
sel<-match(mata_mat$ID,colnames(exp_mat),nomatch = 0)#从基因值中选出对应
exp_mat_filt<-exp_mat[,sel]#
mata_mat_filt<-mata_mat[match(colnames(exp_mat_filt),mata_mat$ID,nomatch = 0),]
#EGFR-mutant EGFR-wild-type
class_dif<-ifelse(mata_mat_filt$condition=="EGFR-mutant","High",'Low')
pho_mat<-data.frame(ID=mata_mat_filt$ID,Class=class_dif)
# 1.构建比较矩阵
design<-model.matrix(~Class,data = pho_mat)
##2.线性模型拟合as.matrix(log2(exp_mat_filt)#在给定一系列阵列的情况下，拟合每个基因的线性模型
fit<-lmFit(log2(exp_mat_filt),design)
#3.贝叶斯检验
fit2<-eBayes(fit)
#差异表达基因计算完成
diff_table<-topTable(fit2,coef=2,n=nrow(exp_mat_filt))#topTable为给定的分类或回归过程提取前n个最重要的特征
geneList<- 0-diff_table$logFC
names(geneList)<-rownames(diff_table)
## 导出所有的差异结果
#nrDEG = na.omit(diff_table) ## 去掉数据中有NA的行或列
#diffsig <- nrDEG  
#write.csv(diffsig, "all.limmaOut.csv")
#foldChange = 1
#padj = 0.05
## 筛选出所有差异基因的结果
#All_diffSig <- diffsig[(diffsig$P.Value < padj & (diffsig$logFC>foldChange | diffsig$logFC < (-foldChange))),]

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

library(ggplot2)#reactome.csv
library(egg)
library(stringr)
reactome<-read.csv('0418/up - 副本.csv')
up<-read.csv('富集分析/up.csv')
down<-read.csv('富集分析/down.csv')
up$Description = factor(up$Description,ordered = T)#
up1<-up[1:15,]
write.csv(up1,'up1.csv')
up1<-read.csv('up1.csv')
#a<-factor(up$Description)
#up$Description<-levels(factor(up$Description))
#up1$Description<-factor(up1$Description,levels = c("Cap-dependent Translation Initiation","Eukaryotic Translation Initiation","rRNA processing","NMD independent of the Exon Junction Complex (EJC)",
#                                                   "L13a-mediated translational silencing of Ceruloplasmin expression","Selenoamino acid metabolism","rRNA processing in the nucleus and cytosol",
#                                                   "Major pathway of rRNA processing in the nucleolus and cytosol","Eukaryotic Translation Termination","Selenocysteine synthesis","Formation of a pool of free 40S subunits",
#                                                   "Eukaryotic Translation Elongation","Response of EIF2AK4 (GCN2) to amino acid deficiency","Peptide chain elongation","Viral mRNA Translation"))
#up1$Description<-factor(up1$Description,levels = c("Toll Like Receptor 7/8 (TLR7/8) Cascade","MyD88 cascade initiated on plasma membrane","Toll Like Receptor 5 (TLR5) Cascade","Toll Like Receptor 10 (TLR10) Cascade","Toll Like Receptor 2 (TLR2) Cascade","Toll Like Receptor TLR6:TLR2 Cascade",
#                                                   "Toll Like Receptor TLR1:TLR2 Cascade","MyD88:MAL(TIRAP) cascade initiated on plasma membrane","Toll Like Receptor 4 (TLR4) Cascade","Metabolism of water-soluble vitamins and cofactors","Interleukin-17 signaling",
#                                                   "Heparan sulfate/heparin (HS-GAG) metabolism","MAP kinase activation","MAPK targets/ Nuclear events mediated by MAP kinases","Metabolism of vitamins and cofactors"))
up1$Description<-factor(up1$Description,levels = c("Metabolism of water-soluble vitamins and cofactors","Tight junction interactions","Peroxisomal lipid metabolism","SLC-mediated transmembrane transport","Glycosaminoglycan metabolism","Muscle contraction","Downregulation of ERBB2 signaling",
                                                   "Fatty acid metabolism","Diseases of metabolism","Phase 0 - rapid depolarisation","O-linked glycosylation","Transport of bile salts and organic acids, metal ions and amine compounds","Interferon alpha/beta signaling","O-linked glycosylation of mucins","Surfactant metabolism"))
#pdf('上调.pdf',width = 10,height = 6.5)
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
      #family = Fon,##坐标轴标签字体
      face='bold', ##字体外形（粗斜体等）
      size=12, ##字体大小
      lineheight = 1),##标签行间距的倍数
      axis.text = element_text(
      #family = Fon,##字体
      face="bold", ##字体外形（粗斜体等）
      #color="blue",
      size=12))
  #theme(axis.title.x = element_text(vjust = 1, hjust = 0.5, angle = 30))+
  #labs(subtitle="Pathway types")+
  #theme(plot.subtitle = element_text(hjust =0.5))+
  
  #facet_grid(class~.,scales = "free_y")
p1
ggsave("p1_01.png",width = 13,height = 6.5, egg::set_panel_size(p1, width=unit(5, "in"), height=unit(5, "in")),units = "in", dpi = 900)
p1
#dev.off()
#
down$Description = factor(down$Description,ordered = T)#
down1<-down[1:15,]
write.csv(down1,'down1.csv')
down1<-read.csv('down1.csv')
#a<-factor(up$Description)
#up$Description<-levels(factor(up$Description))
#down1$Description<-factor(down1$Description,levels = c("Defects in vitamin and cofactor metabolism","Interferon alpha/beta signaling","Fatty acid metabolism","Fatty acyl-CoA biosynthesis","Peroxisomal lipid metabolism","Surfactant metabolism","MAPK targets/ Nuclear events mediated by MAP kinases",
#                                                       "O-linked glycosylation of mucins","ERK/MAPK targets","Transport of bile salts and organic acids, metal ions and amine compounds","IRE1alpha activates chaperones","Tight junction interactions","Downregulation of ERBB2 signaling",
#                                                       "EGR2 and SOX10-mediated initiation of Schwann cell myelination","XBP1(S) activates chaperone genes"))
#down1$Description<-factor(down1$Description,levels = c("rRNA processing in the nucleus and cytosol","DNA Replication","Major pathway of rRNA processing in the nucleolus and cytosol","Synthesis of DNA","DNA Replication Pre-Initiation","Assembly of the pre-replicative complex","Separation of Sister Chromatids",
#                                                       "Switching of origins to a post-replicative state","UCH proteinases","Orc1 removal from chromatin","rRNA modification in the nucleus and cytosol","Response of EIF2AK4 (GCN2) to amino acid deficiency","Influenza Infection","G2/M Checkpoints","L13a-mediated translational silencing of Ceruloplasmin expression"))
down1$Description<-factor(down1$Description,levels = c("Peptide chain elongation","Viral mRNA Translation","Eukaryotic Translation Elongation","Selenocysteine synthesis","Eukaryotic Translation Termination","Activation of APC/C and APC/C:Cdc20 mediated degradation of mitotic proteins","Chromosome Maintenance","Assembly of the pre-replicative complex",
                                                       "APC/C:Cdc20 mediated degradation of mitotic proteins","Switching of origins to a post-replicative state","tRNA processing in the nucleus","Orc1 removal from chromatin","APC/C:Cdh1 mediated degradation of Cdc20 and other APC/C:Cdh1 targeted proteins in late mitosis/early G1","UCH proteinases","Metabolism of polyamines"))
pdf('下调.pdf',width = 13,height = 6.5)
#下调中将名字放到右边，图例放到左边
options(repr.plot.width = 10, repr.plot.height =10)
p2<-ggplot(down1,aes(x = NES,y = Description), showCategory = 15)+
  #aes(y=Description)+
  geom_point(aes(color =pvalue,size = setSize))+#color =pvalue,
  scale_size_continuous(range=c(3,8))+
  scale_color_gradient(low = "#4EB043", high = "#598979")+ #####c("#598979", "#4EB043", "#E69D2A", "#DD4714", "#A61650")
  xlab("NES")+
  ylab("Pathway types") + 
  #par(c(5,10))+
  scale_x_continuous(limits = c(-4,0))+
  #scale_y_discrete(labels = function(x) str_wrap(x, width = 40) )+
  scale_y_discrete(position = "right",labels = function(y)str_wrap(y, width = 40))+
  theme_bw()+#主题
  theme(legend.position = "left") +
  #theme(axis.title.y = element_text(size = 12, face = 'bold'))+
  theme(axis.title = element_text(
    #family = Fon,##坐标轴标签字体
    face='bold', ##字体外形（粗斜体等）
    size=12, ##字体大小
    lineheight = 1),##标签行间距的倍数
    axis.text = element_text(
      #family = Fon,##字体
      face="bold", ##字体外形（粗斜体等）
      #color="blue",
      size=12))
#facet_grid(class~.,scales = "free_y")
ggsave("p2_00.png",width = 16,height = 6.5, egg::set_panel_size(p2, width=unit(5, "in"), height=unit(5, "in")),units = "in", dpi = 600)  ##ggplot 中直接保存
ggsave("p2.pdf",width = 13,height = 6.5,egg::set_panel_size(p2, width=unit(5, "in"), height=unit(5, "in")),family="GB1")
p2
dev.off()
#up2 = up[(order(up$NES)),]
############################################
#实验结果图表1：
#label_data = data.frame(A=c(0,100,200,300,400,500,600,700,800,900),B=c(0,1,2,3,0,1,2,3,4,5))
label_data = data.frame(A=c(0,100,200,300,400,500,600,700,800,900,1000,1100),B=c(0,1,2,3,4,5,0,1,2,3,4,5))
pdf('result3.pdf',width = 13,height = 6.5)
p_result = ggplot(reactome,aes(x=gene.sets,y=pvalue,fill=NES))+#x=setSize,y=p.adjust,fill=NES
  geom_bar(stat = 'identity', position = 'dodge', 
           width = 1)+  
  #scale_fill_gradientn(low = "aquamarine",mid = "orangered",high ="brown" )+#划分了多个结果values =c(-4,-2,0,1.5,3)
  scale_fill_gradientn(colours =c("#598979", "#4EB043", "#E69D2A", "#DD4714", "#A61650"),limits = c(-3, 3))+ ##多颜色
  #geom_text(aes(label=Value),size=4,
  #          position = position_dodge(width = 0.8), 
  #          vjust=-0.3)+ 
  xlab("Gene sets(ranked in ordered list)")+ 
  ylab("Ranked metric P value")+
  #scale_x_continuous(breaks=label_data$A, labels = label_data$A*100)+#坐标轴显示false discovery rate
  scale_x_continuous(breaks=label_data$A, labels = label_data$B*100,guide = guide_axis(position = "top"))+#坐标轴在上面
  theme_bw()+
  theme(axis.title = element_text(
    #family = Fon,##坐标轴标签字体
    face='bold', ##字体外形（粗斜体等）
    size=12, ##字体大小
    lineheight = 1),##标签行间距的倍数
    axis.text = element_text(
      #family = Fon,##字体
      face="bold", ##字体外形（粗斜体等）
      #color="blue",
      size=12))
  #theme(axis.text = element_text(colour = 'black'))+breaks=seq(0, 1100, 100),guide = guide_axis(position = "top"),
p_result
dev.off()
#p_result + scale_x_continuous(breaks=label_data$A, labels = label_data$B*100)                                 
