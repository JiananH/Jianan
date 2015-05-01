#adjacency matrix with spectral decomposition
library(e1071)
library(Matrix)
library(class)
data=read.table("~/Google Drive/R/newsummary.txt",header=T)
data.final=data[,-c(2,3,4)]
data.reduce=na.omit(data.final)

#To preserve the proportion of 0's ans 1's (10522:3571) we choose 10-fold cross validation with 1052 0's and 357 1's in each sample

#create training samples and test samples
void=data.reduce[data.reduce[,1]==0,]
filament=data.reduce[data.reduce[,1]==1,]
sample.vector.void=sample(1:dim(void)[1],10500)
sample.matrix.void=matrix(sample.vector.void,ncol=100,nrow=105)
sample.vector.filament=sample(1:dim(filament)[1],3500)
sample.matrix.filament=matrix(sample.vector.filament,ncol=100,nrow=35)

#scale up a little
# sample.vector.void=sample(1:dim(void)[1],10500)
# sample.matrix.void=matrix(sample.vector.void,ncol=25,nrow=420)
# sample.vector.filament=sample(1:dim(filament)[1],3500)
# sample.matrix.filament=matrix(sample.vector.filament,ncol=25,nrow=140)
group=20
precision.svm=vector('numeric',group)
precision.knn=vector('numeric',group)
precision.svm1=vector('numeric',group)
precision.knn1=vector('numeric',group)
for (itr in 1){
train=rbind(void[sample.matrix.void[,itr],],filament[sample.matrix.filament[,itr],])
#train=rbind(void[setdiff(sample.vector.void,sample.matrix.void[,i]),],filament[setdiff(sample.vector.filament,sample.matrix.filament[,i]),])

dim=dim(train)[1]

combination=combn(dim,2)
mat.com=apply(combination,2,function(t){temp=rep(0,dim); temp[t[1]]=1;temp[t[2]]=-1;return(temp)})
distance=t(mat.com)%*%as.matrix(train[,-1])
norm.2=apply(distance,1,function(x){norm(as.matrix(x),type="F")^2})
weight.matrix=matrix(0,dim,dim)
for (i in 1:dim(combination)[2]){
	weight.matrix[combination[1,i],combination[2,i]]=norm.2[i]
}
weight.matrix=forceSymmetric(weight.matrix)

decompose=lu(weight.matrix)

LU=expand(decompose)
H=LU$L+LU$U-diag(dim)

min=apply(H,1,min)
max=apply(H,1,max)
coef=2/(max-min)
normH=as(coef*H-coef*min-1,"matrix")
# for (i in 1:(dim-1)){
		  # for (j in (i+1):dim){
      # weight.matrix[i,j]=sum((train[i,-1]-train[j,-1])^2)
     # #can be replaced by forcesymmetric()
      # weight.matrix[j,i]=weight.matrix[i,j]
  # }
# }
# decompose=lu(weight.matrix)

# LU=expand(decompose)
# H=LU$L+LU$U-diag(dim)
# #Normalize H
# normH=matrix(0,dim,dim)
# for (i in 1:dim){
  # for (j in 1:dim){
    # normH[i,j]=2*(H[i,j]-min(H[i,]))/(max(H[i,])-min(H[i,]))-1
  # }
# }

#Use normH to build my classifier and test on the test data
cl=as.factor(train[,1])
svp=svm(normH,cl,kernel='radial',cross=dim)
svp1=svm(train[,-1],cl,kernel='radial',cross=dim)

#perform leave one out knn classification
result=knn.cv(normH,cl,k=3,prob=TRUE)
result1=knn.cv(train[,-1],cl,k=3,prob=TRUE)

precision.svm[itr]=mean(train[,1]+as.numeric(svp$fitted)==2)
precision.knn[itr]=mean(train[,1]+as.numeric(result)==2)
precision.svm1[itr]=mean(train[,1]+as.numeric(svp1$fitted)==2)
precision.knn1[itr]=mean(train[,1]+as.numeric(result1)==2)

}

plot(precision.svm,type='b',pch=20, col='red',main="Adjacency Matrix LU decomposition (560x560)",xlab="trial number",ylab="accuracy",ylim=c(0.6,1))
points(precision.knn,type='b',pch=20, col='blue')
points(precision.svm1,type='b',pch=20, col='darkgreen')
points(precision.knn1,type='b',pch=20, col='purple')
grid(10,10)
legend('bottomleft',legend=c('SVM after','SVM before','kNN after','kNN before'),col=c('red','darkgreen','blue','purple'),lty=1)

result=cbind(precision.svm[1:9],precision.svm1[1:9],precision.knn[1:9],precision.knn1[1:9])
write.table(result,"performance_node_560x560.txt")

summary=as.data.frame(matrix(c(mean(precision.svm),mean(precision.knn),sd(precision.svm),sd(precision.knn)),2,2),row.names=c("mean","sd"),colnames=c("SVM","kNN"))
colnames(summary)=c("SVM","kNN")

summary=as.data.frame(matrix(c(mean(precision.svm1[1:2]),mean(precision.knn1[1:2]),sd(precision.svm1[1:2]),sd(precision.knn1[1:2])),2,2),row.names=c("mean","sd"),colnames=c("SVM","kNN"))
colnames(summary)=c("SVM","kNN")
