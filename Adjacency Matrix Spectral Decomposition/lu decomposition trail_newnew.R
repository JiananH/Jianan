#adjacency matrix with spectral decomposition
library(e1071)
library(Matrix)
library(class)
data=read.table("~/Google Drive/R/newnewsummary.txt",header=T)
#data.final=data[,-c(2,3,4)]
data.reduce=na.omit(data)

#To preserve the proportion of 0's ans 1's (10522:3571) we choose 10-fold cross validation with 1052 0's and 357 1's in each sample

#create training samples and test samples
void=data.reduce[data.reduce[,1]==0,]
filament=data.reduce[data.reduce[,1]==1,]
sample.vector.void=sample(1:dim(void)[1],10500)
sample.matrix.void=matrix(sample.vector.void,ncol=100,nrow=105)
sample.vector.filament=sample(1:dim(filament)[1],3500)
sample.matrix.filament=matrix(sample.vector.filament,ncol=100,nrow=35)

group=20
precision.svm=vector('numeric',group)
precision.knn=vector('numeric',group)
precision.svm1=vector('numeric',group)
precision.knn1=vector('numeric',group)
for (itr in 12:group){
train=rbind(void[sample.matrix.void[,itr],],filament[sample.matrix.filament[,itr],])
#train=rbind(void[setdiff(sample.vector.void,sample.matrix.void[,i]),],filament[setdiff(sample.vector.filament,sample.matrix.filament[,i]),])

dim=dim(train)[1]
weight.matrix=matrix(0,dim,dim)
for (i in 1:dim){
	if (i<dim){
		  for (j in (i+1):dim){
      weight.matrix[i,j]=sum((train[i,-1]-train[j,-1])^2)
      weight.matrix[j,i]=weight.matrix[i,j]
  }
	}

}
decompose=lu(weight.matrix)

LU=expand(decompose)
H=LU$L+LU$U-diag(dim)
#Normalize H
normH=matrix(0,dim,dim)
for (i in 1:dim){
  for (j in 1:dim){
    normH[i,j]=2*(H[i,j]-min(H[i,]))/(max(H[i,])-min(H[i,]))-1
  }
}

#Use normH to build my classifier and test on the test data
cl=as.factor(train[,1])
svp=svm(normH,cl,kernel='radial',cross=dim)
svp1=svm(train[,-1],cl,kernel='radial',cross=dim)

#perform leave one out knn classification
result=knn.cv(normH,cl,k=3,prob=TRUE)
result1=knn.cv(train[,-1],cl,k=3,prob=TRUE)
true1=vector('numeric',dim)
true2=vector('numeric',dim)
true3=vector('numeric',dim)
true4=vector('numeric',dim)
for (i in 1:dim){
  true1[i]=ifelse(as.numeric(cl[i])==as.numeric(svp$fitted[i]),1,0)
  true2[i]=ifelse(cl[i]==result[i],1,0)
  true3[i]=ifelse(as.numeric(cl[i])==as.numeric(svp1$fitted[i]),1,0)
  true4[i]=ifelse(cl[i]==result1[i],1,0)
}
precision.svm[itr]=sum(true1)/dim
precision.knn[itr]=sum(true2)/dim
precision.svm1[itr]=sum(true3)/dim
precision.knn1[itr]=sum(true4)/dim

}

plot(precision.svm,type='b',pch=20, col='red',main="Adjacency Matrix LU decomposition",xlab="trial number",ylab="accuracy")
points(precision.knn,type='b',pch=20, col='blue')
points(precision.svm1,type='b',pch=20, col='yellow')
points(precision.knn1,type='b',pch=20, col='purple')
grid(10,10)
legend('topleft',legend=c('SVM after','kNN after','SVM before','kNN before'),col=c('red','blue','yellow','purple'),lty=1)

summary=as.data.frame(matrix(c(mean(precision.svm),mean(precision.knn),sd(precision.svm),sd(precision.knn)),2,2),row.names=c("mean","sd"),colnames=c("SVM","kNN"))
colnames(summary)=c("SVM","kNN")

summary=as.data.frame(matrix(c(mean(precision.svm1[1:2]),mean(precision.knn1[1:2]),sd(precision.svm1[1:2]),sd(precision.knn1[1:2])),2,2),row.names=c("mean","sd"),colnames=c("SVM","kNN"))
colnames(summary)=c("SVM","kNN")
