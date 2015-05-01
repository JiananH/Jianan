#Construct the weight matrix
library(HiDimDA)
library(Matrix)
library(class)
library(SIS)
data(prostate.train)
#regular knn methods
cl=as.factor(prostate.train[,12601])
result=knn.cv(prostate.train[,-12601],cl,k=1,prob=TRUE)
svp=svm(prostate.train[,-12601],cl,kernel='radial',cross=dim)

true=vector('numeric',dim)
for (i in 1:dim){
  true[i]=ifelse(as.numeric(cl[i])==as.numeric(svp$fitted[i]),1,0)
  #true[i]=ifelse(cl[i]==result[i],1,0)
}
precision=sum(true)/dim
precision
#comparison of knn results
before=c(0.7549,0.7941,0.8137,0.7843,0.7941,0.8039)
after=c(0.7549,0.7353,0.6765,0.6961,0.7255,0.7059)
plot(c(1,3,5,7,9,11),before,col="red",pch=20,type='b',ylim=c(0.65,0.85),xlab="k",ylab="accuracy")
lines(c(1,3,5,7,9,11),after,type='b',col="blue")
legend("topright",legend=c("before","after"),col=c("red","blue"),pch=20)

library(e1071)
dim=dim(prostate.train)[1]
weight.matrix=matrix(0,dim,dim)
for (i in 1:dim){
  for (j in 1:dim){
    if (i==j){
      weight.matrix[i,i]=0
    }
    else {
      weight.matrix[i,j]=sum((prostate.train[i,-12601]-prostate.train[j,-12601])^2)
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
#combine the class labels
#newdata=cbind(AlonDS[,1],normH)
svp=svm(normH,cl,kernel='radial',cross=dim)

#perform leave one out knn classification
cl=as.factor(prostate.train[,12601])
result=knn.cv(normH,cl,k=11,prob=TRUE)
true=vector('numeric',dim)
for (i in 1:dim){
  true[i]=ifelse(as.numeric(cl[i])==as.numeric(svp$fitted[i]),1,0)
  #true[i]=ifelse(cl[i]==result[i],1,0)
}
precision=sum(true)/dim
precision

