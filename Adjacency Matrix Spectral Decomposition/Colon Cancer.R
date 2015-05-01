#Construct the weight matrix
library(HiDimDA)
library(Matrix)
library(class)
library(SIS)
library(kernlab)
library(e1071)
data(AlonDS)
#before applying the decomposition
cl=as.factor(AlonDS[,1])
result=knn.cv(AlonDS[,-1],cl,k=1,prob=TRUE)
#svp=svm(AlonDS[,-1],cl,kernel='radial',cross=dim)

true=vector('numeric',dim)
for (i in 1:dim){
  #true[i]=ifelse(as.numeric(cl[i])==as.numeric(svp$fitted[i]),1,0)
  true[i]=ifelse(cl[i]==result[i],1,0)
}
precision=sum(true)/dim
precision

#after
dim=dim(AlonDS)[1]
weight.matrix=matrix(0,dim,dim)
for (i in 1:dim){
  for (j in 1:dim){
    if (i==j){
      weight.matrix[i,i]=0
    }
    else {
      weight.matrix[i,j]=sqrt(sum((AlonDS[i,-c(1,2)]-AlonDS[j,-c(1,2)])^2))
    }
  }
}

#write.table(weight.matrix,"weightmatrix.txt")
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
newdata=cbind(AlonDS[,1],normH)

#perform leave one out knn classification
class=as.factor(AlonDS[,1])
res=vector('numeric',dim)
for (i in 1:dim){
  res[i]=knn(normH[-i,],normH[i,],cl=class[-i],k=11)
}
result=knn.cv(normH,cl,k=1,prob=TRUE)


#svm classifier
svp=svm(normH,class,kernel='radial',cross=dim)
 
true=vector('numeric',dim)
for (i in 1:dim){
  true[i]=ifelse(as.numeric(class[i])==as.numeric(svp$fitted[i]),1,0)
}
precision=sum(true)/dim
precision

