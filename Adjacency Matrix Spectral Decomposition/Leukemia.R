#Construct the weight matrix
library(HiDimDA)
library(Matrix)
library(class)
library(SIS)
library(kernlab)
library(e1071)
data(leukemia.train)
#before applying the decomposition
cl=as.factor(leukemia.train[,7130])
#result=knn.cv(leukemia.train[,-7130],cl,k=11,prob=TRUE)
svp=svm(leukemia.train[,-7130],cl,kernel='radial',cross=dim,gamma=1/38)

true=vector('numeric',dim)
for (i in 1:dim){
  true[i]=ifelse(as.numeric(cl[i])==as.numeric(svp$fitted[i]),1,0)
  #true[i]=ifelse(cl[i]==result[i],1,0)
}
precision=sum(true)/dim
precision


#after
dim=dim(leukemia.train)[1]
weight.matrix=matrix(0,dim,dim)
for (i in 1:dim){
  for (j in 1:dim){
    if (i==j){
      weight.matrix[i,i]=0
    }
    else {
      weight.matrix[i,j]=sqrt(sum((leukemia.train[i,-7130]-leukemia.train[j,-7130])^2))
    }
  }
}

#write.table(weight.matrix,"weightmatrix_leukemia.txt")
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

#perform leave one out knn classification
class=as.factor(leukemia.train[,7130])
res=vector('numeric',dim)
for (i in 1:dim){
  res[i]=knn(normH[-i,],normH[i,],cl=class[-i],k=1)
}
result=knn.cv(normH,cl,k=1,prob=TRUE)


#svm classifier
svp=svm(normH,class,kernel='radial',cross=dim,gamma=1/13)

true=vector('numeric',dim)
for (i in 1:dim){
  true[i]=ifelse(as.numeric(class[i])==as.numeric(svp$fitted[i]),1,0)
  #true[i]=ifelse(as.numeric(class[i])==as.numeric(res[i]),1,0)
}
precision=sum(true)/dim
precision

