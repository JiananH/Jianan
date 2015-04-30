args <- commandArgs(TRUE)

if(length(args)==0){
    stop("No arguments supplied.")
}else{
    for(i in 1:length(args)) eval(parse(text=args[[i]]))
}
#adjacency matrix with spectral decomposition
library(e1071)
library(Matrix)
library(class)
data=read.table("~/ludecompose/newsummary.txt",header=T)
data=data[,-c(2,3,4)]
data=na.omit(data)

#To preserve the proportion of 0's ans 1's (10522:3571) we choose 10-fold cross validation with 1052 0's and 357 1's in each sample

#create training samples and test samples
void=data[data[,1]==0,]
filament=data[data[,1]==1,]
#scale up a little
rm(data)
group=1
precision.svm=vector('numeric',group)
precision.knn=vector('numeric',group)
precision.svm1=vector('numeric',group)
precision.knn1=vector('numeric',group)
for (itr in 1:group){
train=rbind(void[sample(1:dim(void)[1],1050),],filament[sample(1:dim(filament)[1],351),])

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
rm(weight.matrix)
LU=expand(decompose)
H=LU$L+LU$U-diag(dim)

min=apply(H,1,min)
max=apply(H,1,max)
coef=2/(max-min)
normH=as(coef*H-coef*min-1,"matrix")
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

result=cbind(precision.svm,precision.svm1,precision.knn,precision.knn1)
result=1-result
write.table(result,"performance_node_1401_15.txt")
