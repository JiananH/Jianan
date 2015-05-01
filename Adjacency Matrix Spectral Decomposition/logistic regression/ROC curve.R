#draw the ROC curve for my datasets
#define a threshold C where P(y=1|X=x)>C
library(glmnet)
library(kernlab)
data=read.table("~/Google Drive/R/newsummary.txt",header=T)
data.final=data[,-c(2,3,4)]
data.reduce=na.omit(data.final)
# data=read.table("~/Google Drive/R/newnewsummary.txt",header=T)
# data.reduce=na.omite(data)

#To preserve the proportion of 0's ans 1's (10522:3571) we choose 10-fold cross validation with 1052 0's and 357 1's in each sample

#create training samples and test samples
precision=vector('numeric',10)
void=data.reduce[data.reduce[,1]==0,]
filament=data.reduce[data.reduce[,1]==1,]
sample.vector.void=sample(1:dim(void)[1],10520)
sample.matrix.void=matrix(sample.vector.void,ncol=10,nrow=1052)
sample.vector.filament=sample(1:dim(filament)[1],3570)
sample.matrix.filament=matrix(sample.vector.filament,ncol=10,nrow=357)

plot(seq(0,1,0.01),seq(0,1,0.01),type='n',xlab="P(1|0)",ylab="P(1|1)",main="ROC curve for logistic regression")
for (i in 1:4){
test=rbind(void[sample.matrix.void[,i],],filament[sample.matrix.filament[,i],])
train=rbind(void[setdiff(sample.vector.void,sample.matrix.void[,i]),],filament[setdiff(sample.vector.filament,sample.matrix.filament[,i]),])

# #logistic regression
# ytrain=train[,1]
# testtrain=train[,-1]

# logit=glm(ytrain~.,data=testtrain,family="binomial")
# pred=predict(logit,newdata=test,type="response")


# #shrinkage method alpha=1 is the lasoo penalty and alpha=0 is the ridge penalty

# fit=cv.glmnet(as.matrix(train[,-1]),as.factor(train[,1]),alpha=0.95,family="binomial")

# best.fit=glmnet(as.matrix(train[,-1]),as.factor(train[,1]),alpha=0.95,lambda=fit$lambda.min,family="binomial")

# pred=predict(best.fit,newx=as.matrix(test[,-1]),type="response")


#support vector machine
svp <- ksvm(as.matrix(train[,-1]),train[,1],type="C-svc",kernel='rbf',kpar=list(sigma=1),C=1,prob.model=TRUE)
pre=predict(svp,as.matrix(test[,-1]),type="probabilities")
pred=pre[,2]

roc=function(test,pred,i,s){
	ps=as.numeric(pred>s)
	#P(1|0)=False Positive, P(1|1)=True Positive
	fp=sum((ps==1)*(test[,1]==0))/sum(test[,1]==0)
	tp=sum((ps==1)*(test[,1]==1))/sum(test[,1]==1)
	vec=c(fp,tp)
	return(vec)
		
}

M.ROC=sapply(seq(0,1,by=.001),function(x){roc(test,pred,i,x)})
points(M.ROC[1,],M.ROC[2,],col=i,lwd=2,type="l")
#precision.glm[i]=sum(true)/length(true)
}

legend("topleft",legend=c("1","2","3","4"),col=c(1,2,3,4),lty=1)
abline(0,1,col=8)

