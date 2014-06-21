raw0<-read.csv("pml-training.csv", header=TRUE, na.strings=c("NA",""))

# discard NAs
NAs <- apply(raw0,2,function(x) {sum(is.na(x))}) 
raw1 <- raw0[,which(NAs <=1/10*length(raw0[,1]))]
dim(raw1)


require(lubridate)
library(caret)
library(tree)
library(e1071)
library(randomForest)

classe<-raw1$classe
wbh<-as.POSIXct(levels(raw1$cvtd_timestamp)[raw1$cvtd_timestamp], format="%d/%m/%Y %H:%M")
raw1$Date<-as.Date(wbh)
raw1$HM<-hour(wbh)+minute(wbh)/60
raw1<-raw1[, !names(raw1) %in% c("X","cvtd_timestamp", "classe")]
raw<-cbind(raw1,classe)
#str(raw)


Index <- createDataPartition(y = raw$classe, p=0.7,list=FALSE) # 3927 rows
training <- raw[Index,]
testing <- raw[-Index,]
table(training$classe)

rawtest<-read.csv("pml-testing.csv", header=TRUE, na.strings=c("NA",""))
wbh1<-as.POSIXct(levels(rawtest$cvtd_timestamp )[rawtest$cvtd_timestamp], format="%d/%m/%Y %H:%M")
rawtest$Date<-as.Date(wbh1)
rawtest$HM<-hour(wbh1)+minute(wbh1)/60
test<-rawtest[,colnames(rawtest) %in% colnames(raw)]

#Check type match!
#mark=sapply(test,class)==sapply(testing,class)
#mark[mark==FALSE]

test$magnet_dumbbell_z<-as.numeric(test$magnet_dumbbell_z)
test$magnet_forearm_y<-as.numeric(test$magnet_forearm_y)
test$magnet_forearm_z<-as.numeric(test$magnet_forearm_z)

#rf<-train(classe~., data=training, method="rf", preProc=c("center", "scale"))
rf<-randomForest(classe~.,data=training, importance=TRUE, na.action = na.omit)
rf
varImp(rf)

pred<-predict(rf,testing)
length(pred)

check= (testing$classe == pred)
accuracy<-length(check[check == TRUE]) / length(check);
accuracy

## Fake one case with level='yea', otherwise won't work!
one<-testing[which(testing$new_window=='yes')[1],1:59]
test<-rbind(one, test)

final<-predict(rf,test)[-1]
final