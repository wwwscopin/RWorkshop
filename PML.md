---
title: "Predict the Manner of Exercise"
author: "Baohua Wu"
date: "06/20/2014"
output: html_document
---

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, 6 participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways, the relevant data from accelerometers on the belt, forearm, arm, and dumbell was collected. The goal of this project is to predict the manner in which they did the exercise.

## Data and Preprocessing
The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. There are a lot of missing values in many of the variables. Only those variables has less than 10 percent of missing values will be used in our model building.



```r
raw0<-read.csv("pml-training.csv", header=TRUE, na.strings=c("NA",""))
## The raw data size is:
dim(raw0)
```

```
## [1] 19622   160
```

```r
##
NAs <- apply(raw0,2,function(x) {sum(is.na(x))}) 
raw1 <- raw0[,which(NAs <=1/10*length(raw0[,1]))]
## After removing columns with more than 10% missing values, the data size is:
dim(raw1)
```

```
## [1] 19622    60
```

```r
## Load required packages:

require(lubridate)
```

```
## Loading required package: lubridate
```

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(tree)
library(e1071)
library(randomForest)
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

In the raw data, one of the column has date/time in factor format, here we transfere it into 2 variables: Date and Time



The raw data was splitted into training and testing data set with ratio of 7:3:


```
## 
##    A    B    C    D    E 
## 3906 2658 2396 2252 2525
```

The data would be used for prediction was loaded and followed the same preprocessing as the raw data.
Also, only those same columns as raw data was kept for later prediction:



## Model building using Random forest:


```r
rf<-randomForest(classe~.,data=training, importance=TRUE, na.action = na.omit)
## The error rate from Random Forest:
rf
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training, importance = TRUE,      na.action = na.omit) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.1%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3905    1    0    0    0   0.0002560
## B    2 2655    1    0    0   0.0011287
## C    0    3 2392    1    0   0.0016694
## D    0    0    3 2248    1   0.0017762
## E    0    0    0    2 2523   0.0007921
```

```r
## List of Important variables by random Forest:
varImp(rf)
```

```
##                           A        B       C      D      E
## user_name            12.088 14.28505 12.1661 13.139 14.488
## raw_timestamp_part_1 46.268 54.36952 56.0210 58.387 39.128
## raw_timestamp_part_2  5.181 10.11721  8.4692  9.268  7.952
## new_window           -1.344 -0.02926  0.4457  1.021 -1.898
## num_window           30.275 40.19892 42.9738 34.888 34.852
## roll_belt            28.362 37.08301 32.4895 35.131 32.988
## pitch_belt           23.710 36.53859 30.1243 29.840 25.126
## yaw_belt             29.822 34.11065 30.6186 35.769 27.176
## total_accel_belt     13.471 16.43145 12.5900 14.062 13.437
## gyros_belt_x         14.020 14.35531 14.6189 11.430 13.931
## gyros_belt_y         10.657 13.10758 13.6365 15.184 14.392
## gyros_belt_z         16.392 21.57915 19.5221 20.485 20.479
## accel_belt_x         10.924 14.20577 14.2928 13.071 12.648
## accel_belt_y          9.528 12.36736 11.8871 15.260 11.451
## accel_belt_z         17.549 20.30311 18.0695 19.928 16.549
## magnet_belt_x        13.066 22.69306 20.0377 17.460 19.308
## magnet_belt_y        16.460 21.61083 21.0950 22.181 20.207
## magnet_belt_z        17.198 20.69342 18.8703 23.215 20.171
## roll_arm             15.637 21.92315 20.3505 20.393 17.294
## pitch_arm            10.834 17.90313 17.0722 14.873 12.394
## yaw_arm              15.967 20.34234 17.4992 19.187 14.226
## total_accel_arm       8.226 18.17961 14.0993 13.916 13.744
## gyros_arm_x          12.501 19.45524 16.6676 15.848 14.134
## gyros_arm_y          13.181 20.72756 18.1111 17.864 14.967
## gyros_arm_z           8.307 10.59778 11.5516 10.031  8.541
## accel_arm_x          12.231 13.31651 13.9826 14.106 11.345
## accel_arm_y          13.913 16.29296 12.8817 13.824 13.641
## accel_arm_z           9.751 14.42500 13.4262 13.352 11.360
## magnet_arm_x         13.961 14.28291 15.1093 14.692 12.314
## magnet_arm_y         11.133 13.68163 14.2260 14.341 10.867
## magnet_arm_z         15.116 18.16945 17.4529 13.075 13.237
## roll_dumbbell        20.183 23.52397 26.0821 23.153 21.299
## pitch_dumbbell       10.220 16.83391 15.0072 12.044 12.672
## yaw_dumbbell         13.994 21.28834 20.7935 16.574 17.689
## total_accel_dumbbell 14.952 19.25755 17.4821 17.767 17.857
## gyros_dumbbell_x     11.073 17.85093 16.8508 14.695 12.873
## gyros_dumbbell_y     16.163 16.70813 21.2114 16.217 13.006
## gyros_dumbbell_z     12.643 16.09338 14.6436 12.445 11.801
## accel_dumbbell_x     13.236 18.79316 19.0769 16.454 16.469
## accel_dumbbell_y     19.740 23.10324 24.6754 22.308 21.577
## accel_dumbbell_z     15.194 22.10736 21.8188 20.657 21.561
## magnet_dumbbell_x    18.981 22.09559 24.2768 20.379 18.709
## magnet_dumbbell_y    27.882 29.18640 33.6779 27.477 24.599
## magnet_dumbbell_z    32.857 28.61337 33.3750 25.266 24.975
## roll_forearm         22.771 20.47317 25.3172 18.198 17.426
## pitch_forearm        23.957 26.27986 30.7214 28.848 24.694
## yaw_forearm          12.732 18.06977 15.3561 16.189 14.483
## total_accel_forearm  13.187 14.23564 14.0442 12.402 11.422
## gyros_forearm_x       7.402 12.97376 13.4433 11.061 11.456
## gyros_forearm_y      11.758 21.39441 20.4303 15.649 14.744
## gyros_forearm_z      11.757 17.72123 16.5065 12.879 12.538
## accel_forearm_x      13.602 19.50430 17.7022 20.550 18.970
## accel_forearm_y      13.147 15.59671 16.6214 12.362 14.289
## accel_forearm_z      12.819 18.23260 18.1180 15.628 16.444
## magnet_forearm_x     11.186 16.57842 14.4157 15.317 15.209
## magnet_forearm_y     13.544 19.13776 18.3203 18.603 17.282
## magnet_forearm_z     16.077 17.99613 17.1677 18.930 16.645
## Date                  8.600 11.78724  9.9440 12.928 11.701
## HM                   30.958 42.47193 33.1324 41.245 41.840
```

## Check the model with cross validation:

Through cross-validation on testing data, we can get the accuracy from Random Forest model:
The accuracy is pretty high, I doubt it is true or something wrong here?


```r
pred<-predict(rf,testing)
table(pred,testing$classe)
```

```
##     
## pred    A    B    C    D    E
##    A 1672    2    0    0    0
##    B    2 1137    1    0    0
##    C    0    0 1023    0    0
##    D    0    0    2  964    0
##    E    0    0    0    0 1082
```

```r
check= (testing$classe == pred)
accuracy<-length(check[check == TRUE]) / length(check);
accuracy
```

```
## [1] 0.9988
```

## Prediction for the new data:


```r
## Fake one case with level='yea', otherwise won't work!
one<-testing[which(testing$new_window=='yes')[1],1:59]
test<-rbind(test, one)

answer<-predict(rf,test)[1:20]
answer
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

