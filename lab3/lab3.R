#ASSIGNMENT 1: Kernel methods

set.seed(1234567890)
library(geosphere)
stations <- read.csv("stations.csv", fileEncoding = "latin1")
temps <- read.csv("temps50k.csv")
st <- merge(stations,temps,by="station_number")

distKernel = function(cord1, cord2){
  diffDistances <<- distHaversine(cord1, cord2)/10000; #distance i mil
  kernelValueDist = exp(-diffDistances^2 / (2*h_distance^2));
  return (kernelValueDist)
}

dateKernel = function(date1, date2){
  diffDates <<- abs(as.numeric(difftime(date2, date1, units = "days"))) %% 365
  kernelValueDate = exp(-diffDates^2 / (2*h_date^2))
  return(kernelValueDate)
}

timeKernel = function(time1, time2, col){
  diffTime = abs(as.numeric(difftime(strptime(time1, format = "%H:%M:%S"), 
                                     strptime(time2, format = "%H:%M:%S"), 
                                     units = "hours")))
  diffTime = ifelse(diffTime > 12, 24 - diffTime, diffTime)
  diffTimes[,col] <<- diffTime
  kernelValueTime = exp(-diffTime^2 / (2*h_time^2))
  return(kernelValueTime)
}

sumKernels = function(distKernValues, dateKernValues, timeKernValues){
  tempSum = vector(length=length(times))
  for (i in 1:length(times)) {
    sumValues = distKernValues + dateKernValues + timeKernValues[,i];
    sumValues = sumValues / sum(sumValues);
    predSum = sumValues %*% stFiltered$air_temperature;
    tempSum[i] = sum(predSum) 
  }
  return(tempSum)
}

multKernels = function(distKernValues, dateKernValues, timeKernValues){
  tempMult <- vector(length=length(times))
  for (i in 1:length(times)) {
    multValues = distKernValues * dateKernValues * timeKernValues[,i];
    multValues = multValues / sum(multValues);
    predMult = multValues %*% stFiltered$air_temperature;
    tempMult[i] = sum(predMult) 
  }
  return(tempMult)
}

# These three values are up to the students
# Smoothers Values
h_distance <- 30
h_date <- 5
h_time <- 3

# The point to predict (up to the students)
# Västervik
a <- 57.72130 
b <- 16.46830
pred_cords = c(a,b)

date <- "1989-06-23" # The date to predict (up to the students)

# -------- Filter out dates after the choosen date above --------
stFiltered <- st[st$date < as.Date(date),]

# -------- Times we want the sample --------
times <- c("04:00:00", "06:00:00","08:00:00", "10:00:00", "12:00:00","14:00:00", "16:00:00", 
           "18:00:00","20:00:00", "22:00:00","24:00:00")
times_numeric = c(4,6,8,10,12,14,16,18,20,22,24)

# -------- Create a matrix with every stations lat & long --------
stationCords = cbind(stFiltered$latitude, stFiltered$longitude);
# -------- Create a vector with --------
stationRecordingDates = c(stFiltered$date)
diffDistances = c()
diffDates = c()
diffTimes = matrix(nrow = dim(stFiltered)[1], ncol = length(times))

distKernValues = distKernel(pred_cords,stationCords)
dateKernValues = dateKernel(stationRecordingDates, date)
timeKernValues = matrix(nrow = dim(stFiltered)[1], ncol = length(times))

col = 1
for(time in times){
  timeKernValues[,col] = timeKernel(stFiltered$time,time, col)
  col = col + 1
}

tempSum = sumKernels(distKernValues, dateKernValues, timeKernValues)
tempMult = multKernels(distKernValues, dateKernValues, timeKernValues)

dist_data = data.frame(diffDistances, distKernValues)

library(ggplot2)
ggplot(dist_data, aes(x = diffDistances, y = distKernValues)) +
  geom_point() +
  labs(x = "Distances", y = "Kernel Values")


date_data = data.frame(diffDates, dateKernValues)
ggplot(date_data, aes(x = diffDates, y = dateKernValues)) +
  xlim(0,100) + 
  geom_point() +
  labs(x = "Dates", y = "Kernel Values")

diffTimesAt12 = diffTimes[,5]
timeKernValuesAt12 = timeKernValues[,5]

time_data = data.frame(diffTimesAt12,timeKernValuesAt12)
ggplot(time_data, aes(x = diffTimesAt12, y = timeKernValuesAt12)) +
  geom_point() +
  labs(x = "Time", y = "Kernel Values")



plot(times_numeric,tempSum, type="o", ylab = "Temperature (C°)", xlab = "Time of day")
plot(times_numeric,tempMult, type="o", ylab = "Temperature (C°)", xlab = "Time of day")


#ASSIGNMENT 2: Support vector machines

library(kernlab)
set.seed(1234567890)

data(spam)
foo <- sample(nrow(spam))
spam <- spam[foo,]
spam[,-58]<-scale(spam[,-58])
tr <- spam[1:3000, ]
va <- spam[3001:3800, ]
trva <- spam[1:3800, ]
te <- spam[3801:4601, ] 

by <- 0.3
err_va <- NULL
for(i in seq(by,5,by)){
  filter <- ksvm(type~.,data=tr,kernel="rbfdot",kpar=list(sigma=0.05),C=i,scaled=FALSE)
  mailtype <- predict(filter,va[,-58])
  t <- table(mailtype,va[,58])
  err_va <-c(err_va,(t[1,2]+t[2,1])/sum(t))
}

filter0 <- ksvm(type~.,data=tr,kernel="rbfdot",kpar=list(sigma=0.05),C=which.min(err_va)*by,scaled=FALSE)
mailtype <- predict(filter0,va[,-58])
t <- table(mailtype,va[,58])
err0 <- (t[1,2]+t[2,1])/sum(t)
err0

filter1 <- ksvm(type~.,data=tr,kernel="rbfdot",kpar=list(sigma=0.05),C=which.min(err_va)*by,scaled=FALSE)
mailtype <- predict(filter1,te[,-58])
t <- table(mailtype,te[,58])
err1 <- (t[1,2]+t[2,1])/sum(t)
err1

filter2 <- ksvm(type~.,data=trva,kernel="rbfdot",kpar=list(sigma=0.05),C=which.min(err_va)*by,scaled=FALSE)
mailtype <- predict(filter2,te[,-58])
t <- table(mailtype,te[,58])
err2 <- (t[1,2]+t[2,1])/sum(t)
err2

filter3 <- ksvm(type~.,data=spam,kernel="rbfdot",kpar=list(sigma=0.05),C=which.min(err_va)*by,scaled=FALSE)
mailtype <- predict(filter3,te[,-58])
t <- table(mailtype,te[,58])
err3 <- (t[1,2]+t[2,1])/sum(t)
err3

# Questions

# 1. Which filter do we return to the user ? filter0, filter1, filter2 or filter3? Why?

# 2. What is the estimate of the generalization error of the filter returned to the user? err0, err1, err2 or err3? Why?

# 3. Implementation of SVM predictions.

sv<-alphaindex(filter3)[[1]] #support vectors
co<-coef(filter3)[[1]] #weights for support vectors
inte<- - b(filter3)
k<-c()

rbf <- rbfdot(sigma = 0.05)

for(i in 1:10){ # We produce predictions for just the first 10 points in the dataset.
  k2<-0
  for(j in 1:length(sv)){
    k2 <- k2 + co[j] * kernelMatrix(rbf, as.matrix(spam[i,-58]), as.matrix(spam[sv[j],-58]))
  }
  k[i] = k2 + inte
}
k
predict(filter3,spam[1:10,-58], type = "decision")


#ASSIGNMENT 3: Neural networks

library(neuralnet)

set.seed(1234567890)

### 3.1 
# Random initialization of x in the interval [0, 10]
Var <- runif(500, 0, 10)

# Save it as dataframe
mydata <- data.frame(Var, Sin=sin(Var))

# Split the data for training and test
tr <- mydata[1:25,] # Training
te <- mydata[26:500,] # Test

# Random initialization of the weights in the interval [-1, 1]
winit <- runif(31, -1, 1) 

# Sets up the neural network with Sin as target, using defult activation function (sigmoid)
nn <- neuralnet(Sin ~., data = tr, hidden = 10, startweights = winit)

# Plot of the training data (black), test data (blue), and predictions (red)
plot(tr, cex=2)
points(te, col = "blue", cex=1)
points(te[,1],predict(nn,te), col="red", cex=1)

## 3.2
# Defining cusom activation functions to use for the neural net when repeating 3.1 

set.seed(1234567890)

# linear
linear <- function(x){
  x
}

# ReLU
ReLu <- function(x){
  ifelse(x > 0, x, 0)
}

# softplus
softplus <- function(x){
  log(1 + exp(x))
}

# Sets up the neural network with the different activation functions
nn_linear <- neuralnet(Sin ~., data = tr, hidden = 10, startweights = winit, act.fct = linear)
nn_relu <- neuralnet(Sin ~., data = tr, hidden = 10, startweights = winit, act.fct = ReLu)
nn_softplus <- neuralnet(Sin ~., data = tr, hidden = 10, startweights = winit, act.fct = softplus)

# Plots the new models
# linear plot
plot(tr, cex=2)
points(te, col = "blue", cex=1)
points(te[,1],predict(nn_linear,te), col="red", cex=1)

# ReLu plot
plot(tr, cex=2)
points(te, col = "blue", cex=1)
points(te[,1],predict(nn_relu,te), col="red", cex=1)


# softplus plot
plot(tr, cex=2)
points(te, col = "blue", cex=1)
points(te[,1],predict(nn_softplus,te), col="red", cex=1)

### 3.3
set.seed(1234567890)

# Random initialization of x in the interval [0, 50]
Var50 <- runif(500, 0, 50)

# Save it as dataframe
mydata2 <- data.frame(Var = Var50, Sin=sin(Var50))

# Plot of the training data (black), test data (blue), and predictions (red)
plot(mydata2, cex=2, ylim=c(-11, 1))
points(mydata2[,1], predict(nn, mydata2), col="red", cex=1)

### 3.4
plot(nn)
nn$weights

### 3.5
# Random initialization of the x in the interval [0, 10], we use the same Var as defined in 3.1
# mydata defined in 3.1 also works to use as dataframe

set.seed(1234567890)

# Sets up the neural network with Var as target, using defult activation function (sigmoid)
nnX <- neuralnet(Var ~., data = mydata, hidden = 10, startweights = winit, threshold = 0.1)

# Plot of the training data (black), and predictions (red)
plot(mydata$Sin, mydata$Var, cex = 2)
points(mydata$Sin, predict(nnX, mydata), col = "red", cex = 1)

