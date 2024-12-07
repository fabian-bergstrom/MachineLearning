# ASSIGNMENT 1: Handwritten digit recognition with
# K nearest neighbors

library(kknn)
data = read.csv("optdigits.csv", header = FALSE)

# Split the data into training, validation and test sets
n=dim(data)[1]
set.seed(12345)
id=sample(1:n, floor(n*0.5))
train=data[id,]
id1=setdiff(1:n, id)
set.seed(12345)
id2=sample(id1, floor(n*0.25))
valid=data[id2,]
id3=setdiff(id1,id2)
test=data[id3,]

#K nearest neighbors on training set
trainingkknn = kknn(as.factor(V65)~., train = train, test = train,
                    k = 30, kernel = "rectangular")
trainPredictions = trainingkknn$fitted.values
table(train$V65, trainPredictions)

#K nearest neighbors on test set
testkknn = kknn(as.factor(V65)~., train = train, test = test, k =
                  30, kernel = "rectangular")
testPredictions = testkknn$fitted.values
table(test$V65, testPredictions)

#Calculate the missclassification errors
missclass=function(X,X1){
  n=length(X)
  return(1-sum(diag(table(X,X1)))/n)
}
missclass_rate_train = missclass(train$V65, trainPredictions)

missclass_rate_test = missclass(test$V65, testPredictions)

#3
#How probable predictions are
training_probabilities = trainingkknn$prob

#Get all cases where the correct digit is 8
eights = which(train$V65 == "8")

#get the 2 cases where the correct digit was 8 and the prediction
#had the most confidence of the digit being an 8
easiest_recognizable =
  eights[order(-training_probabilities[eights, '8'])[1:2]]

#get the 3 cases where the correct digit was 8 and the prediction
#had the least confidence of the digit being an 8
hardest_recognizable = eights[order(training_probabilities[eights,
                                                           '8'])[1:3]]
#Split into each case
easiest_cases = train[easiest_recognizable, ]
case1_easy= easiest_cases[1,]
case2_easy = easiest_cases[2,]

hardest_cases = train[hardest_recognizable, ]
case1_hard = hardest_cases[1,]
case2_hard = hardest_cases[2,]
case3_hard = hardest_cases[3,]

#Create matrixes for each case
matrix_easy1 = matrix(data = as.numeric(case1_easy[,1:64]), nrow =
                        8, ncol = 8, byrow = TRUE)
matrix_easy2 = matrix(data = as.numeric(case2_easy[,1:64]), nrow =
                        8, ncol = 8, byrow = TRUE)

matrix_hard1 = matrix(data = as.numeric(case1_hard[,1:64]), nrow =
                        8, ncol = 8, byrow = TRUE)
matrix_hard2 = matrix(data = as.numeric(case2_hard[,1:64]), nrow =
                        8, ncol = 8, byrow = TRUE)
matrix_hard3 = matrix(data = as.numeric(case3_hard[,1:64]), nrow =
                        8, ncol = 8, byrow = TRUE)

#Create heatmaps for each matrix to see the handwritten digit
heatmap(matrix_easy1, Colv = NA, Rowv = NA)
heatmap(matrix_easy2, Colv = NA, Rowv = NA)
heatmap(matrix_hard1, Colv = NA, Rowv = NA)
heatmap(matrix_hard2, Colv = NA, Rowv = NA)
heatmap(matrix_hard3, Colv = NA, Rowv = NA)

#4
training_errors = c(1:30)
valid_errors = c(1:30)

#Loop over K values from 1 to 30 and do the K nearest
#neighbors on the train and validation set for each K
for (k_value in c(1:30)){
  trainingkknn = kknn(as.factor(V65)~., train = train, test =
                        train, k = k_value, kernel = "rectangular")
  trainpredictions = trainingkknn$fitted.values
  missclass_rate_train = missclass(train$V65, trainpredictions)
  training_errors[k_value] = missclass_rate_train
  validationkknn = kknn(as.factor(V65)~., train = train, test =
                          valid, k = k_value, kernel = "rectangular")
  validpredictions = validationkknn$fitted.values
  missclass_rate_valid = missclass(valid$V65, validpredictions)
  valid_errors[k_value] = missclass_rate_valid
}

#Plot the training and validation errors depending on the K values
plot(c(1:30),training_errors, col="green", ylim = c(0,0.06), xlab
     = "K", ylab = "error_rate")
points(c(1:30), valid_errors, col="blue")
best_K = which.min(valid_errors)

#Do the K nearest neighbors on the test set with the best K
#, the one that resultet in the lowest validation error
testkknnbestK = kknn(as.factor(V65)~., train = train, test = test,
                     k = best_K, kernel = "rectangular")
testpredictionsbestK = testkknnbestK$fitted.values
missclass_rate_test_bestK = missclass(test$V65,
                                      testpredictionsbestK)
#5

cross_entropy_errors = c(1:30)

#Calculate the cross entropy error on thge validation set for K 
# values from 1 to 30
for (k_val in c(1:30)){
  sum = 0
  validkknnentropy = kknn(as.factor(V65)~., train = train, test =
                            valid, k = k_val, kernel = "rectangular")
  validprobabilities = validkknnentropy$prob
  for (i in c(1:dim(valid)[1])){
    for (c in c(0:9)){
      if (valid[i, 65] == c){
        sum = sum + log(validprobabilities[i, c + 1] + 1e-15)
      } }
  }
  
  cross_entropy_errors[k_val] = -sum
}

#Plot cross entropy errors depending on K values
plot(c(1:30),cross_entropy_errors, col="red", xlab = "K", ylab =
       "cross_error_rate")
best_cross_K = which.min(cross_entropy_errors)

# ASSIGNMENT 2: Linear regression and ridge regression

# -------- Set seed so we get the same division of data every time
set.seed(12345)

# -------- Read the data and split the data 60/40 --------
data = read.csv("parkinsons.csv")

n=dim(data)[1]
id = sample(1:n,floor(n*0.6))
train = data[id,]
test = data[-id,]

# -------- Scale the data --------
library(caret)
scaler = preProcess(train) 
trainScaled = predict(scaler, train) 
testScaled = predict(scaler, test)

# -------- Compute Linear Regression for Voice characteristics
m1 = lm(formula = motor_UPDRS ~ Jitter... + Jitter.Abs. +
  Jitter.RAP +
  Jitter.PPQ5 + Jitter.DDP + Shimmer + Shimmer.dB. +
  Shimmer.APQ3 +
  RPDE +
  Shimmer.APQ5 + Shimmer.APQ11 + Shimmer.DDA + NHR + HNR +
  DFA + PPE , data = trainScaled)

# -------- Variables that contributes significantly to the model
  summary(m1)

# -------- Calculate MSE for Both Training & Test Data --------
PredsTrain=predict(m1, newdata = trainScaled)
trainMSE=mean((trainScaled$motor_UPDRS-PredsTrain)^2)

PredsTest=predict(m1, newdata = testScaled)
testMSE = mean((testScaled$motor_UPDRS-PredsTest)^2)

print(paste("Train MSE: ", trainMSE))
print(paste("Test MSE: ", testMSE))

# -------- Functions --------
library(dplyr)

# -------- Loglikelihood function --------
loglikelihood=function(theta,sigma,X,y){ 
  n = nrow(trainScaled)
return (-(n/2) * log(2*pi*sigma^2)
        - (1/(2 * sigma^2))*sum((y - X %*% theta)^2))
}

# -------- Ridge function --------
ridge=function(theta, lambda, X, y){
  sigma = theta[17]
  theta = theta[-17]
  return (lambda * sum(theta^2) - loglikelihood(theta, sigma, X,
y)) 
}

# -------- Optimal Ridge function --------
ridgeOpt=function(lambda){
  y = as.matrix(trainScaled$motor_UPDRS)
  X = as.matrix(trainScaled %>% select(Jitter...:PPE))
  #initial values for theta & sigma where theta is the first 16 values
  #and sigma is the last value
  n = ncol(X)
  initial_values = rep(1,n+1);
  return(optim(par = initial_values, fn = ridge,lambda = lambda, X
 = X, y = y, method = "BFGS"))
}

df = function(lambda){
  X = as.matrix(trainScaled%>%select(Jitter...:PPE)); n = ncol(X);
  I = diag(n);
  p = X %*% solve(t(X)%*%X + lambda * I, t(X))
  return (sum(diag(p)))
}

# -------- Calculate Optimal Values
optimal_values1 = ridgeOpt(lambda = 1)$par
optimal_values100 = ridgeOpt(lambda = 100)$par
optimal_values1000 = ridgeOpt(lambda = 1000)$par

# -------- Fetch the Optimal Theta Values
theta_opt1 = as.matrix(optimal_values1[-17])
theta_opt100 = as.matrix(optimal_values100[-17])
theta_opt1000 = as.matrix(optimal_values1000[-17])

# -------- Select the Voice Characteristics from both Data Sets
x_train = as.matrix(trainScaled%>%select(Jitter...:PPE))
x_test = as.matrix(testScaled%>%select(Jitter...:PPE))

# -------- Calculate predictions based on the training data
pred_train_opt1 = x_train %*% theta_opt1
pred_train_opt100 = x_train %*% theta_opt100
pred_train_opt1000 = x_train %*% theta_opt1000

# -------- Calculate Predictions Based on the Test Data --------
pred_test_opt1 = x_test %*% theta_opt1
pred_test_opt100 = x_test %*% theta_opt100
pred_test_opt1000 = x_test %*% theta_opt1000

# -------- Calculate MSE for Training Data --------
mse_train_opt1 = mean((trainScaled$motor_UPDRS -
                         pred_train_opt1)^2)
mse_train_opt100 = mean((trainScaled$motor_UPDRS -
                           pred_train_opt100)^2)
mse_train_opt1000 = mean((trainScaled$motor_UPDRS -
                            pred_train_opt1000)^2)
print(paste("Train MSE lamda=1: ", mse_train_opt1))
print(paste("Train MSE lamda=100: ", mse_train_opt100))
print(paste("Train MSE lamda=1000: ", mse_train_opt1000))

# -------- Calculate MSE for Test Data --------
mse_test_opt1 = mean((testScaled$motor_UPDRS - pred_test_opt1)^2)
mse_test_opt100 = mean((testScaled$motor_UPDRS -
                          pred_test_opt100)^2)
mse_test_opt1000 = mean((testScaled$motor_UPDRS -
                           pred_test_opt1000)^2)
print(paste("Test MSE lamda=1: ", mse_test_opt1))
print(paste("Test MSE lamda=100: ", mse_test_opt100))
print(paste("Test MSE lamda=1000: ", mse_test_opt1000))

# -------- Calculate the Degrees of Freedom for the Different Lambdas --------
df1 = df(lambda = 1)
df100 = df(lambda = 100)
df1000 = df(lambda = 1000)
print(paste("df for lamda = 1: ",df1))
print(paste("df for lamda = 100: ", df100))
print(paste("df for lamda = 1000: ",df1000))


#ASSIGNMENT 3: Logistic regression and basis function expansion

### 3.1
# read the file and name the columns
indians = read.csv("pima-indians-diabetes.csv", header = FALSE)
colnames(indians)[1] = "pregnant"
colnames(indians)[2] = "glucose"
colnames(indians)[3] = "blood_pressure"
colnames(indians)[4] = "tricep"
colnames(indians)[5] = "insulin"
colnames(indians)[6] = "bmi"
colnames(indians)[7] = "diabetes_pedegree_function"
colnames(indians)[8] = "age"
colnames(indians)[9] = "diabetes"

#plot the graph, plasma glucose concentration on age and mark the persons with diabetes as blue and the rest as black
plot(indians$age, indians$glucose, xlab = "Age", ylab = "Plasma glucose
concentration",
     col = ifelse(indians$diabetes == 1, "blue", "black"), pch = 16)

### 3.2

# Train and set up the model
train=indians%>%select(glucose, age, diabetes)
m1=glm(as.factor(diabetes)~., train, family = "binomial") # diabetes set as target and the other elements as features
Prob=predict(m1, type="response")
Pred=ifelse(Prob>0.5, 1, 0)
table(train$diabetes, Pred) # set up table for classifications
summary(m1)

m1_coeff = m1$coefficients # determine coefficients for the calculation of probabilistic equation of the estimated model

# Calculate misclassification based on
misclassification_error = (64 + 138)/(436 + 64 + 138 + 130)
misclassification_error

# Plot based on the models prediction
plot(indians$age, indians$glucose, xlab = "Age", ylab = "Plasma glucose
concentration", col = ifelse(Pred, "blue", "black"), pch = 16)

### 3.3
# saves the coefficiants
intercept = coef(m1)[1]
glucose_coef = coef(m1)[2]
age_coef = coef(m1)[3]

# sets up values for x to plot the line, x = x2 which is age
x_range = range(indians$age)
x_values = seq(x_range[1], x_range[2], by = 1)

# function for y dependent on x
y_values = - (intercept + age_coef* x_values) / glucose_coef

# plot the line
lines(x_values, y_values, col = "red")

### 3.4

# r = 0.2
# Set up plot like in step 2
Pred_02 = ifelse(Prob>0.2, 1, 0)
plot(indians$age, indians$glucose, xlab = "Age", ylab = "Plasma glucose
concentration", col = ifelse(Pred_02, "blue", "black"), pch = 16)
# Calculates misclassification to compare with earlier model
table(train$diabetes, Pred_02)
misclassification_error_02 = (262 + 24)/(238 + 262 + 24 + 244)
misclassification_error_02

# r = 0.8
# Set up plot like in step 2
Pred_08 = ifelse(Prob>0.8, 1, 0)
plot(indians$age, indians$glucose, xlab = "Age", ylab = "Plasma glucose
concentration", col = ifelse(Pred_08, "blue", "black"), pch = 16)

# Calculates misclassification to compare with earlier model
table(train$diabetes, Pred_08)
misclassification_error_08 = (10 + 232)/(490 + 10 + 232 + 36)
misclassification_error_08

### 3.5

# Define x1 and x2
x1 = indians$glucose
x2 = indians$age

# Creates all z values
z1 = x1^4
z2 = (x1^3)*x2
z3 = (x1^2)*(x2^2)
z4 = x1*(x2^3)
z5 = x2^4

# Add them to our data frame
indians$z1 = z1
indians$z2 = z2
indians$z3 = z3
indians$z4 = z4
indians$z5 = z5
indians

# Train and set up the model
trainZ = indians%>%select(glucose, age, diabetes, z1, z2, z3, z4, z5)

mZ=glm(as.factor(diabetes)~., trainZ, family = "binomial") # diabetes set as target and the other elements as features
ProbZ=predict(mZ, type="response")
PredZ=ifelse(ProbZ>0.5, 1, 0)
table(trainZ$diabetes, PredZ) # set up table for classifications
summary(mZ)

mZ_coeff = mZ$coefficients # determine coefficients for the calculation of probabilistic equation of the estimated model

# Calculate misclassification based on
misclassification_error_Z = (67 + 121)/(433 + 67 + 121 + 147)
misclassification_error_Z

# Plot based on the models prediction
plot(indians$age, indians$glucose, xlab = "Age", ylab = "Plasma glucose
concentration", col = ifelse(PredZ, "blue", "black"), pch = 16)