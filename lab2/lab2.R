#ASSIGNMENT 1: Lasso and ridge regression

library(dplyr) 
library(glmnet)

tecator = read.csv("tecator.csv", header = TRUE)
n = dim(tecator)[1] 
set.seed(12345)
id = sample(1:n, floor(n * 0.5)) 
train = tecator[id, ]
test = tecator[-id, ]

### 1.1
# Make a linear regression model with Fat as target and the channels as features 
df = train%>%select(Fat, Channel1:Channel100)
model = lm(Fat~., data = df)
summary(model)

# Predicting on training data
train_pred = predict(model, train)

# Predicting on test data
test_pred = predict(model, test)

# Calculating training and test errors (MSE)
train_error = mean((train$Fat - train_pred)^2) 
test_error = mean((test$Fat - test_pred)^2)

# Print the values
train_error 
test_error

# Training Error: The low training error (0.005709117) suggests that the model
# fits the training data well. This indicates that the model can explain a significant
# portion of the variability in the 'Fat' variable within the training set.
# Test Error: The relatively high test error (722.4294) suggests that the model might
# not generalize well to unseen data. Such a large test error may indicate overfitting,
# where the model performs well on the training data but struggles with new, unseen data.
### 1.2

### 1.3
# Extracting the response variable and predictor matrix
y_train = df$Fat
x_train = as.matrix(df[, -1]) # Exclude the 'Fat' column

# Fitting LASSO regression model
lasso_model_train = glmnet(x_train, y_train, family = 'gaussian', alpha = 1) 
plot(lasso_model_train, xvar = "lambda")

#num_variables_cv = which(coef(lasso_model_train)[-1,] != 0) #num_variables_cv

### 1.4
# Fitting Ridge regression model
ridge_model_train = glmnet(x_train, y_train, alpha = 0) 
plot(ridge_model_train, xvar = "lambda")

### 1.5
# Perform cross-validated LASSO regression
lasso_cv_model = cv.glmnet(x_train, y_train, alpha = 1) 

# Plotting CV score vs. log(lambda)
plot(lasso_cv_model)

# Extracting the optimal lambda
optimal_lambda_cv = lasso_cv_model$lambda.min 
optimal_lambda_cv
log_opt = log(optimal_lambda_cv)
log_opt

# Extracting the number of variables (non-zero coefficients) in the model at the optimal lambda
num_variables_cv = sum(coef(lasso_cv_model, s = optimal_lambda_cv) != 0) 
num_variables_cv

# Model for the optimal lambda
lasso_model_best = glmnet(x_train, y_train, family = 'gaussian', alpha = 1, lambda = optimal_lambda_cv) 
#plot(lasso_model_best, xvar = "lambda")

y_test = test$Fat
x_test = as.matrix(test%>%select(Channel1:Channel100))

# Predicting on the test data using the LASSO model with optimal lambda
lasso_test_pred = predict(lasso_model_best, newx = x_test, s = optimal_lambda_cv)

# Scatterplot
plot(y_test, ylab = "y", col="blue") 
points(lasso_test_pred, col="red")


#ASSIGNMENT 2: Decision trees and logistic regression

data = read.csv("bank-full.csv", sep = ";", stringsAsFactors = TRUE)

#1

#remove duration element from data
data$duration = c()

#split data into train, valid and test
n=dim(data)[1] 
set.seed(12345) 
id=sample(1:n, floor(n*0.4)) 
train=data[id,] 
id1=setdiff(1:n, id) 
set.seed(12345) 
id2=sample(id1, floor(n*0.3)) 
valid=data[id2,] 
id3=setdiff(id1,id2) 
test=data[id3,]

#2

library(tree)

#function to Calculate the missclassification errors
missclass=function(X,X1){ 
  n=length(X) 
  return(1-sum(diag(table(X,X1)))/n)
}

#create tree on train data set, tree on train data set with a minimum node size of 7000 #and tree on train data set with a minimum deviance of 0.0005
treeA = tree(as.factor(y) ~ ., data = train)
treeB=tree(as.factor(y)~., data=train, control = tree.control(nrow(train), minsize = 7000)) 
treeC=tree(as.factor(y)~., data=train, control = tree.control(nrow(train), mindev = 0.0005))

#treeA predictions, confusion matrixes and missclass errors
treeAtrainPred=predict(treeA, newdata=train, type="class") 
treeAvalidPred=predict(treeA, newdata=valid, type="class") 
table(train$y,treeAtrainPred)
table(valid$y,treeAvalidPred)

treeAtrainPredmissclass = missclass(train$y, treeAtrainPred) 
treeAvalidPredmissclass = missclass(valid$y, treeAvalidPred)

#treeB predictions, confusion matrixes and missclass errors
treeBtrainPred=predict(treeB, newdata=train, type="class") 
treeBvalidPred=predict(treeB, newdata=valid, type="class") 
table(train$y,treeBtrainPred)
table(valid$y,treeBvalidPred)

treeBtrainPredmissclass = missclass(train$y, treeBtrainPred) 
treeBvalidPredmissclass = missclass(valid$y, treeBvalidPred)

#treeC predictions, confusion matrixes and missclass errors
treeCtrainPred=predict(treeC, newdata=train, type="class") 
treeCvalidPred=predict(treeC, newdata=valid, type="class") 
table(train$y,treeCtrainPred)
table(valid$y,treeCvalidPred)

treeCtrainPredmissclass = missclass(train$y, treeCtrainPred) 
treeCvalidPredmissclass = missclass(valid$y, treeCvalidPred)

#get information on tree sizes and variables etc.
summary(treeA) 
summary(treeB) 
summary(treeC)

#3

#create tree with leaf amount 2-50
testTree=treeC 
trainScore=rep(0,50) 
validScore=rep(0,50) 
for(i in 2:50) {
  prunedTree=prune.tree(testTree,best=i) 
  pred=predict(prunedTree, newdata=valid,
      type="tree") 
  trainScore[i]=deviance(prunedTree) 
  validScore[i]=deviance(pred)
}

#plot tree depending on leaf amount
plot(2:50, trainScore[2:50], type="b", col="red", 
     ylim=c(8000,12000), ylab = "Deviance", xlab = "Leaves")
points(2:50, validScore[2:50], type="b", col="blue")
legend("topright", legend = c("Training", "Validation"), col = c("red", "blue"), lty = 1)

#get leaf amount that leads to the lowest training and validation deviance # + 1 since index 1 in train/validScore[2:50] correspond to tree size 2 #meaning that index + 1 = tree size
bestNrOfLeavesTraining = which.min(trainScore[2:50]) + 1 
bestNrOfLeavesValid = which.min(validScore[2:50]) + 1

bestNrOfLeavesValid

#plot the tree with the leaf amount that leads to the lowest validation deviance
bestTree = prune.tree(testTree, best = bestNrOfLeavesValid) 
plot(bestTree)
text(bestTree, pretty = 0)

#4
#Estimating the confusion matrix, accuracy and F1 score for the test data by using the optimal model from question 3.
bestTreeTestPred=predict(bestTree, newdata=test, type="class") 
conf_matrix = table(test$y,bestTreeTestPred)
conf_matrix
bestTreetestPredmissclass = missclass(test$y, bestTreeTestPred) 
accuracy = 1 - bestTreetestPredmissclass
TP = conf_matrix[2,2] #Number of times true predcitions were correct 
FN = conf_matrix[2,1] #Number of times false predcitions were false 
FP = conf_matrix[1,2] #Number of times true predcitions were false 
P = FN + TP #Number of times true was correct
Recall = TP/P
Precision = TP/(TP+FP)
F1_score = (2*Precision*Recall)/(Precision+Recall)

#5
#Estimating the confusion matrix, accuracy and F1 score again with a loss matrix that punishes wrong false predictions a lot.

loss_matrix = matrix(c(0,5,1,0), nrow = 2, ncol = 2)

#Make predictions on the best tree model with the loss matrix
bestTreePredictedProbabilities = predict(bestTree, newdata=test)
confidenceNeededToPredictNo = loss_matrix[2,1]/(loss_matrix[2,1]+loss_matrix[1,2]) 
bestTreePredictedProbabilitiesWith_LM = ifelse(bestTreePredictedProbabilities[, "no"] > confidenceNeededToPredictNo, "no", "yes")

#Estimate confidence matrix, accuracy and F1 score
conf_matrix = table(test$y,bestTreePredictedProbabilitiesWith_LM)
conf_matrix
bestTreetestWithLossPredmissclass = missclass(test$y, bestTreePredictedProbabilitiesWith_LM ) 
accuracy = 1 - bestTreetestWithLossPredmissclass
TP = conf_matrix[2,2] 
FN = conf_matrix[2,1] 
FP = conf_matrix[1,2] 
P = FN +TP
Recall = TP /P
Precision = TP /(TP +FP )
F1_score = (2*Precision *Recall )/(Precision+Recall) 
accuracy
F1_score

#6
#create logical regression model and calculate probabilities
logistic_reg_model <- glm(as.factor(y) ~ ., data = train, family = "binomial") 
logistic_predicted_probs <- predict(logistic_reg_model, newdata = test, type = "response")

#create empty lists for true positive rates and false positive rates
TPR_tree=rep(0,19) 
FPR_tree=rep(0,19)
TPR_reg=rep(0,19) 
FPR_reg=rep(0,19)

#calculate True positive rate, False positive rate
#for the optimal decision tree and the logical regression model
#for thresholds 0.05-0.95 (threshold = confidence needed to predict yes in this case)

for(i in 1:19) { 
  pi = 0.05 * i
  
  #Doing it for the Tree model
  bestTreePredictedProbabilitiesWith_pi = ifelse(bestTreePredictedProbabilities[, "yes"] > pi, "yes", "no")
  #Add true positive rates and false positive rates to their lists
  conf_matrix = table(test$y,bestTreePredictedProbabilitiesWith_pi) 
  TP = ifelse(dim(conf_matrix)[2] > 1, conf_matrix[2,2], 0)
  FN = conf_matrix[2,1]
  FP = ifelse(dim(conf_matrix)[2] > 1, conf_matrix[1,2], 0)
  TN = conf_matrix[1,1] 
  
  P =FN +TP
  
  N = FP + TN
  Recall = TP /P 
  TPR = Recall
  FPR = FP / N
  
  TPR_tree[i]=TPR 
  FPR_tree[i]=FPR
  
  #Doing the same for the Regression model
  regPredictedProbabilitiesWith_pi = ifelse(logistic_predicted_probs > pi, "yes", "no")
  
  #Add true positive rates and false positive rates to their lists
  conf_matrix = table(test$y,regPredictedProbabilitiesWith_pi) 
  TP = ifelse(dim(conf_matrix)[2] > 1, conf_matrix[2,2], 0)
  FN = conf_matrix[2,1]
  FP = ifelse(dim(conf_matrix)[2] > 1, conf_matrix[1,2], 0)
  TN = conf_matrix[1,1] 
  
  P =FN +TP
  N = FP + TN
  
  Recall = TP /P 
  TPR = Recall 
  FPR = FP / N
  
  TPR_reg[i]=TPR
  FPR_reg[i]=FPR 
}

#plot ROC curves
plot(FPR_tree, TPR_tree, type="l", col="red",
     ylim=c(0,1), xlim=c(0,1), ylab = "True Positive Rate", xlab = "False Positive Rate")
lines(FPR_reg, TPR_reg, type = "l", col = "blue")
legend("bottomright", legend = c("Decision Tree", "Logistic Regression"), col = c("red", "blue"), lty = 1)


#ASSIGNMENT 3: PCA and implicit regularization

set.seed(12345)
comData = as.data.frame(read.csv("communities.csv")) 
pcaData=comData
pcaData$ViolentCrimesPerPop=c()

library(caret)
# ------------ Task 1 ------------

# -------- Scale data --------
scaler=preProcess(pcaData) 
comDataS=predict(scaler,pcaData)

# -------- Calculate cov matrix & EV (PCA implementation) --------
n = nrow(comDataS)
S = (1/n)*t(as.matrix(comDataS))%*%as.matrix(comDataS) 
eigenvalues = eigen(S)

# -------- Calculate proportion of variance for each PC --------
totalSumEv = sum(eigenvalues$values) 
numOfComponents = length(eigenvalues$values) 
proportionOfVarianceVector = c(100)

for (i in 1:numOfComponents) {
  proportionOfVarianceVector[i] = eigenvalues$values[i]/totalSumEv 
}

# -------- Calculate cumlative proportion of variance --------
cumulativeProportionVariance = 0 
k=0

while(cumulativeProportionVariance <= 0.95){
  k = k+1
  cumulativeProportionVariance = cumulativeProportionVariance + proportionOfVarianceVector[k]
}
sprintf("Components needed to obtain atleast 95 percent of variance in the data: %d", k)

pc1 = proportionOfVarianceVector[1] 
pc2 = proportionOfVarianceVector[2] 
sprintf("PC1: %.4f, PC2: %.4f",pc1,pc2)

# ------------ Task 2 ------------

# ------------ PCA Analysis using princomp ------------
res = princomp(comDataS)

# ------------ Traceplot of PC1 ------------
plot(abs(res$loadings[,1]), xlab = "Features", ylab = "PC1", main = "Trace plot of PC1")

# ------------ Get the five most contributing features ------------
contributingFeatures = head(order(abs(res$loadings[,1]),decreasing = TRUE), 5) 
colnames(comDataS)[contributingFeatures]

# ------------ Plot of PC1, PC2, VCPP ------------
PC1 = res$scores[,1]
PC2 = res$scores[,2]
vcpp = comData$ViolentCrimesPerPop

pc_scores_df = data.frame(PC1, PC2, vcpp)

library(ggplot2)
ggplot(pc_scores_df, aes(x = PC1, y = PC2, color = vcpp)) +
  geom_point() +
  labs(title = "Plot of the PC scores PC1, PC2 where color is the level of VCPP "
       , x = "PC1", y = "PC2")

# ------------ Task 3 ------------

# ------------ Partion & Scale data ------------
n=nrow(comData) 
id=sample(1:n, floor(n*0.5)) 
train=comData[id,] 
test=comData[-id,] 
scaler=preProcess(train) 
trainS=predict(scaler,train) 
testS=predict(scaler,test)

# ------------ Create Linear Regression & Calculate MSE ------------
m1 = lm(ViolentCrimesPerPop~., trainS) 
PredsTrain=predict(m1, newdata = trainS) 
trainMSE=mean((trainS$ViolentCrimesPerPop-PredsTrain)^2)
PredsTest=predict(m1, newdata = testS)

testMSE = mean((testS$ViolentCrimesPerPop-PredsTest)^2)

sprintf("Train MSE: %.4f, Test MSE: %.4f", trainMSE, testMSE) # ------------ Calculate Error Percentage ------------
testMSE/trainMSE

# ------------ Task 4 ------------
library(dplyr)

# ------------ Vectors for storing MSE for each iteration ------------
trainMSEValues = c() 
testMSEValues = c() 
i = 0;

# ------------ Cost function for optimizing theta ------------
costFunc = function(theta){
  yTrain = as.matrix(trainS$ViolentCrimesPerPop) 
  yTest = as.matrix(testS$ViolentCrimesPerPop)
  
  xTrain = as.matrix(trainS%>%select(state:LemasPctOfficDrugUn)) 
  xTest = as.matrix(testS%>%select(state:LemasPctOfficDrugUn))
  
  predTrain = xTrain %*% theta 
  predTest = xTest %*% theta
  
  trainMSE = mean((yTrain - predTrain)^2) 
  testMSE = mean((yTest - predTest)^2)
  
  trainMSEValues <<- c(trainMSEValues, trainMSE) 
  testMSEValues <<- c(testMSEValues, testMSE)
  i <<- i+1
  
  return(trainMSE) 
}

theta = rep(0,100)
# ------------ Optimizing theta ------------
optimal_values = optim(par = theta, fn = costFunc, method = "BFGS")

# ------------ Plot Train & Test MSE for each iteration ------------
X = c(500:i)
df_plot = data_frame(x=X,trainMSEValues[500:i],testMSEValues[500:i])

ggplot(df_plot, aes(x = X)) +
  geom_line(aes(y = trainMSEValues[500:i], color = "Train MSE"), linewidth = 1 , linetype = "solid") + 
  geom_line(aes(y = testMSEValues[500:i], color = "Test MSE"), linewidth = 1 , linetype = "solid") + 
  geom_point(aes(x = which.min(testMSEValues), y = testMSEValues[which.min(testMSEValues)],                                                                                                                                                                                                                       color = "Lowest Test MSE"), size = 3, shape = 16, fill = "green") +
  coord_cartesian(xlim = c(0, 10000), ylim = c(0.25, 0.8)) +
  labs(x = "Iterations", y = "MSE", title = "",color = "Legend") +
  scale_color_manual(values = c("Train MSE" = "magenta", "Test MSE" = "black", "Lowest Test MSE" = "red"))

# ------------ Finding optimal iteration & comparing MSE for 3.3/3.4 ------------
xTest = testS%>%select(state:LemasPctOfficDrugUn) 
optTrainMSE = optimal_values$value
optTestIter = which.min(testMSEValues)
optTestMSE = testMSEValues[which.min(testMSEValues)]

sprintf("The optimal iteration number is %d with Test MSE %.4f",optTestIter, optTestMSE) 
sprintf("Lowest Train MSE: %0.4f", optTrainMSE)

sprintf("TrainMSE in 3.3: %0.4f & TrainMSE in 3.4 %0.4f", trainMSE, optTrainMSE) 
sprintf("TestMSE in 3.3: %0.4f & TestMSE in 3.4 %0.4f", testMSE, optTestMSE)