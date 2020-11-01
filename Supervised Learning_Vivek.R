
#Set working directory
setwd("C:\\Users\\vivek_000\\Desktop\\Data Mining\\Assignment-3")
credit_dataset <-read.csv("Credit_Dataset.csv")

#Create Label
credit_dataset$PROFITABLE = ifelse(credit_dataset$PROFIT>=0,1,0)

#Convert Data Types
credit_dataset$CHK_ACCT = as.factor(credit_dataset$CHK_ACCT)
credit_dataset$SAV_ACCT = as.factor(credit_dataset$SAV_ACCT)
credit_dataset$HISTORY = as.factor(credit_dataset$HISTORY)
credit_dataset$JOB = as.factor(credit_dataset$JOB)
credit_dataset$TYPE = as.factor(credit_dataset$TYPE)

#Split Data
set.seed(12345)
test_inst = sample(nrow(credit_dataset),0.3*nrow(credit_dataset))
credit_test = credit_dataset[test_inst,]
credit_rest = credit_dataset[-test_inst,]

valid_inst = sample(nrow(credit_rest),0.25*nrow(credit_rest))
credit_valid = credit_rest[valid_inst,]
credit_train = credit_rest[-valid_inst,]

#Logistic Regression Model
logmodel = glm(PROFITABLE~AGE+DURATION+RENT+TELEPHONE+FOREIGN+CHK_ACCT+SAV_ACCT+HISTORY+JOB+TYPE, data = credit_train, family = "binomial")
summary(logmodel)

#Receiver operating characteristic
library(ROCR)

logmodel_pred = predict(logmodel, newdata=credit_valid, type="response")

pred = prediction(logmodel_pred, credit_valid$PROFITABLE)


tpr = performance(pred, measure = 'tpr')
tnr = performance(pred, measure = 'tnr')
acc = performance(pred, measure = 'acc')

plot(tpr,ylim=c(0,1), col='green', main="Trade off between metrics")
plot(tnr,add=T, col ='red')
plot(acc,add=T)

index = which.max(slot(acc,"y.values")[[1]])
index

max_acc = slot(acc,"y.values")[[1]][index]
max_acc

max_cutoff = slot(acc,"x.values")[[1]][index]
max_cutoff


logmodel_pred2 = predict(logmodel, newdata = credit_train, type = 'response')

pred2 = prediction(logmodel_pred2, credit_train$PROFITABLE)

roc_train = performance(pred2, measure = 'tpr', x.measure = 'fpr')
plot(roc_train, col = 'red', main="ROC Curve")
abline(a=0,b=1, lty=3)

roc_valid = performance(pred, measure = "tpr", x.measure = "fpr")
plot(roc_valid, add = T, col = 'green')

auc_train = performance(pred2, measure = 'auc')
auc_train

auc_valid = performance(pred, measure = 'auc')
auc_valid

#Lift Curve

lift_valid = performance(pred, measure = 'lift', x.measure = 'rpp')
plot(lift_valid, col = 'blue', main="Lift")

index_lift = which.max(slot(lift_valid,"y.values")[[1]])
index_lift

max_lift = slot(lift_valid,"y.values")[[1]][index_lift]
max_lift


df = data.frame(slot(lift_valid,"y.values"),slot(lift_valid,"x.values"))
df

#Decision Tree

library(tree)

credit_train$PROFITABLE = as.factor(credit_train$PROFITABLE)
credit_valid$PROFITABLE = as.factor(credit_valid$PROFITABLE)

credit_tree = tree(PROFITABLE~AGE+DURATION+RENT+TELEPHONE+FOREIGN+CHK_ACCT+SAV_ACCT+HISTORY+JOB+TYPE, data = credit_train)
summary(credit_tree)
plot(credit_tree)
text(credit_tree, pretty = 1)

#12 terminal nodes in full tree
#11 decision nodes in full tree

credit_tree_2 = prune.tree(credit_tree, best = 2)
summary(credit_tree_2)
plot(credit_tree_2)
text(credit_tree_2, pretty = 1)

credit_tree_4 = prune.tree(credit_tree, best = 4)
summary(credit_tree_4)
plot(credit_tree_4)
text(credit_tree_4, pretty = 1)

credit_tree_6 = prune.tree(credit_tree,  best = 6)
summary(credit_tree_6)
plot(credit_tree_6)
text(credit_tree_6, pretty = 1)

credit_tree_8 = prune.tree(credit_tree,  best = 8)
summary(credit_tree_8)
plot(credit_tree_8)
text(credit_tree_8, pretty = 1)

credit_tree_10 = prune.tree(credit_tree,  best = 10)
summary(credit_tree_10)
plot(credit_tree_10)
text(credit_tree_10, pretty = 1)


#Prediction

predict_and_classify <- function(treename, pred_data, actuals, cutoff)
{
  probs <- predict(treename, newdata = pred_data)[,2]
  classifications <- ifelse(probs>cutoff,1,0)
  accuracy <- sum(ifelse(classifications==actuals,1,0))/length(actuals)
  return(accuracy)
}


full_tree_train_acc <- predict_and_classify(credit_tree, credit_train, credit_train$PROFITABLE, 0.5)
pruned_tree2_train_acc <- predict_and_classify(credit_tree_2, credit_train, credit_train$PROFITABLE, 0.5)
pruned_tree4_train_acc <- predict_and_classify(credit_tree_4, credit_train, credit_train$PROFITABLE, 0.5)
pruned_tree6_train_acc <- predict_and_classify(credit_tree_6, credit_train, credit_train$PROFITABLE, 0.5)
pruned_tree8_train_acc <- predict_and_classify(credit_tree_8, credit_train, credit_train$PROFITABLE, 0.5)
pruned_tree10_train_acc <- predict_and_classify(credit_tree_10, credit_train, credit_train$PROFITABLE, 0.5)


full_tree_valid_acc <- predict_and_classify(credit_tree, credit_valid, credit_valid$PROFITABLE, 0.5)
pruned_tree2_valid_acc <- predict_and_classify(credit_tree_2, credit_valid, credit_valid$PROFITABLE, 0.5)
pruned_tree4_valid_acc <- predict_and_classify(credit_tree_4, credit_valid, credit_valid$PROFITABLE, 0.5)
pruned_tree6_valid_acc <- predict_and_classify(credit_tree_6, credit_valid, credit_valid$PROFITABLE, 0.5)
pruned_tree8_valid_acc <- predict_and_classify(credit_tree_8, credit_valid, credit_valid$PROFITABLE, 0.5)
pruned_tree10_valid_acc <- predict_and_classify(credit_tree_10, credit_valid, credit_valid$PROFITABLE, 0.5)

xval = c(2,4,6,8,10,12)
xval
yval_train = c(pruned_tree2_train_acc, pruned_tree4_train_acc, pruned_tree6_train_acc, pruned_tree8_train_acc, pruned_tree10_train_acc, full_tree_train_acc)
yval_train

yval_valid = c(pruned_tree2_valid_acc, pruned_tree4_valid_acc, pruned_tree6_valid_acc, pruned_tree8_valid_acc, pruned_tree10_valid_acc, full_tree_valid_acc)
yval_valid


plot(xval, yval_train, type='b', col = 'red', ylim = c(0,1), xlab="Tree Size", ylab="Accuracy", main="Train: Red, Valid: Green")
lines(xval, yval_valid, type='b', col = 'green')



#Knn
library(class)

credit_train$CHK_ACCT = as.integer(credit_train$CHK_ACCT)
credit_train$SAV_ACCT = as.integer(credit_train$SAV_ACCT)
credit_train$HISTORY = as.integer(credit_train$HISTORY)
credit_train$JOB = as.integer(credit_train$JOB)
credit_train$TYPE = as.integer(credit_train$TYPE)
credit_train$PROFITABLE = as.numeric(credit_train$PROFITABLE)

credit_valid$CHK_ACCT = as.integer(credit_valid$CHK_ACCT)
credit_valid$SAV_ACCT = as.integer(credit_valid$SAV_ACCT)
credit_valid$HISTORY = as.integer(credit_valid$HISTORY)
credit_valid$JOB = as.integer(credit_valid$JOB)
credit_valid$TYPE = as.integer(credit_valid$TYPE)
credit_valid$PROFITABLE = as.numeric(credit_valid$PROFITABLE)

colnames(credit_train)
for (i in 1:length(colnames(credit_train)))
{
  print(cat(i," ",names(credit_train)[i]))

}

#normalise function for training data
normalise <- function (my_column)
{
  my_range = max(my_column)-min(my_column)
  my_column = (my_column-min(my_column))/my_range
    
  return(my_column)
  
}

#another normalise function for validation/testing data
normalise2 <- function (my_column, my_column2)
{
  my_range = max(my_column)-min(my_column)
  my_column2 = (my_column2-min(my_column))/my_range
  
  return(my_column2)
  
}

credit_train$AGE_norm = normalise(credit_train$AGE)
credit_train$CHK_ACCT_norm = normalise(credit_train$CHK_ACCT)
credit_train$SAV_ACCT_norm = normalise(credit_train$SAV_ACCT)
credit_train$DURATION_norm = normalise(credit_train$DURATION)
credit_train$HISTORY_norm = normalise(credit_train$HISTORY)
credit_train$JOB_norm = normalise(credit_train$JOB)
credit_train$RENT_norm = normalise(credit_train$RENT)
credit_train$TELEPHONE_norm = normalise(credit_train$TELEPHONE)
credit_train$FOREIGN_norm = normalise(credit_train$FOREIGN)
credit_train$TYPE_norm = normalise(credit_train$TYPE)


credit_valid$AGE_norm = normalise2(credit_train$AGE, credit_valid$AGE)
credit_valid$CHK_ACCT_norm = normalise2(credit_train$CHK_ACCT, credit_valid$CHK_ACCT)
credit_valid$SAV_ACCT_norm = normalise2(credit_train$SAV_ACCT, credit_valid$SAV_ACCT)
credit_valid$DURATION_norm = normalise2(credit_train$DURATION, credit_valid$DURATION)
credit_valid$HISTORY_norm = normalise2(credit_train$HISTORY, credit_valid$HISTORY)
credit_valid$JOB_norm = normalise2(credit_train$JOB, credit_valid$JOB)
credit_valid$RENT_norm = normalise2(credit_train$RENT, credit_valid$RENT)
credit_valid$TELEPHONE_norm = normalise2(credit_train$TELEPHONE, credit_valid$TELEPHONE)
credit_valid$FOREIGN_norm = normalise2(credit_train$FOREIGN, credit_valid$FOREIGN)
credit_valid$TYPE_norm = normalise2(credit_train$TYPE, credit_valid$TYPE)

train.X = credit_train[,c(25:34)]
valid.X = credit_valid[,c(25:34)]

train.Y = credit_train$PROFITABLE
valid.Y = credit_valid$PROFITABLE

credit_knn1 = knn(train.X, valid.X, train.Y, k=1)
table(valid.Y,credit_knn1)
knn1_valid_acc = (table(valid.Y,credit_knn1)[1] + table(valid.Y,credit_knn1)[4])/sum(table(valid.Y,credit_knn1))
knn1_valid_acc

credit_knn3 = knn(train.X, valid.X, train.Y, k=3)
table(valid.Y,credit_knn3)
knn3_valid_acc = (table(valid.Y,credit_knn3)[1] + table(valid.Y,credit_knn3)[4])/sum(table(valid.Y,credit_knn3))
knn3_valid_acc

credit_knn5 = knn(train.X, valid.X, train.Y, k=5)
table(valid.Y,credit_knn5)
knn5_valid_acc = (table(valid.Y,credit_knn5)[1] + table(valid.Y,credit_knn5)[4])/sum(table(valid.Y,credit_knn5))
knn5_valid_acc

credit_knn7 = knn(train.X, valid.X, train.Y, k=7)
table(valid.Y,credit_knn7)
knn7_valid_acc = (table(valid.Y,credit_knn7)[1] + table(valid.Y,credit_knn7)[4])/sum(table(valid.Y,credit_knn7))
knn7_valid_acc

credit_knn11 = knn(train.X, valid.X, train.Y, k=11)
table(valid.Y,credit_knn11)
knn11_valid_acc = (table(valid.Y,credit_knn11)[1] + table(valid.Y,credit_knn11)[4])/sum(table(valid.Y,credit_knn11))
knn11_valid_acc

credit_knn15 = knn(train.X, valid.X, train.Y, k=15)
table(valid.Y,credit_knn15)
knn15_valid_acc = (table(valid.Y,credit_knn15)[1] + table(valid.Y,credit_knn15)[4])/sum(table(valid.Y,credit_knn15))
knn15_valid_acc

credit_knn21 = knn(train.X, valid.X, train.Y, k=21)
table(valid.Y,credit_knn21)
knn21_valid_acc = (table(valid.Y,credit_knn21)[1] + table(valid.Y,credit_knn21)[4])/sum(table(valid.Y,credit_knn21))
knn21_valid_acc

credit_knn25 = knn(train.X, valid.X, train.Y, k=25)
table(valid.Y,credit_knn25)
knn25_valid_acc = (table(valid.Y,credit_knn25)[1] + table(valid.Y,credit_knn25)[4])/sum(table(valid.Y,credit_knn25))
knn25_valid_acc

credit_knn31 = knn(train.X, valid.X, train.Y, k=31)
table(valid.Y,credit_knn31)
knn31_valid_acc = (table(valid.Y,credit_knn31)[1] + table(valid.Y,credit_knn31)[4])/sum(table(valid.Y,credit_knn31))
knn31_valid_acc

credit_knn35 = knn(train.X, valid.X, train.Y, k=35)
table(valid.Y,credit_knn35)
knn35_valid_acc = (table(valid.Y,credit_knn35)[1] + table(valid.Y,credit_knn35)[4])/sum(table(valid.Y,credit_knn35))
knn35_valid_acc


#training accuracy:


credit_knn1_train = knn(train.X, train.X, train.Y, k=1)
table(train.Y,credit_knn1_train)
knn1_train_acc = (table(train.Y,credit_knn1_train)[1] + table(train.Y,credit_knn1_train)[4])/sum(table(train.Y,credit_knn1_train))
knn1_train_acc

credit_knn3_train = knn(train.X, train.X, train.Y, k=3)
table(train.Y,credit_knn3_train)
knn3_train_acc = (table(train.Y,credit_knn3_train)[1] + table(train.Y,credit_knn3_train)[4])/sum(table(train.Y,credit_knn3_train))
knn3_train_acc

credit_knn5_train = knn(train.X, train.X, train.Y, k=5)
table(train.Y,credit_knn5_train)
knn5_train_acc = (table(train.Y,credit_knn5_train)[1] + table(train.Y,credit_knn5_train)[4])/sum(table(train.Y,credit_knn5_train))
knn5_train_acc

credit_knn7_train = knn(train.X, train.X, train.Y, k=7)
table(train.Y,credit_knn7_train)
knn7_train_acc = (table(train.Y,credit_knn7_train)[1] + table(train.Y,credit_knn7_train)[4])/sum(table(train.Y,credit_knn7_train))
knn7_train_acc

credit_knn11_train = knn(train.X, train.X, train.Y, k=11)
table(train.Y,credit_knn11_train)
knn11_train_acc = (table(train.Y,credit_knn11_train)[1] + table(train.Y,credit_knn11_train)[4])/sum(table(train.Y,credit_knn11_train))
knn11_train_acc

credit_knn15_train = knn(train.X, train.X, train.Y, k=15)
table(train.Y,credit_knn15_train)
knn15_train_acc = (table(train.Y,credit_knn15_train)[1] + table(train.Y,credit_knn15_train)[4])/sum(table(train.Y,credit_knn15_train))
knn15_train_acc

credit_knn21_train = knn(train.X, train.X, train.Y, k=21)
table(train.Y,credit_knn21_train)
knn21_train_acc = (table(train.Y,credit_knn21_train)[1] + table(train.Y,credit_knn21_train)[4])/sum(table(train.Y,credit_knn21_train))
knn21_train_acc

credit_knn25_train = knn(train.X, train.X, train.Y, k=25)
table(train.Y,credit_knn25_train)
knn25_train_acc = (table(train.Y,credit_knn25_train)[1] + table(train.Y,credit_knn25_train)[4])/sum(table(train.Y,credit_knn25_train))
knn25_train_acc

credit_knn31_train = knn(train.X, train.X, train.Y, k=31)
table(train.Y,credit_knn31_train)
knn31_train_acc = (table(train.Y,credit_knn31_train)[1] + table(train.Y,credit_knn31_train)[4])/sum(table(train.Y,credit_knn31_train))
knn31_train_acc

credit_knn35_train = knn(train.X, train.X, train.Y, k=35)
table(train.Y,credit_knn35_train)
knn35_train_acc = (table(train.Y,credit_knn35_train)[1] + table(train.Y,credit_knn35_train)[4])/sum(table(train.Y,credit_knn35_train))
knn35_train_acc

#Plot metrics

Xval = c(1,3,5,7,11,15,21,25,31,35)
  
Yval_valid = c(knn1_valid_acc, knn3_valid_acc, knn5_valid_acc, knn7_valid_acc, knn11_valid_acc, knn15_valid_acc, knn21_valid_acc, knn25_valid_acc, knn31_valid_acc, knn35_valid_acc)
  
Yval_train = c(knn1_train_acc, knn3_train_acc, knn5_train_acc, knn7_train_acc, knn11_train_acc, knn15_train_acc, knn21_train_acc, knn25_train_acc, knn31_train_acc, knn35_train_acc)


plot(Xval, Yval_train, type='b', col = 'red', ylim = c(0,1), xlab="K-neighbours", ylab="Accuracy", main="Train: Red, Valid: Green")
lines(Xval, Yval_valid, type='b', col = 'green')



#Logistic Regression Model:

logmodel_pred3 = predict(logmodel, newdata = credit_test, type = 'response')

pred3 = prediction(logmodel_pred3, credit_test$PROFITABLE)

acc_test = performance(pred3, measure = 'acc')
acc_test


df2 = data.frame(slot(acc_test,"y.values"),slot(acc_test,"x.values"))
df2

#accuracy at our optimal cut-off of 0.66 is 0.68

#max accuracy for our test model of 0.73 occurs at a cutoff 0.28: - not required here !

index2 = which.max(slot(acc_test,"y.values")[[1]])
index2

max_acc2 = slot(acc_test,"y.values")[[1]][index2]
max_acc2

max_cutoff2 = slot(acc_test,"x.values")[[1]][index2]
max_cutoff2


#Tree model accuracy is 0.71:

credit_test$PROFITABLE = as.factor(credit_test$PROFITABLE)

pruned_tree8_test_acc <- predict_and_classify(credit_tree_8, credit_test, credit_test$PROFITABLE, 0.5)
pruned_tree8_test_acc

#knn model accuracy is 0.73 :

#convert to numeric
credit_test$CHK_ACCT = as.integer(credit_test$CHK_ACCT)
credit_test$SAV_ACCT = as.integer(credit_test$SAV_ACCT)
credit_test$HISTORY = as.integer(credit_test$HISTORY)
credit_test$JOB = as.integer(credit_test$JOB)
credit_test$TYPE = as.integer(credit_test$TYPE)
credit_test$PROFITABLE = as.numeric(credit_test$PROFITABLE)

#normalise test data using previously defined function
credit_test$AGE_norm = normalise2(credit_train$AGE, credit_test$AGE)
credit_test$CHK_ACCT_norm = normalise2(credit_train$CHK_ACCT, credit_test$CHK_ACCT)
credit_test$SAV_ACCT_norm = normalise2(credit_train$SAV_ACCT, credit_test$SAV_ACCT)
credit_test$DURATION_norm = normalise2(credit_train$DURATION, credit_test$DURATION)
credit_test$HISTORY_norm = normalise2(credit_train$HISTORY, credit_test$HISTORY)
credit_test$JOB_norm = normalise2(credit_train$JOB, credit_test$JOB)
credit_test$RENT_norm = normalise2(credit_train$RENT, credit_test$RENT)
credit_test$TELEPHONE_norm = normalise2(credit_train$TELEPHONE, credit_test$TELEPHONE)
credit_test$FOREIGN_norm = normalise2(credit_train$FOREIGN, credit_test$FOREIGN)
credit_test$TYPE_norm = normalise2(credit_train$TYPE, credit_test$TYPE)

test.X = credit_test[,c(25:34)]

test.Y = credit_test$PROFITABLE

credit_knn_test = knn(train.X, test.X, train.Y, k=3)
table(test.Y,credit_knn_test)
knn_test_acc = (table(test.Y,credit_knn_test)[1] + table(test.Y,credit_knn_test)[4])/sum(table(test.Y,credit_knn_test))
knn_test_acc


#extra: calculate accuracy at any cut-off for logistic regression model

predict_and_classify2 <- function(modelname, pred_data, actuals, cutoff)
{
  probs = predict(modelname, newdata=pred_data, type="response")
  classifications <- ifelse(probs>cutoff,1,0)
  accuracy <- sum(ifelse(classifications==actuals,1,0))/length(actuals)
  return(accuracy)
}

predict_and_classify2(logmodel, credit_valid, credit_valid$PROFITABLE, 0.5)
predict_and_classify2(logmodel, credit_valid, credit_valid$PROFITABLE, 0.66)
