library(evtree)
library(caret)
library(nnet)
library(reshape2)
library(ggplot2)
library(scales)
library(nnet)
library(MASS)
library(boot)
library(VSURF)
library(mclust)
library(VGAM)
library(class)
library(nnet)

## This code deals with a binary classification. I converted short term and long term to 
## use. 


data("ContraceptiveChoice")
cmc <- ContraceptiveChoice
attach(cmc)
head(cmc)

##Changing "-" to "." so caret package can use it
conv <- c(2:3,5:10)
cmc[,conv] <- lapply(cmc[,conv], make.names)
cmc[,conv] <- lapply(cmc[,conv], factor)

##Converting to use and no use.
cmc$contraceptive_method_used <- as.character(cmc$contraceptive_method_used)
cmc$contraceptive_method_used[cmc$contraceptive_method_used == "short.term" 
                              | cmc$contraceptive_method_used == "long.term"] <- "use"
cmc$contraceptive_method_used <- as.factor(cmc$contraceptive_method_used)

head(cmc)
levels(cmc$contraceptive_method_used)

##Creating a proportion table
prop.table(table(cmc$contraceptive_method_used))


##Stacked bar plots 
plot(contraceptive_method_used ~ wifes_age, data=cmc, main="Contraceptive Method Used According to Age of Wife")
plot(contraceptive_method_used ~ wifes_education, data=cmc, main="Contraceptive Method Used According to Age of Wife")
plot(contraceptive_method_used ~ husbands_education, data=cmc, main="Contraceptive Method Used According to Age of Wife")
plot(contraceptive_method_used ~ number_of_children, data=cmc, main="Contraceptive Method Used According to Age of Wife")
plot(contraceptive_method_used ~ standard_of_living_index, data=cmc, main="Contraceptive Method Used According to Age of Wife")
plot(contraceptive_method_used ~ media_exposure, data=cmc, main="Contraceptive Method Used According to Age of Wife")
plot(contraceptive_method_used ~ wife_now_working, data=cmc, main="Contraceptive Method Used According to Age of Wife")
plot(contraceptive_method_used ~ wifes_religion, data=cmc, main="Contraceptive Method Used According to Age of Wife")


##Should I use a random forest method for variable selection? Is variable selection necessary?
# How VSURF works
# The first is a subset of important
# variables including some redundancy which can be relevant for interpretation, and the second one
# is a smaller subset corresponding to a model trying to avoid redundancy focusing more closely on
# the prediction objective. The two-stage strategy is based on a preliminary ranking of the explanatory
# variables using the random forests permutation-based score of importance and proceeds using a
# stepwise forward strategy for variable introduction.

a <- VSURF_thres(cmc[,1:9],cmc[,10])
plot(a)
##Partitioning the data set for the validation set approach
train_cmc <- createDataPartition(contraceptive_method_used, p=.70, list=FALSE)
names(getModelInfo())
training <- cmc[train_cmc,]
testing <- cmc[-train_cmc,]
head(training)
tail(training)
nrow(training)
nrow(testing)

#####Analysis
##Tuning the parameters
CV <- trainControl(method = "cv", number = 10, classProbs = TRUE)
LOOCV <- trainControl(method = "LOOCV", classProbs = TRUE)

##running a glm logistic regression instead
glm_cv <- train(contraceptive_method_used~., data = cmc, method = "glm", metric = "Accuracy",
                     trControl = CV) ##  10 fold

system.time(train(contraceptive_method_used~., data = cmc, method = "glm", metric = "Accuracy",
                  trControl = CV))

glm_loocv <- train(contraceptive_method_used~., data = cmc, method = "glm", metric = "Accuracy",
                        trControl = LOOCV) ##LOOCV

system.time(train(contraceptive_method_used~., data = cmc, method = "glm", metric = "Accuracy",
                  trControl = LOOCV)) ##LOOCV


glm_cv$results #results for 10 fold method
glm_loocv$results #results for LOOCV

##For validation set approach
fit<- glm(contraceptive_method_used~.,family = binomial(), data = training)
summary(fit)
probs <- predict(fit, testing, type = "response")
predd <- rep('No.use',length(probs))
predd[probs >= 0.5] <- "Use"
t <- table(predd, testing$contraceptive_method_used)
1-sum(diag(table(predd, testing$contraceptive_method_used)))/440

##28.86%

##LDA Approach
lda_CV <- train(contraceptive_method_used~., data = cmc, method = "lda", metric = "Accuracy",
                            trControl = CV)

lda_loocv <- train(contraceptive_method_used~., data = cmc, method = "lda", metric = "Accuracy",
                               trControl = LOOCV)

lda_vs <- lda(contraceptive_method_used~., data = training)

lda_CV$results
lda_loocv$results

lda_pred <- predict(lda_vs, testing)
accuracy_lda <- mean(lda_pred$class == testing$contraceptive_method_used)
accuracy_lda

# > lda_CV$results
# parameter Accuracy     Kappa AccuracySD    KappaSD
# 1      none 0.665991 0.2929669 0.03245357 0.06588878
# > lda_loocv$results
# parameter Accuracy     Kappa
# 1      none  0.67074 0.3033029
# > lda_pred <- predict(lda_vs, testing)
# > accuracy_lda <- mean(lda_pred$class == testing$contraceptive_method_used)
# > accuracy_lda
# [1] 0.6772727


##qda
set.seed(111)
qda_CV <- train(contraceptive_method_used~., data = cmc, method = "qda", metric = "Accuracy",
                            trControl = CV)

qda_loocv <- train(contraceptive_method_used~., data = cmc, method = "qda", metric = "Accuracy",
                               trControl = LOOCV)

qda_vs <- qda(contraceptive_method_used~., data = training)

qda_CV$results
qda_loocv$results

# > qda_CV$results
# parameter  Accuracy     Kappa AccuracySD    KappaSD
# 1      none 0.6463596 0.2441964 0.03454122 0.07325307
# > qda_loocv$results
# parameter  Accuracy     Kappa
# 1      none 0.6429057 0.2372099


qda_pred <- predict(qda_vs, testing)
accuracy_qda <- mean(qda_pred$class == testing$contraceptive_method_used)
accuracy_qda
# 
# > accuracy_qda
# [1] 0.6681819

##KNN
knn_cv <- train(contraceptive_method_used~., data = cmc, method = "knn", metric = "Accuracy",
                            trControl = CV)

knn_loocv <- train(contraceptive_method_used~., data = cmc, method = "knn", metric = "Accuracy",
                               trControl = LOOCV)

knn_cv$results
knn_loocv$results

# > knn_cv$results
# k  Accuracy     Kappa AccuracySD    KappaSD
# 1 5 0.6734987 0.3033459 0.04695363 0.10080245
# 2 7 0.6714670 0.2952737 0.04007754 0.08723251
# 3 9 0.6856836 0.3225717 0.03161035 0.07102405
# > knn_loocv$results
# k  Accuracy     Kappa
# 1 5 0.6782077 0.3111607
# 2 7 0.6849966 0.3242598
# 3 9 0.6877122 0.3252082



##Validation set KNN didn't work'
set.seed(1234)
train_labels <- training$contraceptive_method_used[]
test_labels <- testing$contraceptive_method_used[]
cmc_labels <- cmc$contraceptive_method_used[]
knn_vs <- knn3(contraceptive_method_used~., data = training)


##Works
knn_vs1 <- knn(training[,c(1,4)], testing[,c(1,4)], cl = train_labels, k=2)
table(knn_vs1, test_labels)

1-sum(diag(table(knn_vs1, test_labels)))/440


##MclustDA
cmc_EDDA <- MclustDA(training[,c(1,4)],train_labels,modelType = "EDDA")
cmc_mclustDA <- MclustDA(training[,c(1,4)],train_labels,modelType = "MclustDA")

summary(cmc_mclustDA)
mod <- MclustDA(training[,c(1,4)],train_labels,modelType = "MclustDA")
cvMclustDA(mod, nfold = 10, verbose = FALSE)
pred <- predict(mod, testing[,c(1,4)])

models <- mclust.options()$emModelNames
tab <- matrix(NA, nrow = length(models), ncol = 3)
rownames(tab) <- models
colnames(tab) <- c("BIC", "10-fold CV", "Test error")
for(i in seq(models))
{
  mod <- MclustDA(training[,c(1,4)], train_labels,
                  modelType = "MclustDA", modelNames = models[i])
  tab[i,1] <- mod$bic
  tab[i,2] <- cvMclustDA(mod, nfold = 10, verbose = FALSE)$error
  pred <- predict(mod, testing[,c(1,4)])
  tab[i,3] <- classError(pred$classification, test_labels)$errorRate
}
tab

# BIC 10-fold CV Test error
# EII -11658.82  0.3572120  0.3681818
# VII -11529.22  0.3562439  0.3159091
# EEI -11488.03  0.3301065  0.3159091
# VEI -11420.22  0.3078412  0.2909091
# EVI -11467.22  0.3146176  0.3090909
# VVI -11418.85  0.3155857  0.3022727
# EEE -11472.44  0.3310745  0.3363636
# EVE -11456.44  0.3146176  0.3159091
# VEE -11405.17  0.3126815  0.3181818
# VVE -11442.23  0.3213940  0.2909091
# EEV -11473.43  0.3233301  0.3113636
# VEV -11479.99  0.3107454  0.3022727
# EVV -11467.27  0.3039690  0.3159091
# VVV -11454.66  0.3194579  0.3068182


mod1 <- MclustDA(training[,1:9], train_labels, modelType = "MclustDA", modelNames = "VVE")
cvMclustDA(mod, nfold = 10, verbose = FALSE)
# $error
# [1] 0.3184898

summary(mod1, parameters = TRUE)

tab1 <- matrix(NA, nrow = length(models), ncol = 3)

colnames(tab1) <- c("BIC", "10-fold CV", "Test error")

for(i in seq(models))
{
  mod <- MclustDA(training[,c(1,4)], train_labels,
                  modelType = "EDDA")
  tab1[i,1] <- mod$bic
  tab1[i,2] <- cvMclustDA(mod, nfold = 10, verbose = FALSE)$error
  pred <- predict(mod, testing[,c(1,4)])
  tab1[i,3] <- classError(pred$classification, test_labels)$errorRate
}
tab1

## LOOCV
mod <- MclustDA(cmc[,c(1,4)],cmc_labels,modelType = "MclustDA")
cvMclustDA(mod, nfold = 1473, verbose = FALSE)

##EDDA CV
mod2 <- MclustDA(training[,1:9], train_labels, modelType = "EDDA")
cvMclustDA(mod2, nfold = 10, verbose = FALSE)

##evtree
ev_cv <- train(contraceptive_method_used~., data = cmc, method = "evtree", metric = "Accuracy",
               trControl = CV)

ev_loocv <- train(contraceptive_method_used~., data = cmc, method = "evtree", metric = "Accuracy",
                  trControl = LOOCV)

ev_cv$results

##Validation Set Approach
set.seed(1090)
contt <- evtree(contraceptive_method_used ~ . , data = training) 
plot(contt)
pr <- predict(contt, testing)
1-sum(diag(table(pr, testing$contraceptive_method_used)))/440


ev_cv$results
ev_loocv$results


##Validation set Approach for Random Forest
set.seed(444)
m <- 2:5
tab <- matrix(NA, nrow = length(m), ncol = 1)
rownames(tab) <- m
for (i in 1:length(m)){
  rf_vs <- randomForest(contraceptive_method_used~. , data = training, mtry =m[i] ,
                        ntree = 500)
  pred <- predict(rf_vs, newdata = testing, type = "class")
  table(pred, testing$contraceptive_method_used)
  tab[i,1] <- 1-sum(diag(table(pred, testing$contraceptive_method_used)))/nrow(testing)
  
}
tab

# > tab
# [,1]
# 2 0.2954545
# 3 0.3068182
# 4 0.3090909
# 5 0.3295455

#Regular Tree
tree_vs <- tree(contraceptive_method_used~. , data = training)
rv_vs1 <- cv.tree(tree_vs)
plot(rv_vs1$size, rv_vs1$dev, type = "b")

prune.cmc <- prune.tree(tree_vs, best = 6)
plot(prune.cmc)
text(prune.cmc, pretty = 0)

pred_prune <- predict(prune.cmc, newdata = testing, type = "class")
table(pred_prune, testing$contraceptive_method_used)
1-sum(diag(table(pred_prune, testing$contraceptive_method_used)))/nrow(testing)
#[1] 0.3181818

pred_unprune <- predict(tree_vs, newdata = testing, type = "class")
table(pred_unprune, testing$contraceptive_method_used)
1-sum(diag(table(pred_unprune, testing$contraceptive_method_used)))/nrow(testing)
#[1] 0.3181818

# Bagging
set.seed(1234)
bag.cmc <- randomForest(contraceptive_method_used~., data = training, 
                       mtry = 9, ntree = 1000,
                       importance = TRUE)

bagging_pred <- predict(bag.cmc, newdata = testing, type = "class")
table(bagging_pred, testing$contraceptive_method_used)
1-sum(diag(table(bagging_pred, testing$contraceptive_method_used)))/nrow(testing)
#[1] 0.3272727
importance(bag.cmc)
varImpPlot(bag.cmc)


##SVM
library(e1071)

set.seed(456)
tunelinear <- tune(svm,contraceptive_method_used~., 
                   data=training, kernel="linear",
                   ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100))) 

tunerad <-tune(svm, contraceptive_method_used~., data=training, kernel="radial", 
               ranges=list(cost=c(0.1,1,10,100,1000),gamma=c(0.5,1,2,3,4)))

tunepol <- tune(svm, contraceptive_method_used ~ ., data = training, kernel = "polynomial", 
                ranges = list(cost = c(0.01, 0.1, 1, 5, 10, 100), 
                degree = c(2, 3, 4)))

fit <- svm(contraceptive_method_used~., data = training, kernel = "linear", 
           cost = tunelinear$best.parameters$cost)
fit2 <- svm(contraceptive_method_used~., data = training, kernel = "radial", 
            cost = tunerad$best.parameters$cost)
fit3 <- svm(contraceptive_method_used~., data = training, kernel = "polynomial", 
            cost = tunepol$best.parameters$cost)
summary(fit3)

a  <- function (x){
pred <- predict(x, testing)
t <- table(pred, testing$contraceptive_method_used)
1-sum(diag(t))/440
}
data.frame(cbind(a(fit),a(fit2), a(fit3)))


###10 fold CV
##Linear
svm_cv_linear <- train(contraceptive_method_used~., data = cmc, method = "svmLinear", metric = "Accuracy",cost = 1,
               trControl = CV)

svm_loocv_linear <- train(contraceptive_method_used~., data = cmc, method = "svmLinear", metric = "Accuracy",cost = 1,
                  trControl = LOOCV)

1-svm_cv_linear$results$Accuracy
1-svm_loocv_linear$results$Accuracy

svm_cv_poly <- train(contraceptive_method_used~., data = cmc, method = "svmPoly", metric = "Accuracy",cost = 10,
                      degree = 3,
                       trControl = CV)

svm_loocv_poly <- train(contraceptive_method_used~., data = cmc, method = "svmPoly", metric = "Accuracy",cost = 1,
                          trControl = LOOCV)

1-svm_cv_poly$results
1-svm_loocv_poly$results$Accuracy
