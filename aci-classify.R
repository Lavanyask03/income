# 1. import dataset adult.csv
adult <- read.csv("./adult.csv", header = TRUE)
View(adult)
summary(adult)
# Output:
# age              workclass             fnlwgt         education         education.num   marital.status    
# Min.   :17.00   Length:32561       Min.   :  12285   Length:32561       Min.   : 1.00   Length:32561      
# 1st Qu.:28.00   Class :character   1st Qu.: 117827   Class :character   1st Qu.: 9.00   Class :character  
# Median :37.00   Mode  :character   Median : 178356   Mode  :character   Median :10.00   Mode  :character  
# Mean   :38.58                      Mean   : 189778                      Mean   :10.08                     
# 3rd Qu.:48.00                      3rd Qu.: 237051                      3rd Qu.:12.00                     
# Max.   :90.00                      Max.   :1484705                      Max.   :16.00     

# occupation         relationship           race               sex             capital.gain    capital.loss   
# Length:32561       Length:32561       Length:32561       Length:32561       Min.   :    0   Min.   :   0.0  
# Class :character   Class :character   Class :character   Class :character   1st Qu.:    0   1st Qu.:   0.0  
# Mode  :character   Mode  :character   Mode  :character   Mode  :character   Median :    0   Median :   0.0  
#                                                                             Mean   : 1078   Mean   :  87.3  
#                                                                             3rd Qu.:    0   3rd Qu.:   0.0  
#                                                                             Max.   :99999   Max.   :4356.0 

# hours.per.week  native.country        income         
# Min.   : 1.00   Length:32561       Length:32561      
# 1st Qu.:40.00   Class :character   Class :character  
# Median :40.00   Mode  :character   Mode  :character  
# Mean   :40.44                                        
# 3rd Qu.:45.00                                        
# Max.   :99.00  

# 2. change all "?" to NA
df2 <- adult
df2[df2 == "?"] <- NA
adult <- df2
rm(df2)

# 3. change all character variables to categorical variables
# workclass, education, marital-status, occupation, relationship, race, sex, native-country, income
adult$workclass <- as.factor(adult$workclass)
adult$education <- as.factor(adult$education)
adult$marital.status <- as.factor(adult$marital.status)
adult$occupation <- as.factor(adult$occupation)
adult$relationship <- as.factor(adult$relationship)
adult$race <- as.factor(adult$race)
adult$sex <- as.factor(adult$sex)
adult$native.country <- as.factor(adult$native.country)

# 4. change the output variable income <=50k to 0 and >50k to 1
adult$income <- as.factor(adult$income)
library(plyr)
adult$income <- mapvalues(adult$income, from = c('>50K','<=50K'), to = c(1,0))

# 5. Analyze the histograms and scatter plots
hist(adult$age)
hist(adult$fnlwgt)
pairs(~income + fnlwgt, data = adult) # fnlwgt can be ignored
hist(adult$education.num)
pairs(~income + education.num, data = adult) # education.num can be ignored

# 6. remove unwanted columns - fnlwgt and education.num
adult <- adult[,-3]
adult <- adult[,-4]

# 7. dealing with missing values
sum(is.na(adult$workclass) == TRUE)/32561 # 0.05638647
sum(is.na(adult$occupation) == TRUE)/32561 # 0.05660146
sum(is.na(adult$native.country) == TRUE)/32561 # 0.01790486
# since the number of missing values are very less compared to number of observations we will drop the rows
# remove NA rows from workclass
adult <- adult[!is.na(adult$workclass),]
# 30725 rows
sum(is.na(adult$occupation) == TRUE) # 7 rows with NA for occupation to be deleted
adult <- adult[!is.na(adult$occupation),]
# 30718 rows
sum(is.na(adult$native.country) == TRUE) # 556 rows with NA for native.country
# we can replace NA with mode for native.country
adult$native.country[is.na(adult$native.country)] <- names(which.max(table(adult$native.country)))

# 8. splitting into train set and test set
library("caTools")
set.seed(42)
split <- sample.split(adult, SplitRatio = 0.7) # 70:30
train <- subset(adult, split == TRUE)
test <- subset(adult, split == FALSE)

# 9. training the model - decision tree
library("rpart")
library("rpart.plot")
# build decision tree with train set and method as "class" for classification
decision_tree <- rpart(formula = income~., data = train, method = "class")
# plot the decision tree
rpart.plot(decision_tree, box.palette = 'RdBu')

# 10. predict the income for test set
test$predicted.income <- predict(decision_tree, test, type = "class")
# get the confusion matrix
confMat <- table(test$predicted.income, test$income)
#      0    1
# 0 6740 1169
# 1  350 1192
accuracy <- sum(diag(confMat))/sum(confMat)
# 0.8392763

# 11. random forest
library("randomForest")
random_forest <- randomForest(income~., data = train, method = "class", ntree = 500, do.trace = 100)
# ntree    OOB      1      2
# 100:  14.27%  7.37% 35.09%
# 200:  14.05%  7.24% 34.64%
# 300:  14.12%  7.30% 34.69%
# 400:  14.17%  7.26% 35.05%
# 500:  14.09%  7.22% 34.86%
test$rf.predicted.income <- predict(random_forest, test, type = "class")
rfconfMat <- table(test$rf.predicted.income, test$income)
#      0    1
# 0 6594  829
# 1  496 1532
rfaccuracy <- sum(diag(rfconfMat))/sum(rfconfMat)
# 0.8598032

# 12. plotting ROC curves
# i. decision tree
predictionprobs <- predict(decision_tree, test, type = "prob")
auc <- auc(test$income,predictionprobs[,2])
# Area under the curve: 0.8435
plot(roc(test$income,predictionprobs[,2]))

# ii. random forest
rfpredictionprobs <- predict(random_forest, test, type = "prob")
rfauc <- auc(test$income,rfpredictionprobs[,2])
# Area under the curve: 0.9018
plot(roc(test$income,rfpredictionprobs[,2]))
