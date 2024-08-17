
##############################################################################################
# R code to replicate the machine learning validation from the paper
# "Brain white matter pathways of resilience to chronic back pain: a multisite validation"
# Author: Kyungjin Sohn
##############################################################################################

###################################
## Calling necessary libraries
###################################

library(readxl)
library(R.matlab) # read matlab data
library(ggplot2) 
library(dplyr) 
library(e1071) # SVM
library(randomForest, quietly = TRUE) # Random Forest
library(pROC) # AUC
library(stringr)
library(reshape2)
# library(devtools)
# install_github("jfortin1/neuroCombatData", dependencies = TRUE)
# install_github("jfortin1/neuroCombat_Rpackage")
library(neuroCombat) # harmonization


###################################
## Defining functions
###################################

## 01. Combat Harmonization
harmonFun <- function(cm.tnpca, label.1, label.2, label.3) {
     age <- c(label.1$age, label.2$age, label.3$age)
     gender <- c(label.1$sex, label.2$sex, label.3$sex)
     translation <- c(label.1$Translation, label.2$Translation, label.3$Translation)
     rotation <- c(label.1$Rotation, label.2$Rotation, label.3$Rotation)
     batchID <- c(rep(1, nrow(label.1)), rep(2, nrow(label.2)), rep(3, nrow(label.3)))
     pc.harmonized <- neuroCombat(dat=t(cm.tnpca), batch=batchID)
     return(t(pc.harmonized$dat.combat))
}

## 02. Selecting Train Data
selTrainData <- function(ranSeed, mainData, mixYes, n1, n2, n3) {
     set.seed(ranSeed)
     # select the main data set
     if (mainData == "NewHaven") { 
          rowNum <- 1:n1 
     } else if (mainData == "Chicago") { 
          rowNum <- n1 + 1:n2
     } else if (mainData == "Mannheim") { 
          rowNum <- n1+n2 + 1:n3
     } else {
          print(paste("We don't have the following data set:", mainSet))
          break
     }
     #if mix is true, select 40% of data from the other data set
     if (mixYes) {
          if (mainData == "NewHaven") { 
               # Mix Chicago40
               randNum <- sample(n1 + 1:n2, round(n2*0.4), replace = FALSE)
          } else { 
               # Mix NewHaven40
               randNum <- sample(1:n1, round(n1*0.4), replace = FALSE)
          }
          rowNum <- c(rowNum, randNum)
     } 
     
     return(rowNum)
}

## 03. LOOCV(Leave-one-out cross validation)
logisticLOOCV <- function(cm.train, cm.test.1, cm.test.2) {
     # Start LOOCV
     val.LOOCV <- NULL
     train.auc <- NULL
     test1.auc <- NULL
     test2.auc <- NULL
     for (i in 1:nrow(cm.train)) {
          # 1. Split the data 
          cm.LOOCV <- cm.train[-i,]
          cm.LOOCVtest <- cm.train[i,]
          
          # 2. Create a model and predict the left out data
          # 1) logistic regression
          log.fit <- glm(Groups ~ ., data = cm.LOOCV, family = "binomial")
          log.pred.LOOCVtrain <- predict(log.fit, newdata = cm.LOOCV, type = "response")
          log.pred.LOOCVtest <- predict(log.fit, newdata = cm.LOOCVtest, type = "response")
          log.pred.test.1 <- predict(log.fit, newdata = cm.test.1, type = "response")
          log.pred.test.2 <- predict(log.fit, newdata = cm.test.2, type = "response")
          
          # 3. Save the result
          # 1) train
          auc <- auc(cm.LOOCV$Groups, log.pred.LOOCVtrain)
          train.auc <- c(train.auc, auc)
          # 2) validation
          val.LOOCV <- c(val.LOOCV, log.pred.LOOCVtest)
          # 3) test 1
          auc1 <- auc(cm.test.1$Groups, log.pred.test.1)
          test1.auc <- c(test1.auc, auc1)
          # 4) test2
          auc2 <- auc(cm.test.2$Groups, log.pred.test.2)
          test2.auc <- c(test2.auc, auc2)
     }
     val.auc <- auc(cm.train$Groups, val.LOOCV)
     mean.train.auc <- mean(train.auc)
     mean.test1.auc <- mean(test1.auc)
     mean.test2.auc <- mean(test2.auc)
     
     return(c(mean.train.auc, val.auc, mean.test1.auc, mean.test2.auc))
}

svcLOOCV <- function(cm.train, cm.test.1, cm.test.2) {
     # Start LOOCV
     SVCtrial <- NULL
     costs <- c(5, 10, 30, 50, 80, 100)
     for (cost in costs) {
          val.LOOCV <- NULL
          train.auc <- NULL
          test1.auc <- NULL
          test2.auc <- NULL
          for (i in 1:nrow(cm.train)) {
               # 1. Split the data 
               cm.LOOCV <- cm.train[-i,]
               cm.LOOCVtest <- cm.train[i,]
               
               # 2. Create a model and predict the left out data
               # 2) SVC(linear)
               svc.fit <- svm(Groups ~ ., data = cm.LOOCV, kernel = "linear",
                              cost = cost, scale = FALSE, probability = TRUE)
               svc.pred.LOOCVtrain <- attr(predict(svc.fit, newdata = cm.LOOCV,
                                                   probability = TRUE), "probabilities")[,1]
               svc.pred.LOOCVtest <- attr(predict(svc.fit, newdata = cm.LOOCVtest,
                                                  probability = TRUE), "probabilities")[1]
               svc.pred.test.1 <- attr(predict(svc.fit, newdata = cm.test.1,
                                               probability = TRUE), "probabilities")[,1]
               svc.pred.test.2 <- attr(predict(svc.fit, newdata = cm.test.2,
                                               probability = TRUE), "probabilities")[,1]
               # 3. Save the result
               # 1) train
               auc <- auc(cm.LOOCV$Groups, svc.pred.LOOCVtrain)
               train.auc <- c(train.auc, auc)
               # 2) validation
               val.LOOCV <- c(val.LOOCV, svc.pred.LOOCVtest)
               # 3) test 1
               auc1 <- auc(cm.test.1$Groups, svc.pred.test.1)
               test1.auc <- c(test1.auc, auc1)
               # 4) test2
               auc2 <- auc(cm.test.2$Groups, svc.pred.test.2)
               test2.auc <- c(test2.auc, auc2)
          }
          val.auc <- auc(cm.train$Groups, val.LOOCV)
          mean.train.auc <- mean(train.auc)
          mean.test1.auc <- mean(test1.auc)
          mean.test2.auc <- mean(test2.auc)
          
          SVCtrial <- rbind(SVCtrial, 
                            c(mean.train.auc, val.auc, 
                              mean.test1.auc, mean.test2.auc, cost))
     }
     
     return(SVCtrial)
}

rfLOOCV <- function(cm.train, cm.test.1, cm.test.2) {
     # Start LOOCV
     val.LOOCV <- NULL
     train.auc <- NULL
     test1.auc <- NULL
     test2.auc <- NULL
     for (i in 1:nrow(cm.train)) {
          # 1. Split the data 
          cm.LOOCV <- cm.train[-i,]
          cm.LOOCVtest <- cm.train[i,]
          
          # 2. Create a model and predict the left out data
          # 3) Random Forest
          m <- floor(sqrt(ncol(cm.LOOCV)-1))
          rf.fit <- randomForest(Groups ~ ., data = cm.LOOCV, 
                                 mtry = m, importance = TRUE, ntree = 500)
          rf.pred.LOOCVtrain <- predict(rf.fit, newdata = cm.LOOCV, type ='prob')[,2]
          rf.pred.LOOCVtest <- predict(rf.fit, newdata = cm.LOOCVtest, type ='prob')[,2]
          rf.pred.test.1 <- predict(rf.fit, newdata = cm.test.1, type ='prob')[,2]
          rf.pred.test.2 <- predict(rf.fit, newdata = cm.test.2, type ='prob')[,2]
          
          # 3. Save the result
          # 1) train
          auc <- auc(cm.LOOCV$Groups, rf.pred.LOOCVtrain)
          train.auc <- c(train.auc, auc)
          # 2) validation
          val.LOOCV <- c(val.LOOCV, rf.pred.LOOCVtest)
          # 3) test 1
          auc1 <- auc(cm.test.1$Groups, rf.pred.test.1)
          test1.auc <- c(test1.auc, auc1)
          # 4) test2
          auc2 <- auc(cm.test.2$Groups, rf.pred.test.2)
          test2.auc <- c(test2.auc, auc2)
     }
     val.auc <- auc(cm.train$Groups, val.LOOCV)
     mean.train.auc <- mean(train.auc)
     mean.test1.auc <- mean(test1.auc)
     mean.test2.auc <- mean(test2.auc)
     
     return(c(mean.train.auc, val.auc, mean.test1.auc, mean.test2.auc))
}

## 04. Experiment with Multiple Datasets
multi2 <- function(num.pcs, mainData, mixYes, n1, n2, n3, cm.Total) {
     # 1. Train and test data for multiple times
     num.trial <- ifelse(mixYes, 20, 1)
     seeds <- round(seq(1000, 9999, length.out = num.trial))
     auc.logi <- NULL
     auc.svc <- NULL
     auc.svm <- NULL
     auc.rf <- NULL
     result.AUC <- NULL
     for (ranSeed in seeds) {#
          # 1) randomly select rows for train data set
          rowNum <- selTrainData(ranSeed, mainData, mixYes, n1, n2, n3)
          
          cm.train <- cm.Total[rowNum, ]
          cm.train <- subset(cm.train, select=-c(ID))
          cm.train <- cm.train[,c(1:(num.pcs+1))]
          
          # 2) choose rest as a test data set
          set1 <- (1:n1)[!(1:n1) %in% rowNum]
          set2 <- (n1+1:n2)[!(n1+1:n2) %in% rowNum]
          set3 <- (n1+n2+1:n3)[!(n1++n2+1:n3) %in% rowNum]
          testNum <- list()
          if (length(set1) != 0) {
               testNum[[length(testNum)+1]] = set1
          } 
          if (length(set2) != 0) {
               testNum[[length(testNum)+1]] = set2
          } 
          if (length(set3) != 0) {
               testNum[[length(testNum)+1]] = set3
          } 
          
          cm.test.1 <- cm.Total[testNum[[1]],]
          cm.test.1 <- subset(cm.test.1, select=-c(ID))
          cm.test.1 <- cm.test.1[,c(1:(num.pcs+1))]
          
          cm.test.2 <- cm.Total[testNum[[2]],]
          cm.test.2 <- subset(cm.test.2, select=-c(ID))
          cm.test.2 <- cm.test.2[,c(1:(num.pcs+1))]
          
          ## 3) LOOCV
          ## 3-1) Logistic
          auc.sum <- logisticLOOCV(cm.train, cm.test.1, cm.test.2)
          auc.logi <- cbind(auc.logi, t(round(auc.sum, 2)))
          
          ## 3-2) SVC
          auc.sum <- svcLOOCV(cm.train, cm.test.1, cm.test.2)
          auc.svc <- cbind(auc.svc, round(auc.sum, 2))
          
          ## 3-3) Random Forest
          auc.sum <- rfLOOCV(cm.train, cm.test.1, cm.test.2)
          auc.rf <- cbind(auc.rf, t(round(auc.sum, 2)))
     }
     
     # 2. Save the result
     ## 2-1) Logistic
     result.AUC <- rbind(result.AUC,
                         data.frame(method = "Logistic",
                                    pc = num.pcs,
                                    auc.logi))
     ## 2-2) SVC
     result.AUC <- rbind(result.AUC,
                         data.frame(
                              method = paste0("SVC.c", auc.svc[,5]),
                              pc = num.pcs,
                              auc.svc[, -(seq(1, 5*num.trial, by = 5)+4)]))
     
     ## 2-3) RandomForest
     result.AUC <- rbind(result.AUC,
                         data.frame(method = "RandomForest",
                                    pc = num.pcs,
                                    auc.rf))
     return(result.AUC)
}


###################################
## Main: Machine learning validation
###################################

## 01. Prepare Label and Demographics

# CAUTION: Important to set the correct directory to run the code
dataPath <- getwd() # Fix the address if needed.

# 01) DATA 1 - NewHaven
label.1 <- read_excel(paste0(dataPath, "./labels/GroupLabelsPerSite.xlsx"), sheet = "NewHaven", col_names = TRUE)
label.1 <- label.1 %>% 
     mutate(ID = as.character(ID)) %>% 
     rename(Translation = mTranslation,
            Rotation = mRotation) %>% 
     filter(!ID %in% c("1464", "1525", "1534", "1550", 
                       "1909", "1372", "1374")) # Removed based on 95% quantile

# 02) DATA 2 - Chicago
label.2 <- read_excel(paste0(dataPath, "./labels/GroupLabelsPerSite.xlsx"), sheet = "Chicago", col_names = TRUE)
label.2 <- label.2 %>% 
     mutate(ID = str_sub(as.character(ID), 5, 11)) %>% 
     filter(ID != "008") %>% # Removed because of missing matrices
     filter(!ID %in% c("055", "059", "064", "018", "032", "042")) # Removed based on 95% quantile

# 03) DATA 3 - Mannheim
label.3 <- read_excel(paste0(dataPath, "./labels/GroupLabelsPerSite.xlsx"), sheet = "Mannheim", col_names = TRUE)
label.3 <- label.3 %>% 
     mutate(ID = str_sub(as.character(ID), 5, 7)) %>%
     filter(!ID %in% c("590")) # Removed based on 95% quantile

## 02. Choose Model Setting

# 01) Choose the data combination
mainData <- "Chicago" # Options: "NewHaven", "Chicago"
mixYes <- TRUE # Options: TRUE, FALSE

# 02) Select the matrix to test on

# CAUTION: Before running the following code, you need to estimate structural connectivity using population-based structural connectomes (PSC).
#          Then, you need to run TNPCA for dimension reduction. The following matrix name is an example of that output.
#          Refer to the paper for more details.
bestMatrix <- "abcd_desikan_partbrain_subcort_cm_count_processed"
## Options: "abcd_desikan_partbrain_subcort_cm_processed_volumn_100", 
##          "abcd_destrieux_partbrain_subcort_cm_processed_volumn_100", 
##          "abcd_desikan_partbrain_subcort_cm_count_processed" and etc.

if (grepl("desikan", bestMatrix)) {
     TNPCA.dir <- "TNPCA_threeData_desikan"
     resultTitle <- ifelse(grepl("volumn", bestMatrix), 
                           "desikan_volumn", "desikan_count")
} else {
     TNPCA.dir <- "TNPCA_threeData_destrieux"
     resultTitle <- ifelse(grepl("volumn", bestMatrix), 
                           "destrieux_volumn", "destrieux_count")
}

## 03. Train and Test Models
# 01) Read TNPCA Results
cm.tnpca <- readMat(paste0("./derived_TNPCA/", TNPCA.dir, "/", 
                           bestMatrix, "_TNPCA.mat"))$newU

# 02) Combat Harmonization
pc.harmo <- harmonFun(cm.tnpca, label.1, label.2, label.3)

# 03) Remove Confounding Factors
age <- c(label.1$age, label.2$age, label.3$age)
gender <- c(label.1$sex, label.2$sex, label.3$sex)
translation <- c(label.1$Translation, label.2$Translation, label.3$Translation)
rotation <- c(label.1$Rotation, label.2$Rotation, label.3$Rotation)
lm.pc.harmonized <- lm(pc.harmo ~ age+gender+translation+rotation)

cm.tnpca.harmo <- as.data.frame(lm.pc.harmonized$residuals)

# 04) Add labels (ID & Groups)
bindLabel <- rbind(label.1[,1:2], label.2[,1:2], label.3[,1:2])
cm.Total <- cm.tnpca.harmo %>% 
     mutate(ID = c(sort(label.1$ID), sort(label.2$ID), sort(label.3$ID))) %>% 
     inner_join(bindLabel, by = "ID") %>% 
     mutate(Groups = as.factor(ifelse(Groups == "Rec", 1, 0))) %>% 
     dplyr::select("ID", "Groups", everything())

# 05) LOOCV
K <- dim(cm.tnpca)[2]
min.pc <- seq(1, K, by = 4)
result.AUC <- NULL
for (num.pcs in min.pc) {
     multi.auc <- multi2(num.pcs, mainData, mixYes, 
                         nrow(label.1), nrow(label.2), nrow(label.3), cm.Total)
     result.AUC <- rbind(result.AUC, multi.auc)
}

## 04. Save the result
if (!file.exists("./result")) {
     dir.create(file.path("./result/"))
}
write.csv(result.AUC, paste0("./result/",
                             ifelse(mixYes, paste0(mainData, "Mix"), mainData),
                             "_", resultTitle,".csv"),
          row.names = FALSE)
