# The MIT License (MIT)
# 
# Copyright (c) 2018 Mladen Jovanović
#
#
# For using/citing this code please use the following reference
# --------------------------------------------------------------
# Jovanovic, M.(2018)
# Predicting non-contact hamstring injuries by using training load data
# Complementarytraining.net
# URL: www.complementarytraining.net/predicting-hamstring-injuries
# --------------------------------------------------------------- 
#
# email: coach.mladen.jovanovic@gmail.com
# web: www.complementarytraining.net
# twitter: physical_prep
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#    
#     The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Load Libraries
library(tidyverse)
library(caret)
library(pROC)
library(rlist)
library(reshape2)
library(hrbrthemes)

# GGplot theme for all the graphs
theme_bmbp <- function(...) {
    theme_ipsum(base_family = "Helvetica", axis_text_size = 8, axis_title_size = 10)
}

# Function for calculating ROC
calculateROC <- function(learningModel, predictors, target, fileName) {
    pdf(fileName)
    predictions <- predict(learningModel, data.matrix(predictors), type = "prob")$Injured
    ROC.curve <- pROC::roc(response = target,
                           predictor = predictions,
                           # arguments for ci
                           ci = TRUE, boot.n = 2000, stratified = FALSE,
                           # argument for lotting
                           plot = TRUE, grid = TRUE,
                           print.auc = TRUE, show.thres = TRUE)
    
    dev.off()
    return(as.numeric(ROC.curve$ci))
}

# Function for plotting predictions
plotInjuryPrediction <- function(learningModel, predictors, target,
                                 players, season = 1, dates) {
    predictions <- predict(learningModel, data.matrix(predictors), type = "prob")$Injured
    
    plotDF <- data.frame(season = season,
                         players = players,
                         date = dates,
                         injury = ifelse(target == "Injured", 1, NA),
                         probability = predictions)
    
    plotDF <- plotDF %>%
        filter(date > 0)
    
    gg <- ggplot(plotDF, aes(x = date, y = injury)) +
        geom_bar(stat = "identity", width = 1, alpha = 0.5, color = "red") +
        geom_line(aes(x = date, y = probability), color = "black", alpha = 0.9, size = 0.2) +
        facet_wrap(~players+season) +
        theme_bmbp() +
        xlab("Day") + ylab("Injury likelihood")
    
    return(gg)
}


# Load data
injuryData <- read.csv("Prepared Data.csv",
                         header = TRUE,
                         stringsAsFactors = FALSE)


######################
# Descriptive analysis
######################

# Table with injuries
injuriesSummary <-  injuryData %>%
    group_by(Season, Player.Name) %>%
    summarise(HS.Injury.Count = sum(Injury01 == "Injured")) %>%
    filter(HS.Injury.Count > 0) %>%
    dcast(Player.Name~Season, value.var = "HS.Injury.Count") %>%
    arrange(Player.Name)

names(injuriesSummary) <- c("", "Season 1", "Season 2")    
write.csv(injuriesSummary, "Descriptive/Hamstring Injuries Summary.csv", row.names = FALSE, na = "")

# Season summaries
seasonSummary <- injuryData %>%
    group_by(Season) %>%
    summarize(Total.Athletes = length(unique(Player.Name)),
              Duration = max(Date),
              HS.Injuries = sum(Injury01 == "Injured"),
              Injured.Athletes = length(unique(ifelse(Injury01 == "Injured", Player.Name, NA))) - 1)
write.csv(seasonSummary, "Descriptive/Seasons Summary.csv", row.names = FALSE, na = "")

# Data split summary
dataSplitSummary <- injuryData %>%
    group_by(Data.Split) %>%
    summarize(Total.Athletes = length(unique(Player.Name)),
              Duration = max(Date),
              HS.Injuries = sum(Injury01 == "Injured"),
              Injured.Athletes = length(unique(ifelse(Injury01 == "Injured", Player.Name, NA))) - 1)
write.csv(dataSplitSummary, "Descriptive/Data Split Summary.csv", row.names = FALSE, na = "")


######################
## Prediction
######################

# Data frame to save model performance
modelPerformance <- data.frame(Model = character(0),
                               InjuryLead = character(0),
                               metric = character(0),
                               lower = numeric(0),
                               value = numeric(0),
                               upper = numeric(0),
                               stringsAsFactors = FALSE)

# Object to save actual models
predictionModels <- list()

# Set the seed for the random number generator
set.seed(6667)

# Split the data
trainingData <- filter(injuryData, Data.Split == "Train")
testingData <- filter(injuryData, Data.Split == "Test")

# Control parameters for Smote
trainCtrl <- trainControl(method="repeatedcv",
                          repeats = 10,
                          number = 3,
                          summaryFunction = twoClassSummary,
                          classProbs = TRUE,
                          verboseIter = TRUE,
                          savePredictions = FALSE,
                          returnData = FALSE, 
                          sampling = "smote",
                          allowParallel = TRUE)

# Main LOOP
for (injuryLeadIndex in 2:4) {
    
    trainingTargetVariable <- factor(trainingData[[7 + injuryLeadIndex]])
    testingTargetVariable <- factor(testingData[[7 + injuryLeadIndex]]) 

    injuryLeadName <- names(trainingData[7 + injuryLeadIndex])
    cat("Injury Lead:", injuryLeadName, "\n")
    
    # Create DF for model building and testing
    trainingDF <- trainingData %>% select(-(1:11))
    testingDF <- testingData %>% select(-(1:11))
    
    #################################
    # Model (PCA Logistic Regression)
    model <- train(x = trainingDF, y = trainingTargetVariable,
                  method = "glm",
                  preProcess = c("center", "scale", "pca"),
                  metric = "ROC",
                  trControl = trainCtrl)
    
    # Save performance
    trainROC = calculateROC(model,
                            trainingDF,
                            trainingTargetVariable,
                            fileName = paste("Figures/ROC/", injuryLeadName,
                                             " Training Logistic Regression.pdf",
                                             sep = ""))
    cvROC = max(model$results$ROC)
    
    testROC = calculateROC(model,
                           testingDF,
                           testingTargetVariable,
                           fileName = paste("Figures/ROC/", injuryLeadName,
                                            " Testing Logistic Regression.pdf",
                                            sep = ""))
    
    performanceDF <- data.frame(Model = rep("Logistic Regression", 3),
                                InjuryLead = rep(injuryLeadName, 3),
                                metric = c("Training AUC", "CV AUC", "Testing AUC"),
                                lower = c(trainROC[1], NA, testROC[1]),
                                value = c(trainROC[2], cvROC, testROC[2]),
                                upper = c(trainROC[3], NA, testROC[3]),
                                stringsAsFactors = FALSE)
    modelPerformance <- bind_rows(modelPerformance, performanceDF)
    
    # Save model
    predictionModels <- list.append(predictionModels,
                                    list(modelName = "Logistic Regression",
                                         InjuryLead = injuryLeadName,
                                         Model = model,
                                         Performance = performanceDF))
    # Plot
    gg <- plotInjuryPrediction(model, 
                         trainingDF,
                         trainingTargetVariable,
                         trainingData$Player.Name, 
                         trainingData$Season,
                         trainingData$Date)
    
    ggsave(paste("Figures/Predictions/", injuryLeadName,
                 " Training Logistic Regression.pdf",
                 sep = ""), gg, dpi = "retina", width = 30, height = 20 )
    
    gg <- plotInjuryPrediction(model,
                         testingDF,
                         testingTargetVariable,
                         testingData$Player.Name,
                         testingData$Season,
                         testingData$Date)
    ggsave(paste("Figures/Predictions/", injuryLeadName,
                 " Testing Logistic Regression.pdf",
                 sep = ""), gg, dpi = "retina", width = 20, height = 10)

    #################################
    # Model (Random Forest)
    model <- train(x = trainingDF, y = trainingTargetVariable,
                   method = "rf",
                   ntree = 2000,
                   tuneLength = 10,
                   metric = "ROC",
                   trControl = trainCtrl)
    
    # Save performance
    trainROC = calculateROC(model,
                            trainingDF,
                            trainingTargetVariable,
                            fileName = paste("Figures/ROC/", injuryLeadName,
                                             " Training Random Forest.pdf",
                                             sep = ""))
    cvROC = max(model$results$ROC)
    
    testROC = calculateROC(model,
                           testingDF,
                           testingTargetVariable,
                           fileName = paste("Figures/ROC/", injuryLeadName,
                                            " Testing Random Forest.pdf",
                                            sep = ""))
    
    performanceDF <- data.frame(Model = rep("Random Forest", 3),
                                InjuryLead = rep(injuryLeadName, 3),
                                metric = c("Training AUC", "CV AUC", "Testing AUC"),
                                lower = c(trainROC[1], NA, testROC[1]),
                                value = c(trainROC[2], cvROC, testROC[2]),
                                upper = c(trainROC[3], NA, testROC[3]),
                                stringsAsFactors = FALSE)
    modelPerformance <- bind_rows(modelPerformance, performanceDF)
    
    # Save model
    predictionModels <- list.append(predictionModels,
                                    list(modelName = "Random Forest",
                                         InjuryLead = injuryLeadName,
                                         Model = model,
                                         Performance = performanceDF))
    # Plot
    gg <- plotInjuryPrediction(model, 
                               trainingDF,
                               trainingTargetVariable,
                               trainingData$Player.Name, 
                               trainingData$Season,
                               trainingData$Date)
    
    ggsave(paste("Figures/Predictions/", injuryLeadName,
                 " Training Random Forest.pdf",
                 sep = ""), gg, dpi = "retina", width = 30, height = 20 )
    
    gg <- plotInjuryPrediction(model,
                               testingDF,
                               testingTargetVariable,
                               testingData$Player.Name,
                               testingData$Season,
                               testingData$Date)
    ggsave(paste("Figures/Predictions/", injuryLeadName,
                 " Testing Random Forest.pdf",
                 sep = ""), gg, dpi = "retina", width = 20, height = 10)
    
    #################################
    # Model (GLMNET)
    model <- train(x = trainingDF, y = trainingTargetVariable,
                   method = "glmnet",
                   preProcess = c("center", "scale"),
                   tuneLength = 10,
                   metric = "ROC",
                   trControl = trainCtrl) 
    
    # Save performance
    trainROC = calculateROC(model,
                            trainingDF,
                            trainingTargetVariable,
                            fileName = paste("Figures/ROC/", injuryLeadName,
                                             " Training GLMNET.pdf",
                                             sep = ""))
    cvROC = max(model$results$ROC)
    
    testROC = calculateROC(model,
                           testingDF,
                           testingTargetVariable,
                           fileName = paste("Figures/ROC/", injuryLeadName,
                                            " Testing GLMNET.pdf",
                                            sep = ""))
    
    performanceDF <- data.frame(Model = rep("GLMNET", 3),
                                InjuryLead = rep(injuryLeadName, 3),
                                metric = c("Training AUC", "CV AUC", "Testing AUC"),
                                lower = c(trainROC[1], NA, testROC[1]),
                                value = c(trainROC[2], cvROC, testROC[2]),
                                upper = c(trainROC[3], NA, testROC[3]),
                                stringsAsFactors = FALSE)
    modelPerformance <- bind_rows(modelPerformance, performanceDF)
    
    # Save model
    predictionModels <- list.append(predictionModels,
                                    list(modelName = "GLMNET",
                                         InjuryLead = injuryLeadName,
                                         Model = model,
                                         Performance = performanceDF))
    # Plot
    gg <- plotInjuryPrediction(model, 
                               trainingDF,
                               trainingTargetVariable,
                               trainingData$Player.Name, 
                               trainingData$Season,
                               trainingData$Date)
    
    ggsave(paste("Figures/Predictions/", injuryLeadName,
                 " Training GLMNET.pdf",
                 sep = ""), gg, dpi = "retina", width = 30, height = 20 )
    
    gg <- plotInjuryPrediction(model,
                               testingDF,
                               testingTargetVariable,
                               testingData$Player.Name,
                               testingData$Season,
                               testingData$Date)
    ggsave(paste("Figures/Predictions/", injuryLeadName,
                 " Testing GLMNET.pdf",
                 sep = ""), gg, dpi = "retina", width = 20, height = 10)
    
    #################################
    # Model (Neural Net)
    model <- train(x = trainingDF, y = trainingTargetVariable,
                   method = "nnet",
                   preProcess = c("center", "scale"),
                   tuneLength = 10,
                   maxit = 2000, 
                   metric = "ROC",
                   trControl = trainCtrl) 
    
    # Save performance
    trainROC = calculateROC(model,
                            trainingDF,
                            trainingTargetVariable,
                            fileName = paste("Figures/ROC/", injuryLeadName,
                                             " Training NeuralNet.pdf",
                                             sep = ""))
    cvROC = max(model$results$ROC)
    
    testROC = calculateROC(model,
                           testingDF,
                           testingTargetVariable,
                           fileName = paste("Figures/ROC/", injuryLeadName,
                                            " Testing NeuralNet.pdf",
                                            sep = ""))
    
    performanceDF <- data.frame(Model = rep("NeuralNet", 3),
                                InjuryLead = rep(injuryLeadName, 3),
                                metric = c("Training AUC", "CV AUC", "Testing AUC"),
                                lower = c(trainROC[1], NA, testROC[1]),
                                value = c(trainROC[2], cvROC, testROC[2]),
                                upper = c(trainROC[3], NA, testROC[3]),
                                stringsAsFactors = FALSE)
    modelPerformance <- bind_rows(modelPerformance, performanceDF)
    
    # Save model
    predictionModels <- list.append(predictionModels,
                                    list(modelName = "NeuralNet",
                                         InjuryLead = injuryLeadName,
                                         Model = model,
                                         Performance = performanceDF))
    # Plot
    gg <- plotInjuryPrediction(model, 
                               trainingDF,
                               trainingTargetVariable,
                               trainingData$Player.Name, 
                               trainingData$Season,
                               trainingData$Date)
    
    ggsave(paste("Figures/Predictions/", injuryLeadName,
                 " Training NeuralNet.pdf",
                 sep = ""), gg, dpi = "retina", width = 30, height = 20 )
    
    gg <- plotInjuryPrediction(model,
                               testingDF,
                               testingTargetVariable,
                               testingData$Player.Name,
                               testingData$Season,
                               testingData$Date)
    ggsave(paste("Figures/Predictions/", injuryLeadName,
                 " Testing NeuralNet.pdf",
                 sep = ""), gg, dpi = "retina", width = 20, height = 10)
    
    #################################
    # Model (SVM)
    model <- train(x = trainingDF, y = trainingTargetVariable,
                   method = "svmLinear",
                   preProcess = c("center", "scale"),
                   tuneLength = 10,
                   metric = "ROC",
                   trControl = trainCtrl)
    
    # Save performance
    trainROC = calculateROC(model,
                            trainingDF,
                            trainingTargetVariable,
                            fileName = paste("Figures/ROC/", injuryLeadName,
                                             " Training SVM.pdf",
                                             sep = ""))
    cvROC = max(model$results$ROC)
    
    testROC = calculateROC(model,
                           testingDF,
                           testingTargetVariable,
                           fileName = paste("Figures/ROC/", injuryLeadName,
                                            " Testing SVM.pdf",
                                            sep = ""))
    
    performanceDF <- data.frame(Model = rep("SVM", 3),
                                InjuryLead = rep(injuryLeadName, 3),
                                metric = c("Training AUC", "CV AUC", "Testing AUC"),
                                lower = c(trainROC[1], NA, testROC[1]),
                                value = c(trainROC[2], cvROC, testROC[2]),
                                upper = c(trainROC[3], NA, testROC[3]),
                                stringsAsFactors = FALSE)
    modelPerformance <- bind_rows(modelPerformance, performanceDF)
    
    # Save model
    predictionModels <- list.append(predictionModels,
                                    list(modelName = "SVM",
                                         InjuryLead = injuryLeadName,
                                         Model = model,
                                         Performance = performanceDF))
    # Plot
    gg <- plotInjuryPrediction(model, 
                               trainingDF,
                               trainingTargetVariable,
                               trainingData$Player.Name, 
                               trainingData$Season,
                               trainingData$Date)
    
    ggsave(paste("Figures/Predictions/", injuryLeadName,
                 " Training SVM.pdf",
                 sep = ""), gg, dpi = "retina", width = 30, height = 20 )
    
    gg <- plotInjuryPrediction(model,
                               testingDF,
                               testingTargetVariable,
                               testingData$Player.Name,
                               testingData$Season,
                               testingData$Date)
    ggsave(paste("Figures/Predictions/", injuryLeadName,
                 " Testing SVM.pdf",
                 sep = ""), gg, dpi = "retina", width = 20, height = 10)
}

# Plot performance of the models
modelPerformance$metric <- factor(modelPerformance$metric)
modelPerformance$metric <-  factor(modelPerformance$metric,
                                   levels = levels(modelPerformance$metric)[c(2, 1, 3)])

gg <- ggplot(modelPerformance, aes(y = metric, x = value)) +
    theme_bmbp() + 
    geom_errorbarh(aes(xmax = upper, xmin = lower), color = "black", height = 0, size = 1) + 
    geom_point(shape=21, size=3, fill="white") +
    facet_grid(Model~InjuryLead) +
    geom_vline(xintercept = 0.5, linetype = "dashed", color = "dark grey") +
    xlab("") + ylab("")

ggsave("Figures/Model Performance.pdf", gg, dpi = "retina", width = 10, height = 10)

# Save data
save(modelPerformance, predictionModels,
     file = "results.RData")
