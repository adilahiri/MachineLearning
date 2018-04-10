setwd("C:/Users/adi44/Desktop/Kaggle/DigitRecognizer")
rm(list=ls())
######## Load the training and testing data ####################
train_data <- read.csv("train.csv", header = TRUE)                  
test_data <- read.csv("test.csv", header = TRUE)                     
######## Seperate the Predictors and Response ##################
train_label <- train_data[1:dim(train_data)[1],1]
train_matrix <- as.matrix(train_data[1:dim(train_data)[1],2:dim(train_data)[2]])
test_matrix <- as.matrix(train_data[1:dim(test_data)[1],1:dim(test_data)[2]])

                          
######## Library Load  #########################################
library(readr)