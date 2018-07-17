setwd("C:/Users/adi44/Desktop/data_science_task")
# Needed Libraries
library(ggplot2)
library(reshape2)
library(caTools) # split data set
library(e1071) # SVR
library(rpart)# DT
library(randomForest) # randomforest regressor
library(caret)
library(xgboost)
library(h2o)
# Load the data and process it for regression 
data_set = read.csv("sample_load_weather_data.csv", header = TRUE)
data_matrix <- subset(data_set,select=-c(1,3,9))
data_matrix <- cbind(data_matrix,data_set$load)
colnames(data_matrix)[11]<-"Load"

# Encoding the categorical variables
data_matrix$wkday= as.numeric(factor(data_matrix$wkday, levels= c("sun","mon","twt","fri","sat"),
                          labels=c(1,2,3,4,5)))

data_matrix$weather_type= as.numeric(factor(data_matrix$weather_type, levels= c("cold","mild","hot"),
                           labels=c(1,2,3)))

data_matrix$hr_weather_type= as.numeric(factor(data_matrix$hr_weather_type, levels= c("xcold","cold",
                                                                           "mild","hot",
                                                                           "xhot","xxhot"),
                                    labels = c(1,2,3,4,5,6)))
data_matrix$season=as.numeric(factor(data_matrix$season, levels= c("F","S","Sp","W"),
                                 labels=c(1,2,3,4)))     

# Check for correlations in the dataset
cormat <- round(cor(data_matrix),2)
melted_cormat <- melt(cormat)
p<-ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile()+ggtitle("Correlation Matrix")
plot(p)


# Split the data set
set.seed(666)
split= sample.split(data_matrix$Load, SplitRatio = 0.8)
training_set=subset(data_matrix,split==TRUE)
test_set=subset(data_matrix,split==FALSE)

### BUILD REGRESSOR FROM CARET PACKAGE SO THESE ARE TUNED AND CROSS VALIDATED

# Fit a multiple linear regression to the training set 
regressor_multi=lm(formula=Load ~., data=training_set)
summary(regressor_multi)
regressor_multi=train(form=Load~., data=training_set,method='lm')

# Fit a linear regressor with cdd as the independent variable as it has the lowest Pval from 
# summary of multiple regression analysis
regressor_linear=train(form=Load~cdd, data=training_set,method='lm')


# Fit a svr (linear) regressor using 
regressor_svr=train(form=Load~., data=training_set,method='svmLinear')

# Fit a svr (Gaussian kernel)
regressor_svr_rbf=train(form=Load~., data=training_set,method='svmRadial')
# Fit a decision tree regressor 
regressor_DT=train(form=Load~., data=training_set,method='rpart')
# Fit a random forest regressor 
regressor_RF=train(form=Load~., data=training_set,method='rf')





# Predict the test data
y_pred_multi = predict(regressor_multi,newdata = test_set)
y_pred_linear= predict(regressor_linear,newdata=test_set)
y_pred_svr=predict(regressor_svr,newdata=test_set)
y_pred_svr_rbf=predict(regressor_svr_rbf,newdata=test_set)
y_pred_DT=predict(regressor_DT,newdata=test_set)
y_pred_RF=predict(regressor_RF,newdata=test_set)







#ANN

# Feature Scaling
training_set[-11]=scale(training_set[-11])
test_set[-11]=scale(test_set[-11])

# Connect to h20
h2o.init(nthreads=-1)
regressor_ann = h2o.deeplearning(y = 'Load',
                         training_frame = as.h2o(training_set),
                         activation = 'Rectifier',
                         hidden = c(5,5),
                         epochs = 20,
                         train_samples_per_iteration = -2)
y_pred_ann=h2o.predict(regressor_ann,newdata = as.h2o(test_set[-11]))



# MSE 
rmse_multi=rmse(y_pred_multi,test_set$Load)
rmse_Linear=rmse(y_pred_linear,test_set$Load)
rmse_svr=rmse(y_pred_svr,test_set$Load)
rmse_svr_rbf=rmse(y_pred_svr_rbf,test_set$Load)
rmse_DT=rmse(y_pred_DT,test_set$Load)
rmse_RF=rmse(y_pred_RF,test_set$Load)
rmse_ann<-rmse(as.vector(y_pred_ann),test_set$Load)

# Baplot of RMSE
name1<-c("multi","linear","svr","rbf_svr","DT","RF","ANN")
B<- round(c(rmse_multi,rmse_Linear,rmse_svr,rmse_svr_rbf,rmse_DT,rmse_RF,rmse_ann),3)
y<-barplot(B, beside=T, col=c("red"), ylab="RMSE",xlab="regressors",main="RMSE Plot",
           names.arg=name1, ylim=c(0,0.5),cex.names=1.8,cex.axis=1.8)
text(y, B, label = B, pos = 3, cex = 1, col = c("black"))
