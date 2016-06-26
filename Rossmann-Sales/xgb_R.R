
library(readr)
library(xgboost)

test <- read_csv("test.csv")      ## Reading the Train,Test and Store dataset
store <- read_csv('store.csv')
train <- read_csv('train.csv')
sub <- read_csv('sample_submission.csv')

train <- merge(train,store)   ## Combining the Train and Test data sets with Store
test <- merge(test,store)
train[is.na(train)]   <- 0   ## Eliminating the NA values,replacing them with 0
test[is.na(test)]   <- 0

train <- train[ which(train$Open=='1'),]    ## Training on examples which has non zero Sales and is open
train <- train[ which(train$Sales!='0'),]

train$month <- as.integer(format(train$Date, "%m"))
train$year <- as.integer(format(train$Date, "%y"))  ## COnverting the Month,Year,Day from the Date string to integer variables
train$day <- as.integer(format(train$Date, "%d"))

train <- train[,-c(3,8)]  ## Dropping the columns Date and State Holiday

test$month <- as.integer(format(test$Date, "%m"))
test$year <- as.integer(format(test$Date, "%y"))
test$day <- as.integer(format(test$Date, "%d"))

test <- test[,-c(4,7)]

feature.names <- names(train)[c(1,2,5:19)]      ## Converting Character fields to categorical integers
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

tra <-train[,feature.names]   ## Column names for the Training Set used for Indexing


RMPSE<- function(preds, dtrain) {     ## Evaluation Function
  labels <- getinfo(dtrain, "label")
  elab<- exp(as.numeric(labels))-1
  epreds<- exp(as.numeric(preds))-1
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMPSE", value = err))
}

#RMPSE<- function(preds, dtrain) {
#   labels <- getinfo(dtrain, "label")
#   elab<- (as.numeric(labels))^ 16
#   epreds<- (as.numeric(preds)) ^ 16
#   err <- sqrt(mean((epreds/elab-1)^2))
#   return(list(metric = "RMPSE", value = err))
#   }

h<-sample(nrow(train),8000)  ## CV Hold out

dval<-xgb.DMatrix(data=data.matrix(tra [h,]),label=log(train$Sales+1)[h])
dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=log(train$Sales+1)[-h])

#dval<-xgb.DMatrix(data=data.matrix(tra [h,]),label=(train$Sales ^ (1/16))[h])
#dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=(train$Sales ^ (1/16))[-h])

watchlist<-list(val=dval,train=dtrain)

param <- list (objective = "reg:linear",
                booster = "gbtree",
                eta = "0.02", ## 0.03 , 0.015, 0.02 , 0.01
                min_child_weight = "7",
                subsample = "0.8",
                colsample_bytree = "0.8",
                max_depth = "10"   ## 9 , 10, 
                )

model <- xgb.train(params           = param, 
                   data             = dtrain, 
                   nrounds          = 4000, # 3000 (log(sales)), 3500 (Sales^1/16), 2000 (16) ~ 0.10, 2500(e) ~ 0.11 
                   verbose          = 0,
                   early.stop.round = 200,
                   watchlist        = watchlist,
                   maximize         = FALSE,
                   feval=RMPSE)

prediction = exp(predict(model,data.matrix(test[,feature.names]))) - 1
#prediction = (predict(model,data.matrix(test[,feature.names]))) ^ 16

sub = data.frame(Id = test$Id, Sales = prediction)
write_csv(sub,"R_XGB_V.csv")

