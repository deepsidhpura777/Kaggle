library(readr)
library(h2o)

test <- read_csv("test.csv")
store <- read_csv('store.csv')
train <- read_csv('train.csv')


train <- merge(train,store)
test <- merge(test,store)
train[is.na(train)]   <- 0
test[is.na(test)]   <- 0

train <- train[ which(train$Open=='1'),]
train <- train[ which(train$Sales!='0'),]

train$month <- as.integer(format(train$Date, "%m"))
train$year <- as.integer(format(train$Date, "%y"))
train$day <- as.integer(format(train$Date, "%d"))

train <- train[,-c(3,8)] ## dropping Date & StateHoliday

test$month <- as.integer(format(test$Date, "%m"))
test$year <- as.integer(format(test$Date, "%y"))
test$day <- as.integer(format(test$Date, "%d"))

test <- test[,-c(4,7)]

X.names <- names(train)[c(1,2,5:19)]  ## Not choosing Sales & Customers (X)
train$Sales <- log(train$Sales + 1)
y.names <- names(train)[3]

for (f in X.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}


localH2O = h2o.init(ip="localhost",port=54321,startH2O=TRUE,max_mem_size='2g')
train_h2o = as.h2o(localH2O,train)
test_h2o = as.h2o(localH2O,test)         ## initializing H2O connection variable

forest <- h2o.randomForest(y=y.names,x=X.names,data=train_h2o,
                           ntree=200,depth=20,
                           classification=FALSE,type="BigData")

predictions<-as.data.frame(h2o.predict(forest,test_h2o))
pred <- exp(predictions[,1])- 1
sub = data.frame(Id = test$Id, Sales = pred)
write_csv(sub,"R_RF_H2O.csv")


