#### R Basics ########

#### Data Types in R #########
# 1. character 2. numeric 3. integer 4. complex 5. logical

#### Creating a Vector in R (Stores objects of a single data type) ##########

x <- c(1,2,3,4,5)
print (x)
length(x)
print (x[1]) ### Indexing begins with 1
print (x[x > 2])

y <- 0:5 ## Another way of creating a sequence
print(y)
class(y)
as.logical(y)

z <- vector("numeric",length = 10)
z[1] <- 5.5   ### Can thereafter assign values

z <- c(1.7,"a") ### Mixing objects in a Vector causes coercion

#### Creating a Matrix in R ############################

m <- matrix(1:6,nrow = 2,ncol = 3)
print (m)
dim(m)
attributes(m)
print (m[1,]) ### Retireves the first row
print (m[,1]) ### Retrieves the first col

n <- 1:10
dim(n) <- c(2,5)   ####We can set the dimension of the vector to form a matrix
print (n)

x <- 1:3
y <- 10:12

cbind(x,y) ### Binds x,y by column.
rbind(x,y) ### Binds x,y by row.

#### Creating a List in R #############################

x <- list(1,"a",TRUE,1+4i)
print (x)
x <- list(foo = 1:4,bar = 0.6)
print (x$foo) ### Returns a list type
print ( x[[1]] ) ### Returns the Value
print ( x[[c(1,3)]] )  ### Worth noting the output


##### Factors in R (Giving labels to Integers)###############

x <- factor(c("Male","Female","Male","Female"))
table(x)
unclass(x)
as.integer(x)

#### Dealing with Missing Values ################

x <- c(1,2,3,NA,4,NA)
is.na(x)  ### Another function called is.nan() is also there

#### Data Frames in R (Used to store tabular data) ########

x <- data.frame(ID = 1:4, Name = c("Naruto","Sasuke","Itachi","Madara"), Surname = c("Uzumaki","Uchiha","Uchiha","Uchiha"))
nrow(x)
ncol(x)
colnames(x)
print (x$ID)
print (x["Name"])


########## Vectorized Operations in R ##############

x <- 1:5
y <- 10:14
print (x + y)  ## Other similar operations can be performed.

x <- matrix(1:6,nrow = 2,ncol = 3)
y <- matrix(1:15,nrow = 3,ncol = 5)
print (x %*% y)  ## True matrix multiplication
print (x * x) ## Element wise multiplication

####### Control Structures in R ###############
x <- 5

if (x > 3){
  y <- 10
  
}else{
  y <- 0
  
}

y <- if(x > 3){
  10
}else{
  0
}

for(i in 1:10){
  print (i)
}

x <- c("Naruto","Sakura","Sasuke","Hinata")
for(i in seq_along(x)){  ## seq_along creates a sequence of length x
  print (x[i])
}
for(i in seq_len(5)){   ### Creates a sequence of the length of the passed value
  print (i)
}

for(i in x){  ### Looping over objects
  print (i)
}

count <- 0
while (count < 5){
  print (count)
  count <- count + 1
}

lappl####### Functions in R ########################
myFunction <- function(a=3,b=5){
  a+b
}

answer <- myFunction()
answer <- myFunction(100,200)


########### Reading Data in R and manipulating it ############

setwd("C:/Users/deepsidhpura777/Desktop/R")
airquality <- read.csv("hw1_data.csv")
summary(airquality)
dim(airquality)
head(airquality)

mean(airquality$Ozone)

ozoneNA <- which(is.na(airquality$Ozone)) ### Cleaning NAs in Data. Very Important
airquality$Ozone[ozoneNA] <- 0

solarNA <- which(is.na(airquality$Solar.R))
airquality$Solar.R[solarNA] <- 0

## Finding mean accross all columns
mean <- lapply(airquality,mean)
mean <- sapply(airquality,mean)
mean <- apply(airquality,2,mean) ## Applies functions to Rows(1) or Columns(2)
rowmean <- rowMeans(airquality)  ## Mean of each row
splitByMonth <- split(airquality,airquality$Month)  ## Splitting the Data Set by Months
avgByMonth <- lapply(splitByMonth,function(x) colMeans(x[,c("Ozone","Solar.R","Wind")]))


######## Naive Bayes in R ######################
data(iris)
dim(iris)
summary(iris)
h <- sample(150)
iris <- iris[h,]
train <- iris[1:100,]
test <- iris[-c(1:100),]
library(e1071)
colnames(train)
model <- naiveBayes(Species~.,data = train)
prediction <- predict(model,test[,-5])
table(prediction)
