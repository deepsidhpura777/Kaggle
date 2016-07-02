corr <- function(directory,threshold=0){
  
  compData<-complete(directory)
  tempData<-compData[compData[,2] > threshold ,]
  corVector<-vector()
  
  data<-lapply(dir(),read.csv)
  
  for(i in seq_len(nrow(tempData))){
    
    tempId = tempData[i,1]
    tempFile= na.omit(data[[tempId]])
    corVector[i]<-cor(tempFile$nitrate,tempFile$sulfate)
  }
  
  setwd("C:/Users/deepsidhpura777/Desktop/R")
  corVector
}