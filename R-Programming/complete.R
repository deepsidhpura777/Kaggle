complete <- function(directory,id=1:332){
  
  setwd(directory)
  data<-lapply(dir(),read.csv)
  tempData<-data.frame(id=integer(0),nobs=integer(0))
  
  for(i in seq_along(id)){
    
    tempId<-id[i]
    tempFile<-data[[tempId]]
    tempNum<-nrow(tempFile[complete.cases(tempFile),])
    
    tempData[i,]<-c(tempId,tempNum)
  }
  tempData
  
  
  
}