pollutantmean <- function(directory,pollutant,id=1:332){
  
  setwd(directory)
  data<-lapply(dir(),read.csv)
  tempData<-data.frame()
  
  for(i in seq_along(id)){
    
    ##tempFile <- data[[i]] 
    ##colN <- which(names(tempFile)==pollutant
    temp<-id[i]
    tempData<-rbind(tempData,data[[temp]])
  } 
  colN <- which(names(tempData)==pollutant)
  
  polmean<-mean(tempData[,colN],na.rm=TRUE)
  polmean

}