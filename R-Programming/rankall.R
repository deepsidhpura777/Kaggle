rankall <- function(outcome ,num="best"){
  
  stateList <- split(data,data$State)
  outData<-data.frame()
  
  if(outcome=="heart attack" || outcome=="heart failure" || outcome=="pneumonia"){
    ##print("In if")
    if(outcome == "heart attack")
      colN <- 11
    if(outcome == "heart failure")
      colN <- 17
    if(outcome == "pneumonia")
      colN <- 23
    
    for(i in seq_len(length(stateList))){
      
      tempData <- stateList[[i]]
      
      tempData[,colN]<-as.numeric(tempData[,colN])
      hospOrder <- tempData[order(tempData[,colN],tempData$Hospital.Name,na.last=NA),c("Hospital.Name","State")]
      if(num == "best"){
       outData <- rbind(outData,hospOrder[1,])
      }
      else if(num== "worst"){
        outData<-rbind(outData,hospOrder[nrow(hospOrder),])
      }
      else{
        outData<-rbind(outData,hospOrder[num,])
      }
      
      
    }
    colnames(outData)<-c("hospital","state")
    outData
    
    
  }
  else
    print("Invalid Outcome")
}