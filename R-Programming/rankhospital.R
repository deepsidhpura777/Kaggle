rankhospital <- function(state,outcome,num="best"){
  
  stateD<-data$State==state
  stateData <- data[stateD,]
  
  if(nrow(stateData) != 0){
    if(outcome=="heart attack" || outcome=="heart failure" || outcome=="pneumonia"){
      ##print("In if")
      if(outcome == "heart attack")
        colN <- 11
      else if(outcome == "heart failure")
        colN <- 17
      else if(outcome == "pneumonia")
        colN <- 23
      
      stateData[,colN]<-as.numeric(stateData[,colN])
      hospOrder <- stateData[order(stateData[,colN],stateData$Hospital.Name,na.last=NA),"Hospital.Name"]
      if(num == "best")
        hospOrder[1]
      else if(num== "worst")
        hospOrder[length(hospOrder)]
      else
        hospOrder[num]
    }
    
    else
      stop("Invalid Outcome")
  }
  else
    stop("Invalid State")
  
  
  
  
}