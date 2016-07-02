best <- function(state,outcome) {
  
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
      
      colData <- as.numeric(stateData[,colN])
      ##colData <- na.omit(colData) serious error, shifts row indices
      minValue <- min(colData,na.rm=TRUE)
      minRow <- which(colData == minValue)
      
      hospName <- stateData[minRow,"Hospital.Name"]
      ##print(hospName)
      hospOrder <- order(hospName)
      
      hospital <- hospName[hospOrder[1]]
      
      hospital
    }
  
  else
    stop("Invalid Outcome")
  }
  else
    stop("Invalid State")
  
}