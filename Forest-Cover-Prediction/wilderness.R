wilderness <- function(x){
  
  wild <- 12:15
  w <- vector(length=nrow(x))
  for(i in seq_along(wild)){
    
    rows <- which(x[,i+11]==1)
    w[rows] <- i
    
  }
  
  w
  
}