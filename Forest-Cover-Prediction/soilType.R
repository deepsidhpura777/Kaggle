soilType <- function(x){
  
  soil <- 16:55
  s <- vector(length=nrow(x))
  for(i in seq_along(soil)){
    
    rows <- which(x[,i+15]==1)
    s[rows] <- i
    
  }
  s
  
}