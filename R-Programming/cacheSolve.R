##Accepts the list returned by makeCacheMatrix function as argument

cacheSolve <- function(x) {
  inv <- x$getinverse()  ##Getting the inverse of our matrix if calculated
  if(!is.null(inv)) {
    message("getting cached data")
    return(inv)
  }
  data <- x$get()  ##When inverse is not cached, the inverse is calculated here
  inv <- solve(data)
  x$setinverse(inv)
  inv
}