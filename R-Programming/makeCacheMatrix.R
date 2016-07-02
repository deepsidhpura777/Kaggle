
##Function to cache in the matrix

makeCacheMatrix <- function(x = matrix()) {
  inv <- NULL
  set <- function(y) {
    x <<- y  ##setting the matrix passed
    inv <<- NULL
  }
  get <- function() x ##Retrieving the matrix passed
  setinverse <- function(inverse) inv <<- inverse
  getinverse <- function() inv
  list(set = set, get = get,
       setinverse = setinverse,
       getinverse = getinverse) ##Returns the list which can access the special matrix formed
}