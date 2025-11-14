


A = matrix(c(1,2,3,0,9,4,5,6,0,9), nrow=2, ncol=5, byrow=TRUE)

A
colMeans(A) > 3
A[,colMeans(A) > 3]
