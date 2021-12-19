# Logistic regression solver
# Andrew J. Graves
# 10/01/20

# Implements the Newton-Raphson algorithm to 
# solve for logistic regression coefficients as 
# described in the Elements of Statistical Learning
log_reg <- function(x_mat, y, thresh = 1e-14, max_iter = 100){
  
  # Initialize X matrix, beta vector, weighted X matrix,
  # iterator, and SS difference between betas
  x <- cbind("(Intercept)" = 1, x_mat)
  beta_old <- rep(0, ncol(x))
  iter <- 0
  ss_diff <- thresh + 1
  
  # Continue to re-weight until convergence
  while(ss_diff > thresh && iter < max_iter){
    
    xb <- x %*% beta_old # Linear combination of X and betas
    prob <- 1 / (1 + exp(-1 * xb)) # Compute probability of response 1
    
    # Multiply probability of both events (treat diagonal as vector)
    weight <- as.numeric(prob * (1 - prob))
    
    # Weight the X matrix with apply (save compute time)
    x_weight <- t(apply(x, 2, function(x) x * weight))
    
    neg_hess <- x_weight %*% x # Compute the Hessian (negated)
    z <- xb + (1 / weight) * (y - prob) # Compute z (adjusted response)
    beta_new <- solve(neg_hess) %*% x_weight %*% z # New beta values
    
    # Store the SS difference of betas from previous iteration
    ss_diff <- sum((beta_new - beta_old)^2)
    
    beta_old <- beta_new # Update betas to iteratively solve
    iter <- iter + 1 # Count iterations
  }
  return(as.vector(beta_new))
}