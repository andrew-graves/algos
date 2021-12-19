# 2-D Poisson mixture model algorithm
# Andrew J. Graves
# 11/04/21

# Expectation-maximization algorithm as described in the 
# Elements of Statistical Learning (ESL).
# Parameters were modified from the two-component Gaussian 
# distribution to fit the two-component Poisson distribution.

# Import packages
library(tidyverse)

# 2-component Poisson mixture model function
pois_mix2 <- function(x, 
                      thresh = 1e-16, max_iter = 100, # default convergence parameters
                      seed = 42){
  
  # Initialize parameters for EM algorithm
  set.seed(seed)
  old_theta <- sample(x, 2) # pick two data points from input
  old_pi <- .5 # start at balanced mixtures
  iter <- 0
  ss_diff <- 1
  
  # Continue to maximize expectation until convergence
  while(ss_diff > thresh && iter < max_iter){
    
    # Compute weighted densities (mass) for both Poisson instances
    mass1 <- old_pi * dpois(x, old_theta[1])
    mass2 <- (1 - old_pi) * dpois(x, old_theta[2])
    
    # Expectation step: compute responsibilities
    r_i <- mass1 / (mass1 + mass2)
    r_mat <- cbind(r_i, 1 - r_i) # row margins sum to 1
    
    # Sum the responsibilities for each component
    n_k <- colSums(r_mat)
    
    # Update mixing probability for component 1
    new_pi <- mean(r_i) # 1st responsibility constrains the 2nd
    
    # Maximization step: update theta (lambda for Poisson)
    # MLE for lambda is the mean
    new_theta <- c(sum(r_mat[, 1] * x) / n_k[1],
                   sum(r_mat[, 2] * x) / n_k[2])
    
    # Convergence test: diff in parameters from previous iterations
    ss_diff <- sum((new_pi - old_pi)^2,
                   sum(new_theta - old_theta)^2)
    
    # Update parameters and iterate until solved
    old_pi <- new_pi
    old_theta <- new_theta
    iter <- iter + 1
  }
  
  # Return mixture-model parameters
  return(tibble("$k$" = c(1, 2),
                "theta" = unname(new_theta), 
                "pi" = unname(c(new_pi, 1 - new_pi))
  )
  )
}