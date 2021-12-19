# Hubs and authorities algorithm
# Andrew J. Graves
# 11/17/21

# Import packages
library(tidyverse)
library(igraph)
library(Matrix) # for sparse matrix manipulation

# Hubs and authorities algorithm as described in MMDS
hits <- function(g, max_iter = 1000, tol = 1e-20){
  
  # Convert graph to adjacency matrix
  adj <- as_adj(g)
  # Initialize h, a, and convergence parameters
  h <- rep(1, nrow(adj))
  a <- h
  iter <- 0
  ss_diff <- 1
  
  # Iterate until solved
  while(ss_diff > tol && iter < max_iter){
    
    # Update a
    new_a <- t(adj) %*% h
    a_diff <- (new_a / max(new_a) - a)^2
    a <- new_a / max(new_a)
    
    # Update h
    new_h <- adj %*% a
    h_diff <- (new_h / max(new_h) - h)^2
    h <- new_h / max(new_h)
    
    # Update convergence parameters
    ss_diff <- sum(a_diff + h_diff)
    iter <- iter + 1
  }
  # Store results
  out <- tibble(as.numeric(a), as.numeric(h))
  colnames(out) <- c("authority", "hub")
  return(out)
}