import pandas as pd
import numpy as np

# Function to generate dummy data for now
# matrix that is S matrices deep; N*T rows long; p or k or 1 columns wide
# generate basic time series for each factor
# set parameters to be particular numbers with basic lasso sparsity design
# draw Z how we did with Manu just for each time period
# draw epsilon 
# create returns
# create h t 

# Function to perform lasso; look how sklearn does it so i write it from scratch
# -go scope how Denis runs it in his class

# Function to perform OLS from scratch with closed form

# Function to run DL
# takes argument of set of controls of interest
# return the treatment parameter

# Function to run my procedure
# -call all the above functions
# return the vector of gmma beta j so parallize across the p runs of this within this function

# Function to run the simulation so call my estimation in parallel across all the simulations

# Main function to build the data and then call the function to run the simulation

# TODO:
# -Split out all my estimation code into a separate file (i.e. this file) and then just import those functions into a new file
# --read any formating on how to do this nicely