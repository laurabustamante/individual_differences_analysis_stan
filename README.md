# individual_differences_analysis_stan
This is a tutorial for the 2nd Mental Effort Workshop (2021) on Individual Differences Data Analysis With STAN: A common approach to individual differences research is to collect two or more tasks and correlate them. The problem with this approach is that estimating individual task parameters separately for each task before correlating discards information about uncertainty in each individuals’ parameter estimates. This tutorial will cover an alternative method, using the programming language R, and the Bayesian statistical language STAN, that simultaneously estimates parameters for each task, as well as a covariance matrix of within- and between-task parameters. The correlation between any two parameters is then accessible by examining the covariance matrix for any pair of parameters. This method  explicitly estimates the correlation and maintains the uncertainty in the parameter estimates when estimating the correlation. In this tutorial participants will work through a sample dataset of participants who completed two common tasks, create a model, and evaluate inter-task correlations.

An easy way to run this notebook is to upload all of the files (individual_differences_cmdstanr.ipynb, TwoStep_ITC_covar.stan, TwoStep_ITC_covar_fit.RDS) to Google Collaboratory (https://colab.research.google.com/), from there installing packages and interacting will be easy breezy. 
