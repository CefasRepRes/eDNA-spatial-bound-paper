# # Exploratory analysis of variable importance in eDNA dispersal distance
# model simulations

# Code used in Silva et al 2025. 

# Modelling the spatial bound of an eDNA signal in the marine environment – the effect of local conditions Tiago A. M. Silva, Claire P. C. Beraud, Phillip D. Lamb, Wayne Rostant and Hannah J. Tidbury Frontiers in Marine Systems



# The data file is 'dist_buff_015_0075_v2.csv'

getwd()

# set the path
# e_dna_path <- 

setwd(e_dna_path)

library(tidyverse)
library(hablar)
library(dplyr) 
library(ggplot2) 
sessionInfo()

e_dna_file <- "dist_buff_015_0075_v2.csv"
e_dna_pathfile <- paste0(., "\\data\\", e_dna_file)

dispersaldata <- read.csv(e_dna_pathfile, header = TRUE) 
str(dispersaldata)
# Responses are the p0, p5 etc. which are the distances for each percentile
dispersaldata<- dispersaldata %>%
  convert(fct(X,runname,month,targetid,year))

# Plot data for p50 response (median distance)
ggplot(data = dispersaldata, aes(x = targetid, y = p50, colour = month))+
  geom_point()

pairs(~ year + NAO + bathy + TidalExc + DistCoast, data = dispersaldata)
# no obvious collinearity except DistCoast and bathy (as expected), 
# but conditional random forest allows for marginal importance when collinearity
# is present.


# We use conditional random forest with cforest() because we have variables of 
# different types including one (month) with only two levels and randomForest() 
# can be biased towards continuous data and factors with many levels.

library(party) # for cforest() function
library(caret) # for trainControl() function
set.seed(2024)

# Initial run for p50 response:
dispersal_rf_cond<- cforest(formula =  p50 ~ year + month + NAO + bathy + 
                              TidalExc + DistCoast, data = dispersaldata, 
                            control = cforest_unbiased(mtry = 4, ntree = 500))

# Tuning the mtry parameter using between 1 and 5 variables at each split:
rf_grid <- expand.grid(mtry = c(1:5))

# OOB not appropriate for tuning mtry. Use cross validation. 
# Trees already at high number to make sure stable result:
control <- trainControl(
  method = "repeatedcv", 
  number = 10, 
  repeats = 10,
  returnResamp="all",
  selectionFunction = 'best'
)

set.seed(2024)
(dispersal_rf_cond_tune <- train(p50 ~ year + month + NAO + bathy + 
                                  TidalExc + DistCoast, data = dispersaldata,
                                method="cforest", controls = cforest_unbiased(ntree = 500),
                                tuneGrid=rf_grid, trControl=control))

# Use the dispersal_rf_cond_tune results to report RMSE:
plot(dispersal_rf_cond_tune)

# We refit using mtry = 5
dispersal_rf_cond<- cforest(formula =  p50 ~ year + month + NAO + bathy + 
                              TidalExc + DistCoast, data = dispersaldata, 
                            control = cforest_unbiased(mtry = 5, ntree = 500))
 
barplot(varimp(dispersal_rf_cond,conditional = TRUE))
# Importance with uncertainty:
set.seed(2024)
vi_50 <- t(replicate(50, varimp(dispersal_rf_cond,conditional = TRUE)))
boxplot(vi_50)

# Plot 
library(stablelearner)
dispersal_rf_cond_st <- stablelearner::as.stabletree(dispersal_rf_cond)
summary(dispersal_rf_cond_st, original = FALSE)
barplot(dispersal_rf_cond_st)
image(dispersal_rf_cond_st)


# Repeat procedure for p95 response:
set.seed(2024)
dispersal_rf_p95_cond<- cforest(formula =  p95 ~ year + month + NAO + bathy + 
                                  TidalExc + DistCoast, data = dispersaldata, 
                                control = cforest_unbiased(mtry = 5, ntree = 500))

set.seed(2024)
(dispersal_rf_p95_cond_tune <- train(p95 ~ year + month + NAO + bathy + 
                                      TidalExc + DistCoast, data = dispersaldata,
                                    method="cforest", controls = cforest_unbiased(ntree = 500),
                                    tuneGrid=rf_grid, trControl=control))

plot(dispersal_rf_p95_cond_tune)
barplot(varimp(dispersal_rf_p95_cond,conditional = TRUE))
set.seed(2024)
vi_95 <- t(replicate(50, varimp(dispersal_rf_p95_cond,conditional = TRUE)))
boxplot(vi_95)
dispersal_rf_p95_cond_st <- stablelearner::as.stabletree(dispersal_rf_p95_cond)
summary(dispersal_rf_p95_cond_st, original = FALSE)
barplot(dispersal_rf_p95_cond_st)
image(dispersal_rf_p95_cond_st)
