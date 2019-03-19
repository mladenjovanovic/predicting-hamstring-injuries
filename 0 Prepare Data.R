# The MIT License (MIT)
# 
# Copyright (c) 2018 Mladen Jovanović
#
#
# For using/citing this code please use the following reference
# --------------------------------------------------------------
# Jovanovic, M.(2018)
# Predicting non-contact hamstring injuries by using training load data
# Complementarytraining.net
# URL: www.complementarytraining.net/predicting-hamstring-injuries
# --------------------------------------------------------------- 
#
# email: coach.mladen.jovanovic@gmail.com
# web: www.complementarytraining.net
# twitter: physical_prep
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#    
#     The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

library(plyr)
library(dplyr)
library(reshape2)
library(ggplot2)
library(TTR)
library(zoo)

#######################
# FUNCTIONS
######################

# Acute to Chronic Workload Ratio
ACWR <- function(acute, chronic) {
    acwr <- acute / chronic
    acwr <- ifelse(chronic == 0, 0, acwr) 
    return(acwr)
}

# Acute to Chronic Workload Difference
ACWD <- function(acute, chronic) {
    acwd <- acute - chronic
    return(acwd)
}

# Modify the injury tag based on InjuryLead paramet (global environment)
fixInjury <- function(data) {
    # Convert date to number
    data$dateIndex <- data$Date
    
    injuryList <- dplyr::select(data, dateIndex, Player.Name, Injury) %>%
        filter(Injury == "Injured")
    
    for( i in injuryList$dateIndex) {
        data$Injury <- ifelse(((i - data$dateIndex) < (injuryLead + 1)) &
                                  ((i - data$dateIndex) >= 0) | (data$Injury == "Injured"),
                              "Injured",
                              "Non.Injured")
    }
    return(dplyr::select(data, -dateIndex))    
}


#####################
# MAIN CODE
#####################
# Load Data

monitoringData <- read.csv("Raw Data.csv",
                           header = TRUE,
                           stringsAsFactors = FALSE)

# Create DF for the empty days
emptyDays <- expand.grid(Data.Split = "Train",
                         Season = c("Season 1", "Season 2"),
                         Phase = "Pre Season",
                         Date = seq(-70, 0, 1),
                         Player.Name = unique(monitoringData$Player.Name),
                         Injury.Location = "",
                         Injury.Type = "",
                         LoadMetric01 = 0,
                         LoadMetric02 = 0,
                         LoadMetric03 = 0,
                         stringsAsFactors = FALSE)

# Bind the two
monitoringData <- rbind(emptyDays, monitoringData)

# Sort
monitoringData <- monitoringData %>%
    arrange(Season, Player.Name, Date)

# Select testing fold
testingAthlete <- unique(filter(monitoringData, Data.Split == "Test")$Player.Name)

monitoringData$Data.Split <- ifelse(monitoringData$Player.Name %in% testingAthlete, "Test", "Train")

# Create a "long" format
monitoringDataLong <- melt(monitoringData, id.vars = 1:7)

# Create the exponential rolling averages and ACWR variables
monitoringDataLong <- monitoringDataLong %>%
    group_by(Season, Player.Name, variable) %>%
    arrange(Date) %>%
    # Mean
    mutate(Acute = round(EMA(value, n = 1, ratio = 2/(7+1)), 2),
           Chronic = round(EMA(value, n = 1, ratio = 2/(28+1)), 2),
           
           # Add ACWR & ACWD
           ACWR = round(ACWR(Acute, Chronic), 2),
           ACWD = round(ACWD(Acute, Chronic), 2),
           
           # Get the week max in the last rolling 7 days
           AcuteRollMax = round(rollmax(Acute, 7, fill = NA, align = "right"), 2),
           ChronicRollMax = round(rollmax(Chronic, 7, fill = NA, align = "right"), 2),
           ACWRrollMax = round(rollmax(ACWR, 7, fill = NA, align = "right"), 2),
           ACWDrollMax = round(rollmax(ACWD, 7, fill = NA, align = "right"), 2),
           
           # Get the week mean in the last rolling 7 days
           AcuteRollMean = round(rollmean(Acute, 7, fill = NA, align = "right"), 2),
           ChronicRollMean = round(rollmean(Chronic, 7, fill = NA, align = "right"), 2),
           ACWRrollMean = round(rollmean(ACWR, 7, fill = NA, align = "right"), 2),
           ACWDrollMean = round(rollmean(ACWD, 7, fill = NA, align = "right"), 2)) %>%
    ungroup()

# Convert to long again (to merge the new aggregates with the features name)
monitoringDataLong$value <- NULL
monitoringDataLong <- rename(monitoringDataLong, Metric = variable)
monitoringDataLong <- melt(monitoringDataLong, id.vars = 1:8)
monitoringDataLong$Metric <- paste(monitoringDataLong$Metric, monitoringDataLong$variable, sep = ".")
monitoringDataLong$variable <- NULL

# Create Lag variables
monitoringDataLong <- monitoringDataLong %>%
    group_by(Season, Player.Name, Metric) %>%
    arrange(Date) %>%
    mutate(Lag.0 = lag(value, 0),
           Lag.07 = lag(value, 7),
           Lag.14 = lag(value, 14),
           Lag.21 = lag(value, 21)) %>%
    ungroup()

# Convert to long again
monitoringDataLong$value <- NULL
monitoringDataLong <- melt(monitoringDataLong, id.vars = 1:8)
monitoringDataLong$Metric <- paste(monitoringDataLong$Metric, monitoringDataLong$variable, sep = ".")
monitoringDataLong$variable <- NULL

# Finaly convert to wide format
monitoringData <- dcast(monitoringDataLong,
                        ... ~ Metric,
                        value.var = "value")

# Sort
monitoringData <- arrange(monitoringData, Season, Player.Name, Date)

# Create monitoring feature for non-contact injuries ("Soft Tissue & Overuse")
monitoringData$Injury <- ifelse(monitoringData$Injury.Location == "Hamstring",
                                "Injured", "Non.Injured")

monitoringData$Injury01 <- monitoringData$Injury

# Modify the injury tag using injury lead
injuryLead <- 7
tmpData <- ddply(monitoringData, .variables = c("Season", "Player.Name"), .fun = fixInjury)
monitoringData$Injury07 <- tmpData$Injury

injuryLead <- 14
tmpData <- ddply(monitoringData, .variables = c("Season", "Player.Name"), .fun = fixInjury)
monitoringData$Injury14 <- tmpData$Injury

injuryLead <- 21
tmpData <- ddply(monitoringData, .variables = c("Season", "Player.Name"), .fun = fixInjury)
monitoringData$Injury21 <- tmpData$Injury

# Reorganize the columns 
monitoringData <- monitoringData[c(1:7, 153:156, 8:151)] 

#### Clear up the data 
# Clear up missing values
monitoringData <- filter(monitoringData, Date > 0)

# Save data
write.csv(monitoringData,
          file = "Prepared Data.csv",
          row.names = FALSE)

