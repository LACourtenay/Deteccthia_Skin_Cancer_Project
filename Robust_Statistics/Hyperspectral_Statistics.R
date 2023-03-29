
# Code for Hyperspectral Band Analysis Statistics - DETTECTHIA PROJECT
#
# Code by Lloyd A. Courtenay - ladc1995@gmail.com
# TIDOP Research Group (http://tidop.usal.es/) - University of Salamanca
#
# Last Update: 28/03/2023
#

# Libraries ---------------------------

# check to see whether the required packages are installed on your system. If they are not, they will be installed

for (i in c("ggplot2", "tibble", "dplyr", "tidyverse", "e1071",
            "gridExtra",
            "car", "caret", "pROC")) {
  if(i %in% rownames(installed.packages()) == TRUE) {
    print(paste("The package named", i, "has already been installed", sep = " "))
  } else {install.packages(i)}
}; rm(i)

if ("pValueRobust" %in% rownames(installed.packages() == TRUE)) {
  print("The package names pValueRobust has already been installed")
} else {devtools::install_github("LACourtenay/pValueRobust")}

# load packages

library(ggplot2)
library(tibble)
library(dplyr)
library(tidyverse)
library(e1071)
library(gridExtra)
library(car)
library(pValueRobust) # can install using devtools::install_github("LACourtenay/pValueRobust")

#

# functions ----------------------------

# calculate normality data

normality_tests <- function(data) {
  normality_results <- tibble(
    Band_name = as.character(),
    Shapiro_w = as.numeric(),
    Shapiro_p = as.numeric(),
    Shapiro_Lower_pH0 = as.numeric(),
    Shapiro_pH0 = as.numeric(),
    Shapiro_Upper_pH0 = as.numeric(),
    Skew = as.numeric(),
    Kurtosis = as.numeric()
  )
  BAND_data <- data[,2:ncol(data)]
  for (i in 1:ncol(BAND_data)){
    target <- BAND_data[[paste("BAND_", i, sep = "")]]
    normality_results <- normality_results %>%
      add_row(
        Band_name = paste("BAND_", i, sep = ""),
        Shapiro_w = shapiro.test(target)$statistic[[1]],
        Shapiro_p = shapiro.test(target)$p.value[[1]],
        Shapiro_Lower_pH0 = p_H0(shapiro.test(target)$p.value[[1]], priors = 0.8),
        Shapiro_pH0 = p_H0(shapiro.test(target)$p.value[[1]], priors = 0.5),
        Shapiro_Upper_pH0 = p_H0(shapiro.test(target)$p.value[[1]], priors = 0.2),
        Skew = skewness(target),
        Kurtosis = kurtosis(target)
      )
  }
  normality_results <- normality_results %>%
    mutate(Band_name = factor(1:n())) %>%
    mutate(Band_freq = freq_details$Frequency..nm.)
  return(normality_results)
}

# calculation for residuals

residual_calculation <- function(data){
  
  input_data <- data$Sample
  
  residual_results <- tibble(
    Band_name = as.character(),
    Res_count = as.numeric()
  )
  BAND_data <- data[,2:ncol(data)]
  for (i in 1:ncol(BAND_data)) {
    target <- BAND_data[[paste("BAND_", i, sep = "")]]
    linear <- lm(target ~ input_data)
    res <- 1 - sum(linear$residuals ^ 2) / sum((target - mean(target)) ^ 2)
    residual_results <- residual_results %>%
      add_row(
        Band_name = paste("BAND_", i, sep = ""),
        Res_count = res
      )
  }
  residual_results <- residual_results %>%
    mutate(Band_name = factor(1:n())) %>%
    mutate(Band_freq = freq_details$Frequency..nm.)
  return(residual_results)
}

# calculate signature

signature_data <- function(data, type) {
  BAND_data <- data[,2:ncol(data)]
  if (type == "gaussian") {
    signature <- tibble(
      Band_name = as.character(),
      Lower_CI = as.numeric(),
      Central = as.numeric(),
      Upper_CI = as.numeric(),
      Deviation = as.numeric()
    )
    for (i in 1:ncol(BAND_data)){
      target <- BAND_data[[paste("BAND_", i, sep = "")]]
      signature <- signature %>%
        add_row(
          Band_name = paste("BAND_", i, sep = ""),
          Lower_CI = quantile_CI(target, q = 0.05),
          Central = mean(target),
          Upper_CI = quantile_CI(target, q = 0.95),
          Deviation = sd(target)
        )
    }
    signature <- signature %>%
      mutate(Band_name = factor(1:n()))
    return(signature)
  }
  else if (type == "robust") {
    signature <- tibble(
      Band_name = as.character(),
      Lower_CI = as.numeric(),
      Central = as.numeric(),
      Upper_CI = as.numeric(),
      Deviation = as.numeric()
    )
    for(i in 1:ncol(BAND_data)){
      target <- BAND_data[[paste("BAND_", i, sep = "")]]
      signature <- signature %>%
        add_row(
          Band_name = paste("BAND_", i, sep = ""),
          Lower_CI = quantile_CI(target, q = 0.05),
          Central = median(target),
          Upper_CI = quantile_CI(target, q = 0.95),
          Deviation = biweight_midvariance(target, sqrt_bwmv = TRUE)
        )
    }
    signature <- signature %>%
      mutate(Band_name = factor(1:n())) %>%
      mutate(Band_freq = freq_details$Frequency..nm.)
    return(signature)
  }
  else {"Choose type as either 'robust' or 'gaussian"}
}

# homoscedasticity tests

homoscedasticity_test <- function(data, method) {
  BAND_data <- data[, 2:ncol(data)]
  if (method == "levene") {
    homosc_results <- tibble(
      Band_name = as.character(),
      Test_Statistic = as.numeric(),
      p_Value = as.numeric()
    )
    for (i in 1:ncol(BAND_data)) {
      target <- BAND_data[[paste("BAND_", i, sep = "")]]
      homosc_results <- homosc_results %>%
        add_row(
          Band_name = paste("BAND_", i, sep = ""),
          Test_Statistic = leveneTest(target, data$Sample3, location = "median")$`F value`[[1]],
          p_Value = leveneTest(target, data$Sample3, location = "median")$`Pr(>F)`[[1]]
        )
    }
    
    #vecBFB <- Vectorize(p_BFB)
    
    pBFB <- c(); lower_post_odds <- c(); post_odds <- c()
    upper_post_odds <- c(); lower_FPR <- c(); FPR <- c();
    upper_FPR <- c(); lower_pH0 <- c(); pH0 <- c(); upper_pH0 <- c()
    
    for (result in 1:nrow(homosc_results)) {
      if(homosc_results[result,3] < 0.3681) {
        pBFB <- c(pBFB, p_BFB(as.numeric(homosc_results[result,3])))
        lower_post_odds <- c(lower_post_odds, posterior_odds(
          as.numeric(homosc_results[result,3]),
          priors = 0.8
        ))
        post_odds <- c(post_odds, posterior_odds(
          as.numeric(homosc_results[result,3]),
          priors = 0.5
        ))
        upper_post_odds <- c(upper_post_odds, posterior_odds(
          as.numeric(homosc_results[result,3]),
          priors = 0.2
        ))
        lower_FPR <- c(lower_FPR, FPR(
          as.numeric(homosc_results[result,3]),
          priors = 0.8
        ))
        FPR <- c(FPR, FPR(
          as.numeric(homosc_results[result,3]),
          priors = 0.5
        ))
        upper_FPR <- c(upper_FPR, FPR(
          as.numeric(homosc_results[result,3]),
          priors = 0.2
        ))
      } else {
        pBFB <- c(pBFB, NA)
        lower_post_odds <- c(lower_post_odds, NA)
        post_odds <- c(post_odds, NA)
        upper_post_odds <- c(upper_post_odds, NA)
        lower_FPR <- c(lower_FPR, NA)
        FPR <- c(FPR, NA)
        upper_FPR <- c(upper_FPR, NA)
      }
      lower_pH0 = c(lower_pH0, p_H0(as.numeric(
        homosc_results[result, 3]
      ), priors = 0.8))
      pH0 = c(pH0, p_H0(as.numeric(
        homosc_results[result, 3]
      ), priors = 0.5))
      upper_pH0 = c(upper_pH0, p_H0(as.numeric(
        homosc_results[result, 3]
      ), priors = 0.2))
    }
    
    Band_name <- factor(1:nrow(homosc_results))
    
    homosc_results <- homosc_results %>%
      add_column(pBFB, lower_post_odds, post_odds, upper_post_odds, lower_FPR, FPR, upper_FPR,
                 lower_pH0, pH0, upper_pH0) %>%
      mutate(Band_freq = freq_details$Frequency..nm.)
    
    return(homosc_results)
  }
  else if (method == "bartlett") {
    homosc_results <- tibble(
      Band_name = as.character(),
      Test_Statistic = as.numeric(),
      p_Value = as.numeric()
    )
    for (i in 1:ncol(BAND_data)) {
      target <- BAND_data[[paste("BAND_", i, sep = "")]]
      homosc_results <- homosc_results %>%
        add_row(
          Band_name = paste("BAND_", i, sep = ""),
          Test_Statistic = bartlett.test(target, data$Sample3)$statistic[[1]],
          p_Value = bartlett.test(target, data$Sample3)$p.value[[1]]
        )
    }
    pBFB <- c(); lower_post_odds <- c(); post_odds <- c()
    upper_post_odds <- c(); lower_FPR <- c(); FPR <- c();
    upper_FPR <- c(); lower_pH0 <- c(); pH0 <- c(); upper_pH0 <- c()
    
    for (result in 1:nrow(homosc_results)) {
      if(homosc_results[result,3] < 0.3681) {
        pBFB <- c(pBFB, p_BFB(as.numeric(homosc_results[result,3])))
        lower_post_odds <- c(lower_post_odds, posterior_odds(
          as.numeric(homosc_results[result,3]),
          priors = 0.8
        ))
        post_odds <- c(post_odds, posterior_odds(
          as.numeric(homosc_results[result,3]),
          priors = 0.5
        ))
        upper_post_odds <- c(upper_post_odds, posterior_odds(
          as.numeric(homosc_results[result,3]),
          priors = 0.2
        ))
        lower_FPR <- c(lower_FPR, FPR(
          as.numeric(homosc_results[result,3]),
          priors = 0.8
        ))
        FPR <- c(FPR, FPR(
          as.numeric(homosc_results[result,3]),
          priors = 0.5
        ))
        upper_FPR <- c(upper_FPR, FPR(
          as.numeric(homosc_results[result,3]),
          priors = 0.2
        ))
      } else {
        pBFB <- c(pBFB, NA)
        lower_post_odds <- c(lower_post_odds, NA)
        post_odds <- c(post_odds, NA)
        upper_post_odds <- c(upper_post_odds, NA)
        lower_FPR <- c(lower_FPR, NA)
        FPR <- c(FPR, NA)
        upper_FPR <- c(upper_FPR, NA)
      }
      lower_pH0 = c(lower_pH0, p_H0(as.numeric(
        homosc_results[result, 3]
      ), priors = 0.8))
      pH0 = c(pH0, p_H0(as.numeric(
        homosc_results[result, 3]
      ), priors = 0.5))
      upper_pH0 = c(upper_pH0, p_H0(as.numeric(
        homosc_results[result, 3]
      ), priors = 0.2))
    }
    
    Band_name <- factor(1:nrow(homosc_results))
    
    homosc_results <- homosc_results %>%
      add_column(pBFB, lower_post_odds, post_odds, upper_post_odds, lower_FPR, FPR, upper_FPR,
                 lower_pH0, pH0, upper_pH0) %>%
      mutate(Band_freq = freq_details$Frequency..nm.)
    return(homosc_results)
  }
  else {print("Choose a method between 'levene' and 'bartlett'")}
}

#

# load data ---------------------------

a <- read.csv(url("https://raw.githubusercontent.com/LACourtenay/Deteccthia_Skin_Cancer_Project/main/Robust_Statistics/dettecthia_dataset.csv"))
a$Sample <- as.factor(a$Sample)

a <- a[,c(1, 3:ncol(a))] # remove photo id

# convert to percentages (x 100)

for (band in 2:ncol(a)) {
  a[,band] <- a[,band] * 100
}; rm(band)

# load band frequency data

freq_details <- read.csv(url("https://raw.githubusercontent.com/LACourtenay/Deteccthia_Skin_Cancer_Project/main/Robust_Statistics/Camera_Frequency_Details.csv"))
freq_details$Band <- as.factor(freq_details$Band)
ggplot_freq_scale <- scale_x_continuous(breaks = seq(floor(min(freq_details$Frequency..nm.)),
                                                     ceiling(max(freq_details$Frequency..nm.)),
                                                     111))

# separate data according to sample

SCC <- a[a$Sample == "SCC",]
BCC <- a[a$Sample == "BCC",]
H <- a[a$Sample == "H",]
H_BCC <- rbind(H, BCC); H_BCC <- droplevels(H_BCC)
H_SCC <- rbind(H, SCC); H_SCC <- droplevels(H_SCC)
SCC_BCC <- rbind(SCC, BCC); SCC_BCC <- droplevels(SCC_BCC)
AK <- a[a$Sample == "AK",]
H_AK <- rbind(H, AK); H_AK <- droplevels(H_AK)
SCC_AK <- rbind(SCC, AK); SCC_AK <- droplevels(SCC_AK)
BCC_AK <- rbind(BCC, AK); BCC_AK <- droplevels(BCC_AK)

#

# Check data normality ----------------

# Calculate Shapiro results, skewness and kurtosis

H_normality_results <- normality_tests(H)
BCC_normality_results <- normality_tests(BCC)
SCC_normality_results <- normality_tests(SCC)
AK_normality_results <- normality_tests(AK)

#

# Prepare and view normality plots ----------------

# Shapiro W and P value Plots

shap_theme <- theme(panel.grid.major = element_blank(),
                    panel.grid.minor = element_blank(),
                    panel.background = element_blank(),
                    axis.title.y = element_text(margin = margin(r = 10)),
                    axis.ticks.x = element_blank(),
                    axis.text.x = element_text(margin = margin(t = 5)),
                    plot.margin = margin(0.5,0.5,0.5,0.5,"cm"),
                    plot.title = element_text(face = "bold",
                                              size = 15,
                                              margin=margin(0,0,10,0)),
                    axis.title = element_text(size = 15,
                                              face = "bold"),
                    axis.text = element_text(size = 10,
                                             face = "bold"))

shap_pvalues_SCC<-ggplot(data = SCC_normality_results,
                         aes(x = Band_freq,
                             y = log(Shapiro_p))) +
  geom_bar(stat = "identity", fill = "#999999") +
  xlab("Wavelength (nm)") +
  labs(title = "SCC - Shapiro Test (p Values)") +
  shap_theme +
  ggplot_freq_scale +
  geom_hline(yintercept = log(0.003), colour = "black", size = 1)
shap_pvalues_BCC<-ggplot(data = BCC_normality_results,
                         aes(x = Band_freq,
                             y = log(Shapiro_p))) +
  geom_bar(stat = "identity", fill = "#999999") +
  xlab("Wavelength (nm)") +
  labs(title = "BCC - Shapiro Test (p Values)") +
  shap_theme +
  ggplot_freq_scale +
  geom_hline(yintercept = log(0.003), colour = "black", size = 1)
shap_pvalues_H<-ggplot(data = H_normality_results,
                       aes(x = Band_freq,
                           y = log(Shapiro_p))) +
  geom_bar(stat = "identity", fill = "#999999") +
  xlab("Wavelength (nm)") +
  labs(title = "Healthy - Shapiro Test (p Values)") +
  shap_theme +
  ggplot_freq_scale +
  geom_hline(yintercept = log(0.003), colour = "black", size = 1)
shap_wvalues_H<-ggplot(data = H_normality_results,
                       aes(x = Band_freq,
                           y = Shapiro_w)) +
  geom_bar(stat = "identity", fill = "#999999") +
  xlab("Wavelength (nm)") +
  labs(title = "Healthy - Shapiro Test (w Values)") +
  shap_theme +
  ggplot_freq_scale +
  coord_cartesian(ylim = c(0.91,1))
shap_wvalues_SCC<-ggplot(data = SCC_normality_results,
                         aes(x = Band_freq,
                             y = Shapiro_w)) +
  geom_bar(stat = "identity", fill = "#999999") +
  xlab("Wavelength (nm)") +
  labs(title = "SCC - Shapiro Test (w Values)") +
  shap_theme +
  ggplot_freq_scale +
  coord_cartesian(ylim = c(0.91,1))
shap_wvalues_BCC<-ggplot(data = BCC_normality_results,
                         aes(x = Band_freq,
                             y = Shapiro_w)) +
  geom_bar(stat = "identity", fill = "#999999") +
  xlab("Wavelength (nm)") +
  labs(title = "BCC - Shapiro Test (w Values)") +
  shap_theme +
  ggplot_freq_scale +
  coord_cartesian(ylim = c(0.91,1))
skew_values_BCC<-ggplot(data = BCC_normality_results,
                        aes(x = Band_freq,
                            y = Skew)) +
  geom_bar(stat = "identity", fill = "#999999") +
  xlab("Wavelength (nm)") +
  labs(title = "BCC - Skewness") +
  shap_theme +
  ggplot_freq_scale
skew_values_SCC<-ggplot(data = SCC_normality_results,
                        aes(x = Band_freq,
                            y = Skew)) +
  geom_bar(stat = "identity", fill = "#999999") +
  xlab("Wavelength (nm)") +
  labs(title = "SCC - Skewness") +
  shap_theme +
  ggplot_freq_scale
skew_values_H<-ggplot(data = H_normality_results,
                      aes(x = Band_freq,
                          y = Skew)) +
  geom_bar(stat = "identity", fill = "#999999") +
  xlab("Wavelength (nm)") +
  labs(title = "Healthy - Skewness") +
  shap_theme +
  ggplot_freq_scale
kurt_values_H<-ggplot(data = H_normality_results,
                      aes(x = Band_freq,
                          y = Kurtosis)) +
  geom_bar(stat = "identity", fill = "#999999") +
  xlab("Wavelength (nm)") +
  labs(title = "Healthy - Kurtosis") +
  shap_theme +
  ggplot_freq_scale +
  geom_hline(yintercept = 0, colour = "black", size = 1)
kurt_values_SCC<-ggplot(data = SCC_normality_results,
                        aes(x = Band_freq,
                            y = Kurtosis)) +
  geom_bar(stat = "identity", fill = "#999999") +
  xlab("Wavelength (nm)") +
  labs(title = "SCC - Kurtosis") +
  shap_theme +
  ggplot_freq_scale +
  geom_hline(yintercept = 0, colour = "black", size = 1)
kurt_values_BCC<-ggplot(data = BCC_normality_results,
                        aes(x = Band_freq,
                            y = Kurtosis)) +
  geom_bar(stat = "identity", fill = "#999999") +
  xlab("Wavelength (nm)") +
  labs(title = "BCC - Kurtosis") +
  shap_theme +
  ggplot_freq_scale +
  geom_hline(yintercept = 0, colour = "black", size = 1)

#

base_p_norm_theme <- theme(axis.title = element_text(face = "bold", size = 15),
                           axis.title.x = element_text(margin = margin(t = 10)),
                           axis.title.y = element_text(margin = margin(r = 10)),
                           axis.text = element_text(size = 12, face = "bold"),
                           plot.title = element_text(face = "bold", size = 15),
                           axis.text.y = element_text(margin = margin(r = 5)),
                           axis.text.x = element_text(margin = margin(t = 5)))

H_shap_ph0_plot<-ggplot(data = H_normality_results,
                        aes(x = Band_freq, y = Shapiro_pH0, group = 1)) +
  geom_ribbon(aes(ymin = Shapiro_Lower_pH0, ymax = Shapiro_Upper_pH0),
              alpha = 0.2) +
  geom_line(size = 0.75) +
  theme_bw() +
  xlab("Wavelength (nm)") +
  ylab("Probability of Null Hypothesis") +
  ggtitle("Shapiro Test Results - Healthy") +
  theme(plot.margin = unit(c(1,1,0.5,1), "cm")) +
  base_p_norm_theme +
  ggplot_freq_scale +
  coord_cartesian(ylim = c(0, 1))
BCC_shap_ph0_plot<-ggplot(data = BCC_normality_results,
                          aes(x = Band_freq, y = Shapiro_pH0, group = 1)) +
  geom_ribbon(aes(ymin = Shapiro_Lower_pH0, ymax = Shapiro_Upper_pH0),
              alpha = 0.2) +
  geom_line(size = 0.75) +
  theme_bw() +
  xlab("Wavelength (nm)") +
  ylab("Probability of Null Hypothesis") +
  ggtitle("Shapiro Test Results - BCC") +
  theme(plot.margin = unit(c(0.5,1,1,1), "cm")) +
  base_p_norm_theme +
  ggplot_freq_scale +
  coord_cartesian(ylim = c(0, 1))
SCC_shap_ph0_plot<-ggplot(data = SCC_normality_results,
                          aes(x = Band_freq, y = Shapiro_pH0, group = 1)) +
  geom_ribbon(aes(ymin = Shapiro_Lower_pH0, ymax = Shapiro_Upper_pH0),
              alpha = 0.2) +
  geom_line(size = 0.75) +
  theme_bw() +
  xlab("Wavelength (nm)") +
  ylab("Probability of Null Hypothesis") +
  ggtitle("Shapiro Test Results - SCC") +
  theme(plot.margin = unit(c(0.5,1,1,1), "cm")) +
  base_p_norm_theme +
  ggplot_freq_scale +
  coord_cartesian(ylim = c(0, 1))

shap_pvalues_AK<-ggplot(data = AK_normality_results,
                        aes(x = Band_freq,
                            y = log(Shapiro_p))) +
  geom_bar(stat = "identity", fill = "#999999") +
  xlab("Wavelength (nm)") +
  labs(title = "AK - Shapiro Test (p Values)") +
  shap_theme +
  ggplot_freq_scale +
  geom_hline(yintercept = log(0.003), colour = "black", size = 1)
shap_wvalues_AK<-ggplot(data = AK_normality_results,
                        aes(x = Band_freq,
                            y = Shapiro_w)) +
  geom_bar(stat = "identity", fill = "#999999") +
  xlab("Wavelength (nm)") +
  labs(title = "AK - Shapiro Test (w Values)") +
  shap_theme +
  ggplot_freq_scale +
  coord_cartesian(ylim = c(0.91,1))
skew_values_AK<-ggplot(data = AK_normality_results,
                       aes(x = Band_freq,
                           y = Skew)) +
  geom_bar(stat = "identity", fill = "#999999") +
  xlab("Wavelength (nm)") +
  labs(title = "AK - Skewness") +
  shap_theme +
  ggplot_freq_scale
kurt_values_AK<-ggplot(data = AK_normality_results,
                       aes(x = Band_freq,
                           y = Kurtosis)) +
  geom_bar(stat = "identity", fill = "#999999") +
  xlab("Wavelength (nm)") +
  labs(title = "AK - Kurtosis") +
  shap_theme +
  ggplot_freq_scale +
  geom_hline(yintercept = 0, colour = "black", size = 1)

AK_shap_ph0_plot<-ggplot(data = AK_normality_results,
                         aes(x = Band_freq, y = Shapiro_pH0, group = 1)) +
  geom_ribbon(aes(ymin = Shapiro_Lower_pH0, ymax = Shapiro_Upper_pH0),
              alpha = 0.2) +
  geom_line(size = 0.75) +
  theme_bw() +
  xlab("Wavelength (nm)") +
  ylab("Probability of Null Hypothesis") +
  ggtitle("Shapiro Test Results - AK") +
  theme(plot.margin = unit(c(0.5,1,1,1), "cm")) +
  base_p_norm_theme +
  ggplot_freq_scale +
  coord_cartesian(ylim = c(0, 1))

#

X11(); grid.arrange(shap_pvalues_H, shap_wvalues_H,
                    shap_pvalues_SCC, shap_wvalues_SCC,
                    shap_pvalues_BCC, shap_wvalues_BCC,
                    shap_pvalues_AK, shap_wvalues_AK,
                    nrow = 4,
                    ncol = 2)

X11(); grid.arrange(skew_values_H, kurt_values_H,
                    skew_values_SCC, kurt_values_SCC,
                    skew_values_BCC, kurt_values_BCC,
                    skew_values_AK, kurt_values_AK,
                    nrow = 4,
                    ncol = 2)

X11(); grid.arrange(H_shap_ph0_plot, BCC_shap_ph0_plot, SCC_shap_ph0_plot, AK_shap_ph0_plot,
                    ncol = 2, nrow = 2)

# clean memory

rm(H_shap_ph0_plot, BCC_shap_ph0_plot, SCC_shap_ph0_plot, shap_theme, base_p_norm_theme,
   skew_values_H, kurt_values_H, skew_values_SCC, kurt_values_SCC, skew_values_BCC, kurt_values_BCC,
   shap_pvalues_H, shap_wvalues_H, shap_pvalues_SCC, shap_wvalues_SCC, shap_pvalues_BCC, shap_wvalues_BCC,
   shap_pvalues_AK, shap_wvalues_AK, skew_values_AK, kurt_values_AK, AK_shap_ph0_plot)


# Calculate and plot residuals -----------------------

residual_plot_theme <- theme(plot.margin = unit(c(1,0.5,1,1), "cm"),
                             axis.title = element_text(face = "bold", size = 20),
                             axis.title.x = element_text(margin = margin(t = 10)),
                             axis.title.y = element_text(margin = margin(r = 10)),
                             axis.text = element_text(size = 13.5, face = "bold"),
                             plot.title = element_text(face = "bold", size = 20),
                             axis.text.y = element_text(margin = margin(r = 5)),
                             axis.text.x = element_text(margin = margin(t = 5)),
                             plot.subtitle = element_text(face = "bold", size = 15)
)

residual_plot_H_SCC <- ggplot(data = residual_calculation(rbind(H, SCC)),
                            aes(x = Band_freq, y = Res_count, group = 1)) +
  geom_line(stat = "identity", colour = "black", size = 1) +
  theme_bw() + residual_plot_theme +
  xlab("Wavelength (nm)") +
  ylab("Residuals") +
  ggplot_freq_scale +
  labs(title = "H vs SCC")

residual_plot_H_BCC <- ggplot(data = residual_calculation(rbind(H, BCC)),
                            aes(x = Band_freq, y = Res_count, group = 1)) +
  geom_line(stat = "identity", colour = "black", size = 1) +
  theme_bw() + residual_plot_theme +
  xlab("Wavelength (nm)") +
  ylab("Residuals") +
  ggplot_freq_scale +
  labs(title = "H vs BCC")

residual_plot_SCC_BCC <- ggplot(data = residual_calculation(rbind(SCC, BCC)),
                                                      aes(x = Band_freq, y = Res_count, group = 1)) +
  geom_line(stat = "identity", colour = "black", size = 1) +
  theme_bw() + residual_plot_theme +
  xlab("Wavelength (nm)") +
  ylab("Residuals") +
  ggplot_freq_scale +
  labs(title = "SCC vs BCC")

residual_plot_H_AK <- ggplot(data = residual_calculation(rbind(H, AK)),
                                aes(x = Band_freq, y = Res_count, group = 1)) +
  geom_line(stat = "identity", colour = "black", size = 1) +
  theme_bw() + residual_plot_theme +
  xlab("Wavelength (nm)") +
  ylab("Residuals") +
  ggplot_freq_scale +
  labs(title = "H vs AK")

residual_plot_SCC_AK <- ggplot(data = residual_calculation(rbind(SCC, AK)),
                             aes(x = Band_freq, y = Res_count, group = 1)) +
  geom_line(stat = "identity", colour = "black", size = 1) +
  theme_bw() + residual_plot_theme +
  xlab("Wavelength (nm)") +
  ylab("Residuals") +
  ggplot_freq_scale +
  labs(title = "SCC vs AK")

residual_plot_BCC_AK <- ggplot(data = residual_calculation(rbind(BCC, AK)),
                               aes(x = Band_freq, y = Res_count, group = 1)) +
  geom_line(stat = "identity", colour = "black", size = 1) +
  theme_bw() + residual_plot_theme +
  xlab("Wavelength (nm)") +
  ylab("Residuals") +
  ggplot_freq_scale +
  labs(title = "BCC vs AK")

residual_plot_H_AK <- ggplot(data = residual_calculation(rbind(H, AK)),
                             aes(x = Band_freq, y = Res_count, group = 1)) +
  geom_line(stat = "identity", colour = "black", size = 1) +
  theme_bw() + residual_plot_theme +
  xlab("Wavelength (nm)") +
  ylab("Residuals") +
  ggplot_freq_scale +
  labs(title = "H vs AK")

#

X11(); grid.arrange(residual_plot_H_SCC,
             residual_plot_H_BCC,
             residual_plot_H_AK,
             residual_plot_SCC_AK,
             residual_plot_BCC_AK,
             residual_plot_SCC_BCC,
             ncol = 3)

rm(residual_plot_H_AK, residual_plot_theme,
   residual_plot_H_SCC, residual_plot_H_BCC,
   residual_plot_SCC_AK, residual_plot_BCC_AK,
   residual_plot_SCC_BCC
   )

#

# Calculate signatures ---------------

H_signature <- signature_data(H, type = "robust") # select either robust signature or gaussian signature
SCC_signature <- signature_data(SCC, type = "robust")
BCC_signature <- signature_data(BCC, type = "robust")
AK_signature <- signature_data(AK, type = "robust")

# Visualise plot

sig_theme <- theme(axis.title = element_text(face = "bold", size = 20),
                   axis.title.x = element_text(margin = margin(t = 10)),
                   axis.title.y = element_text(margin = margin(r = 10)),
                   axis.text = element_text(size = 12.5, face = "bold"),
                   plot.title = element_text(face = "bold", size = 22.5),
                   axis.text.y = element_text(margin = margin(r = 5)),
                   axis.text.x = element_text(margin = margin(t = 5)))

X11(); grid.arrange(ggplot() +
                      geom_line(data = SCC_signature, aes(x = Band_freq, y = Central, group = 1),
                                size = 0.75, colour = "red") +
                      geom_line(data = SCC_signature, aes(x = Band_freq, y = Central + Upper_CI, group = 1),
                                size = 0.5, colour = "red", linetype = "solid") +
                      geom_line(data = SCC_signature, aes(x = Band_freq, y = Central - Lower_CI, group = 1),
                                size = 0.5, colour = "red", linetype = "solid") +
                      geom_line(data = BCC_signature, aes(x = Band_freq, y = Central, group = 1),
                                size = 0.75, colour = "black") +
                      geom_line(data = BCC_signature, aes(x = Band_freq, y = Central + Upper_CI, group = 1),
                                size = 0.5, colour = "black", linetype = "solid") +
                      geom_line(data = BCC_signature, aes(x = Band_freq, y = Central - Lower_CI, group = 1),
                                size = 0.5, colour = "black", linetype = "solid") +
                      geom_line(data = H_signature, aes(x = Band_freq, y = Central, group = 1),
                                size = 0.75, colour = "blue") +
                      geom_line(data = H_signature, aes(x = Band_freq, y = Central + Upper_CI, group = 1),
                                size = 0.5, colour = "blue", linetype = "solid") +
                      geom_line(data = H_signature, aes(x = Band_freq, y = Central - Lower_CI, group = 1),
                                size = 0.5, colour = "blue", linetype = "solid") +
                      geom_line(data = AK_signature, aes(x = Band_freq, y = Central, group = 1),
                                size = 0.75, colour = "orange") +
                      geom_line(data = AK_signature, aes(x = Band_freq, y = Central + Upper_CI, group = 1),
                                size = 0.5, colour = "orange", linetype = "solid") +
                      geom_line(data = AK_signature, aes(x = Band_freq, y = Central - Lower_CI, group = 1),
                                size = 0.5, colour = "orange", linetype = "solid") +
                      theme_bw() +
                      theme(plot.margin = unit(c(1,0.5,1,1), "cm")) +
                      sig_theme + 
                      ggtitle("Robust Signature") +
                      xlab("Wavelength (nm)") +
                      ylab("Reflectance (%)") +
                      coord_cartesian(ylim = c(0, 175)) +
                      ggplot_freq_scale, ggplot() +
                      geom_line(data = SCC_signature, aes(x = Band_freq, y = Deviation, group = 1),
                                size = 0.75, colour = "red") +
                      geom_line(data = BCC_signature, aes(x = Band_freq, y = Deviation, group = 1),
                                size = 0.75, colour = "black") +
                      geom_line(data = H_signature, aes(x = Band_freq, y = Deviation, group = 1),
                                size = 0.75, colour = "blue") +
                      geom_line(data = AK_signature, aes(x = Band_freq, y = Deviation, group = 1),
                                size = 0.75, colour = "orange") +
                      theme_bw() +
                      theme(plot.margin = unit(c(1,1,1,0.5), "cm")) +
                      sig_theme +
                      ggtitle("Robust Variance") +
                      xlab("Wavelength (nm)") +
                      ylab("Reflectance (%)") +
                      coord_cartesian(ylim = c(0, 40)) +
                      ggplot_freq_scale,
                    ncol = 2, nrow = 1)

# clear memory

rm(H_signature, SCC_signature, BCC_signature, AK_signature, sig_theme)

#

# Hypothesis Testing ---------------

# homoscedasticity

H_BCC_homosc <- homoscedasticity_test(H_BCC, method = "levene") # select either levene or bartlett test
H_SCC_homosc <- homoscedasticity_test(H_SCC, method = "levene")
SCC_BCC_homosc <- homoscedasticity_test(SCC_BCC, method = "levene")

# prepare homoscedasticity p-value plots

homosc_theme <- theme(plot.margin = unit(c(1,1,0.5,1), "cm"),
                      axis.title = element_text(face = "bold", size = 15),
                      axis.title.x = element_text(margin = margin(t = 10)),
                      axis.title.y = element_text(margin = margin(r = 10)),
                      axis.text = element_text(size = 12, face = "bold"),
                      plot.title = element_text(face = "bold", size = 15),
                      axis.text.y = element_text(margin = margin(r = 5)),
                      axis.text.x = element_text(margin = margin(t = 5)))

homosc_H_BCC_plot<-ggplot(data = H_BCC_homosc,
                          aes(x = Band_freq, y = pH0, group = 1)) +
  geom_ribbon(aes(ymin = lower_pH0, ymax = upper_pH0),
              alpha = 0.2) +
  geom_line(size = 0.75) +
  theme_bw() +
  xlab("Wavelength (nm)") +
  ylab("P(H0)") +
  ggtitle("H vs BCC") +
  homosc_theme +
  coord_cartesian(ylim = c(0, 1)) +
  ggplot_freq_scale
homosc_H_SCC_plot<-ggplot(data = H_SCC_homosc,
                          aes(x = Band_freq, y = pH0, group = 1)) +
  geom_ribbon(aes(ymin = lower_pH0, ymax = upper_pH0),
              alpha = 0.2) +
  geom_line(size = 0.75) +
  theme_bw() +
  xlab("Wavelength (nm)") +
  ylab("P(H0)") +
  ggtitle("H vs SCC") +
  homosc_theme +
  coord_cartesian(ylim = c(0, 1)) +
  ggplot_freq_scale
homosc_SCC_BCC_plot<-ggplot(data = SCC_BCC_homosc,
                            aes(x = Band_freq, y = pH0, group = 1)) +
  geom_ribbon(aes(ymin = lower_pH0, ymax = upper_pH0),
              alpha = 0.2) +
  geom_line(size = 0.75) +
  theme_bw() +
  xlab("Wavelength (nm)") +
  ylab("P(H0)") +
  ggtitle("SCC vs BCC") +
  homosc_theme +
  coord_cartesian(ylim = c(0, 1)) +
  ggplot_freq_scale

# prepare homoscedasticity test statistic plots

homosc_H_SCC_test_stat_plot<-ggplot(data = H_SCC_homosc, 
                                    aes(x = Band_freq,
                                        group = 1,
                                        y = Test_Statistic)) +
  geom_line(stat = "identity", size = 0.75) +
  theme_bw() +
  xlab("Wavelength (nm)") +
  ylab("F") +
  homosc_theme +
  labs(title = "H vs SCC") +
  theme(plot.title = element_text(face = "bold")) +
  ggplot_freq_scale
homosc_H_BCC_test_stat_plot<-ggplot(data = H_BCC_homosc, 
                                    aes(x = Band_freq,
                                        group = 1,
                                        y = Test_Statistic)) +
  geom_line(stat = "identity", size = 0.75) +
  theme_bw() +
  xlab("Wavelength (nm)") +
  ylab("F") +
  homosc_theme +
  labs(title = "H vs BCC") +
  theme(plot.title = element_text(face = "bold")) +
  ggplot_freq_scale
homosc_SCC_BCC_test_stat_plot<-ggplot(data = SCC_BCC_homosc, 
                                      aes(x = Band_freq,
                                          group = 1,
                                          y = Test_Statistic)) +
  geom_line(stat = "identity", size = 0.75) +
  theme_bw() +
  xlab("Wavelength (nm)") +
  ylab("F") +
  homosc_theme +
  labs(title = "SCC vs BCC") +
  theme(plot.title = element_text(face = "bold")) +
  ggplot_freq_scale

# View plots

X11(); grid.arrange(homosc_H_SCC_plot, homosc_H_SCC_test_stat_plot,
                    homosc_H_BCC_plot, homosc_H_BCC_test_stat_plot,
                    homosc_SCC_BCC_plot, homosc_SCC_BCC_test_stat_plot,
                    nrow = 3, ncol = 2)

# clear memory

rm(homosc_H_SCC_plot, homosc_H_SCC_test_stat_plot,
   homosc_H_BCC_plot, homosc_H_BCC_test_stat_plot,
   homosc_SCC_BCC_plot, homosc_SCC_BCC_test_stat_plot,
   H_BCC_homosc, H_SCC_homosc, SCC_BCC_homosc,
   homosc_theme)

#

# Hypothesis Testing AK --------------

# homoscedasticity

H_AK_homosc <- homoscedasticity_test(H_AK, method = "levene") # select either levene or bartlett test
SCC_AK_homosc <- homoscedasticity_test(SCC_AK, method = "levene") # select either levene or bartlett test
BCC_AK_homosc <- homoscedasticity_test(BCC_AK, method = "levene") # select either levene or bartlett test

#write.table(
#  BCC_AK_homosc,
#  "Recalibrated Results V2\\BCC_AK_homosc_cal.csv", sep = ";", row.names = FALSE, col.names = TRUE
#)

# view numeric homoscedasticity results

View(H_AK_homosc)
View(SCC_AK_homosc)
View(BCC_AK_homosc)

# prepare homoscedasticity p-value plots

homosc_theme <- theme(plot.margin = unit(c(1,1,0.5,1), "cm"),
                      axis.title = element_text(face = "bold", size = 15),
                      axis.title.x = element_text(margin = margin(t = 10)),
                      axis.title.y = element_text(margin = margin(r = 10)),
                      axis.text = element_text(size = 12, face = "bold"),
                      plot.title = element_text(face = "bold", size = 15),
                      axis.text.y = element_text(margin = margin(r = 5)),
                      axis.text.x = element_text(margin = margin(t = 5)))

homosc_H_AK_plot<-ggplot(data = H_AK_homosc,
                         aes(x = Band_freq, y = pH0, group = 1)) +
  geom_ribbon(aes(ymin = lower_pH0, ymax = upper_pH0),
              alpha = 0.2) +
  geom_line(size = 0.75) +
  theme_bw() +
  xlab("Wavelength (nm)") +
  ylab("P(H0)") +
  ggtitle("H vs AK") +
  homosc_theme +
  coord_cartesian(ylim = c(0, 1)) +
  ggplot_freq_scale
homosc_SCC_AK_plot<-ggplot(data = SCC_AK_homosc,
                           aes(x = Band_freq, y = pH0, group = 1)) +
  geom_ribbon(aes(ymin = lower_pH0, ymax = upper_pH0),
              alpha = 0.2) +
  geom_line(size = 0.75) +
  theme_bw() +
  xlab("Wavelength (nm)") +
  ylab("P(H0)") +
  ggtitle("SCC vs AK") +
  homosc_theme +
  coord_cartesian(ylim = c(0, 1)) +
  ggplot_freq_scale
homosc_BCC_AK_plot<-ggplot(data = BCC_AK_homosc,
                           aes(x = Band_freq, y = pH0, group = 1)) +
  geom_ribbon(aes(ymin = lower_pH0, ymax = upper_pH0),
              alpha = 0.2) +
  geom_line(size = 0.75) +
  theme_bw() +
  xlab("Wavelength (nm)") +
  ylab("P(H0)") +
  ggtitle("BCC vs AK") +
  homosc_theme +
  coord_cartesian(ylim = c(0, 1)) +
  ggplot_freq_scale

# prepare homoscedasticity test statistic plots

homosc_SCC_AK_test_stat_plot<-ggplot(data = SCC_AK_homosc, 
                                     aes(x = Band_freq,
                                         group = 1,
                                         y = Test_Statistic)) +
  geom_line(stat = "identity", size = 0.75) +
  theme_bw() +
  xlab("Wavelength (nm)") +
  ylab("F") +
  homosc_theme +
  labs(title = "SCC vs AK") +
  theme(plot.title = element_text(face = "bold")) +
  ggplot_freq_scale
homosc_H_AK_test_stat_plot<-ggplot(data = H_AK_homosc, 
                                   aes(x = Band_freq,
                                       group = 1,
                                       y = Test_Statistic)) +
  geom_line(stat = "identity", size = 0.75) +
  theme_bw() +
  xlab("Wavelength (nm)") +
  ylab("F") +
  homosc_theme +
  labs(title = "H vs AK") +
  theme(plot.title = element_text(face = "bold")) +
  ggplot_freq_scale
homosc_BCC_AK_test_stat_plot<-ggplot(data = BCC_AK_homosc, 
                                     aes(x = Band_freq,
                                         group = 1,
                                         y = Test_Statistic)) +
  geom_line(stat = "identity", size = 0.75) +
  theme_bw() +
  xlab("Wavelength (nm)") +
  ylab("F") +
  homosc_theme +
  labs(title = "BCC vs AK") +
  theme(plot.title = element_text(face = "bold")) +
  ggplot_freq_scale

# View plots

X11(); grid.arrange(homosc_H_AK_plot, homosc_H_AK_test_stat_plot,
                    homosc_SCC_AK_plot, homosc_SCC_AK_test_stat_plot,
                    homosc_BCC_AK_plot, homosc_BCC_AK_test_stat_plot,
                    nrow = 3, ncol = 2)

# clear memory

rm(homosc_H_AK_plot, homosc_H_AK_test_stat_plot,
   homosc_SCC_AK_plot, homosc_SCC_AK_test_stat_plot,
   homosc_BCC_AK_plot, homosc_BCC_AK_test_stat_plot,
   H_AK_homosc, SCC_AK_homosc, BCC_AK_homosc,
   homosc_theme)

#

# Multivariate Hypothesis entire spectrum ---------------------------------------

# check final windows

select_data_set <- rbind(H, SCC, BCC, AK)
select_data_set <- droplevels(select_data_set)
select_data <- select_data_set[,c(1:40, 45:66, 72:90, 103:118) + 1]
pairwise.wilcox.test(as.matrix(select_data), select_data_set$Sample,
                     p.adjust.method = "BH")

# Final windows - without melanin bands

select_data_set <- rbind(H, SCC, BCC, AK)
select_data_set <- droplevels(select_data_set)
select_data <- select_data_set[,c(45:66, 72:90, 103:118) + 1]
pairwise.wilcox.test(as.matrix(select_data), select_data_set$Sample,
                     p.adjust.method = "BH")

# Best three bands

select_data_set <- rbind(H, SCC, BCC, AK)
select_data_set <- droplevels(select_data_set)
select_data <- select_data_set[,c(7, 30, 47) + 1]
pairwise.wilcox.test(as.matrix(select_data), select_data_set$Sample,
                     p.adjust.method = "BH")


#
