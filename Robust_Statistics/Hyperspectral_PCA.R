
# Code for Hyperspectral Kernel-PCA - DETTECTHIA PROJECT
#
# Code by Lloyd A. Courtenay - ladc1995@gmail.com
# TIDOP Research Group (http://tidop.usal.es/) - University of Salamanca
#
# Last Update: 28/03/2023
#

# libraries ----------------------

library(GraphGMM) # can install using devtools::install_github("LACourtenay/GraphGMM")
library(ggplot2)

#

# load data -----------------

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

# select optimal channels ---------------------

select_data_set <- rbind(H, SCC, BCC, AK)
select_data_set <- droplevels(select_data_set)
select_data <- select_data_set[,c(1:40, 45:66, 72:90, 103:118) + 1]
select_data <- select_data[-c(2873, 2874),] # filter out two anomalies for visualisation
select_data_set <- select_data_set[-c(2873, 2874),] # filter out two anomalies for visualisation

#

# Kernel Principal Components Analysis -------------------------

pca_plot(select_data, kernel = "spline", hyperparm = 0, main = "Kernel-PCA (Spline)")
pca_biplot(select_data, n_variables = 50, kernel = "spline", hyperparm = 0, main = "Kernel-PCA (Spline)")

#

# visualising samples seperately ----------------------------

pca_data <- kernel_pca(select_data, select_data_set$Sample3, kernel = "spline", hyperparam = 1,
                       main = "Kernel-PCA (Spline)")
pc_scores <- pca_data$pc_scores

xlim = range(pc_scores[,1])
ylim = range(pc_scores[,2])
target_sample <- "H"

if (target_sample == "BCC") {
  target_colour <- "black"
  target_name <- "BCC"
} else if (target_sample == "SCC") {
  target_colour <- "red"
  target_name <- "SCC"
} else if (target_sample == "AK") {
  target_colour <- "orange"
  target_name <- "AK"
} else if (target_sample == "H") {
  target_colour <- "blue"
  target_name <- "H"
}

target_data <- pc_scores[select_data_set$Sample == target_sample, 1:2]
colnames(target_data) <- c("x", "y")

ggplot(data = as.data.frame(target_data), aes(x = x, y = y)) +
  geom_point(stat = "identity", size = 1, color = target_colour) +
  xlab("PC1") +
  ylab("PC2") +
  ggtitle(target_name) +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    plot.margin = unit(c(1,1,1,1), "cm"),
    plot.title = element_text(face = "bold", size = 20),
    plot.subtitle = element_text(size = 15),
    panel.border = element_rect(colour = "black", fill = NA),
    axis.title.x = element_text(face = "bold", size = 18,
                                margin = margin(t = 10, r = 0, b = 5, l = 0)),
    axis.title.y = element_text(face = "bold", size = 18,
                                margin = margin(t = 0, r = 10, b = 0, l = 0)),
    axis.text.x = element_text(angle = 90, size = 15, face = "bold"),
    axis.text.y = element_text(size = 15, face = "bold"),
    axis.ticks.x = element_blank(),
    axis.ticks.y = element_blank()
  ) +
  geom_vline(xintercept = 0,
             colour = "black",
             size = 0.5,
             linetype = "dashed") +
  geom_hline(yintercept = 0,
             colour = "black",
             linetype = "dashed",
             size = 0.5) +
  coord_cartesian(xlim = xlim, ylim = ylim)

#

