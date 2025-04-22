#Don't forget to install all of the library using: install.packages('Library name')

install.packages(c("fst", "dplyr", "stringr", "crayon", "tibble", "ResourceSelection", 
                   "caret", "pROC", "Metrics", "ggplot2", "readr", "readxl", 
                   "mice", "factoextra"))
{
  library(fst)
  library(dplyr)
  library(stringr)
  library(crayon)
  library(tibble)
  library(ResourceSelection)
  library(caret)
  library(pROC)
  library(Metrics)
  library(ggplot2)
  library(readr)
  library(readxl)
  library(mice)
  library(factoextra)
}
setwd("C:\\Users\\Alan\\OneDrive\\桌面\\利率债择时模型")
getwd()

data <- read_excel("宏观因子库.xlsx")


#Select only numerical data
data_numeric <- data %>% 
  select(where(is.numeric))

#Standardise the data
scaled_data <- scale(data_numeric)

#Find constant/zero variance column (remove Near-Zero Variance Columns)
nzv_cols <- nearZeroVar(scaled_data, saveMetrics = TRUE)
problem_cols <- nzv_cols[nzv_cols$zeroVar | nzv_cols$nzv, ]
cols_to_remove <- rownames(problem_cols) 
nzv_filtered_data <- scaled_data[, !(colnames(scaled_data) %in% cols_to_remove), drop = FALSE]

#Impute missing data using predictive mean matching (PPM)
colSums(is.na(nzv_filtered_data))
#Create 5 separate imputed datasets for variability in plausible values and improve accuracy
imputed_data <- mice(nzv_filtered_data, m = 5, method = "pmm")
combined_data <- complete(imputed_data, 1)

#Check the number of na
sum(is.na(combined_data))
colSums(is.na(combined_data))
#imputed_data$method
#sapply(combined_data, class)
#error<- as.tibble(colSums(is.na(combined_data)), imputed_data$method)
#sum(error$`colSums(is.na(combined_data))`!= 0 ,error$`imputed_data$method` != 'pmm',  na.rm =TRUE)

#Fill na with 0
combined_data[is.na(combined_data)] <- 0

#Run PCA
pca_result <- prcomp(combined_data, center = TRUE, scale. = TRUE)
summary(pca_result)

#PCA Visualisation
variance <- pca_result$sdev^2 / sum(pca_result$sdev^2)

#Variance plot
ggplot(data = NULL, aes(x = 1:length(variance), y = variance)) +
  geom_line(aes(y = variance), color = "blue") +
  geom_point(aes(y = variance), color = "red") +
  labs(x = "Principal Component", y = "Proportion of Variance", title = "Scree Plot")

#Score plot (Projection onto PCs)
scores <- as.tibbe(pca_result$x)
ggplot(scores, aes(x = PC1, y = PC2, color = "PC")) +
  geom_point() +
  labs(title = "PCA Scores Plot")
fviz_eig(pca_result, addlabels = TRUE)
fviz_pca_biplot(pca_result, repel = TRUE, col.var = "red", label = FALSE)

#3 Identify Top Contributors to PC1
head(scores[, 1:3])

loadings <- pca_result$rotation
head(loadings[, 1:2])

top_vars <- loadings[order(abs(loadings[, 1]), decreasing = TRUE), 1]
head(top_vars, 10)

#Add the date back to the data
date <- data[[1]]
combined_data <- cbind(指標名稱 = date, combined_data)

#Save the data
saveRDS(combined_data, "../cleaned_macro_data.rds")
