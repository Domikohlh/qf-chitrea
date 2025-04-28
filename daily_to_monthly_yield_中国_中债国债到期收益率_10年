library(readxl)
library(dplyr)
library(lubridate)
library(openxlsx)


df <- read_excel("C:\\Users\\Alan\\OneDrive\\桌面\\利率债择时模型\\中国_中债国债到期收益率_10年.xlsx")


print(colnames(df))


df <- df %>%
  mutate(date = as.Date(`time`)) %>%
  mutate(month = floor_date(date, "month"))


monthly_mean <- df %>%
  group_by(month) %>%
  summarise(monthly_avg_yield = mean(`中债国债到期收益率_10年`, na.rm = TRUE))


write.xlsx(monthly_mean, "C:\\Users\\Alan\\OneDrive\\桌面\\利率债择时模型\\monthly_avg_yield.xlsx")


print(head(monthly_mean))
