# library(xlsx)
# df <- read.xlsx('../../data/爬取的数据_cleaned_Price缺失较多-无地区信息.xlsx')
df <- read.csv('yecanming/爬虫数据特征工程/r-data1.csv')

str(df)
summary(df$Price)
hist(df$Price)

table(df$Features.Electronics.Vcr)

continu <- c("Price", "Basics.Length..ft.","Propulsion.TotalPower..hp.","Specifications.Accommodations.DoubleBerths","Specifications.Accommodations.Cabins","Specifications.Accommodations.Heads","Propulsion.EngineYear","Specifications.Speed.Distance.CruisingSpeed..kn.","Specifications.Speed.Distance.MaxSpeed..kn.","Specifications.Dimensions.LengthOnDeck..ft.","Specifications.Dimensions.LengthAtWaterline..ft.","Specifications.Dimensions.MaxBridgeClearance..ft.")
cor(df[continu])

library(psych)

# pairs.panels(df[continu])

df$LogPrice<-log(df$Price)
hist(df$LogPrice)

continu2 <- c("LogPrice", "Basics.Length..ft.","Propulsion.TotalPower..hp.","Specifications.Accommodations.DoubleBerths","Specifications.Accommodations.Cabins","Specifications.Accommodations.Heads","Propulsion.EngineYear","Specifications.Speed.Distance.CruisingSpeed..kn.","Specifications.Speed.Distance.MaxSpeed..kn.","Specifications.Dimensions.LengthOnDeck..ft.","Specifications.Dimensions.LengthAtWaterline..ft.","Specifications.Dimensions.MaxBridgeClearance..ft.")

pairs.panels(df[continu2])

model <- lm (LogPrice~Basics.Length..ft.+Propulsion.TotalPower..hp.+Specifications.Accommodations.DoubleBerths+Specifications.Accommodations.Cabins+Specifications.Accommodations.Heads+Propulsion.EngineYear+Specifications.Speed.Distance.CruisingSpeed..kn.+Specifications.Speed.Distance.MaxSpeed..kn.+Specifications.Dimensions.LengthOnDeck..ft.+Specifications.Dimensions.LengthAtWaterline..ft.+Specifications.Dimensions.MaxBridgeClearance..ft., data = df)
summary(model)
# model <- lm (Price~Basics.Length..ft.+Propulsion.TotalPower..hp.+Specifications.Accommodations.DoubleBerths+Specifications.Accommodations.Cabins+Specifications.Accommodations.Heads+Propulsion.EngineYear+Specifications.Speed.Distance.CruisingSpeed..kn.+Specifications.Speed.Distance.MaxSpeed..kn.+Specifications.Dimensions.LengthOnDeck..ft.+Specifications.Dimensions.LengthAtWaterline..ft.+Specifications.Dimensions.MaxBridgeClearance..ft., data = df)

