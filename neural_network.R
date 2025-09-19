library("dplyr")
library("data.table")
library("ggplot2")
library("foreach")
library("doSNOW")
library("e1071")
library("neuralnet")

load("C:/Users/fiko1/Desktop/UWr/Z2024/Wierzytelności masowe/Lab2024-main/KrukUWr2024.RData")

# ZMIENNA CELU
# zakodować binarnie, że sprawa nie jest opłacalna w miesiącach 1-3 
# np. brak przychodu pomimo wykonanych czynności, np. wizyta z klientem;
# podejście per sprawa: 1 - sprawa nieopłacalna; 0 - sprawa opłacalna

target <- events %>% 
  group_by(CaseId) %>%
  filter(Month <= 3) %>%
  summarize(S3M = sum(PaymentAmount, na.rm = T) - 
              sum(NumberOfCalls * 0.14, na.rm = T) -
              sum(NumberOfLettersSent * 2.5, na.rm = T) -
              sum(NumberOfVisits * 210, na.rm = T)) %>%
  mutate(NotProfitable = ifelse(S3M < 0, 1, 0))

cases_target <- cases %>% left_join(target, by = "CaseId")

# Imputation, NAs, outliers

cases_target <- data.table(cases_target)


# Handling missing data

Variables = c(         "LoanAmount",
                       "TOA",
                       "Principal",
                       "Interest",
                       "Other",
                       "D_ContractDateToImportDate",
                       "DPD",
                       "PopulationInCity",
                       "Age",
                       "LastPaymentAmount",
                       "M_LastPaymentToImportDate",
                       "GDPPerCapita",
                       "MeanSalary"
                       #"CreditCard",
                       #"Gender"
                       #"Bailiff",
                       #"ClosedExecution",
                       #"ExternalAgency
)

nullCounts <- lapply(cases_target[,.SD,.SDcols=Variables], function(x) sum(is.na(x)))



# Imputation with avg

variables <- c(        "LoanAmount",
                       "TOA",
                       "Principal",
                       "Interest",
                       "Other",
                       "D_ContractDateToImportDate",
                       "DPD",
                       "PopulationInCity",
                       "Age",
                       "LastPaymentAmount",
                       "M_LastPaymentToImportDate",
                       "GDPPerCapita",
                       "MeanSalary"
)

for (variable in Variables) {      ## variable = 'Age'
  if (eval(parse(text=paste("nullCounts$",variable,sep=""))) > 0) {
    avg <- eval(parse(text=paste("mean(cases_target[,",variable,"],na.rm=TRUE)",sep="")))
    eval(parse(text=paste("cases_target[is.na(",variable,"), ",variable,":=avg]",sep="")))
  }           
}



# Other imputation

cases_target[is.na(Bailiff),Bailiff:= ifelse(runif(cases_target[is.na(Bailiff),.N],0,1)<cases_target[,mean(Bailiff,na.rm=TRUE)],1,0)]
cases_target[is.na(ExternalAgency),ExternalAgency:= ifelse(runif(cases_target[is.na(ExternalAgency),.N],0,1)<cases_target[,mean(ExternalAgency,na.rm=TRUE)],1,0)]
cases_target[is.na(ClosedExecution) & Bailiff==0, ClosedExecution:= 0]
cases_target[is.na(ClosedExecution), ClosedExecution:= ifelse(runif(cases_target[is.na(ClosedExecution),.N],0,1)<cases_target[,mean(ClosedExecution,na.rm=TRUE)],1,0)]

#  Proportion of tail data to be removed from the dataset

Proportion = 0.001

cases_target <- cases_target[LoanAmount<quantile(cases_target[,LoanAmount], probs=1-Proportion),]
cases_target <- cases_target[DPD<quantile(cases_target[,DPD], probs=1-Proportion),]
cases_target <- cases_target[LastPaymentAmount<quantile(cases_target[,LastPaymentAmount], probs=1-Proportion),]

# Categorial to dummy variables

table(cases_target[,Gender])
cases_target[is.na(Gender), Gender:='brak']
cases_target[, isMale := ifelse(Gender=="MALE",1,ifelse(Gender=="FEMALE",0,-1))]
table(cases_target$isMale)


table(cases_target[,Product])
cases_target[, CashLoan := ifelse(Product=="Cash loan",1,0)]
table(cases_target$CashLoan)


# Remove the remaining variables

cases_target[,c('Gender',"Product") := NULL] 

#  Variables selection

Variables = c(         
  #"CaseId",
  #"LoanAmount",
  "TOA",
  #"Principal",
  #"Interest",
  #"Other",
  #"D_ContractDateToImportDate",
  "DPD",                       
  "ExternalAgency",
  #"Bailiff",                   
  #"ClosedExecution",
  #"Land",
  #"PopulationInCity",
  "Age",
  "LastPaymentAmount",         
  "M_LastPaymentToImportDate",
  #"GDPPerCapita",
  #"MeanSalary",
  #"S3M",                       
  #"NotProfitable",
  #"isMale",                    
  "CashLoan"  
)

NotProfitable <- cases_target$NotProfitable
cases_selected <- cases_target[,.SD,.SDcols = Variables]
cases_selected$NotProfitable <- NotProfitable

# Standardize numeric columns to min-max scale

num_cols = c(         
  #"CaseId",
  "LoanAmount",
  "TOA",
  "Principal",
  "Interest",
  "Other",
  #"D_ContractDateToImportDate",
  "DPD",                       
  #"ExternalAgency",
  #"Bailiff",                   
  #"ClosedExecution",
  #"Land",
  "PopulationInCity",
  "Age",
  "LastPaymentAmount",         
  "M_LastPaymentToImportDate",
  #"GDPPerCapita",
  "MeanSalary"
  #"S3M",                       
  #"NotProfitable",
  #"isMale",                    
  #"CashLoan"  
)

num_cols <- generics::intersect(Variables,num_cols)

cases_selected[, (num_cols) := lapply(.SD, function(x) (x - min(x)) / (max(x) - min(x))), .SDcols = num_cols]

# splitting into training and test set

library("caret")

set.seed(42)

# test set and validation set
# note: folds are disjoint sets

n = dim(cases_selected)[1]
K = 2
folds <- createFolds(1:n, k = K, list = TRUE, returnTrain = FALSE)
index.learn = folds[[1]]
cases_tst = cases_selected[index.learn,]
cases_trn = cases_selected[-index.learn,]

models <- list()
Formula <- as.formula(paste("NotProfitable ~",paste(Variables,collapse="+")))

# first nn

system.time({ # 1min 4s
  set.seed(42)
  models[[1]] <- neuralnet(Formula, data=cases_trn, hidden=c(1), 
                           stepmax=2e6, threshold = 0.1,
                           linear.output = FALSE, 
                           lifesign = "full")
})

# second nn

system.time({ # 3min 10s
  set.seed(42)
  models[[2]] <- neuralnet(Formula, data=cases_trn, hidden=c(2,1), 
                           stepmax=2e6, threshold = 0.1,
                           linear.output = FALSE, 
                           lifesign = "full")
})

# third nn

system.time({ # 24min 43s
  set.seed(42)
  models[[3]] <- neuralnet(Formula, data=cases_trn, hidden=c(3,2,1), 
                           stepmax=2e6, threshold = 0.1,
                           linear.output = FALSE, 
                           lifesign = "full")
})

# fourth nn

system.time({
  set.seed(42)
  models[[4]] <- neuralnet(Formula, data=cases_trn, hidden=c(4,3,2,1), 
                           stepmax=1e6, threshold = 0.05,
                           linear.output = FALSE, 
                           lifesign = "full")
})


# wizualizacja sieci
plot(models[[1]])

# profits

{
gains_df <- events %>% 
  group_by(CaseId) %>%
  summarize(S12M = sum(PaymentAmount, na.rm = T) - 
              sum(NumberOfCalls * 0.14, na.rm = T) -
              sum(NumberOfLettersSent * 2.5, na.rm = T) -
              sum(NumberOfVisits * 210, na.rm = T))

mean_gain <- gains_df %>%
  filter(S12M > 0) %>%
  summarize(mean_gain = mean(S12M))
mean_gain <- unname(unlist(as.vector(mean_gain)))

mean_loss <- gains_df %>%
  filter(S12M < 0) %>%
  summarize(mean_loss = mean(S12M))
mean_loss <- unname(unlist(as.vector(mean_loss)))
}

# predykcja
library("caret")
library("pROC")
library("ROCR")
library("Metrics")

recap = data.table()

for (k in 1:4) {
  nnetFitClPred <- compute(models[[k]], cases_tst[, .SD, .SDcols=Variables])
  p_pred <- nnetFitClPred$net.result[, 1]
  actual <- cases_tst$NotProfitable
  roc_obj <- roc(actual, p_pred)
  
  optimal_coords <- coords(roc_obj, "best", ret = c("threshold", "npv", "tnr", "closest.topleft"), best.method = "closest.topleft")
  
  threshold = optimal_coords$threshold
  distance = optimal_coords$closest.topleft
  npv = optimal_coords$npv
  tnr = optimal_coords$tnr
  predicted <- ifelse(p_pred >= threshold, 1, 0)
  gain_loss <- table(actual, predicted)
  
  recap = rbind(recap,
                cbind(data.frame(hidden = paste(as.character(k:1),collapse = ", ")),
                      threshold = threshold,
                      distance = distance,
                      auc_test = Metrics::auc(actual = actual, predicted = p_pred),
                      acc_test = Metrics::accuracy(actual = actual, predicted = predicted),
                      npv_test = npv,
                      tnr_test = tnr,
                      precision_test = Metrics::precision(actual = actual, predicted = predicted),
                      recall_test = Metrics::recall(actual = actual, predicted = predicted),
                      profits = sum(gain_loss[,1] * c(mean_gain, mean_loss))
                ))
}

# round values

cols <- colnames(recap)[-1]
recap[, (cols) := lapply(.SD, round, digits = 4), .SDcols = cols]

save(recap, file = "C:/Users/fiko1/Desktop/UWr/Z2024/Wierzytelności masowe/Projekt/lab06/models/models_nn.RData")


