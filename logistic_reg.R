# Bazując na trzech pierwszych miesiącach obsługi (cechy behawioralne)
# oraz cechach aplikacyjnych zidentyfikuj sprawy,
# w których nie jest opłacalne wykonywanie działań procesowych.

library("dplyr")
library("data.table")
library("ggplot2")


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

#  Correlation analysis

Variables = c(         "LoanAmount",
                       "TOA",
                       "Principal",
                       "Interest",
                       "Other",
                       "D_ContractDateToImportDate",
                       "DPD",
                       "ExternalAgency",
                       "Bailiff",
                       "ClosedExecution",
                       "PopulationInCity",
                       "Age",
                       "LastPaymentAmount",
                       "M_LastPaymentToImportDate",
                       "GDPPerCapita",
                       "MeanSalary"
)



library(corrplot)
corrplot(cor(cases_target[,.SD,.SDcols = Variables]), order = "hclust", tl.col='black', tl.cex=.95)
detach("package:corrplot", unload = TRUE)

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
  "Bailiff",                   
  #"ClosedExecution",
  #"Land",
  #"PopulationInCity",
  "Age",
  "LastPaymentAmount",         
  "M_LastPaymentToImportDate",
  #"GDPPerCapita",
  "MeanSalary",
  #"S3M",                       
  #"NotProfitable",
  "isMale",                    
  "CashLoan"  
)

NotProfitable <- cases_target$NotProfitable
cases_target <- cases_target[,.SD,.SDcols = Variables]
cases_target$NotProfitable <- NotProfitable

#  Training, test and validation sets

library("caret")
library("pROC")
library("ROCR")
library("Metrics")

set.seed(42)

# indices for 10 test sets and validation set
# note: folds are disjoint sets

n = dim(cases_target)[1]
K = 10
folds <- createFolds(1:n, k = K + 1, list = TRUE, returnTrain = FALSE)
index.valid = folds[[11]]

recap_folds <- data.table()
models <- list()

# 10 potential features

sets <- expand.grid(rep(list(0:1), 10))[-1,]
for (i in 1:10) {
  sets[,i] = sets[,i]*i
}

n <- 2^10-1

for (k in 1:K) {
  
  index.learn = folds[[k]]
  cases_tst = cases_target[index.learn,]
  cases_trn = cases_target[-c(index.learn, index.valid),]

  # Logistic regression of NotProfitable:
  # seeking the best variable subset for k-th fold

  recap <- data.table()
  
  for (i in 1:n) {
    
    # model building
    
    variables <- c()
    
    for (j in 1:10) {  
      if (sets[i,j] > 0) (variables <- c(variables,Variables[sets[i,j]]))
    }
    
    Formula <- as.formula(paste("NotProfitable~",paste(variables,collapse="+")))
    
    model <- glm(formula = Formula, data = cases_trn, family = binomial)
    
    # model efficiency characteristics
    
    {
    noOfVars <- length(variables)
    
    p_pred <- predict(model, newdata = cases_tst, type = "response")
    actual <- cases_tst$NotProfitable
    roc_obj <- roc(actual, p_pred)
    
    optimal_coords <- coords(roc_obj, "best", ret = c("threshold", "npv", "closest.topleft"), best.method = "closest.topleft")
    
    threshold = optimal_coords$threshold
    distance = optimal_coords$closest.topleft
    npv = optimal_coords$npv
    predicted <- ifelse(p_pred >= threshold, 1, 0)
    
    recap = rbind(recap,
                  cbind(data.frame(vars = paste(variables,collapse=","), k = noOfVars),
                        threshold = threshold,
                        distance = distance,
                        acc_test = Metrics::accuracy(actual = actual, predicted = predicted),
                        auc_test = Metrics::auc(actual = actual, predicted = p_pred),
                        recall_test = Metrics::recall(actual = actual, predicted = predicted),
                        npv = npv,
                        precision_test = Metrics::precision(actual = actual, predicted = predicted)
                        ))
    }
    
    print(i)
  }
  
  # best model for each fold 
  
  {
  best_m <- recap[which.min(recap$distance),]
  
  best_vars <- strsplit(best_m$vars, split = ",")[[1]]
  Formula <- as.formula(paste("NotProfitable~",paste(best_vars, collapse = "+")))
  
  models[[k]] <- glm(formula = Formula, data = cases_trn, family = binomial)
  recap_folds <- rbind(recap_folds, best_m)
  }
  
  print(paste(c("Fold:", k), collapse = " "))
}

# best models checked on their test sets

recap_folds <- data.table()

for (k in 1:K) {
  
  index.learn = folds[[k]]
  cases_tst = cases_target[index.learn,]
  
  model <- models[[k]]
  
  variables <- names(model$coefficients)[-1]
  noOfVars <- length(variables)
  
  p_pred <- predict(model, newdata = cases_tst, type = "response")
  actual <- cases_tst$NotProfitable
  roc_obj <- roc(actual, p_pred)
  
  optimal_coords <- coords(roc_obj, "best", ret = c("threshold", "npv", "tnr", "closest.topleft"), best.method = "closest.topleft")
  
  threshold = optimal_coords$threshold
  distance = optimal_coords$closest.topleft
  npv = optimal_coords$npv
  tnr = optimal_coords$tnr
  predicted <- ifelse(p_pred >= threshold, 1, 0)
  
  recap_folds = rbind(recap_folds,
                      cbind(data.frame(vars = paste(variables,collapse=","), k = noOfVars),
                            threshold = threshold,
                            distance = distance,
                            auc_test = Metrics::auc(actual = actual, predicted = p_pred),
                            acc_test = Metrics::accuracy(actual = actual, predicted = predicted),
                            npv_test = npv,
                            tnr_test = tnr,
                            precision_test = Metrics::precision(actual = actual, predicted = predicted),
                            recall_test = Metrics::recall(actual = actual, predicted = predicted)
                      ))
  
}

cols <- colnames(recap_folds)[-(1:2)]
recap_folds[, (cols) := lapply(.SD, round, digits = 4), .SDcols = cols]

# best models checked on valid set
# note: based on 3 months model we want check
# whether the case is at all profitable (all 12 months)

# calculate NotProfitable from 12 months
{
target <- events %>% 
  group_by(CaseId) %>%
  summarize(S12M = sum(PaymentAmount, na.rm = T) - 
              sum(NumberOfCalls * 0.14, na.rm = T) -
              sum(NumberOfLettersSent * 2.5, na.rm = T) -
              sum(NumberOfVisits * 210, na.rm = T)) %>%
  mutate(NotProfitable = ifelse(S12M < 0, 1, 0)) %>%
  select(-S12M)


cases_valid = cases_target[index.valid,]
cases_valid$NotProfitable <- target$NotProfitable[index.valid]
recap_valid <- data.table()

for (k in 1:K) {
  
  model <- models[[k]]
  
  variables <- names(model$coefficients)[-1]
  noOfVars <- length(variables)
  
  p_pred <- predict(model, newdata = cases_valid, type = "response")
  actual <- cases_valid$NotProfitable
  roc_obj <- roc(actual, p_pred)
  
  optimal_coords <- coords(roc_obj, "best", ret = c("threshold", "npv", "tnr", "closest.topleft"), best.method = "closest.topleft")
  
  threshold = optimal_coords$threshold
  distance = optimal_coords$closest.topleft
  npv = optimal_coords$npv
  tnr = optimal_coords$tnr
  predicted <- ifelse(p_pred >= threshold, 1, 0)
  
  recap_valid = rbind(recap_valid,
                      cbind(data.frame(vars = paste(variables,collapse=","), k = noOfVars),
                            threshold = threshold,
                            distance = distance,
                            auc_valid = Metrics::auc(actual = actual, predicted = p_pred),
                            acc_valid = Metrics::accuracy(actual = actual, predicted = predicted),
                            npv_valid = npv,
                            tnr_valid = tnr,
                            precision_valid = Metrics::precision(actual = actual, predicted = predicted),
                            recall_valid = Metrics::recall(actual = actual, predicted = predicted)
                      ))
  
}
}

# round the values obtained
{
cols <- colnames(recap_valid)[-(1:2)]
recap_valid[, (cols) := lapply(.SD, round, digits = 4), .SDcols = cols]
}

# calculate potential profits
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


for (k in 1:K) {
  
  model <- models[[k]]
  p_pred <- predict(model, newdata = cases_valid, type = "response")
  actual <- cases_valid$NotProfitable
  roc_obj <- roc(actual, p_pred)
  
  optimal_coords <- coords(roc_obj, "best", ret = c("threshold"), best.method = "closest.topleft")
  
  threshold = optimal_coords$threshold
  predicted <- ifelse(p_pred >= threshold, 1, 0)
  
  gain_loss <- table(actual, predicted)
  recap_valid$profits[k] <- sum(gain_loss[,1] * c(mean_gain, mean_loss))
}
}

# Data tables for test sets and validation set

View(recap_folds)
View(recap_valid)

save(recap_valid, file = "C:/Users/fiko1/Desktop/UWr/Z2024/Wierzytelności masowe/Projekt/lab06/models/models.RData")

# NotProfitable probability for each variable
# note: we take model with the most regressors

pred_data = data.table()
model <- models[[6]]
var_names <- names(model$coefficients)[-1]

for (var in var_names) {
  temp <- copy(cases_valid[,.SD,.SDcols = var_names])
  
  new_data <- data.frame(var = seq(min(temp[[var]], na.rm = TRUE),
                                   max(temp[[var]], na.rm = TRUE),
                                   length.out = 100))
  colnames(new_data) <- var
  
  other_vars <- setdiff(var_names, var)
  for (other_var in other_vars) {
    new_data[[other_var]] <- median(temp[[other_var]], na.rm = T)
  }
  
  predicted <- predict(model, newdata = new_data, type = "response")
  
  pred_data <- rbind(pred_data, data.table(x = new_data[[var]], p = predicted, name = var))
  
}

ggplot(pred_data, aes(x = x, y = p)) +
  geom_line() +                           
  facet_wrap(~ name, scales = "free_x") +    
  labs(title = "Plots of success probability for each variable", 
       x = "x", 
       y = "Probability") +
  theme_bw() 

# ROC curve

p_pred <- predict(model, newdata = cases_valid, type = "response")
actual <- cases_valid$NotProfitable

pred <- prediction(predictions = p_pred, labels = actual)
perf <- performance(pred, "tpr", "fpr")

roc_obj <- roc(actual, p_pred)
optimal_coords <- coords(roc_obj, "best", ret = c("fpr", "tpr"), best.method = "closest.topleft")


plot(perf, col = "blue", lwd = 2, main = "ROC Curve")
rect(xleft = optimal_coords$fpr, xright = 1, ybottom = 0, ytop = optimal_coords$tpr,
     col = adjustcolor("blue", alpha.f = 0.2))
abline(a = 0, b = 1, lty = 2, lwd = 2, col = "black")
grid(col = "gray", lty = "dotted")
auc <- performance(pred, "auc")@y.values[[1]]
legend("bottomright", legend = sprintf("AUC = %.3f", auc), col = "black", lty = 0, cex = 1.2, bty = "n")
