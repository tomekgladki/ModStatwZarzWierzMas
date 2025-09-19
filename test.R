# Zad. 3 ------------------------------------------------------------------

## przygotowaine danych


# usunięcie spacji z nazw produktu
cases_train[,Product:=gsub("\\s", "", Product)]
cases_valid[,Product:=gsub("\\s", "", Product)]
cases_test[,Product:=gsub("\\s", "", Product)]

reference <- rbindlist(list(cases_train, cases_valid))
reference_targets <- rbindlist(list(targets_train, targets_valid))
test_set <- copy(cases_test)

# Przekształcenie zmiennych nienumerycznych w numeryczne
dummy <- caret::dummyVars(~ Gender + Product, data=reference, levelsOnly=FALSE, fullRank=TRUE)
dummy_vars_reference <- data.table(predict(dummy, newdata=reference))
dummy_vars_test <- data.table(predict(dummy, newdata=test_set))

reference <- cbind(reference, dummy_vars_reference)
test_set <- cbind(test_set, dummy_vars_test)

reference[,`:=`(Product=NULL, Gender=NULL, SegmentModel=NULL)]
test_set[,`:=`(Product=NULL, Gender=NULL, SegmentModel=NULL)]


# uzupełnienie NA i transfomracja zmiennych do tej samej skali

cols_to_preprocess <- setdiff(names(reference), c("CaseId", "SegmentExpert"))

preprocess <- preProcess(reference[,.SD,.SDcols=cols_to_preprocess], method = c("medianImpute", "range"))

reference <- predict(preprocess, newdata=reference)
test_set <- predict(preprocess, newdata=test_set)

summary(reference)
summary(test_set)


## Wyznaczenie ważności zmiennych do modelu knn:

### korelacje z jedną ze zmiennych objaśnianych


abscorr_with_target <- abs(cor(
  x = reference[,.SD,.SDcols=cols_to_preprocess], 
  y = reference_targets[,IfPayment]
))


variables_cor <- data.table(
  Name = rownames(abscorr_with_target),
  Cor = abscorr_with_target[,1]
)

setorder(variables_cor, -Cor)

print(variables_cor)

### Grupy zmiennych wg korelacji. Szukamy grup zmiennych skorelowanych między sobą.

cors_between_variables <- cor(reference[,.SD,.SDcols=cols_to_preprocess])

dist_vars <- dist(cors_between_variables, method="euclidean")

hcl_model <- hclust(dist_vars, method = "centroid")

plot(hcl_model)

k <- 5

variable_groups <- cutree(hcl_model, k=k)

variable_groups <- data.table(
  Name = names(variable_groups),
  Group = variable_groups
)

variable_groups


### Zebranie powyższych danych w całość:

variables <- variables_cor[variable_groups, on="Name"]
setorder(variables, Group, -Cor)
variables


## Model knn w segmentach. Referencją w modelowaniu segmentu NoSegment jest cały zbiór referencyjny.

segments <- reference[,unique(SegmentExpert)]

# bierzemy pierwszą ("najlepsza") zmienną z każdej grupy
selected_variables <- variables[,.(Variables = head(Name, 1)), .(Group) ]
selected_variables <- selected_variables$Variables

reference_model <- reference[reference_targets[,.(CaseId, SumOfPayments)], on = c("CaseId" = "CaseId")]
test_set_model <- test_set[targets_test[,.(CaseId, SumOfPayments)], on = c("CaseId" = "CaseId")]

predictions <- data.table()

for(seg in segments) {
  
  # Test set in segment
  seg_test_caseids <- test_set_model[SegmentExpert==seg, CaseId]
  seg_test <- test_set_model[SegmentExpert==seg, .SD, .SDcols=selected_variables]
  seg_test_targets <- test_set_model[SegmentExpert==seg, SumOfPayments]
  
  # Reference in segment
  if(seg == "NoSegment") {
    
    seg_ref <- reference_model[, .SD, .SDcols=selected_variables]
    seg_ref_targets <- reference_model[, SumOfPayments]
    
  } else {
    
    seg_ref <- reference_model[SegmentExpert==seg, .SD, .SDcols=selected_variables]
    seg_ref_targets <- reference_model[SegmentExpert==seg, SumOfPayments]
    
  }
  
  # Model
  knn_model <- caret::knnregTrain(
    train=seg_ref, 
    test=seg_test, 
    y = seg_ref_targets, 
    k = 5)
  
  predictions_in_segment <- data.table(
    CaseId = seg_test_caseids, 
    Segment=seg, 
    Real = seg_test_targets, 
    Pred = knn_model
  )
  
  predictions <- rbindlist(list(predictions, predictions_in_segment))
  
}

# Errors:
Metrics::mae(predictions$Real, predictions$Pred)
Metrics::rmse(predictions$Real, predictions$Pred)