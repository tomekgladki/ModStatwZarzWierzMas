
setExpertSegment <- function(last_payment_amount) {
  
  glosary <- c("NoPmt", "Pmt", "NoSegment")
  
  last_payment_num <- as.integer(last_payment_amount > 0) + 1L
  last_payment_num[is.na(last_payment_num)] <- 3L
  # 1 - NoPmt, 2 - Pmt, 3 - NoSegment
  
  segment_values <- sapply(last_payment_num, function(i, gl=glosary) {
    return(gl[i])
  })
  
  return(segment_values)
  
}

minMaxScaler <- function(x) {
  
  y <- (max(x) - x) / (max(x) - min(x))
  
  return(y)
  
}


# funkcja została zmodyfikowana względem tego co znajduje się w skrypcie ta
predict.kmeans <- function(model, newdata) {
  
  col_names <- colnames(model$centers)
  newdata <- newdata[,.SD, .SDcols=col_names]
  
  y <- apply(newdata, 1, function(r) {
    which.min(colSums((t(model$centers) - r)^2))
  })
  return(y)
}


predictKMeans <- function(model, newdata, preprocess=NULL) {
  
  if(!is.null(preprocess)) {
    newdata <- predict(preprocess, newdata = newdata)
  }
  
  y <- predict(model, newdata=newdata)
  
  return(y)
  
}
