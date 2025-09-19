required_packages <- c(
  "tidyverse", "caret", "randomForest", 
  "nnet", "mgcv", "pROC", "ggplot2"
)

to_install <- required_packages[!required_packages %in% installed.packages()]

if(length(to_install) > 0) {
  install.packages(to_install)
  cat("Zainstalowano pakiety:", paste(to_install, collapse = ", "), "\n")
} else {
  cat("Wszystkie pakiety są już zainstalowane.\n")
}

lapply(required_packages, library, character.only = TRUE)