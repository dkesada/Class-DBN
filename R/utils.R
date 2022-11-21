# Obsolete
read_fjm_data <- function(path){
  dt <- fread(file = path, encoding = "UTF-8", check.names = T) # Load the data using UTF-8 and substituting variable-illegal characters with dots
  names(dt) <- sapply(names(dt), iconv, from = "", to = "ASCII//TRANSLIT", USE.NAMES = F) # Remove special characters, like accents
  names(dt) <- sapply(names(dt), gsub, pattern = "[^0-9A-Za-z///' ]", replacement = "_", ignore.case = T, USE.NAMES = F) # Replace dots with underscores
  names(dt) <- sapply(names(dt), gsub, pattern = "_$", replacement = "", ignore.case = T, USE.NAMES = F) # Remove ending underscores
  names(dt) <- sapply(names(dt), gsub, pattern = "__", replacement = "_", ignore.case = T, USE.NAMES = F) # Remove double underscores
  
  return(dt)
}

# Obsolete
choose_date <- function(fecha_uci, fecha_prueba){
  res <- fecha_uci
  if(is.na(res[1]))
    res <- fecha_prueba
  
  return(res)
}

factorize_character <- function(dt){
  dt[, lapply(.SD, function(x){
    if(is.character(x))
      x <- as.numeric(factor(x))-1
    return(x)
  })]
}

factorize_dt <- function(dt){
  dt[, lapply(.SD, function(x){
    x <- factor(x)
    return(x)
  })]
}

conf_matrix <- function(orig, preds){
  tp <- sum(orig == 1 & preds == 1)
  fp <- sum(orig == 0 & preds == 1)
  tn <- sum(orig == 0 & preds == 0)
  fn <- sum(orig == 1 & preds == 0)
  
  cat(sprintf("Confusion matrix:
                  Orig
                1      0\n
      Pred  1   %d     %d\n
            0   %d     %d\n", tp, fp, fn, tn))
}

evalerror <- function(preds, dtrain){
  labels <- getinfo(dtrain, "label")
  #err <- as.numeric(sum(labels != (preds > 0)))/length(labels)
  err <- as.numeric(sum(labels == 1 & preds >= 0.5) / sum(labels == 1))
  return(list(metric = "tp_rate", value = err))
}

fscore <- function(preds, dtrain){
  labels <- getinfo(dtrain, "label")
  tp <- sum(labels == 1 & preds >= 0.5)
  fp <- sum(labels == 0 & preds >= 0.5)
  fn <- sum(labels == 1 & preds < 0.5)
  err <- as.numeric(tp / (tp + 0.5 * (fp + fn)))
  
  return(list(metric = "fscore", value = err))
}

f1score <- function(orig, preds){
  tp <- sum(orig == 1 & preds == 1)
  fp <- sum(orig == 0 & preds == 1)
  fn <- sum(orig == 1 & preds == 0)
  precision <- tp / (tp + fp)
  recall <- tp / (tp + fn)
  res <- 2 * precision * recall / (precision + recall)
  
  return(res)
}

# Three parameters to optimize: the weight of critical cases, the max.depth and the number of rounds
eval_xgboost <- function(params, dt_train, dt_train_red, dt_test, dt_test_red, eval_metric){
  print(params)
  params[2] <- round(params[2])
  params[3] <- round(params[3])
  weights <- rep(1, dim(dt_train)[1])
  weights[dt_train$Critico == 1] <- params[1]
  bstSparse <- xgboost(data = as.matrix(dt_train_red), label = dt_train$Critico, weight = weights, 
                       eval_metric = eval_metric, max.depth = params[2], nrounds = params[3],
                       eta = 1, nthread = 2, objective = "binary:logistic", verbose = 0)
  preds <- as.numeric(predict(bstSparse, as.matrix(dt_test_red)) > 0.5)
  acc <- mean(dt_test$Critico != preds) # Mean error
  #conf_matrix(dt_test$Critico, preds)
  
  return(acc)
}

# Create a combination of m disjoint sets of k elements each from the original set n
# It will only create 1 combination, not all the existing ones like a leave-3-out cross-validation
cross_sets <- function(n, k){
  res <- vector(mode = "list", length = floor(length(n) / k))
  
  for(i in 1:length(res)){
    ids <- sample.int(length(n), size = k, replace = F)
    res[[i]] <- n[ids]
    n <- n[-ids]
  }
  
  return(res)
}

# Shuffle the instances in a dt to a random order
shuffle_dt <- function(dt){
  order <- sample(dim(dt)[1])
  
  return(dt[order])
}

# Perform SMOTE over a dataset with the provided parameters
# This function ensures that the objective variable is a factor, 
# and returns it to numeric in case it was provided that way
smote_dt <- function(dt, obj_var, perc_over, perc_under, shuffle = T){
  num <- F
  if(is.numeric(dt[, get(obj_var)])){
    num <- T
    dt <- copy(dt)
    dt[, eval(obj_var) := as.factor(get(obj_var))]
  }
  
  form <- as.formula(paste0(obj_var, " ~ ." ))
  dt <- DMwR::SMOTE(form, dt, perc.over = perc_over, perc.under = perc_under)
  
  if(num)
    dt[, eval(obj_var) := as.numeric(as.character(get(obj_var)))]
  
  if(shuffle)
    dt <- shuffle_dt(dt)
  
  return(dt)
}