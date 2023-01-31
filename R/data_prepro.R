library(data.table)
library(jsonlite)
library(xgboost)
library(dbnR)
library(pso)
library(DEoptim)

#################################################################
# Tests
#################################################################

#' @export
run_all <- function(){
  print("Executing the xgb model:")
  main_cv(main_xgb, horizon = 10, suffix = "xgb")
  print("Executing the svm model:")
  main_cv(main_svm, horizon = 10, suffix = "svm")
  print("Executing the nn model:")
  main_cv(main_nn, horizon = 10, suffix = "nn")
  print("Executing the naive Bayes model:")
  main_cv(main_bncl_single, horizon = 10, suffix = "nb", cl_params = c(0,0,0,0))
  print("Executing the TAN CL model:")
  main_cv(main_bncl_single, horizon = 10, suffix = "cl", cl_params = c(1,0,0,0))
  print("Executing the TAN HC model:")
  main_cv(main_bncl_single, horizon = 10, suffix = "tanhc", cl_params = c(2,4,0.5,0.5))
  print("Executing the TAN HCSP model:")
  main_cv(main_bncl_single, horizon = 10, suffix = "tanhcsp", cl_params = c(3,4,0.5,0.5))
}

#' @export
main_cv <- function(foo, k = 100, horizon = 10, suffix = "nb", seed = 42, ...){ # k = 300
  sink(paste0("./output/cv_res_", Sys.Date(), "_", horizon, "_", suffix, ".txt"))
  
  set.seed(seed)
  id_var <- "REGISTRO"
  dt <- fread("./data/FJD_6.csv")
  dt[, Crit := as.numeric(EXITUS == "S" | UCI == "S")]
  #dt[, Crit := as.numeric(EXITUS == "S")]
  dt <- factorize_character(dt)
  
  # Only crit at the last instance
  # regs <- dt[Crit == 1, unique(REGISTRO)]
  # dt[, id := .I]
  # for(i in regs){
  #   ids <- dt[REGISTRO == i, id]
  #   dt[REGISTRO == i, Crit := 0]
  #   dt[max(ids), Crit := 1]
  # }
  # dt[, id := NULL]
  
  cv_sets <- cross_sets(dt[, unique(get(id_var))], k)
  
  crono <- unique(dt[order(Fecha_emision), REGISTRO])
  cv_sets[[1]] <- crono[900:993]
  cv_sets <- cv_sets[1]
  
  cat("Generated folds:\n")
  print(cv_sets)
  
  res_matrix <- matrix(nrow = horizon+1, ncol = length(cv_sets), 0) # Each cv up to the desired horizon 
  res_matrix <- matrix(nrow = horizon+1, ncol = 4, 0) # Acc, f_scr, train_t, exec_t up to horizon 20 
  rm(dt)
  
  for(i in 1:length(cv_sets)){
    cat(paste0("Currently on the fold number ", i, " out of ", length(cv_sets), "\n"))
    res <- foo(cv_sets[[i]], horizon, ...)
    cat(paste0(c("Results of the fold:\n")))
    print(res)
    
    res_matrix <- res_matrix + res
  }
  
  res_matrix <- res_matrix / length(cv_sets)
  
  cat("Final results matrix:\n")
  print(res_matrix)
  
  cat("Final mean error by horizon:\n")
  cat(paste0("Average training time of the model: ", res_matrix[1,4], " seconds.\n"))
  cat("Final results by horizon:\n")
  for(i in 1:nrow(res_matrix)){
    cat(paste0("Error in horizon ", i-1, ": ", res_matrix[i, 1], "\n"))
    cat(paste0("F1score in horizon ", i-1, ": ", res_matrix[i, 2], "\n"))
    cat(paste0("Execution time for horizon ", i-1, ": ", res_matrix[i, 3], "\n"))
  }
  
  sink()
}

#' Main body with R6 encapsulation and testing of horizons for the XGB
#' 
#' Starting point of the whole process
#' @import data.table jsonlite xgboost dbnR pso DEoptim
#' @export
main_xgb <- function(cv_sets, horizon){
  res <- matrix(nrow = horizon +1, ncol = 4)
  size <- 2
  method <- "dmmhc"
  id_var <- "REGISTRO"
  dt <- fread("./data/FJD_6.csv")
  dt[, Crit := as.numeric(EXITUS == "S" | UCI == "S")]
  #dt[, Crit := as.numeric(EXITUS == "S")]
  dt <- factorize_character(dt)
  var_sets <- read_json("./data/var_sets.json", simplifyVector = T)
  var_sets$cte[2] <- "SEXO"
  var_sets$analit_full <- var_sets$analit_full[var_sets$analit_full %in% names(dt)]
  var_sets$analit_red <- var_sets$analit_red[var_sets$analit_red %in% names(dt)]
  dbn_obj_vars <- c(var_sets$vitals, var_sets$analit_red)
  xgb_obj_var <- "Crit"
  dt_red <- dt[, .SD, .SDcols = c(var_sets$cte, var_sets$vitals, var_sets$analit_red, id_var, xgb_obj_var)]  # Can also try analit_full
  # dt_red <- dt[, .SD, .SDcols = c(var_sets$cte, var_sets$analit_red, id_var, xgb_obj_var)]
  # dbn_obj_vars <- c(var_sets$analit_red)
  
  dt_train <- dt_red[!(get(id_var) %in% eval(cv_sets))]
  dt_test <- dt_red[get(id_var) %in% eval(cv_sets)]
  
  # dt_train <- dt_train[Edad > 50]
  # dt_test <- dt_test[Edad > 50]
  
  model <- XGDBN::XGDBN$new(itermax = 100)
  train_t <- Sys.time()
  model$fit_model(dt_train, id_var, size, method, xgb_obj_var, 
                  dbn_obj_vars, seed = 42, optim = T)
  train_t <- Sys.time() - train_t
  res[,4] <- train_t
  
  model$print_params()
  
  cat("Baseline results: \n")
  exec_t <- Sys.time()
  preds <- model$predict_cl(dt_test)
  exec_t <- Sys.time() - exec_t
  res[1,1] <- mean(dt_test[, get(xgb_obj_var)] == preds)
  res[1,2] <- f1score(dt_test[, get(xgb_obj_var)], preds)
  res[1,3] <- exec_t
  
  for(i in 1:horizon){
    cat(sprintf("Horizon %d results:\n", i))
    exec_t <- Sys.time()
    preds <- model$predict(dt_test, horizon = i)
    exec_t <- Sys.time() - exec_t
    res[i+1,1] <- mean(dt_test[, get(xgb_obj_var)] == preds)
    res[i+1,2] <- f1score(dt_test[, get(xgb_obj_var)], preds)
    res[i+1,3] <- exec_t
  }
  
  return(res)
}

#' Main body with R6 encapsulation and testing of horizons for the SVM
#' 
#' Starting point of the whole process
#' @import data.table jsonlite e1071 dbnR pso DEoptim
#' @export
main_svm <- function(cv_sets, horizon){
  res <- matrix(nrow = horizon +1, ncol = 4)
  size <- 2
  method <- "dmmhc"
  id_var <- "REGISTRO"
  dt <- fread("./data/FJD_6.csv")
  dt[, Crit := as.numeric(EXITUS == "S" | UCI == "S")]
  #dt[, Crit := as.numeric(EXITUS == "S")]
  dt <- factorize_character(dt)
  var_sets <- read_json("./data/var_sets.json", simplifyVector = T)
  var_sets$cte[2] <- "SEXO"
  var_sets$analit_full <- var_sets$analit_full[var_sets$analit_full %in% names(dt)]
  var_sets$analit_red <- var_sets$analit_red[var_sets$analit_red %in% names(dt)]
  dbn_obj_vars <- c(var_sets$vitals, var_sets$analit_red)
  svm_obj_var <- "Crit"
  dt_red <- dt[, .SD, .SDcols = c(var_sets$cte, var_sets$vitals, var_sets$analit_red, id_var, svm_obj_var)]  # Can also try analit_full
  
  dt_train <- dt_red[!(get(id_var) %in% eval(cv_sets))]
  dt_test <- dt_red[get(id_var) %in% eval(cv_sets)]
  
  model <- XGDBN::SVDBN$new(itermax = 100)
  train_t <- Sys.time()
  model$fit_model(dt_train, id_var, size, method, svm_obj_var, 
                  dbn_obj_vars, seed = 42, optim = T)
  train_t <- Sys.time() - train_t
  res[,4] <- train_t
  
  model$print_params()
  
  cat("Baseline results: \n")
  exec_t <- Sys.time()
  preds <- model$predict_cl(dt_test)
  exec_t <- Sys.time() - exec_t
  res[1,1] <- mean(dt_test[, get(svm_obj_var)] == preds)
  res[1,2] <- f1score(dt_test[, get(svm_obj_var)], preds)
  res[1,3] <- exec_t
  
  for(i in 1:horizon){
    cat(sprintf("Horizon %d results:\n", i))
    exec_t <- Sys.time()
    preds <- model$predict(dt_test, horizon = i)
    exec_t <- Sys.time() - exec_t
    res[i+1,1] <- mean(dt_test[, get(svm_obj_var)] == preds)
    res[i+1,2] <- f1score(dt_test[, get(svm_obj_var)], preds)
    res[i+1,3] <- exec_t
  }
  
  return(res)
}

#' Main body with R6 encapsulation and testing of horizons for the NN
#' 
#' Starting point of the whole process
#' @import data.table jsonlite keras dbnR DEoptim
#' @export
main_nn <- function(cv_sets, horizon){
  tensorflow::set_random_seed(42)
  res <- matrix(nrow = horizon +1, ncol = 4)
  size <- 2
  method <- "dmmhc"
  id_var <- "REGISTRO"
  dt <- fread("./data/FJD_6.csv")
  dt[, Crit := as.numeric(EXITUS == "S" | UCI == "S")]
  # dt[, Crit := as.numeric(EXITUS == "S")]
  dt <- factorize_character(dt)
  var_sets <- read_json("./data/var_sets.json", simplifyVector = T)
  var_sets$cte[2] <- "SEXO"
  var_sets$analit_full <- var_sets$analit_full[var_sets$analit_full %in% names(dt)]
  var_sets$analit_red <- var_sets$analit_red[var_sets$analit_red %in% names(dt)]
  dbn_obj_vars <- c(var_sets$vitals, var_sets$analit_red)
  nn_obj_var <- "Crit"
  dt_red <- dt[, .SD, .SDcols = c(var_sets$cte, var_sets$vitals, var_sets$analit_red, id_var, nn_obj_var)]  # Can also try analit_full
  
  dt_train <- dt_red[!(get(id_var) %in% eval(cv_sets))]
  dt_test <- dt_red[get(id_var) %in% eval(cv_sets)]
  
  model <- XGDBN::NNDBN$new(itermax = 2)
  train_t <- Sys.time()
  model$fit_model(dt_train, id_var, size, method, nn_obj_var, 
                  dbn_obj_vars, seed = 42, optim = F)
  train_t <- Sys.time() - train_t
  res[,4] <- train_t
  
  model$print_params()
  
  cat("Baseline results: \n")
  exec_t <- Sys.time()
  preds <- model$predict_cl(dt_test)
  exec_t <- Sys.time() - exec_t
  res[1,1] <- mean(dt_test[, get(nn_obj_var)] == preds)
  res[1,2] <- f1score(dt_test[, get(nn_obj_var)], preds)
  res[1,3] <- exec_t
  
  for(i in 1:horizon){
    cat(sprintf("Horizon %d results:\n", i))
    exec_t <- Sys.time()
    preds <- model$predict(dt_test, horizon = i)
    exec_t <- Sys.time() - exec_t
    res[i+1,1] <- mean(dt_test[, get(nn_obj_var)] == preds)
    res[i+1,2] <- f1score(dt_test[, get(nn_obj_var)], preds)
    res[i+1,3] <- exec_t
  }
  
  return(res)
}

#' Main body trying out model tree dynamic Bayesian networks
#' 
#' Starting point of the whole process
#' @import mtDBN
#' @export
main_mtdbn <- function(){
  stop("Not implemented yet.")
}

#' Main body trying bayesian classifiers
#' 
#' Starting point of the whole process
#' @import bnclassify arules
#' @export
main_bncl_single <- function(cv_sets, horizon, cl_params = c(0, 0, 0, 0)){
  res <- matrix(nrow = horizon +1, ncol = 4)
  size <- 2
  method <- "dmmhc"
  id_var <- "REGISTRO"
  dt <- fread("./data/FJD_6.csv")
  dt[, Crit := as.numeric(EXITUS == "S" | UCI == "S")]
  #dt[, Crit := as.numeric(EXITUS == "S")]
  dt <- factorize_character(dt)
  var_sets <- read_json("./data/var_sets.json", simplifyVector = T)
  var_sets$cte[2] <- "SEXO"
  var_sets$analit_full <- var_sets$analit_full[var_sets$analit_full %in% names(dt)]
  var_sets$analit_red <- var_sets$analit_red[var_sets$analit_red %in% names(dt)]
  dbn_obj_vars <- c(var_sets$vitals, var_sets$analit_red)
  cl_obj_var <- "Crit"
  dt_red <- dt[, .SD, .SDcols = c(var_sets$cte, var_sets$vitals, var_sets$analit_red, id_var, cl_obj_var)]  # Can also try analit_full
  
  dt_train <- dt_red[!(get(id_var) %in% eval(cv_sets))]
  dt_test <- dt_red[get(id_var) %in% eval(cv_sets)]
  
  model <- XGDBN::BNCDBN$new(itermax = 100)
  train_t <- Sys.time()
  model$fit_model(dt_train, id_var, size, method, cl_obj_var, 
                  dbn_obj_vars, seed = 42, optim = T, cl_params = cl_params)
  train_t <- Sys.time() - train_t
  res[,4] <- train_t
  
  model$print_params()
  
  cat("Baseline results: \n")
  exec_t <- Sys.time()
  preds <- model$predict_cl(dt_test)
  exec_t <- Sys.time() - exec_t
  res[1,1] <- mean(dt_test[, get(cl_obj_var)] == preds)
  res[1,2] <- f1score(dt_test[, get(cl_obj_var)], preds)
  res[1,3] <- exec_t

  for(i in 1:horizon){
    cat(sprintf("Horizon %d results:\n", i))
    exec_t <- Sys.time()
    preds <- model$predict(dt_test, horizon = i)
    exec_t <- Sys.time() - exec_t
    res[i+1,1] <- mean(dt_test[, get(cl_obj_var)] == preds)
    res[i+1,2] <- f1score(dt_test[, get(cl_obj_var)], preds)
    res[i+1,3] <- exec_t
  }
  
  return(res)
}

#' Main body trying bayesian classifiers
#' 
#' Starting point of the whole process
#' @import bnclassify arules
#' @export
main_bncl_full <- function(){
  sink(paste0("./output/bncl_", Sys.Date(), ".txt"))
  size <- 2
  method <- "dmmhc"
  id_var <- "REGISTRO"
  dt <- fread("./data/FJD_6.csv")
  dt[, Crit := as.numeric(EXITUS == "S" | UCI == "S")]
  #dt[, Crit := as.numeric(EXITUS == "S")]
  dt <- factorize_character(dt)
  var_sets <- read_json("./data/var_sets.json", simplifyVector = T)
  var_sets$cte[2] <- "SEXO"
  var_sets$analit_full <- var_sets$analit_full[var_sets$analit_full %in% names(dt)]
  var_sets$analit_red <- var_sets$analit_red[var_sets$analit_red %in% names(dt)]
  dbn_obj_vars <- c(var_sets$vitals, var_sets$analit_red)
  cl_obj_var <- "Crit"
  dt_red <- dt[, .SD, .SDcols = c(var_sets$cte, var_sets$vitals, var_sets$analit_red, id_var, cl_obj_var)]  # Can also try analit_full
  
  dt_train <- dt_red[1:2925]
  dt_test <- dt_red[2926:3656]
  
  print("--------------------")
  print("Naive Bayes approach")
  print("--------------------")
  
  model <- XGDBN::BNCDBN$new(itermax = 100)
  model$fit_model(dt_train, id_var, size, method, cl_obj_var, 
                  dbn_obj_vars, seed = 42, optim = T, cl_params = c(0, 0, 0, 0))
  
  model$print_params()
  
  print("Baseline results: ")
  print(model$predict_cl(dt_test))
  
  for(i in 1:20){
    print(sprintf("Horizon %d results:", i))
    model$predict(dt_test, horizon = i)
  }
  
  print("--------------------")
  print("TAN cl approach")
  print("--------------------")
  
  model <- XGDBN::BNCDBN$new(itermax = 100)
  model$fit_model(dt_train, id_var, size, method, cl_obj_var, 
                  dbn_obj_vars, seed = 42, optim = T, cl_params = c(1, 0, 0, 0))
  
  model$print_params()
  
  print("Baseline results: ")
  print(model$predict_cl(dt_test))
  
  for(i in 1:20){
    print(sprintf("Horizon %d results:", i))
    model$predict(dt_test, horizon = i)
  }
  
  print("--------------------")
  print("TAN HC approach")
  print("--------------------")
  
  model <- XGDBN::BNCDBN$new(itermax = 100)
  model$fit_model(dt_train, id_var, size, method, cl_obj_var, 
                  dbn_obj_vars, seed = 42, optim = T, cl_params = c(2, 4, 0.5, 0.5))
  
  model$print_params()
  
  print("Baseline results: ")
  print(model$predict_cl(dt_test))
  
  for(i in 1:20){
    print(sprintf("Horizon %d results:", i))
    model$predict(dt_test, horizon = i)
  }
  
  print("--------------------")
  print("TAN HCSP approach")
  print("--------------------")
  
  model <- XGDBN::BNCDBN$new(itermax = 100)
  model$fit_model(dt_train, id_var, size, method, cl_obj_var, 
                  dbn_obj_vars, seed = 42, optim = T, cl_params = c(3, 4, 0.5, 0.5))
  
  model$print_params()
  
  print("Baseline results: ")
  print(model$predict_cl(dt_test))
  
  for(i in 1:20){
    print(sprintf("Horizon %d results:", i))
    model$predict(dt_test, horizon = i)
  }
  
  sink()
}
