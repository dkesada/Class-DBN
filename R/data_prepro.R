library(data.table)
library(jsonlite)
library(xgboost)
library(dbnR)
library(pso)
library(DEoptim)

#################################################################
# Tests
#################################################################

# - Get the hybrid model inside a portable R6 class
# - Get a decent pipeline for the experiment, not all in a single script

#' Main body with R6 encapsulation
#' 
#' Starting point of the whole process
#' @import data.table jsonlite xgboost dbnR pso DEoptim
#' @export
main_r6 <- function(){
  size <- 2
  method <- "dmmhc"
  id_var <- "REGISTRO"
  dt <- fread("./data/FJD_6.csv")
  dt[, Crit := as.numeric(EXITUS == "S" | UCI == "S")]
  dt <- factorize_character(dt)
  var_sets <- read_json("./data/var_sets.json", simplifyVector = T)
  var_sets$cte[2] <- "SEXO"
  var_sets$analit_full <- var_sets$analit_full[var_sets$analit_full %in% names(dt)]
  var_sets$analit_red <- var_sets$analit_red[var_sets$analit_red %in% names(dt)]
  dbn_obj_vars <- c(var_sets$vitals, var_sets$analit_red)
  xgb_obj_var <- "Crit"
  dt_red <- dt[, .SD, .SDcols = c(var_sets$cte, var_sets$vitals, var_sets$analit_red, id_var, xgb_obj_var)]  # Can also try analit_full
  
  dt_train <- dt_red[1:2925]
  dt_test <- dt_red[2926:3656]
  
  model <- XGDBN::XGDBN$new(itermax = 1)
  model$fit_model(dt_train, id_var, size, method, xgb_obj_var, 
                  dbn_obj_vars, seed = 42, optim = T)
  
  model$predict(dt_test)
  model$predict_xgb(dt_test)
}

#' Main body with R6 encapsulation and testing of horizons
#' 
#' Starting point of the whole process
#' @import data.table jsonlite xgboost dbnR pso DEoptim
#' @export
main_hor <- function(){
  size <- 2
  method <- "dmmhc"
  id_var <- "REGISTRO"
  dt <- fread("./data/FJD_6.csv")
  dt[, Crit := as.numeric(EXITUS == "S" | UCI == "S")]
  dt <- factorize_character(dt)
  var_sets <- read_json("./data/var_sets.json", simplifyVector = T)
  var_sets$cte[2] <- "SEXO"
  var_sets$analit_full <- var_sets$analit_full[var_sets$analit_full %in% names(dt)]
  var_sets$analit_red <- var_sets$analit_red[var_sets$analit_red %in% names(dt)]
  dbn_obj_vars <- c(var_sets$vitals, var_sets$analit_red)
  xgb_obj_var <- "Crit"
  dt_red <- dt[, .SD, .SDcols = c(var_sets$cte, var_sets$vitals, var_sets$analit_red, id_var, xgb_obj_var)]  # Can also try analit_full
  
  dt_train <- dt_red[1:2925]
  dt_test <- dt_red[2926:3656]
  
  model <- XGDBN::XGDBN$new(itermax = 100)
  model$fit_model(dt_train, id_var, size, method, xgb_obj_var, 
                  dbn_obj_vars, seed = 42, optim = T)
  
  print("Baseline results: ")
  model$predict_xgb(dt_test)
  
  for(i in 1:20){
    print(sprintf("Horizon %d results:", i))
    model$predict(dt_test, horizon = i)
  }
  
}

#' Main body trying out model tree dynamic Bayesian networks
#' 
#' Starting point of the whole process
#' @import mtDBN
#' @export
main_mtdbn <- function(){
  size <- 2
  method <- "dmmhc"
  id_var <- "REGISTRO"
  dt <- fread("./data/FJD_6.csv")
  dt[, Crit := as.numeric(EXITUS == "S" | UCI == "S")]
  dt <- factorize_character(dt)
  var_sets <- read_json("./data/var_sets.json", simplifyVector = T)
  var_sets$cte[2] <- "SEXO"
  var_sets$analit_full <- var_sets$analit_full[var_sets$analit_full %in% names(dt)]
  var_sets$analit_red <- var_sets$analit_red[var_sets$analit_red %in% names(dt)]
  dbn_obj_vars <- c(var_sets$vitals, var_sets$analit_red)
  xgb_obj_var <- "Crit"
  dt_red <- dt[, .SD, .SDcols = c(var_sets$cte, var_sets$vitals, var_sets$analit_red, id_var, xgb_obj_var)]  # Can also try analit_full
  
  dt_train <- dt_red[1:2925]
  dt_test <- dt_red[2926:3656]
  
  model <- XGDBN::XGDBN$new(itermax = 100)
  model$fit_model(dt_train, id_var, size, method, xgb_obj_var, 
                  dbn_obj_vars, seed = 42, optim = T)
  
  print("Baseline results: ")
  model$predict_xgb(dt_test)
  
  for(i in 1:20){
    print(sprintf("Horizon %d results:", i))
    model$predict(dt_test, horizon = i)
  }
  
}

#' Main body trying bayesian classifiers
#' 
#' Starting point of the whole process
#' @import bnclassify infotheo
#' @export
main_bncl <- function(){
  size <- 2
  method <- "dmmhc"
  id_var <- "REGISTRO"
  dt <- fread("./data/FJD_6.csv")
  dt[, Crit := as.numeric(EXITUS == "S" | UCI == "S")]
  dt <- factorize_character(dt)
  var_sets <- read_json("./data/var_sets.json", simplifyVector = T)
  var_sets$cte[2] <- "SEXO"
  var_sets$analit_full <- var_sets$analit_full[var_sets$analit_full %in% names(dt)]
  var_sets$analit_red <- var_sets$analit_red[var_sets$analit_red %in% names(dt)]
  dbn_obj_vars <- c(var_sets$vitals, var_sets$analit_red)
  xgb_obj_var <- "Crit"
  dt_red <- dt[, .SD, .SDcols = c(var_sets$cte, var_sets$vitals, var_sets$analit_red, id_var, xgb_obj_var)]  # Can also try analit_full
  
  #discretize
  dt_disc <- copy(dt_red)
  dbn_vars_disc <-  discretize(dt_disc[, .SD, .SDcols = dbn_obj_vars])
  dt_disc[, eval(dbn_obj_vars) := dbn_vars_disc]
  dt_disc[, IMC := round(IMC)]
  dt_disc <- factorize_dt(dt_disc)
  
  
  dt_train <- dt_disc[1:2925]
  dt_test <- dt_disc[2926:3656]
  
  browser()
  
  
  model <- bnclassify::nb(class = xgb_obj_var, dataset = dt_train)
  fit <- lp(model, dt_train, 0.01) # Optimize parameters and model TAN/NB
  
  print("Baseline results: ")
  preds <- predict(fit, dt_test)
  conf_matrix(dt_test$Crit, preds)
  cv(fit, dt_test, k=10)
  #model$predict_xgb(dt_test)
  lapply(dbn_obj_vars, function(x){
    arules::discretize(dt_red[, .SD, .SDcols = x], method = "frequency", breaks = 15, onlycuts=T)
  })
  
  for(i in 1:20){
    print(sprintf("Horizon %d results:", i))
    model$predict(dt_test, horizon = i)
  }
  
}

#' Main body of the experiment
#' 
#' For now, a long script with the first experiment
#' @import data.table jsonlite xgboost dbnR pso DEoptim
#' @export
main <- function(){
  size <- 2
  id_var <- "REGISTRO"
  dt <- fread("./data/FJD_6.csv")
  dt[, Critico := as.numeric(EXITUS == "S" | UCI == "S")]
  dt_old <- dt[Ola != 6]
  dt_6 <- dt[Ola == 6]
  dt_old <- factorize_character(dt_old)
  dt_6 <- factorize_character(dt_6)
  var_sets <- read_json("./data/var_sets.json", simplifyVector = T)
  var_sets$cte[2] <- "SEXO"
  var_sets$analit_full <- var_sets$analit_full[var_sets$analit_full %in% names(dt)]
  var_sets$analit_red <- var_sets$analit_red[var_sets$analit_red %in% names(dt)]
  
  # 6th wave
  set.seed(42)
  test_id <- sample(unique(dt_6[, get(id_var)]), length(unique(dt_6[, get(id_var)]))*0.2) # Get 20% of patients for the test dataset
  dt_train <- dt_6[!(get(id_var) %in% test_id)]
  dt_test <- dt_6[get(id_var) %in% test_id]
  dt_train_red <- dt_train[, .SD, .SDcols = c(var_sets$cte, var_sets$vitals, var_sets$analit_red)]
  dt_test_red <- dt_test[, .SD, .SDcols = c(var_sets$cte, var_sets$vitals, var_sets$analit_red)]
  
  weights <- rep(1, dim(dt_train)[1])
  weights[dt_train$Critico == 1] <- 1
  
  bstSparse <- xgboost(data = as.matrix(dt_train_red), label = dt_train$Critico,
                       weight = weights, eval_metric = fscore, max.depth = 4,
                       eta = 1, nthread = 2, nrounds = 100, objective = "binary:logistic")
  preds <- as.numeric(predict(bstSparse, as.matrix(dt_test_red)) > 0.5)
  mean(dt_test$Critico != preds)
  conf_matrix(dt_test$Critico, preds)
  
  print(xgb.importance(model = bstSparse))
  
  # Var subset selection TODO
  
  # Optim params xgboost
  
  par <- c(2, 2, 20) # Initial parameters: weight 2, max.depth 2 and nrounds 20
  lower <- c(0.1, 1, 10)
  upper <- c(5, 10, 500)
  ndeps <- c(0.7, 10, 200)
  
  # With regular L-BFGS-B optimization. Awful results with the rounding of parameters
  # res <- optim(par = par, fn = eval_xgboost, gr = NULL, dt_train, dt_train_red, dt_test, dt_test_red, fscore,
  #              method = "L-BFGS-B", lower = lower, upper = upper, control = list(maxit = 100, ndeps = ndeps))
  
  # With pso 
  # res <- pso::psoptim(par = par, fn = eval_xgboost, gr = NULL, dt_train, dt_train_red, dt_test, dt_test_red, fscore,
  #                     lower = lower, upper = upper, control = list(maxit = 100))
  # Best val: 0.1938, par:  0.2369057   9.9978955 485.5463872
  # Confusion matrix:
  #             Orig
  #           1      0
  # 
  # Pred  1   74     55
  # 
  #       0   153     791
  
  # With differential evolution. Best option
  res <- DEoptim::DEoptim(fn = eval_xgboost, lower = lower, upper = upper,
                          control = DEoptim::DEoptim.control(itermax = 100),
                          dt_train, dt_train_red, dt_test, dt_test_red, fscore)
  # Iteration: 100 bestvalit: 0.174278 bestmemit:    0.677691    8.486887   85.252797
  # Iteration: 100 bestvalit: 0.163094 bestmemit:    0.162297    4.962632  424.167746
  # Confusion matrix:
  #             Orig
  #           1      0
  # 
  # Pred  1   91     51
  # 
  #       0   136     795
  
  browser()
  
  # Full data
  dt_old_red <- dt_old[, .SD, .SDcols = c(var_sets$cte, var_sets$vitals, var_sets$analit)]
  dt_6_red <- dt_6[, .SD, .SDcols = c(var_sets$cte, var_sets$vitals, var_sets$analit)]
  
  weights <- rep(1, dim(dt_old)[1])
  weights[dt_old$Critico == 1] <- 1
  
  bstSparse <- xgboost(data = as.matrix(dt_old_red), label = dt_old$Critico,
                       weight = weights, eval_metric = evalerror, max.depth = 2,
                       eta = 1, nthread = 2, nrounds = 150, objective = "binary:logistic")
  preds <- as.numeric(predict(bstSparse, as.matrix(dt_6_red)) > 0.5)
  mean(dt_6$Critico != preds)
  conf_matrix(dt_6$Critico, preds)
  
  print(xgb.importance(model = bstSparse))
  
  # ---------------------------------------------------------------------------------------
  
  del_vars <- c("orig_row_t_0", "orig_row_t_1")
  obj_vars <- sapply(var_sets$analit_red, function(x){paste0(x, "_t_0")}, USE.NAMES = F)
  dt_test_mod <- copy(dt_test)
  dt_test_mod[, orig_row := .I]
  f_dt_train <- dbnR::filtered_fold_dt(dt_train[, .SD, .SDcols = c(var_sets$analit_red, "REGISTRO")], size, id_var) 
  #dt_train[, eval(id_var) := NULL]
  f_dt_test <- dbnR::filtered_fold_dt(dt_test_mod[, .SD, .SDcols = c(var_sets$analit_red, "REGISTRO", "orig_row")], size, id_var)
  orig_rows <- f_dt_test$orig_row_t_1
  f_dt_test[, (del_vars) := NULL]
  
  net <- dbnR::learn_dbn_struc(dt = NULL, size = size, method = "natPsoho", f_dt = f_dt_train, n_inds = 50)
  fit <- dbnR::fit_dbn_params(net, f_dt_train)
  preds_net <- suppressWarnings(predict_dt(fit, f_dt_test, obj_nodes = obj_vars))
  preds_net[, nrow := NULL]
  preds_net <- setnames(preds_net, old = names(preds_net), new = var_sets$analit_red)
  dt_test_mod <- copy(dt_test)
  dt_test_mod[orig_rows, eval(names(preds_net)) := preds_net,]
  
  bstSparse <- xgboost(data = as.matrix(dt_train_red), label = dt_train$Critico,
                       weight = weights, eval_metric = fscore, max.depth = 4,
                       eta = 1, nthread = 2, nrounds = 20, objective = "binary:logistic")
  preds <- as.numeric(predict(bstSparse, as.matrix(dt_test_mod[, .SD, .SDcols = names(dt_test_red)])) > 0.5)
  mean(dt_test$Critico != preds)
  conf_matrix(dt_test$Critico, preds) # Comprobar cambio con respecto al original
  
  return(0)
}