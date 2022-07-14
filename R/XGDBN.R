#' R6 class that defines the XGBoost + DBN model
#'
#' The model is defined as a DBN model that forecasts the state of a time-series
#' based system and an XGBoost model that classifies that predicted state.
#' @export
XGDBN <- R6::R6Class("XGDBN",
  public = list(
    #' @description
    #' Initialize the object with some modifiable parameters of the optimization
    #' @param lower lower bounds of the weight, max tree depth and number of rounds
    #' @param upper upper bounds of the weight, max tree depth and number of rounds
    #' @param itermax maximum number of iterations of the optimization process
    #' @param test_per percentage of instances assigned as test in the optimization
    initialize = function(lower = c(0.1, 1, 10), upper = c(5, 10, 500), 
                          itermax = 100, test_per = 0.2){
      private$optim_lower <- lower 
      private$optim_upper <- upper
      private$optim_itermax <- itermax
      private$optim_test_per = test_per
    },
    
    # --ICO-Merge: No f_dt option allowed, maybe improve upon merging
    #' @description
    #' Fit the XGBoost and the DBN models to some provided data
    #' @param dt_train a data.table with the training dataset
    #' @param id_var an index variable that identifies different time series in the data
    #' @param size the size of the DBN 
    #' @param method the structure learning method used
    #' @param xgb_obj_var the objective variable for the XGBoost
    #' @param dbn_obj_vars the objective variables for the DBN
    #' @param optim boolean that determines wheter or not the XGBoost parameters should be optimized
    #' @param xgb_params vector with the parameters of the XGBoost. c(weight, max_depth, n_rounds)
    #' @param ... additional parameters for the DBN structure learning
    fit_model = function(dt_train, id_var, size, method, 
                         xgb_obj_var, dbn_obj_vars, seed = NULL,
                         optim = TRUE, xgb_params = c(1.26, 7, 258), ...){
      private$xgb_obj_var <- xgb_obj_var
      private$dbn_obj_vars <- sapply(dbn_obj_vars, function(x){paste0(x, "_t_0")}, USE.NAMES = F)  # Do not expect the user to input the vars with "_t_0" appended. Could allow both
      
      if(!is.null(seed))
        set.seed(seed)
      
      private$fit_dbn(dt_train, id_var, dbn_obj_vars, size, method, ...)
      
      private$fit_xgb(dt_train, id_var, optim, xgb_params)
    },
    
    predict = function(){
      stop("Not implemented yet.")
      
      del_vars <- c("orig_row_t_0", "orig_row_t_1")
      obj_vars <- sapply(var_sets$analit_red, function(x){paste0(x, "_t_0")}, USE.NAMES = F)
      dt_test_mod <- copy(dt_test)
      dt_test_mod[, orig_row := .I]
      f_dt_train <- dbnR::filtered_fold_dt(dt_train[, .SD, .SDcols = c(var_sets$analit_red, "REGISTRO")], size, id_var) 
      #dt_train[, eval(id_var) := NULL]
      f_dt_test <- dbnR::filtered_fold_dt(dt_test_mod[, .SD, .SDcols = c(var_sets$analit_red, "REGISTRO", "orig_row")], size, id_var)
      orig_rows <- f_dt_test$orig_row_t_1
      f_dt_test[, (del_vars) := NULL]
      
      preds_net <- suppressWarnings(predict_dt(fit, f_dt_test, obj_nodes = obj_vars))
      preds_net[, nrow := NULL]
      preds_net <- setnames(preds_net, old = names(preds_net), new = var_sets$analit_red)
      dt_test_mod <- copy(dt_test)
      dt_test_mod[orig_rows, eval(names(preds_net)) := preds_net,]
      preds <- as.numeric(predict(bstSparse, as.matrix(dt_test_mod[, .SD, .SDcols = names(dt_test_red)])) > 0.5)
    }
   
  ),
  
  private = list(
    #' @field xgb the XGBoost model
    xgb = NULL,
    #' @field xgb_params the XGBoost parameters. Weight of the '1' labels, maximum depth of the trees and number of rounds.
    xgb_params = NULL,
    #' @field xgb_obj_var the variable to be classified with the XGBoost
    xgb_obj_var = NULL,
    #' @field net the DBN graph object
    net = NULL,
    #' @field fit the DBN fitted model
    fit = NULL,
    #' @field dbn_obj_vars the variables to be predicted with the DBN model
    dbn_obj_vars = NULL,
    #' @field optim_lower lower bounds of the weight, max tree depth and number of rounds
    optim_lower = NULL,
    #' @field optim_upper bounds of the weight, max tree depth and number of rounds
    optim_upper = NULL,
    #' @field optim_itermax maximum number of iterations of the optimization process
    optim_itermax = NULL,
    #' @field optim_test_per percentage of instances assigned as test in the optimization
    optim_test_per = NULL,
    
    #' @description
    #' Fit the internal DBN
    #' @param dt_train a data.table with the training dataset
    #' @param id_var an index variable that identifies different time series in the data
    #' @param dbn_obj_vars the objective variables for the DBN
    #' @param size the size of the DBN 
    #' @param method the structure learning method used
    #' @param ... additional parameters for the DBN structure learning
    fit_dbn = function(dt_train, id_var, dbn_obj_vars, size, method, ...){
      dt <- dt_train[, .SD, .SDcols = c(id_var, dbn_obj_vars)]
      f_dt <- dbnR::filtered_fold_dt(dt, size, id_var)
      dt[, eval(id_var) := NULL]
      
      private$net <- dbnR::learn_dbn_struc(dt = dt, size = size, method = method, f_dt = f_dt, ...)
      private$fit <- dbnR::fit_dbn_params(private$net, f_dt)
    },
    
    #' @description
    #' Fit the internal XGBoost
    #' @param dt_train a data.table with the training dataset
    #' @param id_var an index variable that identifies different time series in the data
    #' @param method the structure learning method used
    #' @param ... additional parameters for the DBN structure learning
    fit_xgb = function(dt_train, id_var, optim, xgb_params){
      obj_col <- dt_train[, get(private$xgb_obj_var)]
      dt_train_red <- copy(dt_train)
      dt_train_red[, eval(id_var) := NULL]
      
      if(optim)
        xgb_params <- private$optimize_xgb(dt_train, id_var)$optim$bestmem
      
      browser()
      
      private$xgb_params <- xgb_params
      
      weights <- rep(1, dim(dt_train)[1])
      weights[dt_train[get(private$xgb_obj_var) == 1, .I]] <- xgb_params[1]
      dt_train_red[, eval(private$xgb_obj_var) := NULL]
      
      private$xgb <- xgboost(data = as.matrix(dt_train_red), label = obj_col,
                           weight = weights, eval_metric = fscore, max.depth = 4,
                           eta = 1, nthread = 2, nrounds = 20, objective = "binary:logistic")
    },
    
    #' @description
    #' Optimize the parameters of the internal XGBoost
    #' @param dt a data.table with the training dataset
    #' @param id_var an index variable that identifies different time series in the data
    optimize_xgb = function(dt, id_var){
      test_id <- sample(unique(dt[, get(id_var)]), 
                        length(unique(dt[, get(id_var)])) * private$optim_test_per)
      dt_train <- dt[!(get(id_var) %in% test_id)]
      dt_test <- dt[get(id_var) %in% test_id]
      labels <- dt[, .SD, .SDcols = c(id_var, private$xgb_obj_var)]
      labels[, test := 0]
      labels[get(id_var) %in% test_id, test := 1]
      dt_train[, eval(private$xgb_obj_var) := NULL]
      dt_test[, eval(private$xgb_obj_var) := NULL]
      dt_train[, eval(id_var) := NULL]
      dt_test[, eval(id_var) := NULL]
      
      
      res <- DEoptim::DEoptim(fn = private$eval_xgboost, lower = private$optim_lower, upper = private$optim_upper,
                              control = DEoptim::DEoptim.control(itermax = private$optim_itermax),
                              dt_train, dt_test, labels, private$fscore)
      
      return(res)
    },
    
    # Three parameters to optimize: the weight of critical cases, the max.depth and the number of rounds
    eval_xgboost = function(params, dt_train, dt_test, labels, eval_metric){
      print(params)
      params[2] <- round(params[2])
      params[3] <- round(params[3])
      weights <- rep(1, dim(labels[test == 0])[1])
      weights[labels[test == 0 & get(private$xgb_obj_var) == 1, .I]] <- params[1]
      xgb <- xgboost(data = as.matrix(dt_train), 
                           label = labels[test == 0, get(private$xgb_obj_var)], weight = weights, 
                           eval_metric = eval_metric, max.depth = params[2], nrounds = params[3],
                           eta = 1, nthread = 2, objective = "binary:logistic", verbose = 0)
      preds <- as.numeric(predict(xgb, as.matrix(dt_test)) > 0.5)
      acc <- mean(labels[test == 1, get(private$xgb_obj_var)] != preds) # Mean error
      
      return(acc)
    },
    
    fscore = function(preds, dtrain){
      labels <- getinfo(dtrain, "label")
      tp <- sum(labels == 1 & preds >= 0.5)
      fp <- sum(labels == 0 & preds >= 0.5)
      fn <- sum(labels == 1 & preds < 0.5)
      err <- as.numeric(tp / (tp + 0.5 * (fp + fn)))
      
      return(list(metric = "fscore", value = err))
    }
    
  )
)