#' R6 class that defines the XGBoost + DBN model
#'
#' The model is defined as a DBN model that forecasts the state of a time-series
#' based system and an XGBoost model that classifies that predicted state.
#' @export
XGDBN <- R6::R6Class("XGDBN",
  public = list(
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
                         xgb_obj_var, dbn_obj_var,
                         optim = TRUE, xgb_params = c(1.26, 7, 258), ...){
      
      private$xgb_obj_var <- xgb_obj_var
      private$dbn_obj_vars <- sapply(obj_var_dbn, function(x){paste0(x, "_t_0")}, USE.NAMES = F)  # Do not expect the user to input the vars with "_t_0" appended. Could allow both
      
      private$fit_dbn(dt_train, f_dt, id_var, dbn_obj_var, size, method, ...)
      
      private$fit_xgb(dt_train, id_var, obj_var_xgb, optim, xgb_params)
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
    
    #' @description
    #' Fit the internal DBN
    #' @param dt_train a data.table with the training dataset
    #' @param id_var an index variable that identifies different time series in the data
    #' @param dbn_obj_vars the objective variables for the DBN
    #' @param size the size of the DBN 
    #' @param method the structure learning method used
    #' @param ... additional parameters for the DBN structure learning
    fit_dbn = function(dt_train, id_var, dbn_obj_var, size, method, ...){
      f_dt <- dbnR::filtered_fold_dt(dt_train[, .SD, .SDcols = c(id_var, dbn_obj_vars)], size, id_var)
      
      private$net <- dbnR::learn_dbn_struc(dt = NULL, size = size, method = method, f_dt = f_dt, ...)
      private$fit <- dbnR::fit_dbn_params(private$net, f_dt)
    },
    
    #' @description
    #' Fit the internal XGBoost
    #' @param dt_train a data.table with the training dataset
    #' @param id_var an index variable that identifies different time series in the data
    #' @param method the structure learning method used
    #' @param ... additional parameters for the DBN structure learning
    fit_xgb = function(dt_train, id_var, optim, xgb_params){
      del_vars <- c("orig_row_t_0", "orig_row_t_1")
      dt_test_mod <- copy(dt_test)
      dt_test_mod[, orig_row := .I]
      f_dt_train <- dbnR::filtered_fold_dt(dt_train[, .SD, .SDcols = c(var_sets$analit_red, "REGISTRO")], size, id_var) 
      #dt_train[, eval(id_var) := NULL]
      
      bstSparse <- xgboost(data = as.matrix(dt_train_red), label = dt_train$Critico,
                           weight = weights, eval_metric = fscore, max.depth = 4,
                           eta = 1, nthread = 2, nrounds = 20, objective = "binary:logistic")
    },
    
    optimize_xgb = function(xgb_params){
      
    }
    
  )
)