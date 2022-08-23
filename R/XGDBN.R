#' R6 class that defines the XGBoost + DBN model
#'
#' The model is defined as a DBN model that forecasts the state of a time-series
#' based system and an XGBoost model that classifies that predicted state.
#' @export
XGDBN <- R6::R6Class("XGDBN",
  inherit = "HDBN",
  public = list(
    #' @description
    #' Initialize the object with some modifiable parameters of the optimization
    #' @param lower lower bounds of the weight, max tree depth and number of rounds
    #' @param upper upper bounds of the weight, max tree depth and number of rounds
    #' @param itermax maximum number of iterations of the optimization process
    #' @param test_per percentage of instances assigned as test in the optimization
    #' @param optim_trace whether or not to print the progress of each optimization iteration
    initialize = function(lower = c(0.1, 1, 10), upper = c(5, 10, 500), 
                          itermax = 100, test_per = 0.2, trace = TRUE){
      super$initialize(lower, upper, itermax, test_per, trace)
    },
    
    # --ICO-Merge: No f_dt option allowed, maybe improve upon merging
    #' @description
    #' Fit the XGBoost and the DBN models to some provided data
    #' @param dt_train a data.table with the training dataset
    #' @param id_var an index variable that identifies different time series in the data
    #' @param size the size of the DBN 
    #' @param method the structure learning method used
    #' @param cl_obj_var the objective variable for the XGBoost
    #' @param dbn_obj_vars the objective variables for the DBN
    #' @param optim boolean that determines wheter or not the XGBoost parameters should be optimized
    #' @param cl_params vector with the parameters of the XGBoost. c(weight, max_depth, n_rounds)
    #' @param ... additional parameters for the DBN structure learning
    fit_model = function(dt_train, id_var, size, method, 
                         cl_obj_var, dbn_obj_vars, seed = NULL,
                         optim = TRUE, cl_params = c(1.26, 7, 258), ...){
      private$cl_obj_var <- cl_obj_var
      private$dbn_obj_vars_raw <- dbn_obj_vars
      private$dbn_obj_vars <- sapply(dbn_obj_vars, function(x){paste0(x, "_t_0")}, USE.NAMES = F)  # Do not expect the user to input the vars with "_t_0" appended. Could allow both
      private$id_var <- id_var
      private$size <- size
      
      if(!is.null(seed))
        set.seed(seed)
      
      private$fit_dbn(dt_train, size, method, ...)
      
      private$fit_cl(dt_train, optim, cl_params)
    },
    
    #' @description
    #' Predict the objective variable in all the rows in a dataset with the 
    #' XGBoost augmented by the DBN forecasting. The horizon sets the length of 
    #' the forecasting with the DBN model.
    #' @param dt_test a data.table with the test dataset
    #' @param horizon integer value that defines the lenght of the forecasting with the DBN model
    #' @param print_res a boolean that determines whether or not should the results of the prediction be printed
    #' @param conf_mat a boolean that determines whether or not should a confusion matrix be printed
    #' @return the prediction result vector
    predict = function(dt_test, horizon = 1, print_res = T, conf_mat=F){
      del_vars <- c("orig_row_t_0", "orig_row_t_1")
      dt_test_mod <- copy(dt_test)
      dt_test_mod[, orig_row := .I]
      f_dt_test <- dbnR::filtered_fold_dt(dt_test_mod[, .SD, .SDcols = c(private$dbn_obj_vars_raw, private$id_var, "orig_row")], private$size, private$id_var)
      orig_rows <- f_dt_test$orig_row_t_1
      f_dt_test[, (del_vars) := NULL]
      
      #preds_net <- suppressWarnings(dbnR::predict_dt(private$fit, f_dt_test, obj_nodes = private$dbn_obj_vars, verbose = F))
      preds_net <- f_dt_test[, private$predict_row(.SD, horizon), by = seq_len(nrow(f_dt_test))]
      preds_net[, seq_len := NULL]
      preds_net <- setnames(preds_net, old = names(preds_net), new = private$dbn_obj_vars_raw)
      dt_test_mod <- copy(dt_test)
      dt_test_mod[orig_rows, eval(names(preds_net)) := preds_net,]
      dt_test_mod[, eval(private$cl_obj_var) := NULL]
      dt_test_mod[, eval(private$id_var) := NULL]
      preds <- as.numeric(predict(private$cl, as.matrix(dt_test_mod)) > 0.5)
      
      if(print_res)
        cat(paste0("Mean accuracy: ", mean(dt_test[, get(private$cl_obj_var)] == preds)))
      
      if(conf_mat)
        private$conf_matrix(dt_test[, get(private$cl_obj_var)], preds)
      
      return(preds)
    },
    
    #' @description
    #' Predict the objective variable in all the rows in a dataset using only
    #' the XGBoost model. 
    #' @param dt_test a data.table with the test dataset
    #' @param print_res a boolean that determines whether or not should the results of the prediction be printed
    #' @param conf_mat a boolean that determines whether or not should a confusion matrix be printed
    #' @return the prediction result vector
    predict_cl = function(dt_test, print_res = T, conf_mat=F){
      browser()
      dt_test_mod <- copy(dt_test)
      dt_test_mod[, eval(private$cl_obj_var) := NULL]
      dt_test_mod[, eval(private$id_var) := NULL]
      preds <- as.numeric(predict(private$cl, as.matrix(dt_test_mod)) > 0.5)
      
      if(print_res)
        cat(paste0("Mean accuracy: ", mean(dt_test[, get(private$cl_obj_var)] == preds)))
      
      if(conf_mat)
        private$conf_matrix(dt_test[, get(private$cl_obj_var)], preds)
      
      return(preds)
    }
   
  ),
  
  private = list(
    #' @description
    #' Fit the internal XGBoost
    #' @param dt_train a data.table with the training dataset
    #' @param optim boolean that determines wheter or not the classifier parameters should be optimized
    #' @param cl_params the classifier parameters.
    fit_cl = function(dt_train, optim, cl_params){
      obj_col <- dt_train[, get(private$cl_obj_var)]
      dt_train_red <- copy(dt_train)
      dt_train_red[, eval(private$id_var) := NULL]
      
      if(optim)
        cl_params <- private$optimize_cl(dt_train)$optim$bestmem
      
      private$cl_params <- cl_params
      
      weights <- rep(1, dim(dt_train)[1])
      weights[dt_train[get(private$cl_obj_var) == 1, .I]] <- cl_params[1]
      dt_train_red[, eval(private$cl_obj_var) := NULL]
      
      private$cl <- xgboost(data = as.matrix(dt_train_red), label = obj_col,
                           weight = weights, eval_metric = fscore, max.depth = 4,
                           eta = 1, nthread = 2, nrounds = 20, objective = "binary:logistic")
    },
    
    # Three parameters to optimize: the weight of critical cases, the max.depth and the number of rounds
    eval_cl = function(params, dt_train, dt_test, labels, eval_metric){
      print(params)
      params[2] <- round(params[2])
      params[3] <- round(params[3])
      weights <- rep(1, dim(labels[test == 0])[1])
      weights[labels[test == 0 & get(private$cl_obj_var) == 1, .I]] <- params[1]
      cl <- xgboost(data = as.matrix(dt_train), 
                           label = labels[test == 0, get(private$cl_obj_var)], weight = weights, 
                           eval_metric = eval_metric, max.depth = params[2], nrounds = params[3],
                           eta = 1, nthread = 2, objective = "binary:logistic", verbose = 0)
      preds <- as.numeric(predict(cl, as.matrix(dt_test)) > 0.5)
      acc <- mean(labels[test == 1, get(private$cl_obj_var)] != preds) # Mean error
      
      return(acc)
    }
    
  )
)