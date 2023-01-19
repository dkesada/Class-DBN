#' R6 class that defines the XGBoost + DBN model
#'
#' The model is defined as a DBN model that forecasts the state of a time-series
#' based system and an XGBoost model that classifies that predicted state.
#' @export
XGDBN <- R6::R6Class("XGDBN",
  inherit = HDBN,
  public = list(
    #' @description
    #' Initialize the object with some modifiable parameters of the optimization
    #' @param lower lower bounds of the weight, max tree depth and number of rounds
    #' @param upper upper bounds of the weight, max tree depth and number of rounds
    #' @param itermax maximum number of iterations of the optimization process
    #' @param test_per percentage of instances assigned as test in the optimization
    #' @param optim_trace whether or not to print the progress of each optimization iteration
    initialize = function(lower = c(0.5, 1, 500), upper = c(2, 10, 1000), # 0.1, 5
                          itermax = 100, test_per = 0.2, trace = TRUE, optim_f = "gmean"){
      super$initialize(lower, upper, itermax, test_per, trace)
      private$optim_f <- optim_f
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
    predict = function(dt_test, horizon = 1, print_res = T, conf_mat=T){
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
      
      dt_scaled <- scale(dt_test_mod[, .SD, .SDcols = private$dbn_obj_vars_raw],
                         center = private$center, scale = private$scale)
      dt_test_mod[, eval(private$dbn_obj_vars_raw) := as.data.table(dt_scaled)]
      
      preds <- as.numeric(predict(private$cl, as.matrix(dt_test_mod)) > 0.5)
      
      if(print_res){
        cat(paste0("Mean accuracy: ", mean(dt_test[, get(private$cl_obj_var)] == preds), "\n"))
        cat(paste0("F1score: ", 1-private$fscore(dt_test[, get(private$cl_obj_var)], preds), "\n"))
        cat(paste0("G-mean score: ", 1-private$gmean(dt_test[, get(private$cl_obj_var)], preds), "\n"))
      }
      
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
    predict_cl = function(dt_test, print_res = T, conf_mat=T){
      dt_test_mod <- copy(dt_test)
      dt_test_mod[, eval(private$cl_obj_var) := NULL]
      dt_test_mod[, eval(private$id_var) := NULL]
      
      dt_scaled <- scale(dt_test_mod[, .SD, .SDcols = private$dbn_obj_vars_raw],
                         center = private$center, scale = private$scale)
      dt_test_mod[, eval(private$dbn_obj_vars_raw) := as.data.table(dt_scaled)]
      
      preds <- as.numeric(predict(private$cl, as.matrix(dt_test_mod)) > 0.5)
      
      if(print_res){
        cat(paste0("Mean accuracy: ", mean(dt_test[, get(private$cl_obj_var)] == preds), "\n"))
        cat(paste0("F1score: ", 1-private$fscore(dt_test[, get(private$cl_obj_var)], preds), "\n"))
        cat(paste0("G-mean score: ", 1- private$gmean(dt_test[, get(private$cl_obj_var)], preds), "\n"))
      }
      
      if(conf_mat)
        private$conf_matrix(dt_test[, get(private$cl_obj_var)], preds)
      
      return(preds)
    }
   
  ),
  
  private = list(
    #' @field optim_f the internal optimization function for the XGBoost
    optim_f = NULL,
    
    #' @description
    #' Fit the internal XGBoost
    #' @param dt_train a data.table with the training dataset
    #' @param optim boolean that determines whether or not the XGBoost parameters should be optimized
    #' @param cl_params vector with the parameters of the XGBoost. c(weight, max_depth, n_rounds)
    fit_cl = function(dt_train, optim, cl_params){
      dt_train <- copy(dt_train)
      obj_col <- dt_train[, get(private$cl_obj_var)]
      # dt_train_red <- copy(dt_train)
      
      dt_scaled <- scale(dt_train[, .SD, .SDcols = private$dbn_obj_vars_raw])
      private$center <- attr(dt_scaled, "scaled:center")
      private$scale <- attr(dt_scaled, "scaled:scale")
      dt_train[, eval(private$dbn_obj_vars_raw) := as.data.table(dt_scaled)]
      
      # SMOTE should be optional
      dt_train <- smote_dt(dt_train, obj_var = private$cl_obj_var, perc_over = 100, perc_under = 400)
      obj_col <- dt_train[, get(private$cl_obj_var)]
      # dt_train <- copy(dt_train_red)
      
      dt_train[, eval(private$id_var) := NULL]
      
      if(optim)
        cl_params <- private$optimize_cl(dt_train)$optim$bestmem
      
      if(is.null(cl_params))  # No optimization and no params provided by the user
        cl_params <- c(0.866, 5.008, 701.397)  #cl_params <- c(1.2, 3, 337)
      
      private$cl_params <- cl_params
      
      dt_train[, index := .I]
      weights <- rep(1, dim(dt_train)[1])
      weights[dt_train[get(private$cl_obj_var) == 1, index]] <- cl_params[1]
      dt_train[, eval(private$cl_obj_var) := NULL]
      dt_train[, index := NULL]
      
      private$cl <- xgboost(data = as.matrix(dt_train), label = obj_col,
                            weight = weights, eval.metric = private$gmean_xgb, max.depth = round(cl_params[2]),
                            eta = 1, nthread = 2, nrounds = round(cl_params[3]), objective = "binary:logistic")
    },
    
    #' @description
    #' Optimize the parameters of the internal classifier. Generic version without using the id_var
    #' @param dt a data.table with the training dataset
    optimize_cl = function(dt){
      dt[, Idx := .I]
      test_id <- sample(dt[, Idx], nrow(dt) * private$optim_test_per)
      dt_train <- dt[!(Idx %in% test_id)]
      dt_test <- dt[Idx %in% test_id]
      labels <- dt[, .SD, .SDcols = c("Idx", private$cl_obj_var)]
      labels[, test := 0]
      labels[Idx %in% test_id, test := 1]
      dt_train[, eval(private$cl_obj_var) := NULL]
      dt_test[, eval(private$cl_obj_var) := NULL]
      dt_train[, Idx := NULL]
      dt_test[, Idx := NULL]
      dt[, Idx := NULL]
      
      res <- DEoptim::DEoptim(fn = private$eval_cl, lower = private$optim_lower, upper = private$optim_upper,
                              control = DEoptim::DEoptim.control(itermax = private$optim_itermax, trace = private$optim_trace),
                              dt_train, dt_test, labels, private$gmean)
      
      return(res)
    },
    
    # Three parameters to optimize: the weight of critical cases, the max.depth and the number of rounds
    eval_cl = function(params, dt_train, dt_test, labels, eval_metric){
      print(params)
      params[2] <- round(params[2])
      params[3] <- round(params[3])
      weights <- rep(1, dim(labels[test == 0])[1])
      labels[test == 0, index := .I]
      weights[labels[test == 0 & get(private$cl_obj_var) == 1, index]] <- params[1]
      labels[, index := NULL]
      
      cl <- xgboost(data = as.matrix(dt_train), 
                           label = labels[test == 0, get(private$cl_obj_var)], weight = weights, 
                           eval.metric = private$gmean_xgb, max.depth = params[2], nrounds = params[3],
                           eta = 1, nthread = 2, objective = "binary:logistic", verbose = 0)
      preds <- as.numeric(predict(cl, as.matrix(dt_test)) > 0.5)
      
      acc <- eval_metric(labels[test == 1, get(private$cl_obj_var)], preds)
      
      return(acc)
    },
    
    get_optim_f = function(){
      res <- private$fscore_xgb
      if(private$optim_f == "gmean")
        res <- private$gmean_xgb
      if(private$optim_f == "dummy")
        res <- private$dummy_xgb
      
      return(res)
    },
    
    # For the xgb internal optimization
    fscore_xgb = function(preds, dtrain){
      labels <- getinfo(dtrain, "label")
      tp <- sum(labels == 1 & preds >= 0.5)
      fp <- sum(labels == 0 & preds >= 0.5)
      fn <- sum(labels == 1 & preds < 0.5)
      err <- as.numeric(tp / (tp + 0.5 * (fp + fn)))
      if(is.na(err))
        err <- 0
      
      return(list(metric = "fscore", value = err))
    },
    
    # For the xgb internal optimization
    gmean_xgb = function(preds, dtrain){
      labels <- getinfo(dtrain, "label")
      tp <- sum(labels == 1 & preds >= 0.5)
      fp <- sum(labels == 0 & preds >= 0.5)
      fn <- sum(labels == 1 & preds < 0.5)
      tn <- sum(labels == 0 & preds < 0.5)
      
      recall <- tp / (tp + fn)
      spec <- tn / (fp + tn)
      err <- 1 - as.numeric(sqrt(spec * recall))
      if(is.na(err))
        err <- 0
      
      return(list(metric = "g-mean", value = err))
    },
    
    dummy_xgb = function(preds, dtrain){
      labels <- getinfo(dtrain, "label")
      tp <- 1 - (sum(labels == 1 & preds >= 0.5) / length(labels))
      
      err <- as.numeric(tp)
      if(is.na(err))
        err <- 0
      
      return(list(metric = "dummy", value = err))
    }
    
  )
)