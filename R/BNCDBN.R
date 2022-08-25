#' R6 class that defines the Bayesian network classifier + DBN model
#'
#' The model is defined as a DBN model that forecasts the state of a time-series
#' based system and a Bayesian network classifier that classifies that predicted state.
#' Predicting with bnclassify requires the gRain package, which has unavailable dependencies
#' on CRAN that must be downloaded from bioconductor. It's kind of a pain, but there's not much I can
#' do about it
#' @export
BNCDBN <- R6::R6Class("BNCDBN",
  inherit = HDBN,
  public = list(
    #' @description
    #' Initialize the object with some modifiable parameters of the optimization
    #' @param lower lower bounds of the number of folds, epsilon and smooth
    #' @param upper upper bounds of the number of folds, epsilon and smooth
    #' @param itermax maximum number of iterations of the optimization process
    #' @param test_per percentage of instances assigned as test in the optimization
    #' @param optim_trace whether or not to print the progress of each optimization iteration
    initialize = function(lower = c(1, 0, 0), upper = c(10, 5, 5), 
                          itermax = 100, test_per = 0.2, trace = TRUE){
      super$initialize(lower, upper, itermax, test_per, trace)
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
    #' @field mcuts master set of cut points of the discretization process
    mcuts = NULL,
    
    #' @description
    #' Fit the internal Bayesian network classifier
    #' @param dt_train a data.table with the training dataset
    #' @param optim boolean that determines whether or not the Bayesian network classifier parameters should be optimized. 
    #' Only the TAN models have parameters that can be optimized.
    #' @param cl_params vector with the parameters of the Bayesian network classifier. c(n_folds, epsilon, smooth)
    fit_cl = function(dt_train, optim, cl_params){
      obj_col <- dt_train[, get(private$cl_obj_var)]
      dt_train_red <- copy(dt_train)
      dt_train_red[, eval(private$id_var) := NULL]
      
      private$mcuts <- dt_train_red[, lapply(.SD, arules::discretize, breaks = 4, onlycuts = T), 
                                    .SDcols = c("Edad", "IMC", dbn_obj_vars)]
      dt_train_red <- private$discretize_dt(dt_train_red)
      
      if(cl_params[1] == 1){ # The tan_cl model only has the smooth parameter
        private$optim_lower <- private$optim_lower[3] 
        private$optim_upper <- private$optim_upper[3]
      }
      
      browser()
      
      if(optim & (cl_params[1] > 0)){
        private$cl_params <- cl_params # I need to know my model type cl_params[1] inside the optimization process
        cl_params <- private$optimize_cl(dt_train)$optim$bestmem
      }
      
      model <-private$build_bnc(dt_train_red, cl_params)
      private$cl <- lp(model, dt_disc, cl_params[4]) 
      private$cl_params <- cl_params
    },
    
    # One to three parameters to optimize: the number of folds, the epsilon and the smooth
    eval_cl = function(params, dt_train, dt_test, labels, eval_metric){
      print(params)
      
      if(cl_params[1] == 1){ # The tan_cl model only has the smooth parameter
        model <-private$build_bnc(dt_train_red, cl_params)
        private$cl <- lp(model, dt_disc, cl_params[4]) 
      }#TODO
      
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
    },
    
    # Builds the appropriate Bayesian netwock classifier from the bnclassify package: 0-nb, 1-tan_cl, 2-tan_hc, 3-tan_hcsp
    build_bnc = function(dt, cl_params){
      res <- bnclassify::nb(private$cl_obj_var, dt)
      if(cl_params[1] == 1)
        res <- bnclassify::tan_cl(private$cl_obj_var, dt)
      else if(cl_params[1] == 2)
        res <- bnclassify::tan_hc(private$cl_obj_var, dt, cl_params[2], cl_params[3], cl_params[4])
      else if (cl_params[1] == 3)
        res <- bnclassify::tan_hcsp(private$cl_obj_var, dt, cl_params[2], cl_params[3], cl_params[4])
      
      return(res)
    },
    
    # Discretizes a data.table based on the master set of cuts stored inside this object
    discretize_dt = function(dt){
      for(i in names(private$mcuts))
        dt[, eval(i) := cut(get(i), breaks = private$mcuts[, get(i)], include.lowest = T)]
      
      return(factorize_dt(dt))
    }
    
  )
)