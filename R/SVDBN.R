#' R6 class that defines the SVM + DBN model
#'
#' The model is defined as a DBN model that forecasts the state of a time-series
#' based system and an SVM model that classifies that predicted state.
#' @export
SVDBN <- R6::R6Class("SVDBN",
  inherit = HDBN,
  public = list(
    #' @description
    #' Initialize the object with some modifiable parameters of the optimization
    #' @param lower lower bounds of the weight, gamma and cost
    #' @param upper upper bounds of the weight, gamma and cost
    #' @param itermax maximum number of iterations of the optimization process
    #' @param test_per percentage of instances assigned as test in the optimization
    #' @param optim_trace whether or not to print the progress of each optimization iteration
    initialize = function(lower = c(1, -8, -8), upper = c(5, 4, 4), 
                          itermax = 100, test_per = 0.2, trace = TRUE){
      super$initialize(lower, upper, itermax, test_per, trace)
    },
    
    #' @description
    #' Predict the objective variable in all the rows in a dataset with the 
    #' SVM augmented by the DBN forecasting. The horizon sets the length of 
    #' the forecasting with the DBN model.
    #' @param dt_test a data.table with the test dataset
    #' @param horizon integer value that defines the lenght of the forecasting with the DBN model
    #' @param print_res a boolean that determines whether or not should the results of the prediction be printed
    #' @param conf_mat a boolean that determines whether or not should a confusion matrix be printed
    #' @return the prediction result vector
    predict = function(dt_test, horizon = 1, print_res = T, conf_mat=T){
      # --ICO-MERGE very similar to the XGBoost one, maybe include in HDBN and inherit?
      del_vars <- c("orig_row_t_0", "orig_row_t_1")
      dt_test_mod <- copy(dt_test)
      dt_test_mod[, orig_row := .I]
      f_dt_test <- dbnR::filtered_fold_dt(dt_test_mod[, .SD, .SDcols = c(private$dbn_obj_vars_raw, private$id_var, "orig_row")], private$size, private$id_var)
      orig_rows <- f_dt_test$orig_row_t_1
      f_dt_test[, (del_vars) := NULL]
      
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
      
      preds <- predict(private$cl, dt_test_mod)
      
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
    #' the SVM model. 
    #' @param dt_test a data.table with the test dataset
    #' @param print_res a boolean that determines whether or not should the results of the prediction be printed
    #' @param conf_mat a boolean that determines whether or not should a confusion matrix be printed
    #' @return the prediction result vector
    predict_cl = function(dt_test, print_res = T, conf_mat=T){
      # --ICO-MERGE very similar to the XGBoost one, maybe include in HDBN and inherit?
      dt_test_mod <- copy(dt_test)
      dt_test_mod[, eval(private$cl_obj_var) := NULL]
      dt_test_mod[, eval(private$id_var) := NULL]
      
      dt_scaled <- scale(dt_test_mod[, .SD, .SDcols = private$dbn_obj_vars_raw],
                         center = private$center, scale = private$scale)
      dt_test_mod[, eval(private$dbn_obj_vars_raw) := as.data.table(dt_scaled)]
      
      preds <- predict(private$cl, dt_test_mod)
      
      if(print_res){
        cat(paste0("Mean accuracy: ", mean(dt_test[, get(private$cl_obj_var)] == preds), "\n"))
        cat(paste0("F1score: ", 1-private$fscore(dt_test[, get(private$cl_obj_var)], preds), "\n"))
        cat(paste0("G-mean score: ", 1-private$gmean(dt_test[, get(private$cl_obj_var)], preds), "\n"))
      }
      
      if(conf_mat)
        private$conf_matrix(dt_test[, get(private$cl_obj_var)], preds)
      
      return(preds)
    }
   
  ),
  
  private = list(
    #' @field formula the formula for the SVM model
    formula = NULL,
    
    #' @description
    #' Fit the internal SVM
    #' @param dt_train a data.table with the training dataset
    #' @param optim boolean that determines whether or not the SVM parameters should be optimized
    #' @param cl_params vector with the parameters of the SVM c(weight, gamma, cost)
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
      private$formula <- as.formula(paste0(private$cl_obj_var, " ~ ." ))
      
      # dt_train_red <- shuffle_dt(dt_train_red)
      
      if(optim)
        cl_params <- private$optimize_cl(dt_train)$optim$bestmem
      
      if(is.null(cl_params))  # No optimization and no params provided by the user
        cl_params <- c(3.9742, -6.7470, -3.441)
      
      private$cl_params <- cl_params
      
      weights <- c("0" = 1, "1" = cl_params[[1]])
      
      private$cl <- e1071::svm(private$formula, data = dt_train, 
                               type = "C", class.weights = weights, 
                               gamma = 2^cl_params[2], cost = 2^cl_params[3])
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
      weights <- c("0" = 1, "1" = params[1])
      dt_train[, eval(private$cl_obj_var) := labels[test == 0, get(private$cl_obj_var)]]
      cl <- e1071::svm(private$formula, data = dt_train, 
                       type = "C", class.weights = weights, 
                       gamma = 2^params[2], cost = 2^params[3])
      preds <- predict(cl, dt_test)
      
      acc <- eval_metric(labels[test == 1, get(private$cl_obj_var)], preds)
      
      return(acc)
    }
    
  )
)