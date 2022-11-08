#' R6 abstract class that defines the classifier + DBN model
#'
#' The model is defined as a DBN model that forecasts the state of a time-series
#' based system and a classifier that classifies that predicted state.
#' This is a very simplified abstract class for all the different hybrid models
#' that I would like to try. It could be more generic, but I do not have the
#' time to encapsulate it any further.
#' @export
HDBN <- R6::R6Class("HDBN",
  public = list(
    #' @description
    #' Initialize the object with some modifiable parameters of the optimization
    #' @param lower vector of lower bounds of the variables to optimize
    #' @param upper vector of upper bounds of the variables to optimize
    #' @param itermax maximum number of iterations of the optimization process
    #' @param test_per percentage of instances assigned as test in the optimization
    #' @param optim_trace whether or not to print the progress of each optimization iteration
    initialize = function(lower, upper, itermax = 100, test_per = 0.2, trace = TRUE){
      private$optim_lower <- lower 
      private$optim_upper <- upper
      private$optim_itermax <- itermax
      private$optim_test_per <- test_per
      private$optim_trace <- trace
    },
    
    # --ICO-Merge: No f_dt option allowed, maybe improve upon merging
    #' @description
    #' Fit the classifier and the DBN models to some provided data
    #' @param dt_train a data.table with the training dataset
    #' @param id_var an index variable that identifies different time series in the data
    #' @param size the size of the DBN 
    #' @param method the structure learning method used
    #' @param cl_obj_var the objective variable for the classifier
    #' @param dbn_obj_vars the objective variables for the DBN
    #' @param optim boolean that determines whether or not the classifier parameters should be optimized
    #' @param cl_params vector with the parameters of the classifier
    #' @param ... additional parameters for the DBN structure learning
    fit_model = function(dt_train, id_var, size, method, 
                         cl_obj_var, dbn_obj_vars, seed = NULL,
                         optim = TRUE, cl_params = NULL, ...){
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
    #' classifier augmented by the DBN forecasting. The horizon sets the length of 
    #' the forecasting with the DBN model.
    #' @param dt_test a data.table with the test dataset
    #' @param horizon integer value that defines the lenght of the forecasting with the DBN model
    #' @param print_res a boolean that determines whether or not should the results of the prediction be printed
    #' @param conf_mat a boolean that determines whether or not should a confusion matrix be printed
    #' @return the prediction result vector
    predict = function(dt_test, horizon, print_res, conf_mat){
      stop("To be implemented.")
    },
    
    #' @description
    #' Predict the objective variable in all the rows in a dataset using only
    #' the classifier model. 
    #' @param dt_test a data.table with the test dataset
    #' @param print_res a boolean that determines whether or not should the results of the prediction be printed
    #' @param conf_mat a boolean that determines whether or not should a confusion matrix be printed
    #' @return the prediction result vector
    predict_cl = function(dt_test, print_res, conf_mat){
      stop("To be implemented.")
    },
    
    #' @description
    #' Print the parameters used to train the model 
    print_params = function(){
      cat("Parameters used during training: \n")
      cat("Classifier params: ", private$cl_params, "\n")
      cat("Classifier objective var: ", private$cl_obj_var, "\n")
      cat("DBN objective vars: ", private$dbn_obj_vars_raw, "\n")
      cat("Id variable: ", private$id_var, "\n")
      cat("Size of the DBN: ", private$size, "\n")
      cat("Lower bounds for the optimization: ", private$optim_lower, "\n")
      cat("Upper bounds for the optimization: ", private$optim_upper, "\n")
      cat("Maximum iterations of the optimizer: ", private$optim_itermax, "\n")
      cat("Percentage of instances in test: ", private$optim_test_per, "\n")
    }
   
  ),
  
  private = list(
    #' @field cl the classifier model
    cl = NULL,
    #' @field cl_params the classifier parameters.
    cl_params = NULL,
    #' @field cl_obj_var the variable to be classified with the classifier
    cl_obj_var = NULL,
    #' @field net the DBN graph object
    net = NULL,
    #' @field fit the DBN fitted model
    fit = NULL,
    #' @field dbn_obj_vars_raw the variables to be predicted with the DBN model without the appended 't_0'
    dbn_obj_vars_raw = NULL,
    #' @field dbn_obj_vars the variables to be predicted with the DBN model
    dbn_obj_vars = NULL,
    #' @field id_var the variable that differentiates each independent individual TS in the dataset
    id_var = NULL,
    #' @field size size argument of the DBN
    size = NULL,
    #' @field optim_lower lower bounds of the parameters
    optim_lower = NULL,
    #' @field optim_upper bounds of the parameters
    optim_upper = NULL,
    #' @field optim_itermax maximum number of iterations of the optimization process
    optim_itermax = NULL,
    #' @field optim_test_per percentage of instances assigned as test in the optimization
    optim_test_per = NULL,
    #' @field optim_trace whether or not to print the progress of each optimization iteration
    optim_trace = NULL,
    
    #' @description
    #' Fit the internal DBN
    #' @param dt_train a data.table with the training dataset
    #' @param size the size of the DBN 
    #' @param method the structure learning method used
    #' @param ... additional parameters for the DBN structure learning
    fit_dbn = function(dt_train, size, method, ...){
      dt <- dt_train[, .SD, .SDcols = c(private$id_var, private$dbn_obj_vars_raw)]
      f_dt <- dbnR::filtered_fold_dt(dt, size, private$id_var)
      dt[, eval(private$id_var) := NULL]
      
      private$net <- dbnR::learn_dbn_struc(dt = dt, size = size, method = method, f_dt = f_dt, ...)
      private$fit <- dbnR::fit_dbn_params(private$net, f_dt)
    },
    
    #' @description
    #' Fit the internal classifier
    #' @param dt_train a data.table with the training dataset
    #' @param optim boolean that determines whether or not the classifier parameters should be optimized
    #' @param cl_params the classifier parameters.
    fit_cl = function(dt_train, optim, cl_params){
      stop("To be implemented.")
    },
    
    #' @description
    #' Optimize the parameters of the internal classifier
    #' @param dt a data.table with the training dataset
    optimize_cl = function(dt){
      test_id <- sample(unique(dt[, get(private$id_var)]), 
                        length(unique(dt[, get(private$id_var)])) * private$optim_test_per)
      dt_train <- dt[!(get(private$id_var) %in% test_id)]
      dt_test <- dt[get(private$id_var) %in% test_id]
      labels <- dt[, .SD, .SDcols = c(private$id_var, private$cl_obj_var)]
      labels[, test := 0]
      labels[get(private$id_var) %in% test_id, test := 1]
      dt_train[, eval(private$cl_obj_var) := NULL]
      dt_test[, eval(private$cl_obj_var) := NULL]
      dt_train[, eval(private$id_var) := NULL]
      dt_test[, eval(private$id_var) := NULL]
      
      
      res <- DEoptim::DEoptim(fn = private$eval_cl, lower = private$optim_lower, upper = private$optim_upper,
                              control = DEoptim::DEoptim.control(itermax = private$optim_itermax, trace = private$optim_trace),
                              dt_train, dt_test, labels, private$fscore)
      
      return(res)
    },
    
    eval_cl = function(params, dt_train, dt_test, labels, eval_metric){
      stop("To be implemented.")
    },
    
    predict_row = function(orig_row, horizon){
      pred <- dbnR::forecast_ts(dt = orig_row, fit = private$fit, 
                                obj_vars = private$dbn_obj_vars, ini = 1, 
                                len = horizon, print_res = F, plot_res = F)
      pred$pred[, exec := NULL]
      return(pred$pred[horizon])
    },
    
    fscore = function(preds, dtrain){
      labels <- getinfo(dtrain, "label")
      tp <- sum(labels == 1 & preds >= 0.5)
      fp <- sum(labels == 0 & preds >= 0.5)
      fn <- sum(labels == 1 & preds < 0.5)
      err <- as.numeric(tp / (tp + 0.5 * (fp + fn)))
      
      return(list(metric = "fscore", value = err))
    },
    
    dummyscore = function(preds, dtrain){
      labels <- getinfo(dtrain, "label")
      tp <- sum(labels == 1 & preds >= 0.5)
      
      return(list(metric = "dummyscore", value = tp))
    },
    
    conf_matrix = function(orig, preds){
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
    
  )
)