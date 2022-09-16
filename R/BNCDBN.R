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
    initialize = function(lower = c(1.5, 0, 0), upper = c(10, 5, 5), 
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
      
      preds_net <- f_dt_test[, private$predict_row(.SD, horizon), by = seq_len(nrow(f_dt_test))]
      preds_net[, seq_len := NULL]
      preds_net <- setnames(preds_net, old = names(preds_net), new = private$dbn_obj_vars_raw)
      dt_test_mod <- copy(dt_test)
      dt_test_mod[orig_rows, eval(names(preds_net)) := preds_net,]
      dt_test_mod[, eval(private$cl_obj_var) := NULL]
      dt_test_mod[, eval(private$id_var) := NULL]
      dt_test_mod <- private$discretize_dt(dt_test_mod)
      
      preds <- predict(private$cl, dt_test_mod)
      
      if(print_res)
        cat(paste0("Mean accuracy: ", mean(dt_test[, get(private$cl_obj_var)] == preds), "\n"))
      
      if(conf_mat)
        private$conf_matrix(dt_test[, get(private$cl_obj_var)], preds)
      
      return(preds)
    },
    
    #' @description
    #' Predict the objective variable in all the rows in a dataset using only
    #' the Bayseian network classifier. 
    #' @param dt_test a data.table with the test dataset
    #' @param print_res a boolean that determines whether or not should the results of the prediction be printed
    #' @param conf_mat a boolean that determines whether or not should a confusion matrix be printed
    #' @return the prediction result vector
    predict_cl = function(dt_test, print_res = T, conf_mat=F){
      dt_test_mod <- copy(dt_test)
      dt_test_mod[, eval(private$id_var) := NULL]
      dt_test_mod <- private$discretize_dt(dt_test_mod)
      preds <- predict(private$cl, dt_test_mod)
      
      if(print_res)
        cat(paste0("Mean accuracy: ", mean(dt_test[, get(private$cl_obj_var)] == preds), "\n"))
      
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
                                    .SDcols = c("Edad", "IMC", private$dbn_obj_vars_raw)]
      dt_train_red <- private$discretize_dt(dt_train_red)
      
      if(is.null(cl_params))
        cl_params <- c(0, 0, 0, 0)
      
      if(optim & (cl_params[1] > 1)){
        private$cl_params <- cl_params # I need to know my model type cl_params[1] inside the optimization process
        cl_params <- private$optimize_cl(dt_train)$optim$bestmem
        cl_params <- c(private$cl_params[1], cl_params)
      }
      
      model <-private$build_bnc(dt_train_red, cl_params)
      private$cl <- lp(model, dt_train_red, cl_params[4]) 
      private$cl_params <- cl_params
    },
    
    #' @description
    #' Optimize the parameters of the internal classifier. 
    #' Slight changes with respect to the inherited function
    #' @param dt a data.table with the training dataset
    optimize_cl = function(dt){
      test_id <- sample(unique(dt[, get(private$id_var)]), 
                        length(unique(dt[, get(private$id_var)])) * private$optim_test_per)
      dt_train <- dt[!(get(private$id_var) %in% test_id)]
      dt_test <- dt[get(private$id_var) %in% test_id]
      labels <- dt[, .SD, .SDcols = c(private$id_var, private$cl_obj_var)]
      labels[, test := 0]
      labels[get(private$id_var) %in% test_id, test := 1]
      dt_train[, eval(private$id_var) := NULL]
      dt_test[, eval(private$id_var) := NULL]
      dt_train <- private$discretize_dt(dt_train)
      dt_test <- private$discretize_dt(dt_test)
      
      res <- DEoptim::DEoptim(fn = private$eval_cl, lower = private$optim_lower, upper = private$optim_upper,
                              control = DEoptim::DEoptim.control(itermax = private$optim_itermax, trace = private$optim_trace),
                              dt_train, dt_test, private$fscore)
    
      return(res)
    },
    
    # One to three parameters to optimize: the number of folds, the epsilon and the smooth
    eval_cl = function(params, dt_train, dt_test, eval_metric){
      print(params)

      params[1] <- round(params[1])  # Number of folds
      model <-private$build_bnc(dt_train, c(private$cl_params[1], params))
      cl <- lp(model, dt_train, params[3]) 
      
      preds <- predict(cl, dt_test)
      acc <- mean(dt_test[, get(private$cl_obj_var)] != preds)
      
      return(acc)
    },
    
    # Builds the appropriate Bayesian netwock classifier from the bnclassify package: 0-nb, 1-tan_cl, 2-tan_hc, 3-tan_hcsp
    build_bnc = function(dt, cl_params){
      res <- bnclassify::nb(private$cl_obj_var, dt)
      if(cl_params[1] == 1)
        res <- bnclassify::tan_cl(private$cl_obj_var, dt)
      else if(cl_params[1] == 2)
        res <- bnclassify::tan_hc(private$cl_obj_var, dt, k = cl_params[2], epsilon = cl_params[3], smooth = cl_params[4])
      else if (cl_params[1] == 3)
        res <- bnclassify::tan_hcsp(private$cl_obj_var, dt, k = cl_params[2], epsilon = cl_params[3], smooth = cl_params[4])
      
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