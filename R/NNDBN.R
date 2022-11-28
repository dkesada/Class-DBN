#' R6 class that defines the NN + DBN model
#'
#' The model is defined as a DBN model that forecasts the state of a time-series
#' based system and a neural network model that classifies that predicted state.
#' @export
NNDBN <- R6::R6Class("NNDBN",
  inherit = HDBN,
  public = list(
    #' @description
    #' Initialize the object with some modifiable parameters of the optimization
    #' @param lower lower bounds of the learning rate, epochs, batch size and recall
    #' @param upper upper bounds of the learning rate, epochs, batch size and recall
    #' @param itermax maximum number of iterations of the optimization process
    #' @param test_per percentage of instances assigned as test in the optimization
    #' @param optim_trace whether or not to print the progress of each optimization iteration
    initialize = function(lower = c(0.5, 20, 2.5, 0.5), upper = c(6.49, 100, 8.49, 0.75), 
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
    predict = function(dt_test, horizon = 1, print_res = T, conf_mat=F){
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
      
      preds <- predict(private$cl, scale(as.matrix(dt_test_mod), center = private$center, scale = private$scale)) > 0.5
      preds <- as.numeric(preds[,2])
      
      if(print_res){
        cat(paste0("Mean accuracy: ", mean(dt_test[, get(private$cl_obj_var)] == preds), "\n"))
        cat(paste0("F1score: ", f1score(dt_test[, get(private$cl_obj_var)], preds), "\n"))
        cat(paste0("G-mean score: ", g_mean(dt_test[, get(private$cl_obj_var)], preds), "\n"))
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
    predict_cl = function(dt_test, print_res = T, conf_mat=F){
      # --ICO-MERGE very similar to the XGBoost one, maybe include in HDBN and inherit?
      dt_test_mod <- copy(dt_test)
      dt_test_mod[, eval(private$cl_obj_var) := NULL]
      dt_test_mod[, eval(private$id_var) := NULL]
      
      preds <- predict(private$cl, scale(as.matrix(dt_test_mod), center = private$center, scale = private$scale)) > 0.5
      preds <- as.numeric(preds[,2])
      
      if(print_res){
        cat(paste0("Mean accuracy: ", mean(dt_test[, get(private$cl_obj_var)] == preds), "\n"))
        cat(paste0("F1score: ", f1score(dt_test[, get(private$cl_obj_var)], preds), "\n"))
        cat(paste0("G-mean score: ", g_mean(dt_test[, get(private$cl_obj_var)], preds), "\n"))
      }
      
      if(conf_mat)
        private$conf_matrix(dt_test[, get(private$cl_obj_var)], preds)
      
      return(preds)
    }
   
  ),
  
  private = list(
    #' @field formula the formula for the SVM model
    formula = NULL,
    #' @field center the mean of the variables for normalization
    center = NULL,
    #' @field scale the standard deviation of the variables for normalization
    scale = NULL,
    
    keras_structure = function(in_shape){
      model <- keras_model_sequential() %>%
        layer_dense(units = 64, activation = "relu", input_shape = in_shape) %>%
        layer_dense(units = 32, activation = "relu") %>%
        layer_dense(units = 2, activation = "softmax")
      
      return(model)
    },
    
    #' @description
    #' Fit the internal NN
    #' @param dt_train a data.table with the training dataset
    #' @param optim boolean that determines whether or not the NN parameters should be optimized
    #' @param cl_params vector with the parameters of the NN c(learn_rate, epochs, batch_size, recall)
    fit_cl = function(dt_train, optim, cl_params){
      obj_col <- dt_train[, get(private$cl_obj_var)]
      dt_train_red <- copy(dt_train)
      dt_train_red[, eval(private$id_var) := NULL]
      dt_train_red[, eval(private$cl_obj_var) := NULL]
      private$formula <- as.formula(paste0(private$cl_obj_var, " ~ ." ))
      
      # Optional shuffle
      dt_train_red <- shuffle_dt(dt_train_red)

      dt_scaled <- scale(as.matrix(dt_train_red))
      private$center <- attr(dt_scaled, "scaled:center")
      private$scale <- attr(dt_scaled, "scaled:scale")
      
      if(optim)
        cl_params <- private$optimize_cl(dt_train)$optim$bestmem
      
      if(is.null(cl_params))  # No optimization and no params provided by the user
        cl_params <- c(4, 50, 5, 0.5)
      
      private$cl_params <- cl_params
      
      # Internal structure was built manually, could be optimized
      private$cl <- private$keras_structure(dim(dt_train_red)[2])
      
      compile(private$cl, loss = "categorical_crossentropy", optimizer = optimizer_rmsprop(learning_rate = 10^-round(cl_params[1])),
              metrics = keras::metric_precision_at_recall(recall=cl_params[4]))
      
      history = fit(private$cl, dt_scaled, to_categorical(obj_col, num_classes = 2), 
                    epochs = round(cl_params[2]), batch_size = 2^round(cl_params[3]), validation_split = 0.2)
      
      plot(history)
    },
    
    # Four parameters to optimize: the learning rate, the epochs, the batch size and the min recall metric
    eval_cl = function(params, dt_train, dt_test, labels, eval_metric){
      print(params)
      
      dt_train_sc <- scale(as.matrix(dt_train), center = private$center, scale = private$scale)
      dt_test_sc <- scale(as.matrix(dt_train), center = private$center, scale = private$scale)
      
      cl <- private$keras_structure(dim(dt_train_sc)[2])
      
      compile(cl, loss = "categorical_crossentropy", optimizer = optimizer_rmsprop(learning_rate = 10^-round(params[1])),
              metrics = keras::metric_precision_at_recall(recall=params[4]))
      
      history = fit(cl, dt_train_sc, to_categorical(labels[test == 0, get(private$cl_obj_var)], num_classes = 2), 
                    epochs = round(params[2]), batch_size = 2^round(params[3]), validation_split = 0.2)
      
      preds <- predict(cl, dt_test_sc) > 0.5
      preds <- as.numeric(preds[,2])

      acc <- eval_metric(labels[test == 0, get(private$cl_obj_var)], preds) # Check this
      
      return(acc)
    }
    
  )
)