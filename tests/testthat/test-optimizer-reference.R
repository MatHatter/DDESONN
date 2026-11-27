# Independent 60-digit Python oracle, computed from full gradient histories.
# This checks optimizer arithmetic, NOT prediction accuracy or learning-rate tuning.
optimizer_oracle <- read.csv(test_path("fixtures", "optimizer-reference.csv"), na.strings = "")
for (method in ddesonn_optimizer_options()) local({
  optimizer <- method
  test_that(paste(optimizer, "matches independent 40-step trajectories and state"), {
    oracle <- optimizer_oracle[optimizer_oracle$method == optimizer, ]
    for (case in unique(oracle$case)) for (target in c("weights", "biases")) {
      rows <- oracle[oracle$case == case, ]
      observed <- reference <- vector("list", 40L)
      start <- rows[rows$step==1L,]
      shape <- if (case=="zero_scalar") c(1L,1L) else if (target=="weights") c(2L,3L) else c(1L,6L)
      model <- new.env(parent=emptyenv())
      model[[target]] <- matrix(start$initial,shape[1L],shape[2L])
      state <- DDESONN:::initialize_optimizer_params(optimizer,shape,3L,1L,parameter=model[[target]])
      for (step in 1:40) {
        expected <- rows[rows$step==step,]
        g <- matrix(expected$gradient,shape[1L],shape[2L])
        # Epoch deliberately differs from the per-parameter update count.
        result <- DDESONN:::apply_optimizer_update(optimizer,list(state),g,
          expected$lr[1L],.8,.9,1e-6,999L,model,1L,target)
        state <- result$updated_optimizer_params
        # Keep each step and state field separate for tolerance checks and diagnostics.
        values <- list(parameter = as.numeric(model[[target]]), shape = dim(model[[target]]),
          state_parameter = state$param, counter = state$t)
        want <- list(parameter = expected$parameter, shape = shape,
          state_parameter = model[[target]], counter = step)
        for (nm in c("m","v","r","momentum","z","n","slow")) {
          if (all(is.na(expected[[nm]]))) next
          actual <- if (nm=="slow") state$slow_weights else state[[nm]][[1L]]
          values[[nm]] <- list(value = as.numeric(actual), shape = dim(actual))
          want[[nm]] <- list(value = expected[[nm]], shape = shape)
        }
        if (optimizer == "lookahead") {
          values$lookahead_counter <- state$lookahead_counter
          want$lookahead_counter <- step %% 3L
        }
        observed[[step]] <- values
        reference[[step]] <- want
        # Resume from serialized state midway; following updates must still match.
        if (step==19L) state <- unserialize(serialize(state,NULL))
      }
      expect_equal(observed, reference, tolerance = 1e-10, info = paste(case, target))
    }
  })
})
