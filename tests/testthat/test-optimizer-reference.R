# Independent 60-digit Python oracle, computed from full gradient histories.
# This checks optimizer arithmetic, NOT prediction accuracy or learning-rate tuning.
for (method in ddesonn_optimizer_options()) local({
  optimizer <- method
  test_that(paste(optimizer, "matches independent 40-step trajectories and state"), {
    oracle <- read.csv(test_path("fixtures", "optimizer-reference.csv"), na.strings="")
    for (case in unique(oracle$case)) for (target in c("weights", "biases")) {
      rows <- oracle[oracle$method==optimizer & oracle$case==case,]
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
        expect_equal(as.numeric(model[[target]]),expected$parameter,tolerance=1e-10)
        expect_identical(dim(model[[target]]),shape)
        expect_equal(state$param,model[[target]],tolerance=1e-10)
        expect_equal(state$t,step)
        for (nm in c("m","v","r","momentum","z","n","slow")) {
          if (all(is.na(expected[[nm]]))) next
          actual <- if (nm=="slow") state$slow_weights else state[[nm]][[1L]]
          expect_equal(as.numeric(actual),expected[[nm]],tolerance=1e-10,info=nm)
          expect_identical(dim(actual),shape)
        }
        if (optimizer=="lookahead") expect_equal(state$lookahead_counter,step%%3L)
        # Resume from serialized state midway; following updates must still match.
        if (step==19L) state <- unserialize(serialize(state,NULL))
      }
    }
  })
})
