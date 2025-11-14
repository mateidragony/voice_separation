library(reticulate)
reticulate::use_python("~/.pyenv/versions/3.11.9/bin/python3.11", required = TRUE)

library(tensorflow)
library(keras3)

library(tuneR)
setWavPlayer('paplay')

source('utils.r')

get_train_data = function(a1, a2, N, H) {
    ## left channel only
    a1 = a1@left
    a2 = a2@left

    ## set audio to same length
    length(a1) = min(length(a1), length(a2))
    length(a2) = min(length(a1), length(a2))

    ## get stft
    A1 = stft(a1, H, N)
    A2 = stft(a2, H, N)

    ## get percentage density of each audio
    p = 1
    eps = 1e-8 # no divide by 0 error
    den = Mod(A1)^p + Mod(A2)^p + eps
    A1p = Mod(A1)^p / den
    A2p = Mod(A2)^p / den

    ## Build weight map to reduce impact of silence
    amps = Mod(A1 + A2)
    cutoff = mean(amps)
    weights = matrix(1, nrow = nrow(A1p), ncol = ncol(A1p))
    ## weights = (amps^p / max(amps^p))^1  # scale weights by loudness of amplitude
    weights[amps < cutoff] = 0 # quiet coeffs have lower training impact (1%)
    weights = t(weights)
    ## A1p[amps < cutoff] = 0.5
    ## A2p[amps < cutoff] = 0.5
    
    ## prepare combined input audio
    x = a1 + a2
    X = stft(x, H, N)
    X = Mod(X)        # only amps
    X = X / max(X)    # normalize
    X = t(X)          # transpose

    ## prepare output separation percentages
    A1p = t(A1p)
    A2p = t(A2p)
    Y = array(0, dim = c(nrow(A1p), ncol(A1p), 2))
    Y[,,1] = A1p
    Y[,,2] = A2p

    ## train only on loud enough data
    X = X[t(colMeans(amps) > cutoff),]    
    Y = Y[t(colMeans(amps) > cutoff),,]
    weights = weights[colMeans(amps) > cutoff,]    
    
    return(list(X, Y, weights, x))
}

train_model = function(x_train, y_train, weights, n_input, epochs, model_file='') {

    n_output = 2

    ## model = keras_model_sequential() %>%
    ##     layer_reshape(input_shape = c(n_input), target_shape = c(n_input, 1)) %>%
    ##     layer_conv_1d(filters = 32, kernel_size = 9, activation = "relu", padding = "same") %>%
    ##     layer_dropout(0.2) %>%
    ##     layer_conv_1d(filters = 32, kernel_size = 9, activation = "relu", padding = "same") %>% 
    ##     layer_dropout(0.2) %>%
    ##     layer_conv_1d(filters = n_output, kernel_size = 1, activation = "sigmoid", padding = "same")
    
    model = keras_model_sequential() %>%
        layer_reshape(input_shape = c(n_input), target_shape = c(n_input, 1)) %>%
        layer_conv_1d(filters = 32, kernel_size = 9, activation = "relu", padding = "same") %>%
        layer_dropout(0.3) %>%
        layer_conv_1d(filters = n_output, kernel_size = 1, activation = "softmax", padding = "same")
    
    ## model = keras_model_sequential() %>%
    ##     layer_dense(units = 512, activation = "relu", input_shape = c(n_input)) %>%
    ##     layer_dropout(0.3) %>%
    ##     layer_dense(units = n_input * n_output) %>%
    ##     layer_reshape(target_shape = c(n_input, n_output)) %>%
    ##     layer_activation("sigmoid")  # softmax over the 2 classes for each coeff

    ## model <- keras_model_sequential() %>%
    ##     layer_reshape(input_shape = c(n_input), target_shape = c(n_input, 1)) %>%
    ##     layer_conv_1d(filters = 64, kernel_size = 9, activation = "relu", padding = "same") %>%
    ##     layer_conv_1d(filters = 64, kernel_size = 9, activation = "relu", padding = "same") %>%
    ##     layer_dropout(0.3) %>%
    ##     layer_conv_1d(filters = n_output, kernel_size = 1, activation = "softmax", padding = "same")
    
    ## model = keras_model_sequential() %>%
    ##     layer_reshape(input_shape = c(n_input), target_shape = c(n_input, 1)) %>%
    ##     layer_conv_1d(filters = 32, kernel_size = 9, activation = "relu", padding = "same") %>%
    ##     layer_conv_1d(filters = 32, kernel_size = 9, activation = "relu", padding = "same") %>%
    ##     layer_dropout(0.4) %>%
    ##     layer_conv_1d(filters = n_output, kernel_size = 1, activation = "softmax", padding = "same")

    model %>% compile(
                  optimizer = optimizer_rmsprop(learning_rate = 0.0005),
                  loss = "binary_crossentropy", # mean squared error
                  metrics = c("accuracy") # mean absolute error
              )

    callback = callback_early_stopping(monitor = "val_accuracy",
                                        patience = 10,
                                        restore_best_weights = TRUE)
    
    history = model %>% fit(x_train,
                            y_train,
                            sample_weight = weights,
                            epochs = epochs,
                            batch_size = 64,
                            validation_split = 0.15,
                            callbacks = list(callback)
                            )

    if (model_file != '') {
        save_model(model, model_file, overwrite=TRUE)
    }

    return(model)
}


## constants
N = 1024
H = N/4
n_input = N
n_output = 2

## read audio files
p1_train_audio  = readWave('audio/matei/train/i547_notes.wav')
p1_test_audio  = readWave('audio/matei/test/glunker_stew.wav')
p2_train_audio = readWave('audio/shulin/train/i547_notes.wav')
p2_test_audio = readWave('audio/shulin/test/glunker_stew.wav')

## get sr and bits
sr = p1_train_audio@samp.rate
bits = p1_train_audio@bit

## get train data
train_data = get_train_data(p1_train_audio, p2_train_audio, N, H)
x_train        = train_data[[1]]
y_train        = train_data[[2]]
weights        = train_data[[3]]
train_combined = train_data[[4]]

C = stft(train_combined, H, N)
plot(colMeans(Mod(C)))
abline(h = mean(Mod(C)), col = 'coral2', lwd = 2)

## train/load model
epochs = 100
model_file = 'models/shulin_matei_sep.keras'
cat("Dim x train:", dim(x_train), "\n")
cat("Dim y train:", dim(y_train), "\n")
model = train_model(x_train, y_train, weights, N, epochs, model_file)
## model = load_model(model_file)

## load test data
test_data = get_train_data(p1_test_audio, p2_test_audio, N, H)
x_test        = test_data[[1]]
y_test        = test_data[[2]]
test_combined = test_data[[4]]

## evaluate and predict from model
model %>% evaluate(x_test, y_test)
preds = model %>% predict(x_test)

y_train[1000,,2]
preds[1000,,2]
## ## max(y_train[1000,,2])
## ## max(preds[1000,,2])

## ## separate sources
## Ap = t(preds[,,2])
## C = stft(test_combined, H, N)

## amps = Mod(C)
## cutoff = mean(amps)
## Ap[amps < cutoff] = 0.5
## ahat = istft(Ap * C, H, N)

## ## play pre recorded, combined, then separated
## ## print("Playing audio")
## ## play(shulin_test_audio)
## ## play(create_wav(test_combined, sr, bits))
## ## play(create_wav(ahat, sr, bits))


## ## play(create_wav(a2, sr, bits))
## ## play(create_wav(y, sr, bits))
## ## play(create_wav(yhat, sr, bits))
