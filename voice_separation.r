library(reticulate)
reticulate::use_python("~/.pyenv/versions/3.11.9/bin/python3.11", required = TRUE)

library(tensorflow)
library(keras3)

library(tuneR)
setWavPlayer('paplay')

source('utils.r')

print_message <- function(str) {
    cat("\n==================================\n", str, "\n\n")
}

get_dir_audio <- function(dirname) {
    audio_files = list.files(dirname, pattern = "\\.wav$", full.names = TRUE)
    audio_list = lapply(audio_files, readWave)
    combined_audio = do.call(bind, audio_list)

    cat("Read", length(audio_files), "files\n")
    
    return(combined_audio)
}

get_train_data <- function(a1, a2, N, H, bin_cutoff, cutoff_method='mean') {
    ## left channel only
    a1 = a1@left
    a2 = a2@left

    ## set audio to same length
    length(a1) = min(length(a1), length(a2))
    length(a2) = min(length(a1), length(a2))

    ## get stft
    A1 = stft(a1, H, N)
    A2 = stft(a2, H, N)

    cat("\nDim input:", dim(t(A1)), "\n")

    ## cutoff non-voice frequencies 
    A1 = A1[1:bin_cutoff,]
    A2 = A2[1:bin_cutoff,]
    
    ## get percentage density of each audio
    p = 1
    eps = 1e-8 # no divide by 0 error
    den = Mod(A1)^p + Mod(A2)^p + eps
    A1p = Mod(A1)^p / den
    A2p = Mod(A2)^p / den

    ## Build weight map to reduce impact of silence
    amps = Mod(A1 + A2)
    cutoff = switch(cutoff_method,
                    'mean'=mean(amps),
                    'median'=median(amps),
                    'none'=0)

    weights = matrix(1, nrow = nrow(A1p), ncol = ncol(A1p))
    ## weights = (amps^p / max(amps^p))^1  # scale weights by loudness of amplitude
    ## weights[amps < cutoff] = 0 # quiet coeffs have lower training impact (1%)
    weights = t(weights)
    ## A1p[amps < cutoff] = 0.5
    ## A2p[amps < cutoff] = 0.5
    
    ## prepare combined input audio
    x = a1 + a2
    X = stft(x, H, N)
    X = X[1:bin_cutoff,] # cutoff freq bins
    X = Mod(X)           # only amps
    X = X / max(X)       # normalize
    X = t(X)             # transpose

    ## prepare output separation percentages
    A1p = t(A1p)
    A2p = t(A2p)
    Y = array(0, dim = c(nrow(A1p), ncol(A1p), 2))
    Y[,,1] = A1p
    Y[,,2] = A2p

    ## train only on loud enough data
    filter = colMeans(amps) >= cutoff
    X = X[t(filter),]
    Y = Y[t(filter),,]
    weights = weights[filter,]

    cat("Dim train:", dim(X), "\n")
    ## cat("Dim output:", dim(Y), "\n")
    
    return(list(X, Y, weights, x, filter))
}

train_model <- function(x_train, y_train, weights, n_input, epochs, model_file='') {

    n_output = 2

    ## ## Dense neural net
    ## model <- keras_model_sequential() %>%
    ##     layer_dense(units = 512, activation = "relu", input_shape = c(n_input)) %>%
    ##     layer_dropout(0.3) %>%
    ##     layer_dense(units = 256, activation = "relu") %>%
    ##     layer_dropout(0.3) %>%
    ##     layer_dense(units = 128, activation = "relu") %>%
    ##     layer_dropout(0.2) %>%
    ##     layer_dense(units = n_input * n_output) %>%
    ##     layer_reshape(target_shape = c(n_input, n_output)) %>%
    ##     layer_activation_softmax(axis = -1)

    ## 1D CNN
    model <- keras_model_sequential() %>%
        layer_reshape(input_shape = c(n_input), target_shape = c(n_input, 1)) %>%
        layer_conv_1d(filters = 64, kernel_size = 3, activation = "relu", padding = "same") %>%
        layer_batch_normalization() %>%
        layer_dropout(0.3) %>%
        layer_conv_1d(filters = 128, kernel_size = 3, activation = "relu", padding = "same") %>%
        layer_batch_normalization() %>%
        layer_conv_1d(filters = 64, kernel_size = 3, activation = "relu", padding = "same") %>%
        layer_batch_normalization() %>%
        layer_dropout(0.3) %>%
        layer_conv_1d(filters = n_output, kernel_size = 1, padding = "same") %>%
        layer_reshape(target_shape = c(n_input, n_output)) %>%
        layer_activation_softmax(axis = -1)
    
    model %>% compile(
                  optimizer = optimizer_adam(learning_rate = 0.001),
                  loss = "categorical_crossentropy",  # or "sparse_categorical_crossentropy"
                  metrics = c("accuracy")
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
bin_cutoff = 30
n_input = N
n_output = 2

## read audio files
print_message("Reading audio files")
p1_train_audio = get_dir_audio('audio/matei-high/train/')
## p2_train_audio = get_dir_audio('audio/matei-low/train/')
## p2_train_audio = readWave('audio/random/crying.wav')
p2_train_audio = readWave('audio/random/talking.wav')
## p2_train_audio = get_dir_audio('audio/shulin/train/')

p1_test_audio = readWave('audio/matei-high/test/glunker_stew.wav')
## p2_test_audio = readWave('audio/random/crying.wav')
p2_test_audio = readWave('audio/random/talking.wav')
## p2_test_audio = readWave('audio/shulin/test/glunker_stew.wav')
## p2_test_audio = readWave('audio/matei-low/test/glunker_stew.wav')

## get sr and bits
sr = p1_train_audio@samp.rate
bits = p1_train_audio@bit

## get train data
print_message("Loading train data")
train_data = get_train_data(p1_train_audio, p2_train_audio, N, H, bin_cutoff, cutoff_method='mean')
x_train        = train_data[[1]]
y_train        = train_data[[2]]
weights        = train_data[[3]]
train_combined = train_data[[4]]

C = stft(train_combined, H, N)

## train/load model
print_message("Training model")
epochs = 1000
model_file = 'models/matei_high_talking_sep.keras'
## model = train_model(x_train, y_train, NULL, bin_cutoff, epochs, model_file)
model = load_model(model_file)

## load test data
print_message("Loading test data")
test_data = get_train_data(p1_test_audio, p2_test_audio, N, H, bin_cutoff, cutoff_method='none')
x_test        = test_data[[1]]
y_test        = test_data[[2]]
test_combined = test_data[[4]]
test_filter   = test_data[[5]]

## evaluate and predict from model
print_message("Evaluating model")
model %>% evaluate(x_test, y_test)
preds = model %>% predict(x_test)

## separate sources
print_message("Separating sources")
C = stft(test_combined, H, N)

A1p = matrix(0.5, bin_cutoff, ncol(C))
A2p = matrix(0.5, bin_cutoff, ncol(C))
A1ptest = matrix(0.5, bin_cutoff, ncol(C))
A2ptest = matrix(0.5, bin_cutoff, ncol(C))

A1p[,test_filter] = t(preds[,,1])
A2p[,test_filter] = t(preds[,,2])
A1ptest[,test_filter] = t(y_test[,,1])
A2ptest[,test_filter] = t(y_test[,,2])

Y1hat = matrix(0, nrow(C), ncol(C))
Y2hat = matrix(0, nrow(C), ncol(C))
Y1 = matrix(0, nrow(C), ncol(C))
Y2 = matrix(0, nrow(C), ncol(C))

Y1hat[1:bin_cutoff,] = C[1:bin_cutoff,] * A1p
Y2hat[1:bin_cutoff,] = C[1:bin_cutoff,] * A2p
Y1[1:bin_cutoff,] = C[1:bin_cutoff,] * A1ptest
Y2[1:bin_cutoff,] = C[1:bin_cutoff,] * A2ptest

y1hat = istft(Y1hat, H, N)
y2hat = istft(Y2hat, H, N)
y1 = istft(Y1, H, N)
y2 = istft(Y2, H, N)

## graph
## png(filename="demo/plots/matei-high-crying-predict-plot.png")
plot(rowMeans(t(y_test[,,1])), type='l', col='red4', ylim=c(0,1), lwd=2, ylab="Mean Mod Percent Contribution", xlab="Bin Number")
lines(rowMeans(t(preds[,,1])), col='red', lwd=2)
lines(rowMeans(t(y_test[,,2])), col='seagreen', lwd=2)
lines(rowMeans(t(preds[,,2])), col='seagreen1', lwd=2)
legend(x='topright',
       legend=c('Matei test', 'Matei predict', 'Talking test', 'Talking predict'),
       lty=c(1,1,1,1),
       col=c('red4', 'red', 'seagreen', 'seagreen1'),
       lwd=2)
## dev.off()

## play audios
print_message("Playing audio")
play(create_wav(test_combined, sr, bits))
## play(p1_test_audio)
play(create_wav(y1, sr, bits))
play(create_wav(y2, sr, bits))
play(create_wav(y1hat, sr, bits))
play(create_wav(y2hat, sr, bits))

## writeWave(create_wav(test_combined, sr, bits), "demo/audio/matei-high-crying-combined.wav")
## writeWave(create_wav(y1, sr, bits), "demo/audio/matei-high-cyring-sep-h-actual.wav")
## writeWave(create_wav(y2, sr, bits), "demo/audio/matei-high-cyring-sep-c-actual.wav")
## writeWave(create_wav(y1hat, sr, bits), "demo/audio/matei-high-cyring-sep-h-predict.wav")
## writeWave(create_wav(y2hat, sr, bits), "demo/audio/matei-high-cyring-sep-c-predict.wav")

## ## play(create_wav(a2, sr, bits))
## ## play(create_wav(y, sr, bits))
## ## play(create_wav(yhat, sr, bits))
