##
## Voice detection
##
## First attempt of playing around with ml models analyzing audio.
## This model analyzes stft frames individually and categorizes them
## as either person 1 or person 2. This works great as a voice
## detection algorithm, but not so great as voice source separation.
##

library(reticulate)
reticulate::use_python("~/.pyenv/versions/3.11.9/bin/python3.11", required = TRUE)

library(tensorflow)
library(keras3)

library(tuneR)
setWavPlayer('paplay')

create_wav <- function(f, sample_rate=32000, bit_depth=16) {
    n <- f / max(f)
    u <- (2^(bit_depth - 2)) * n
    w <- Wave(round(u), samp.rate=sample_rate, bit=bit_depth)
    return(w)
}

stft = function(y,H,N) {
  v = seq(from=0,by=2*pi/N,length=N)     
  win = (1 + cos(v-pi))/2
  cols = floor((length(y)-N)/H) + 1
  stft = matrix(0,N,cols)
  for (t in 1:cols) {
    range = (1+(t-1)*H): ((t-1)*H + N)
    chunk  = y[range]
    stft[,t] = fft(chunk*win)
  } 
  ph = Arg(stft)
  for (k in 1:nrow(ph)) {
    ph[k,] = c(ph[k,1],diff(ph[k,]))
  }
  stft = matrix(complex(modulus = Mod(stft), argument = ph),nrow(stft),ncol(stft)) 
  return(stft)
}

istft = function(Y,H,N) {
  ph = Arg(Y)
  for (k in 1:nrow(Y)) {
    ph[k,] = cumsum(ph[k,])
  }
  Y = matrix(complex(modulus = Mod(Y), argument = ph),nrow(Y),ncol(Y)) 
  v = seq(from=0,by=2*pi/N,length=N)     
  win = (1 + cos(v-pi))/2
  y = rep(0,N + H*ncol(Y))
  for (t in 1:ncol(Y)) {
    chunk  = fft(Y[,t],inverse=T)/N
    range = (1+(t-1)*H): ((t-1)*H + N)
    y[range]  = y[range]  + win*Re(chunk)
  }
  return(y)
}

clean_voice_data = function(w, N, H, cutoff_method = "median") {
    bits = w@bit
    sr = w@samp.rate
    y = w@left
    Y = stft(y, H, N)

    amps = colMeans(Mod(Y)) / (N / 2)
    cutoff = switch(cutoff_method,
                    "mean"=mean(amps),
                    "median"=median(amps),
                    "none"=0)

    Ybar = matrix(0,nrow(Y),ncol(Y))
    cnt = 0

    for (i in 1:ncol(Y)) {
        if (amps[i] > cutoff) {
            cnt = cnt + 1
            Ybar[,cnt] = Y[,i]
        }
    }

    Ybar = Ybar[,1:cnt]
    ybar = istft(Ybar, H, N)
    
    X_train = t(Mod(Ybar))

    return(list(X_train, ybar))
}

N = 1024
H = N/4

n_input = N
n_output = 2

matei_train_audio  = readWave('audio/matei/train/glunker_stew.wav')
shulin_train_audio = readWave('audio/shulin/train/glunker_stew.wav')
matei_test_audio   = readWave('audio/matei/test/i547_notes.wav')
shulin_test_audio  = readWave('audio/shulin/test/i547_notes.wav')

cutoff_method = "median"

matei_train  = clean_voice_data(matei_train_audio, N, H, cutoff_method)[[1]]
shulin_train = clean_voice_data(shulin_train_audio, N, H, cutoff_method)[[1]]
x_train      = rbind(matei_train, shulin_train) 

matei_test  = clean_voice_data(matei_test_audio, N, H, cutoff_method)[[1]]
shulin_test = clean_voice_data(shulin_test_audio, N, H, cutoff_method)[[1]]
x_test      = rbind(matei_test, shulin_test)

y_train = c(rep(0, nrow(matei_train)), rep(1, nrow(shulin_train)))
y_train = to_categorical(y_train, num_classes = n_output)
y_test  = c(rep(0, nrow(matei_test)), rep(1, nrow(shulin_test)))
y_test  = to_categorical(y_test, num_classes = n_output)

## shuffle train data
perm <- sample(nrow(x_train))
x_train <- x_train[perm, ]
y_train <- y_train[perm, ]

## normalize vectors
x_train = x_train / max(abs(x_train))
x_test  = x_test  / max(abs(x_test))

epochs = 100
model_file = "models/voice_recognition.keras"
train = FALSE

if (train) {
    model = keras_model_sequential() %>%
        layer_dense(units = 64, activation="relu", input_shape=c(n_input)) %>%
        layer_dropout(0.5) %>%
        layer_dense(units = n_output, activation="softmax")
    
    ## model = keras_model_sequential() %>%
    ##     layer_dense(units = 256, activation = "relu", input_shape = c(n_input)) %>%
    ##     layer_dropout(rate = 0.25) %>% 
    ##     layer_dense(units = 128, activation = "relu") %>%
    ##     layer_dropout(rate = 0.25) %>% 
    ##     layer_dense(units = 64, activation = "relu") %>%
    ##     layer_dropout(rate = 0.25) %>%
    ##     layer_dense(units = n_output, activation = "softmax")

    model %>% compile(
                  loss = "categorical_crossentropy",
                  optimizer = optimizer_rmsprop(learning_rate = 0.0005),
                  metrics = c("accuracy")
              )

    callback <- callback_early_stopping(monitor = "val_accuracy",
                                        patience = 10,
                                        restore_best_weights = TRUE)

    history <- model %>%
        fit(x_train, y_train,
            epochs = epochs,
            batch_size = 64,
            validation_split = 0.15,
            callbacks = list(callback))

    save_model(model, model_file)
} else {
    model = load_model(model_file)
}

model %>% evaluate(x_test, y_test)


bits = matei_train_audio@bit
sr = matei_train_audio@samp.rate

a1 = matei_train_audio@left
a2 = shulin_train_audio@left

a1 = clean_voice_data(matei_train_audio, N, H, cutoff_method)[[2]]
a2 = clean_voice_data(shulin_train_audio, N, H, cutoff_method)[[2]]

y = c(a1, a2)

Y = stft(y, H, N)

X = t(Mod(Y))
X = X / max(abs(X))
perm <- sample(nrow(X))
X <- X[perm, ]

preds = model %>% predict(X)
classes = max.col(preds)  # gives 1 or 2 per frame

A1 = matrix(0, nrow = nrow(Y), ncol = ncol(Y))
A2 = matrix(0, nrow = nrow(Y), ncol = ncol(Y))

## logical masks for each class
mask1 = which(classes == 1)
mask2 = which(classes == 2)

## assign all frames in one go
A1[, mask1] = Y[, mask1]
A2[, mask2] = Y[, mask2]

A1 = A1[,1:length(mask1)]
A2 = A2[,1:length(mask2)]

print(length(mask1))
print(length(mask2))

a1hat = istft(A1, H, N)
a2hat = istft(A2, H, N)
