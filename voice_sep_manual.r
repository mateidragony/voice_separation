library(tuneR)			
library(ggplot2)

source('utils.r')

setWavPlayer("paplay")			# R needs to know what utility to use to play a wave file


get_train_data = function(a1, a2, N, H, bin_cutoff, cutoff_method='mean') {
    ## left channel only
    a1 = a1@left
    a2 = a2@left

    ## set audio to same length
    length(a1) = min(length(a1), length(a2))
    length(a2) = min(length(a1), length(a2))

    ## get stft
    A1 = stft(a1, H, N)
    A2 = stft(a2, H, N) * 0

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
    A1p[amps < cutoff] = 0.5
    A2p[amps < cutoff] = 0.5
    
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
    
    return(list(X, Y, weights, x, filter))
}

N = 1024
H = N/4

w2 = readWave("audio/matei-high/test/glunker_stew.wav")
w1 = readWave("audio/matei-low/test/glunker_stew.wav")
y1 = w1@left
y2 = w2@left

bits = w1@bit
sr = w1@samp.rate

length(y1) = min(length(y1), length(y2))
length(y2) = min(length(y1), length(y2))

Y1 = stft(y1, H, N)
Y2 = stft(y2, H, N)

A1 = Mod(Y1)
A2 = Mod(Y2)

bin_cutoff = 50
plot(rowSums(A1)[1:bin_cutoff], type='l', col='red', ylab="Sum Mod", xlab="Bin Number")
lines(rowSums(A2)[1:bin_cutoff], col='green')

C = Y1 + Y2 * 0

## Option 1
p = 1
eps = 1e-8 # no divide by 0 error
den = A1^p + A2^p + eps
Y1p = A1^p / den
Y2p = A2^p / den

Y1hat = C * Y1p
Y2hat = C * Y2p
for (i in bin_cutoff:N) {
    Y1hat[i,] = 0
    Y2hat[i,] = 0
}

## Option 2
A1 = A1[1:bin_cutoff,]
A2 = A2[1:bin_cutoff,]
amps = A1 + A2
cutoff = mean(amps) * 0
filter = colMeans(amps) >= cutoff

png(filename="model-out-ex-plot.png")
plot(rowMeans(Y1p[1:bin_cutoff,filter]), type='l', col='red', ylim=c(0, 1), lwd=2, ylab="Mean Mod Percent Contribution", xlab="Bin Number")
lines(rowMeans(Y2p[1:bin_cutoff,filter]), col='seagreen1', lwd=2)
legend(x='topright',
       legend=c('Speaker 1', 'Speaker 2'),
       lty=c(1,1),
       col=c('red', 'seagreen1'),
       lwd=2)
dev.off()

A1p = matrix(0.5, bin_cutoff, ncol(C))
A1p[,filter] = Y1p[1:bin_cutoff, filter]
Y1hat2 = matrix(0, nrow(C), ncol(C))
Y1hat2[1:bin_cutoff,] = C[1:bin_cutoff,] * A1p

## Option 3
test_data = get_train_data(w1, w2, N, H, bin_cutoff, cutoff_method='median')
x_test        = test_data[[1]]
y_test        = test_data[[2]]
test_combined = test_data[[4]]
test_filter   = test_data[[5]]

A1p = matrix(0.5, bin_cutoff, ncol(C))
A1p[,test_filter] = t(y_test[,,1])
## ## A1p[,test_filter] = Y1p[1:bin_cutoff,filter]

Y1hat3 = matrix(0, nrow(C), ncol(C))
Y1hat3[1:bin_cutoff,] = C[1:bin_cutoff,] * A1p


y1hat = istft(Y1hat, H, N)
y1hat2 = istft(Y1hat2, H, N)
y1hat3 = istft(Y1hat3, H, N)


## play(create_wav(y1+y2, sr, bits))
## play(create_wav(y1, sr, bits))
## ## play(create_wav(y1hat, sr, bits))
## play(create_wav(y1hat3, sr, bits))

## writeWave(create_wav(y1hat3, sr, bits), "demo/audio/matei-low-train-clean.wav")

