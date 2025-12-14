library(tuneR)			
library(ggplot2)

source('utils.r')

setWavPlayer("paplay")			# R needs to know what utility to use to play a wave file

spectrogram = function(y,N) {
  frames = floor(length(y)/N)             # number of "frames" (like in movie)
  num_freqs = N/2
  
  spect = matrix(0,frames,num_freqs)	          # initialize frames x N/2 spectrogram matrix to  0
					  # N/2 is # of freqs we compute in fft (as usual)
  v = seq(from=0,by=2*pi/N,length=N)      # N evenly spaced pts 0 -- 2*pi
  win = (1 + cos(v-pi))/2		  # Our Hann window --- could use something else (or nothing)
  for (t in 1:frames) {
    chunk  = y[(1+(t-1)*N):(t*N)]         # the  frame t of  audio data
    Y = fft(chunk*win)
#    Y = fft(chunk)
    spect[t,] = Mod(Y[1:num_freqs]) 
#    spect[t,] = Arg(Y[1:(N/2)]) 
#    spect[t,] = log(1+Mod(Y[1:(N/2)])/1000)  # log(1 + x/1000) transformation just changes contrast
  }
  return(spect)
}

norm01 <- function(x) {
  x <- x - min(x)
  x / max(x + 1e-12)
}

N = 2048

w1 = readWave("audio/matei-high/test/glunker_stew.wav")
w2 = readWave("audio/matei-low/test/glunker_stew.wav")
y1 = w1@left
y2 = w2@left

bits = w1@bit
sr = w1@samp.rate

length(y1) = min(length(y1), length(y2))
length(y2) = min(length(y1), length(y2))

spect1 = spectrogram(y1,N)
spect2 = spectrogram(y2,N)

s1 = norm01(spect1^0.5)
s2 = norm01(spect2^0.5)

freq_hz <- (0:(ncol(s1) - 1)) * sr / N

df <- expand.grid(
  time = seq_len(nrow(s1)),
  k    = seq_len(ncol(s1))
)

df$freq_hz <- freq_hz[df$k]

m = max(as.vector(s1) + as.vector(s2))

df$red   <- (as.vector(s1) + as.vector(s2)) / m
df$green <- (as.vector(s1) + as.vector(s2)) / m
df$blue  <- (as.vector(s1) + as.vector(s2)) / m

df$col <- rgb(
  red   = df$red,
  green = df$green,
  blue  = df$blue
)

## X11(width = 20, height = 6)
p = ggplot(df, aes(time, freq_hz, fill = col)) +
    geom_raster() +
    scale_fill_identity() +
    coord_cartesian(ylim = c(0, 5000)) +
    labs(x = "Time frame", y = "Frequency (Hz)") +
    theme_minimal()
ggsave("demo/plots/matei-high-low-spectrogram-bw.png", plot = p, width = 20, height = 6)
