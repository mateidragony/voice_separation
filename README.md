# 🗣️ Voice Source Separation

Source separation involves taking an audio file with multiple sources contributing sound and separating
that out into different audio files for each source.

I'm going to separate voices! ([I547](https://academics.iu.edu/courses/bloomington/info-i-547-music-information-processing-audio.html) final project btw)

## 🤔 How the hell?

Glad you asked! See raw audio files are lame and hard to analyze, but if you look at that audio file in
the frequency domain you can get a lot of information about that audio. 

Enter [Joseph Fourier](https://archiveofourown.org/works/1073009). Using a fourier transform you can enter
into the new frequency dimension and the short time fourier transform gives you little snapshots of said
domain over your entire audio.

Now that we have this portal under our belt we can << 🤓 train a model 🤓 >> to read that spectral frequency
information and label audio points as either person 1 or person 2 talking. Then we collect those points,
concatenate them and voila, source separated audio!

In theory.

## 😐 How far along are we?

Ummm... We have a model modeling something 🥳. Is is modeling what we want correctly?... Up to debate.

I'm hoping my issue is not enough audio training data, but we'll see. I know that I have faced problems
with filtering out quiet data from my training set but that seems to make my model implode so who knows.

## 🤕 What have we learned from this << ongoing >> ordeal?

R is great. But also awful sometimes. I LOVE all the vector jazz. So fun, but dynamic scope 👿...

Love hate relationship with machine learning. Yeah its cool I guess but I hate it so...


