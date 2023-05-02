import pandas as pd
import librosa as lb
import numpy as np

dataset = pd.read_pickle('songs1.pickle')

X = dataset.values.tolist()

MFCCdata = []

for i in range(1000):
    chroma_stft = lb.feature.chroma_stft(y = np.asarray(X[i]), sr=22050)
    spec_cent = lb.feature.spectral_centroid(y = np.asarray(X[i]), sr=22050)
    spec_bw = lb.feature.spectral_bandwidth(y = np.asarray(X[i]), sr=22050)
    rolloff = lb.feature.spectral_rolloff(y = np.asarray(X[i]), sr=22050)
    zcr = lb.feature.zero_crossing_rate(np.asarray(X[i]))
    m = lb.feature.mfcc(y= np.asarray(X[i]), sr = 22050, n_mfcc = 13)
    tempMFCC = []
    for e in m:
        tempMFCC.append(np.mean(e))
    tempMFCC.append(np.mean(chroma_stft))
    tempMFCC.append(np.mean(spec_cent))
    tempMFCC.append(np.mean(spec_bw))
    tempMFCC.append(np.mean(rolloff))
    tempMFCC.append(np.mean(zcr))
    MFCCdata.append(tempMFCC)

MFCCdata = np.reshape(MFCCdata,(1000,19))

df = pd.DataFrame(MFCCdata)

df.to_csv('MFCCValues31.csv', index = False, header = False)

df.to_pickle('MFCCValues31.pickle')
