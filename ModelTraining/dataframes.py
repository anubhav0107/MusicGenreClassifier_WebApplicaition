import os
import numpy as np
import librosa as lb
import pandas as pd

df1 = pd.read_pickle('rock.pickle')
df2 = pd.read_pickle('jazz.pickle')
df3 = pd.read_pickle('disco.pickle')
df4 = pd.read_pickle('blues.pickle')
df5 = pd.read_pickle('reggae.pickle')
df6 = pd.read_pickle('classical.pickle')
df7 = pd.read_pickle('country.pickle')
df8 = pd.read_pickle('hiphop.pickle')
df9 = pd.read_pickle('metal.pickle')
df10 = pd.read_pickle('pop.pickle')

frames = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10]

df = pd.concat(frames)

df.to_pickle('songs1.pickle')
