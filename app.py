from flask import Flask, render_template, url_for, request, flash, redirect
import pandas as pd
import numpy as np
import librosa as lb
from sklearn.externals import joblib
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
file1 = ''
Songs = os.path.basename('uploads')
app.config['Songs'] = Songs

@app.route('/')
def index():
    return render_template('page1.html')

@app.route('/upload', methods = ['GET', 'POST'])
def upload():
   if request.method == 'POST':
        file = request.files['File']
        f = os.path.join(app.config['Songs'], file.filename)
        file.save(f)
        global file1 
        file1 = file.filename
        return render_template('page2.html')
    
@app.route('/predict', methods = ['POST'])
def predict():
    
    data_folder = r"E:\music genre classification\Flask music genre classifier\uploads"

    file_to_open = data_folder + "\\" 
    file_to_open = file_to_open + file1

    y1, sr1 = lb.load(file_to_open, sr = 22050, duration = 10)
    
    chroma_stft = lb.feature.chroma_stft(y = y1, sr=22050)
    spec_cent = lb.feature.spectral_centroid(y = y1, sr=22050)
    spec_bw = lb.feature.spectral_bandwidth(y = y1, sr=22050)
    rolloff = lb.feature.spectral_rolloff(y = y1, sr=22050)
    zcr = lb.feature.zero_crossing_rate(y1)
    m = lb.feature.mfcc(y= y1, sr = 22050, n_mfcc = 13)
    MFCC = []
    for e in m:
        MFCC.append(np.mean(e))
    MFCC.append(np.mean(chroma_stft))
    MFCC.append(np.mean(spec_cent))
    MFCC.append(np.mean(spec_bw))
    MFCC.append(np.mean(rolloff))
    MFCC.append(np.mean(zcr))
    
    MFCC = np.reshape(MFCC,(1,18))
    
    knn_model = open("models/knnjoblib.pkl","rb")
    knn = joblib.load(knn_model)
    
    #MFCCdata = np.reshape(MFCC,(1,-1))
    
    my_prediction = knn.predict(MFCC)
    
    print(my_prediction)
    
    return render_template('result.html',prediction = my_prediction)

if __name__=='__main__':
    app.run(debug=True)
    