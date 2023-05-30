from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import mne
import numpy as np
import mne
import pandas as pd
import sys

import csv
from scipy.stats import skew, kurtosis
import pyeeg as p
from numpy import nan
import math

app = Flask(__name__,template_folder=r'C:/Users/HP/template')
def predict_edf(data):
    import numpy as np   
   
    #features what we are extracting
    # Variance and kurtosis
    # Skewness
    # Petrosian Fractal Dimension (PFD)
    # Hjorth parameters (mobility and complexity)
    # Power Spectral Density (PSD) with EEG bands (delta, theta, alpha, beta, and gamma)
    # Spectral Entropy
#-----------------FFeatures----------------------------------------------------------------------------------------
    def mean_variance(df):
        import numpy as np
        variance_vals = np.var(df)
        return np.mean(variance_vals)

    def mean_kurtosis(df):
        import numpy as np
        kurtosis_vals = kurtosis(df)
        return np.mean(kurtosis_vals)

    def mean_skewness(df):
        import numpy as np
        skew_vals = skew(df)
        return np.mean(skew_vals)
    #mean Higuchi's fractal dimension (PFD)
    def mean_pfd(df):
        pfd_vals = []
        for col in df.columns:
            col = df[col].to_numpy()
            pfd_val = p.pfd(col)
            pfd_vals.append(pfd_val)
        return np.mean(pfd_vals)
    #mean Hjorth mobility and complexity
    def mean_hjorth_mob_comp(df):
        mob_vals = []
        comp_vals = []
        for col in df.columns:
            col = df[col].to_numpy()
            mob_col, comp_col = p.hjorth(col)
            mob_vals.append(mob_col)
            comp_vals.append(comp_col)
        return np.mean(mob_vals), np.mean(comp_vals)

    def all_psd(data):
        fs = 256                                
        N = data.shape[1] # total num of points 

        # Get only in postive frequencies
        fft_vals = np.absolute(np.fft.rfft(data))

        n_rows = fft_vals.shape[0]
        n_cols = fft_vals.shape[1]
        psd_vals = np.zeros(shape=(n_rows, n_cols))

        for i in range(n_rows):
            for j in range(n_cols):
                psd_vals[i][j] = (N/fs) * fft_vals[i][j] * fft_vals[i][j];


        # Get frequencies for amplitudes in Hz
        fft_freq = np.fft.rfftfreq(data.shape[1], 1.0/fs)

        # Define EEG bands
        eeg_bands = {'Delta': (0, 4),
                    'Theta': (4, 8),
                    'Alpha': (8, 12),
                    'Beta': (12, 30),
                    'Gamma': (30, 45)}

        # Take the mean of the fft amplitude for each EEG band
        eeg_band_fft = dict()
        psd_vals_list = []
        for band in eeg_bands:  
            freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & 
                            (fft_freq <= eeg_bands[band][1]))[0]
            eeg_band_fft[band] = np.mean(psd_vals[:,freq_ix])
            psd_vals_list.append(eeg_band_fft[band] * 1000000)
        return psd_vals_list

    def sum_psd(data):
        psd_vals = all_psd(data)
        return np.sum(psd_vals)

    def mean_spectral_entropy(data):
        psd_vals = all_psd(data)
        power_ratio = []
        sum_psd_vals = sum_psd(data)
        for val in psd_vals:
            power_ratio.append(val/sum_psd_vals)
        bands = [0,4,8,12,30,45]
        Fs = 256
        spec_entropy = p.spectral_entropy(data, bands, Fs, power_ratio)
        return spec_entropy
#-------------------Add row to csv-------------------------------------------------------------------
    def add_row(df_input, psd_ip, start, end, index):
        row_to_add = []
        d = df_input[index:index + duration]
        psd_ip = psd_ip[:, start:end]
        psd_ip = psd_ip[:][0]
        
        mean_var = mean_variance(d)
        mean_k = mean_kurtosis(d)
        mean_skew = mean_skewness(d)
        pfd = mean_pfd(d)
        h_mob, h_comp = mean_hjorth_mob_comp(d)
        mean_spec = mean_spectral_entropy(psd_ip)
        
        row_to_add.append(mean_var)
        row_to_add.append(mean_k)
        row_to_add.append(mean_skew)
        row_to_add.append(pfd)
        row_to_add.append(h_mob)
        row_to_add.append(h_comp)
        row_to_add.append(mean_spec)
        #Label: 1 = seizure, 0 = non-seizure. Change before running.
        row_to_add.append(1)
        
        
        return row_to_add
#---------------------Filter-----------------------------------------------------------------------------
    #data = mne.io.read_raw_edf(file,preload = True)

    data = data.filter(l_freq = 4 , h_freq = 8)

    #data.plot()

    data = data.filter(l_freq = 8 , h_freq = 12)
    #data.plot()

    data = data.filter(l_freq = 0.5 , h_freq = 40)
    #data.plot()

#----------------Dropping-----------------------------------------------------------------------------------
    import pandas as pd
    #exclude =  Channel order : 'FP1-F7','F7-T7','T7-P7','P7-O1','FP1-F3','F3-C3','C3-P3','P3-O1','FP2-F4','F4-C4','C4-P4','P4-O2','FP2-F8','F8-T8','T8-P8','P8-O2','FZ-CZ','CZ-PZ','P7-T7','T7-FT9','FT9-FT10','FT10-T8','T8-P8'
    header = ','.join(data.ch_names)
    df = pd.DataFrame(data[:][0])
    df = df.transpose()
    # df = df.iloc[1467*256:1477*256]
    # df3

    #for 1 and 8
    df.drop([1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,20,21,22],axis=1,inplace=True)
    df.shape

    start = temp = 500*256 
    duration = 10*256
    end = 3500*256

    df1 = df.iloc[start:end, :]
    df1 = pd.DataFrame(df1)
    df1.shape

    #processes one seizure in 10s windows
    #adds rows with features extracted from these windows
    # print(df1)
    index = 0

    res = pd.DataFrame()
#-----------------Call add row and others---------------------------------------------------------------------------
    #first iteration run in 'w' mode, all subsequent iteration run in 'a' mode
    with open('EpilepticSeizuredetectionDatasetCHBMIT.csv', 'a') as file:
        writer = csv.writer(file)
        while temp < end:    
            row = add_row(df1, data, temp, temp + duration, index)
            #res=res.append(pd.Series(row),ignore_index=True)
            res = pd.concat([res, pd.Series(row)], ignore_index=True)
            writer.writerow(row)
            temp += duration
            index += duration

    #res.columns = ['Variance', 'Kurtosis', 'Skewness', 'Petrosian Fractal Dimension', 'Hjorth Mobility', 'Hjorth Complexity', 'Spectral Entropy', 'Label']
    res

    
#--------------CSV----------------------------------------------------------------------
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    d = pd.read_csv("EpilepticSezuredetectionDatasetCHBMIT.csv", index_col=0)

    X = d.iloc[:, :-1].values
    y = d.iloc[:, -1].values

    def acc_metric(y,pred):
        df = pd.DataFrame()
        df['test']=y
        df['pred']=pred
        df['sum']=df[['test','pred']].sum(axis=1)
        s =0
        total = 0
        for i in df.index:
            if (df['test'][i]==1):
                total+=1
            if (df['sum'][i]==2):
                s+=1
        return (s/total)

    pd.DataFrame(X)

    import time
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 6)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
#---------------SVM-------------------------------------------
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    from sklearn.svm import SVC

    #SVM without GridSearch
    classifier = SVC()
    start = time.time() #elapsed time
    classifier.fit(X_train, y_train) #train
    stop = time.time()
    print(f"Training time: {stop - start}s")


    #SVM with GridSearch
    params = {'kernel':('linear','rbf','sigmoid'),'C':[1,5,10,20]}#Radial Basis Function
    svc = SVC()
    svm_gs = GridSearchCV(svc,params,cv=10)
    svm_gs.fit(X_train,y_train)#train
    print(svm_gs.best_params_)

    classifier2 = SVC(kernel = 'rbf', random_state = 6,C=20)
    start = time.time()
    classifier2.fit(X_train, y_train)
    stop = time.time()
    print(f"test time: {stop - start}s")
    #SVM
    start = time.time()
    y_pred = classifier.predict(X_test)#test
    stop = time.time()
    print(f"test time: {stop - start}s")
    #SVM after GridSearch
    start = time.time()
    y_pred2 = classifier2.predict(X_test)
    stop = time.time()
    print(f"test time: {stop - start}s")
    #SVM Confusion Matrix
    from sklearn.metrics import confusion_matrix, accuracy_score,cohen_kappa_score
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    accuracy_score(y_test, y_pred)
    acc_metric(y_test,y_pred)
    cohen_kappa_score(y_test,y_pred)
    
    #SVM GS Confusion Matrix
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_test, y_pred2)
    print(cm)
    accuracy=accuracy_score(y_test, y_pred2)##
    acc_metric(y_test,y_pred2)
    cohen_kappa_score(y_test,y_pred2)





     #lstm
    
    import tensorflow as tf

    from sklearn.metrics import cohen_kappa_score
    import time

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from tensorflow import keras
    from keras.layers import Dropout

    d = pd.read_csv("EpilepticSezuredetectionDatasetCHBMIT.csv", index_col=0)


    X = d.iloc[:, :-1].values
    y = d.iloc[:, -1].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 6)


    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    
    X_train

    X_train = X_train.reshape(X_train.shape[0],6 , 1)
    X_test = X_test.reshape(X_test.shape[0],6 , 1)

    y_train
    lstm = keras.Sequential()
        

    lstm.add(keras.layers.LSTM(20,batch_input_shape = (None,6,1), return_sequences=False, recurrent_activation='relu'))
        

    lstm.add(keras.layers.Dense(3,activation="relu"))
        

    lstm.add(keras.layers.Dense(1,activation="sigmoid"))
        

    lstm.compile(optimizer='adam',loss='binary_crossentropy',metrics=['TruePositives','TrueNegatives','FalsePositives','FalseNegatives','accuracy'])

    tf.random.set_seed(69)

    lstm.summary()
    start = time.time()
    his = lstm.fit(X_train, y_train, epochs=100, validation_data=(X_test,y_test))
    stop = time.time()
    print(f"Training time: {stop - start}s")
    start = time.time()
    pred = lstm.predict(X_test)
    stop = time.time()
    print(f"Training time: {stop - start}s")



    for i in range(len(pred)):
        if pred[i] <0.5:
            pred[i]=0
        else:
            pred[i]=1
        

    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_test, pred)
    print(cm)
    accuracy1=accuracy_score(y_test, pred)





    #rnn
    import tensorflow as tf

    from sklearn.metrics import cohen_kappa_score
    import time

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from tensorflow import keras
    from keras.layers import Dropout

    d = pd.read_csv("EpilepticSezuredetectionDatasetCHBMIT.csv", index_col=0)


    X = d.iloc[:, :-1].values
    y = d.iloc[:, -1].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 6)


    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    
    X_train

    X_train = X_train.reshape(X_train.shape[0],6 , 1)
    X_test = X_test.reshape(X_test.shape[0],6 , 1)

    y_train
    rnn = keras.Sequential()

    rnn.add(keras.layers.SimpleRNN(20, input_shape=(6, 1), activation='relu', return_sequences=False))

    rnn.add(keras.layers.Dense(3, activation='relu'))

    rnn.add(keras.layers.Dense(1, activation='sigmoid'))

    rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['TruePositives', 'TrueNegatives', 'FalsePositives', 'FalseNegatives', 'accuracy'])
    rnn.summary()
    start = time.time()
    his = rnn.fit(X_train, y_train, epochs=100, validation_data=(X_test,y_test))
    stop = time.time()
    print(f"Training time: {stop - start}s")
    start = time.time()
    pred = rnn.predict(X_test)
    stop = time.time()
    print(f"Training time: {stop - start}s")



    for i in range(len(pred)):
        if pred[i] <0.5:
            pred[i]=0
        else:
            pred[i]=1
        

    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_test, pred)
    print(cm)
    accuracy2=accuracy_score(y_test, pred)


    #cnn-rnn

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from keras.models import Sequential
    from keras.layers import Dense, Conv1D, MaxPooling1D, LSTM, Dropout
    from keras.callbacks import EarlyStopping

    # Load data
    data = pd.read_csv("EpilepticSeizuredetectionDatasetCHBMIT.csv")

    # Split data into input features and labels
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Split data into training, validation, and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Preprocess data
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # Reshape data for CNN input
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Build CNN-RNN model
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units=32, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))

    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train model
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stop])

    # Evaluate model
    loss, accuracy3 = model.evaluate(X_test, y_test)
    

    # Make predictions
    y_pred = model.predict(X_test)



    # Return the prediction
    return accuracy,accuracy1,accuracy2,accuracy3




@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file
    file = request.files['file']
    file_path = r'C:/Users/HP/Downloads/dataset-20230409T104440Z-001/dataset/' + file.filename
    file.save(file_path)
    # Get the prediction
    data = mne.io.read_raw_edf(file_path, preload=True)
    accuracy,accuracy1,accuracy2,accuracy3 = predict_edf(data)
     
    # Return the prediction as JSON
    #return jsonify({'prediction': prediction.tolist()})
     # Convert prediction to list if it's not already a NumPy array
    #prediction_list = prediction.tolist()

    # Return the prediction as JSON
    #return jsonify({ 'accuracy': accuracy})
    return render_template('result.html',accuracy=accuracy,accuracy1=accuracy1,accuracy2=accuracy2,accuracy3=accuracy3)
@app.route('/')
def index():
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)




