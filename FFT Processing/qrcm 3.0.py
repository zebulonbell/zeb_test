# Deeper dive into featurization methods for GA vibration data
# We ultimately want something predective . . . where we can see start to move into "scary" land
# We'll compare blunt frequency analysis methods (band pass energy aggregation) and PCA




import pandas as pd
from numpy.fft import fft, fftshift
from scipy.signal import welch
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#from plotnine import *
from sklearn.cluster import KMeans, AgglomerativeClustering

# %matplotlib notebook
import matplotlib
matplotlib.rcParams['figure.figsize'] = [6, 2]
from random import randint

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# %matplotlib notebook
import matplotlib
matplotlib.rcParams['figure.figsize'] = [6, 2]
from random import randint

from matplotlib import animation



def read_ga_file(file_path):
    df = pd.read_csv(file_path, header=None)
    ncol = df.shape[1]
    start_cols = ["id", 'path', "something", "axis", "unix_time", "some_number", "number_samples"] 
    end_cols = range(0, ncol-len(start_cols))
    cols = start_cols + list(end_cols)
    df.columns = cols
    df['datetime'] = pd.to_datetime(df['unix_time'], unit='ms')
    return df



def compute_spectra_from_df(df):
    nums = np.arange(0, 1650)
    S = df[nums].values
    X = np.zeros((S.shape[0], 413))
    for ii in range(S.shape[0]):
        Ft, Xt = welch(S[ii, :], nfft=825)
        X[ii, :] = Xt
    return Ft, X




def compute_spectra_from_df_parseval(df):
    
    fs = 4096
    num_samples=df['number_samples'].iloc[0]
    nums = list(range(num_samples))
    S = df[nums].values
    
    Ft = np.linspace(-fs/2, fs/2, num_samples)
    X = np.zeros((S.shape[0], num_samples))
    for ii in range(S.shape[0]):
        Xt = fftshift(fft(S[ii, :]))
        Xt = Xt * np.conj(Xt)
        Xt = np.real(Xt) / len(Xt)
        X[ii, :] = Xt
        #print(len(X))
        #print(np.sum(Xt))
        #print(np.sum(S[ii, :]**2))
        #print()
    return Ft, X, num_samples




#data_files_path = r"C:\MyData\GP\Waveform"
#results_path = r"C:\MyData\GP\FFT"
#data_files_path = r"D:\Projects\GP QRCM\Waveform Data"
data_files_path = r"P:\Open Projects\Georgia Pacific\Open Projects\3022\Vibration Data\Vibration 7.21\GA and JD"
results_path = r"C:\Users\Zebulon.Bell\OneDrive - Pinnacle\Desktop\python test recieve"

os.chdir(data_files_path)
current_path = os.getcwd()
print(f"Current Path is set to {current_path}")

df_final = pd.DataFrame(columns = ['datetime','date','axis','path']) #zeb

for root, dirs, files in os.walk(data_files_path):
    #print(root)
    for file in files:
        file_path = os.path.join(root, file)
        #print(f"Root is {root}")
        full_results_path = root.replace(data_files_path, results_path)
        #print(f"FFT Results path is {full_results_path}")
        os.makedirs(full_results_path, exist_ok = True)
        
        # Get file extension
        #print(os.path.splitext(file_path)[1])
        
        
        print(f"\t{file_path}")
        
        output_file_path = os.path.join(full_results_path, file)
            
        df = read_ga_file(file_path)
        print(df.shape)
        
        Ft, X, num_samples = compute_spectra_from_df_parseval(df)
        freq_dft = pd.DataFrame(X)
        freq_dft["axis"] = df["axis"].values
        freq_dft["path"] = df["path"].values
        freq_dft["datetime"] = df["datetime"].values

        # Need this to translate integer dataframe columns to frequencies while not messing up the non-integer ones
        in_cols = freq_dft.columns.values
        out_cols = []
        for col in in_cols:
            if col in ['axis', 'datetime', 'date', 'path']:
                out_cols.append(col)
            else:
                # out_cols.append((float(col)/1650) * (4096/2))
                out_cols.append(Ft[col])

        conv_dict = dict(zip(in_cols, out_cols))

        freq_dft = freq_dft.rename(columns=conv_dict)
        freq_dft["date"] = freq_dft["datetime"].apply(lambda x: x.date())
        
        # freq_dft.to_csv(output_file_path, index=None)
        
        # takes the full datatable and shrinks it down to what we need.
        df=freq_dft
        
        column_list = list(df)
        column_list.remove('axis')
        column_list.remove('path') 
        column_list.remove('datetime')
        column_list.remove('date')
        if num_samples == 1650:
            hz30=[ 28.565191024863452,31.049120679199405, 33.53305033353536]
            hz60=[58.372346876895335, 60.85627653123129, 63.34020618556724]
        else:
            hz30=[ 29.507203907203802,30.507448107448, 31.507692307692196]
            hz60=[ 59.5145299145297, 60.5147741147739, 61.5150183150181]
        # print(df)
        df["sum"]=df[column_list].sum(axis=1)
        df['30hz']=df[hz30].sum(axis=1)
        df['60hz']=df[hz60].sum(axis=1)


        bol_max_filter = df.groupby(['date'])['sum'].transform(max) == df['sum']
        z= df[bol_max_filter][['sum','datetime','date','axis','path','30hz','60hz']]

        df_final =  pd.concat([df_final, z])



levels = df_final['path']
df_final[['L1', 'L2', 'L3', 'L4', 'L5']] = levels.str.split("/", n = 5, expand = True) 


output_file_path = os.path.join(results_path, 'fft_out.csv')
df_final.to_csv(output_file_path, index = None)