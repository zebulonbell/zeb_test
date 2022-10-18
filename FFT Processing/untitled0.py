import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_ga_file():
    df = pd.read_csv(r"C:\Users\Zebulon.Bell\OneDrive - Pinnacle\Desktop\python test\8.9\Book1.csv", header=None)
    ncol = df.shape[1]
    start_cols = ["id", 'path', "something", "axis", "unix_time", "some_number", "number_samples"] 
    end_cols = range(0, ncol-len(start_cols))
    cols = start_cols + list(end_cols)
    df.columns = cols
    df['datetime'] = pd.to_datetime(df['unix_time'], unit='ms')
    return df


sdf

# data = pd.read_csv(r"C:\Users\Zebulon.Bell\OneDrive - Pinnacle\Desktop\python test\8.9\Book1.csv")

df = read_ga_file()

wave=np.array(df)
wave=wave[0,:]
wave=wave[7:4103]

t=np.arange(0,len(wave))
plt.plot(t,wave,color = 'c',  label = 'Noisy')


dt = 0.001
# t= np.arange(0,1,dt)
# f = np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t)
# f_clean = f
# f = f + 2.5*np.random.randn(len(t))

# plt.plot(t,f,color = 'c',  label = 'Noisy')
# plt.plot(t,f_clean,color = 'k',  label = 'Clean')
# plt.xlim(t[0],t[-1])
# plt.legend()

f=wave

n = len(t)
fhat = np.fft.fft(f,n)
PSD = fhat * np.conj(fhat) / n
freq = (1/(dt*n)) * np.arange(n)
L = np.arange(1,np.floor(n/2),dtype='int')

# fig,axs = plt.subplots(2,1)

# plt.sca(axs[0])
# plt.plot(t,f,color='c', linewidth=1.5, label='noisy')
# plt.plot(t,f_clean,color='k', linewidth=2,  label='clean')

# plt.xlim(t[0],t[-1])
# plt.legend()

# plt.sca(axs[1])
plt.plot(freq[L],PSD[L],color='c',label='noisy')
plt.xlim(freq[L[0]],freq[L[-1]])

plt.show()