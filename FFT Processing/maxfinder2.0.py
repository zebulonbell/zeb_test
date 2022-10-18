import pandas as pd
import datetime as dt
import os
import sys
import pathlib

path = pathlib.Path().resolve()
print(path)
df_final = pd.DataFrame(columns = ['file',"date", "reading", 'time'])
# print(df_final)
dirs = os.listdir()

# for file in dirs:
#     print(file)
#     filename, ext = os.path.splitext(file)
#     if ext == '.csv':
#         # print(file)
#         df = pd.read_csv(file)
#         df.columns = ["date", "reading"]
#         df["date"]= df["date"].map(lambda a: pd.Timestamp(a).date())
#         df['file']=file


#         # print(df)
#         maxx=df.groupby(['file','date']).max().reset_index()
#         # print(maxx)
        
#         # frames = [df_final, maxx]
#         # df_final = pd.concat(frames)
#         df_final = df_final.append(maxx, ignore_index = True)
#         # print(df_final)
        
        
df = pd.read_csv(r"P:\Open Projects\Exxon\Exxon QRO Pilots\Upstream\Hebron\Time Series Data\104_UPS-HBR-HBR.VEY657202_02.PZ.X.csv")
df.columns = ["time", "reading"]
df["date"]= df["time"].map(lambda a: pd.Timestamp(a).date())
df['file']='file'


        # print(df)
maxx=df.groupby(['file','date']).max().reset_index()
        # print(maxx)
        
        # frames = [df_final, maxx]
        # df_final = pd.concat(frames)
df_final = df_final.append(maxx, ignore_index = True)
        # print(df_final)







output_file_path = os.path.join(path, 'max_export.csv')
# df_final.to_csv(output_file_path)
dfsmall=df
dfsmall=df.drop(dfsmall[df.date != dt.date(2022, 1, 31)].index)

plt.plot(dfsmall['time'],dfsmall['reading'])



