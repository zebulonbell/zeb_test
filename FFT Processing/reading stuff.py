# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 13:40:21 2022

@author: Zebulon.Bell
"""

import pandas as pd
data_files_path = r"C:\Users\Zebulon.Bell\OneDrive - Pinnacle\Desktop\python test\reading csv"


typ = pd.read_csv(r"C:\Users\Zebulon.Bell\OneDrive - Pinnacle\Desktop\python test\reading csv\type.csv")
data = pd.read_csv(r"C:\Users\Zebulon.Bell\OneDrive - Pinnacle\Desktop\python test\reading csv\data.csv")

df=pd.DataFrame(columns = ['asset','component','date','type','reading'])

for i in range(len(typ.index)): #len(typ.index)
    asset= typ.iloc[i].asset
    # print(asset)
    column= typ.iloc[i].datatype
    # print(column)
    z=pd.DataFrame(columns = ['asset','component','date','type','reading'])
    if column=='x':
        z['reading']=data[(data['asset'] =='i')].x

    else:
        z['reading']=data[(data['asset'] =='i')].y
    
    z['asset']=asset
    z['component'] = typ.iloc[i].component
    z['type'] = column
    
    
    z['date']=data[(data['asset'] =='i')].date
    df =  pd.concat([df, z])
    # print(df)
    


