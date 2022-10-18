import pandas as pd
import os

df_final = pd.DataFrame(columns = ['datetime','date','axis','path'])

results_path = r"C:\Users\Zebulon.Bell\OneDrive - Pinnacle\Desktop\fft results"

os.chdir(results_path)
current_path = os.getcwd()
print(f"Current Path is set to {current_path}")

def summarizer():
    print("hello world")


for root, dirs, files in os.walk(results_path):
    # print(root)
    for file in files:
        file_path = os.path.join(root, file)
        print(file_path)
        
        df = pd.read_csv(file_path)
        column_list = list(df)
        column_list.remove('axis')
        column_list.remove('path') 
        column_list.remove('datetime')
        column_list.remove('date')
        hz30=[ '24.854368932038835','29.825242718446603', '34.79611650485437']
        hz60=['54.679611650485434','59.650485436893206','64.62135922330097']
        # print(df)
        df["sum"]=df[column_list].sum(axis=1)
        df['30hz']=df[hz30].max(axis=1)
        df['60hz']=df[hz60].max(axis=1)


        bol_max_filter = df.groupby(['date'])['sum'].transform(max) == df['sum']
        z= df[bol_max_filter][['sum','datetime','date','axis','path','30hz','60hz']]

        df_final =  pd.concat([df_final, z])

     
levels = df_final['path']
df_final[['L1', 'L2', 'L3', 'L4', 'L5']] = levels.str.split("/", n = 5, expand = True) 


# output_file_path = os.path.join(results_path, 'fft_out.csv')
# df_final.to_csv(output_file_path, index = None)


