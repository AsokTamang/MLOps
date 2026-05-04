import os

def save_data(df, dirname,filename):
    os.makedirs(dirname, exist_ok=True)
    df.to_csv(f'{dirname}/{filename}', index=False)