import pandas as pd
import matplotlib.pyplot as plt

# Misalkan 'df' adalah DataFrame Anda
df = pd.read_excel('fitur\seleksi\glcm_edge_pure.xlsx')

# Pisahkan data berdasarkan label
df_matang = df[df['Label'] == 'matang']
df_mentah = df[df['Label'] == 'mentah']
df_bukan = df[df['Label'] == 'bukan']

# Urutkan data berdasarkan nilai total dari setiap baris
for df in [df_matang, df_mentah, df_bukan]:
    df['total'] = df.sum(axis=1)
    df.sort_values('total', ascending=False, inplace=True)

# Buat plot untuk setiap label
for df, label in zip([df_matang, df_mentah, df_bukan], ['matang', 'mentah', 'bukan']):
    plt.figure(figsize=(10,6))
    for column in df.columns:
        if column not in ['Label', 'total']:
            plt.plot(df[column], label=column)
    plt.title(f'Diagram garis untuk label {label}')
    plt.legend()
    plt.show()
