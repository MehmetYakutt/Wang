# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 11:43:17 2025

@author: Administrator
"""

import pandas as pd
df = pd.read_csv("ornek_e_ticaret_verisi.csv", 
                 encoding='cp1254', 
                 delimiter=';')
print(df.head())

# Veri tipleri
print(df.dtypes)

# Eksik veri kontrolü
print(df.isnull().sum())

# Sayısal özet istatistikler
print(df.describe())

# Kategorik sütunların dağılımı
print(df['kategori'].value_counts())
print(df['kampanya_var_mı'].value_counts())
df['tarih'] = pd.to_datetime(df['tarih'])

import matplotlib.pyplot as plt

# Günlük toplam satış
gunluk_satis = df.groupby("tarih")["satış_adedi"].sum()

# Grafik çiz
plt.figure(figsize=(10, 5))
gunluk_satis.plot(marker="o")
plt.title("Günlük Toplam Satış Adedi")
plt.xlabel("Tarih")
plt.ylabel("Satış Adedi")
plt.grid(True)
plt.show()


# Kampanyaya göre ortalama satış
kampanya_ortalama = df.groupby("kampanya_var_mı")["satış_adedi"].mean()
print(kampanya_ortalama)



from scipy.stats import ttest_ind

kampanya_var = df[df['kampanya_var_mı'] == "EVET"]['satış_adedi']
kampanya_yok = df[df['kampanya_var_mı'] == "HAYIR"]['satış_adedi']

t_stat, p_value = ttest_ind(kampanya_var, kampanya_yok, equal_var=False)

print("T-istatistiği:", t_stat)
print("p-değeri:", p_value)

import statsmodels.formula.api as smf

# Kampanya sütununu binary formata çevir
df['kampanya_binary'] = df['kampanya_var_mı'].map({'EVET': 1, 'HAYIR': 0})

# Regresyon modeli
model = smf.ols('satış_adedi ~ fiyat + kampanya_binary + tıklanma', data=df).fit()

# Sonuçları göster
print(model.summary())
