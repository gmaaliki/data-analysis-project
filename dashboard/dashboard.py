import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

df_day = pd.read_csv('../data/day.csv')
df_hour = pd.read_csv('../data/hour.csv')

st.title('Proyek Analisis Data : Bike Sharing data')
st.write("Nama: Ghifari Maaliki Syafa Syuhada")
st.write("Email: maalikighifari@gmail.com")
st.write("Dicoding ID: ghifari_maaliki_cyhr")

st.subheader('Pertanyaan bisnis')
st.write("1. Bagaimana suhu dapat mempengaruhi penyewaan sepeda?")
st.write("2. Pada jam berapa rata-rata penyewaan sepeda paling banyak?")

tab1, tab2, tab3 = st.tabs(["Pengenalan data", "Exploratory data analysis", "Hasil"])

with tab1:
    st.write('data dibagi menjadi 2 yaitu day.csv dan hour.csv. File day.csv berisi data penyewaan sepeda per hari sedangkan hour.csv berisi data penyewaan sepeda per jam. day.csv digunakan untuk menilai pengaruh cuaca, temperatur, dll sedangkan hour.csv digunakan untuk melihat kondisi penyewaan untuk setiap jam per hari')

    st.subheader('Sampel dataset')
    st.dataframe(df_hour.head())
    st.write('*day.csv tidak memiliki kolom hr')

    st.subheader('Deskripsi dataset')
    st.dataframe(df_hour.describe())

    st.subheader('Penjelasan fitur')
    feature = [
        'instant: index',
        'dteday: tanggal',
        'season: musim (1: semi, 2: panas,3: kemaraum, 4: dingin)',
        'yr: tahun (0: 2011, 1: 2012)',
        'mnth: bulan (1 sampai 12)',
        'hr: jam (0 sampai 23)',
        'holiday: menadakan hari libur atau tidak',
        'weekday: hari dalam seminggu',
        'workingday: apabila bukan akhir pekan atau hari libur maka 1, selain itu 0',
        'weathersit: 1: cerah, sedikit awan; 2: kabut+berawan; 3: hujan ringan, salju ringan; 4: hujan deras, salju + kabut',
        'temp: temperatur dalam celsius yang ternormalisasi',
        'atemp: temperatur \'feels like\' dalam celsius yang ternormalisasi',
        'hum: kelembaban yang ternormalisasi',
        'windspeed: kecepatan angin yang ternormalisasi',
        'casual: jumlah pengguna casual',
        'registered: jumlah pengguna terdaftar',
        'cnt: total penyewaan casual dan terdaftar',
    ]

    for i in feature:
        st.write(f'- {i}')

with tab2:
    st.write('Pemikiran pertama yang muncul saat menganalisis jumlah penyewaan sepeda adalah pengaruh dari cuaca. Karena apabila dipikirkan, orang-orang tidak akan bersepeda disaat hujan karena jalan yang licin akan membahayakan para pesepeda. Untuk mengonfirmasi dugaan tersebut, kita dapat melihat rata rata dari penyewaan sepeda di cuaca tertentu')

    df = df_hour[['weathersit', 'cnt']].copy()
    df = df.groupby(by='weathersit').mean().reset_index()
    st.write(df)

    st.write('Dari tabel rata-rata diatas, jelas bahwa semakin buruknya cuaca maka penyewaan sepeda akan semakin menurun.')
    st.write('Selanjutnya, kita akan melihat pengaruh dari temperatur, kecepatan angin, dan kelembaban udara dalam penyewaan sepeda. Karena keempat variabel tersebut bukan tipe data categorical, kita dapat mencari korelasinya.')

    selected_columns = ['temp', 'windspeed', 'hum', 'cnt']

    correlation_matrix = df_day[selected_columns].corr()
    st.write(correlation_matrix)

    st.write('Dapat dilihat dari tabel diatas bahwa temperatur dan jumlah sewa mempunyai nilai korelasi positif sebesar 0.627494 sedangkan windspeed dan jumlah sewa mempunyai nilai korelasi negatif sebesar -0.234545. Selain itu, penyewaan sepeda juga memiliki nilai korelasi negatif sebesar -0.100659 dengan kelembaban udara. Hal ini dapat menimbulkan dugaan bahwa para penggemar bersepeda lebih menyukai keadaan temperatur tinggi, kecepatang angin yang rendah dan kelembaban yang rendah. Untuk memastikan hubungan dari variabel tersebut dengan jumlah penyewaan sepeda, kita perlu menelaah lagi data-datanya.')

    st.write('data hour.csv memiliki data penyewaan sepeda per jam. Kita dapat melakukan pengelompokkan data berdasarkan jam untuk menentukan rata-rata penyewaan sepeda per jam. Lalu melakukan sorting untuk menentukan jam dimana penyewaan sepeda paling banyak.')

    df = df_hour[['hr', 'cnt']].copy()
    df = df.groupby(by='hr').mean().reset_index()
    df.sort_values(by='cnt', ascending=False, inplace=True)
    st.write(df)

    st.write('Dilihat dari hasil sorting penyewaan sepeda diatas, orang-orang cenderung menyewa sepeda di waktu sore. Akan tetapi penyewaan sepeda pada jam 8 mengalami lonjakan yang tinggi. Hal ini menimbulkan anomali. Dari yang sudah ditelaah tadi, bisa jadi kondisi temperatur, kecepatan angin, dan kelembaban sangat cocok di waktu sore dan pada jam 8 pagi. Akan tetapi, hal ini tidak masuk akal karena kecil kemungkinannya bahwa kondisi tersebut berubah drastis khusus pada jam 8. Maka dari itu perlu kita lihat kondisi suhu, kelembaban, dan temperatur per jamnya untuk memastikan.')

    st.subheader('Rata-rata suhu setiap jam')
    df = df_hour[['hr', 'temp']]
    df = df.groupby(by='hr').mean().reset_index()
    df.sort_values(by='temp', ascending=False, inplace=True)
    st.write(df)

    st.subheader('Rata-rata kecepatan angin setiap jam')
    df = df_hour[['hr', 'windspeed']]
    df = df.groupby(by='hr').mean().reset_index()
    df.sort_values(by='windspeed', ascending=False, inplace=True)
    st.write(df)

    st.subheader('Rata-rata kelembaban setiap jam')
    df = df_hour[['hr', 'hum']]
    df = df.groupby(by='hr').mean().reset_index()
    df.sort_values(by='hum', ascending=False, inplace=True)
    st.write(df)

    st.write('Setelah memeriksa kondisi temperatur, kecepatan angin, dan kelembaban pada setiap jamnya, bisa dilihat bahwa kondisi pada jam 8 bertolak balik dari yang sudah kita duga. Biarpun kecepatan anginnya tergolong rendah yang mendukung dugaan kenaikan sewa sepeda karena korelasinya yang negatif, akan tetapi suhu dan kelembabannya bertolak balik dari korelasi yang telah dibuat tadi. Maka dari itu, kita perlu mencari informasi lain untuk mendapat kesimpulan.')
    st.write('data yang digunakan adalah berdasarkan sistem Capital Bikeshare yang berasal dari Washington D.C. Menurut wikipedia, jam kerja tradisional USA adalah pada jam 09:00-05:00. Hal ini dapat menjelaskan kenapa pada jam 8 banyak yang menyewa sepeda karena para pekerja menggunakannya untuk berangkat kerja dan pada sore hari para pekerja pulang dengan sepeda juga. Akan tetapi, fakta itu saja tidak dapat menjelaskan selisih penyewaan sepeda pada jam 8 dan jam 17 yang lumayan besar. Dari perbandingan diatas dan juga korelasi yang telah dibuat, saya menyimpulkan bahwa sebagian besar orang yang menyewa sepeda adalah pekerja dan menurut korelasi yang dibuat tadi, banyak orang lain pula yang lebih memilih sepeda sebagai metode transportasi pada sore hari.')

with tab3:
    st.subheader('Hasil')
    st.subheader('Pertanyaan 1:')
    st.write('Bagaimana suhu dapat mempengaruhi penyewaan sepeda?')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x=df_day['temp'], y=df_day['cnt'])
    plt.title('Perbandingan Suhu dan Jumlah Penyewaan Sepeda')
    plt.xlabel('Suhu')
    plt.ylabel('Jumlah Penyewaan Sepeda')
    plt.show()
    st.pyplot(fig)
    st.write('Dapat dilihat bahwa semakin tinggi temperatur, maka penyewaan sepeda semakin tinggi pula. Salah satu faktor yang dapat mempengaruhi kenaikan tersebut adalah semakin panas cuacanya, orang-orang semakin enggan untuk berjalan kaki dan lebih suka untuk berpergian cepat dengan menggunakan sepeda.')

    st.subheader('Pertanyaan 2:')
    st.write('Pada jam berapa rata-rata penyewaan sepeda paling banyak?')

    fig, ax = plt.subplots(figsize=(10, 6))
    df = df_hour[['hr', 'cnt']].copy()
    df = df.groupby(by="hr").mean().reset_index()
    average_cnt = df['cnt'].mean()
    plt.title('Rata-rata sewa sepeda setiap jam')
    plt.axhline(y=average_cnt, color='r', linestyle='--', label='Rata-rata keseluruhan')
    plt.bar(x=df['hr'], height=df['cnt'])
    plt.xlabel('Jam')
    plt.ylabel('Rata-rata penyewaan sepeda')
    plt.xticks(df['hr'])
    plt.legend()
    st.pyplot(fig)
    st.write('Dari grafik diatas, dapat dilihat bahwa penyewaan sepeda memiliki lonjakan pada jam 8 dan juga sore hari pada jam 17 dan 18. Hal ini dapat dihubungkan dengan waktu kerja tradisional 09:00-17:00 di USA, tempat data diambil. Hal ini juga dapat dikaitkan dengan orang-orang yang hobi bersepeda di sore hari, dapat dijelaskan dengan selisih yang besar antara penyewaan sepeda di jam 8 dan jam 17.')

    st.subheader('Kesimpulan')
    st.write('Temperatur yang tinggi menyebabkan orang-orang lebih suka menyewa sepeda')
    st.write('Penyewaan sepeda memiliki angka paling tinggi pada jam pulang kerja yaitu pukul 17:00-18:00')
