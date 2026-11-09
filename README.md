# Klaster Pola Penjualan Mingguan dengan DTW (Streamlit)

Aplikasi eksplorasi untuk analisis klaster *Sales Transactions Dataset Weekly* (811 produk × 52 minggu).
Metode, K, dan parameter dipilih di sidebar; hasil dan visualisasi langsung diperbarui.

## Struktur folder

```
04_Streamlit_App/
├── app.py                     # antarmuka: input sidebar, 5 tab visualisasi
├── analisis.py                # logika: penanganan nol, praproses, DTW, 6 metode klaster
├── data/sales_transaction.csv # data asli
├── requirements.txt           # versi paket yang sudah diuji
├── .streamlit/config.toml     # tema terang
└── README.md
```

## Alur analisis di aplikasi

1. **Penanganan nilai nol** (tab *Penanganan nilai nol*): tidak ada imputasi. Produk dipilah menurut nol beruntun di awal/akhir deret (siklus hidup) dan klasifikasi ADI–CV² (Syntetos dkk., 2005). Hanya produk reguler (*smooth*/*erratic*) yang diklaster.
2. **Praproses**: rata-rata bergerak (bawaan 4 minggu) lalu z-normalisasi per produk, agar yang dibandingkan adalah bentuk pola, bukan volume.
3. **Jarak**: DTW dengan pita Sakoe-Chiba (bawaan r = 4 minggu). Hasilnya identik dengan `tslearn.metrics.cdist_dtw`.
4. **Metode klaster** (tanpa K-Means):

| Metode | Keluarga | Input di sidebar |
|---|---|---|
| Spectral clustering (rekomendasi) | graf | K, skala σ |
| Fuzzy C-Medoids | fuzzy | K, derajat kekaburan m |
| Complete linkage | hierarki | K |
| Average linkage | hierarki | K |
| DBSCAN | kepadatan | eps, min_samples |
| HDBSCAN | kepadatan | min_cluster_size |

## Isi tab

| Tab | Isi |
|---|---|
| Hasil klaster | Pola anggota per klaster (small multiples), rata-rata pola vs volume asli, tabel profil, interpretasi otomatis |
| Kualitas dan pilihan K | Silhouette menurut K, ukuran klaster terkecil, silhouette per produk, peta 2D (MDS), diagnostik khusus metode (nilai eigen, keanggotaan fuzzy, k-distance) |
| Penanganan nilai nol | Tabel segmen, sebaran ADI–CV², peta penjualan 811 produk |
| DTW dua produk | Pilih dua produk: jarak Euclidean vs DTW, pasangan minggu, jalur warping dan pita |
| Tabel dan unduh | Label akhir seluruh produk, unduh CSV |

## Menjalankan di laptop

```bash
pip install -r requirements.txt
streamlit run app.py
```

Perhitungan matriks DTW pertama memerlukan beberapa detik; selanjutnya hasil disimpan di cache.

## Deploy ke Streamlit Community Cloud

1. Buat repositori GitHub baru, lalu unggah **isi** folder ini (termasuk folder `data/` dan `.streamlit/`).
2. Buka [share.streamlit.io](https://share.streamlit.io), masuk dengan akun GitHub, pilih **Create app**.
3. Isi: repositori, branch `main`, *Main file path* `app.py`.
4. Di **Advanced settings**, pilih Python **3.12**.
5. Klik **Deploy**. Pemasangan paket pertama memerlukan beberapa menit.

Jika folder ini diletakkan sebagai subfolder di repositori, isi *Main file path* dengan `04_Streamlit_App/app.py`.

## Rujukan metode

- Campello, Moulavi & Sander (2013), HDBSCAN. https://doi.org/10.1007/978-3-642-37456-2_14
- Damle, Minden & Ying (2019), `cluster_qr`. https://doi.org/10.1093/imaiai/iay008
- Ester, Kriegel, Sander & Xu (1996), DBSCAN. KDD-96, 226–231.
- Krishnapuram, Joshi, Nasraoui & Yi (2001), Fuzzy C-Medoids. https://doi.org/10.1109/91.940971
- Sakoe & Chiba (1978), DTW berpita. https://doi.org/10.1109/TASSP.1978.1163055
- Syntetos, Boylan & Croston (2005), klasifikasi ADI–CV². https://doi.org/10.1057/palgrave.jors.2601841
- von Luxburg (2007), spectral clustering. https://doi.org/10.1007/s11222-007-9033-z
