# DTW Clustering Pola Penjualan Mingguan (Streamlit)

Aplikasi eksplorasi untuk clustering *Sales Transactions Dataset Weekly* (811 produk × 52 minggu).
Alur input di sidebar: **1. clustering approach** (Hierarchical / Non-hierarchical) → **2. method** → **3. K** dan parameter. Hasil dan visualisasi langsung diperbarui.

## Struktur folder

```
04_Streamlit_App/
├── app.py                     # antarmuka: input sidebar, 6 tab
├── analisis.py                # logika: zero handling, preprocessing, DTW, 9 metode clustering
├── data/sales_transaction.csv # data asli
├── requirements.txt           # versi paket yang sudah diuji
├── .streamlit/config.toml     # tema terang
└── README.md
```

## Alur analisis

1. **Zero handling** (tab *Zero handling*): tidak ada imputation. Produk dipilah menurut leading/trailing zeros (lifecycle) dan demand pattern ADI–CV² (Syntetos dkk., 2005). Hanya produk Regular (*smooth*/*erratic*) yang di-cluster.
2. **Preprocessing**: moving average (default 4 minggu) lalu z-normalization per produk, agar yang dibandingkan adalah shape, bukan volume.
3. **Distance**: DTW dengan Sakoe-Chiba band (default r = 4 minggu). Hasilnya identik dengan `tslearn.metrics.cdist_dtw`.
4. **Clustering**:

| Approach | Method | Family | Input |
|---|---|---|---|
| Non-hierarchical | K-Means (DTW + DBA) | centroid-based | K, random seed |
| Non-hierarchical | K-Medoids | medoid-based | K, random seed |
| Non-hierarchical | Fuzzy C-Medoids | fuzzy | K, fuzziness m, random seed |
| Non-hierarchical | Spectral clustering (rekomendasi) | graph-based | K, kernel bandwidth σ |
| Non-hierarchical | DBSCAN | density-based | eps, min_samples |
| Non-hierarchical | HDBSCAN | density-based | min_cluster_size |
| Hierarchical | Complete linkage | agglomerative | K (cut dendrogram) |
| Hierarchical | Average linkage | agglomerative | K (cut dendrogram) |
| Hierarchical | Single linkage | agglomerative | K (cut dendrogram) |

*Ward linkage* tidak disediakan karena rumusnya mengasumsikan Euclidean distance, sedangkan DTW bukan Euclidean.

Hasil pada pengaturan default (K = 2) sama dengan analisis di folder 02 dan 03:

| Method | Ukuran cluster | Silhouette |
|---|---|---|
| K-Means (DTW + DBA) | 302 / 212 | 0,193 |
| K-Medoids | 310 / 204 | 0,191 |
| Fuzzy C-Medoids (m = 1,3) | 263 / 251 | 0,173 |
| Spectral clustering | 321 / 193 | 0,202 |
| Complete linkage | 263 / 251 | 0,160 |
| Average / Single linkage | 513 / 1 | 0,222 (hanya memisahkan outlier) |
| DBSCAN | 1 cluster | – |
| HDBSCAN | 67 / 35 + 412 noise | 0,398 (tanpa noise) |

## Isi tab

| Tab | Isi |
|---|---|
| Hasil cluster | Dendrogram + cut line (hierarchical), pola anggota per cluster dengan rata-rata / DBA centroid dan medoid, shape vs volume, cluster profile, interpretasi otomatis |
| Validasi & pilihan K | Silhouette per K, ukuran cluster terkecil, robustness (ARI antar-seed atau antar-σ), silhouette per produk, 2D map (classical MDS), diagnostik khusus metode (eigengap, membership, k-distance) |
| Zero handling | Tabel segment, ADI–CV² scatter, sales presence map 811 produk |
| DTW explorer | Pilih dua produk: Euclidean vs DTW, week alignment, warping path dan band |
| Tabel & unduh | Label akhir seluruh produk, unduh CSV |
| Glossary | Arti istilah teknis (DTW, medoid, DBA, silhouette, ARI, robust, dll.) |

## Menjalankan di laptop

```bash
pip install -r requirements.txt
streamlit run app.py
```

DTW distance matrix dihitung sekitar 3 detik lalu di-cache. K-Means DTW memerlukan sekitar 30 detik per run (n_init = 3, max_iter = 20, sama dengan analisis folder 02); karena itu silhouette per K dan uji robustness untuk K-Means tidak otomatis, tetapi lewat checkbox.

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
- Kaufman & Rousseeuw (1990), K-Medoids. https://doi.org/10.1002/9780470316801
- Krishnapuram, Joshi, Nasraoui & Yi (2001), Fuzzy C-Medoids. https://doi.org/10.1109/91.940971
- Petitjean, Ketterlin & Gançarski (2011), DTW Barycenter Averaging. https://doi.org/10.1016/j.patcog.2010.09.013
- Sakoe & Chiba (1978), Sakoe-Chiba band. https://doi.org/10.1109/TASSP.1978.1163055
- Syntetos, Boylan & Croston (2005), klasifikasi ADI–CV². https://doi.org/10.1057/palgrave.jors.2601841
- von Luxburg (2007), spectral clustering. https://doi.org/10.1007/s11222-007-9033-z
