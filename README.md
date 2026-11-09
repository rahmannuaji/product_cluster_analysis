# DTW Clustering Pola Penjualan Mingguan (Streamlit)

Aplikasi eksplorasi untuk clustering *Sales Transactions Dataset Weekly* (811 produk × 52 minggu).
Alur input di sidebar: **1. jenis DTW** → **2. clustering approach** (Hierarchical / Non-hierarchical) → **3. method** → **4. K** dan parameter. Hasil dan visualisasi langsung diperbarui.

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
3. **Distance (jenis DTW)**, dipilih di sidebar. Semua jenis dihitung dengan satu kernel numba (area warping yang diizinkan + bobot per time shift); Sakoe-Chiba, Itakura, dan DTW klasik identik dengan `tslearn.metrics.cdist_dtw`.

| Jenis DTW | Input | Ide |
|---|---|---|
| DTW + Sakoe-Chiba band (default) | radius r (1–8 minggu) | time shift dibatasi ±r minggu |
| DTW + Itakura parallelogram | max slope s (1,5–4) | batas jajaran genjang: sempit di awal/akhir, lebar di tengah |
| DTW klasik (unconstrained) | – | time shift bebas |
| Derivative DTW (DDTW) | radius r | DTW pada turunan deret (arah naik/turun) |
| Weighted DTW (WDTW) | penalty g (0,01–0,5) | time shift jauh diberi penalti logistik |
| Euclidean (tanpa warping) | – | baseline tanpa time shift |

K-Means (DBA) tidak tersedia untuk WDTW; untuk Euclidean, K-Means menjadi K-Means biasa. Tampilan *Validasi & pilihan K* dapat menjalankan metode terpilih pada keenam jenis DTW sekaligus untuk dibandingkan.
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
| DTW explorer | Pilih dua produk: Euclidean vs DTW klasik vs jenis DTW terpilih, week alignment, warping path dan area yang diizinkan, kurva bobot WDTW |
| Tabel & unduh | Label akhir seluruh produk, unduh CSV |
| Glossary | Arti istilah teknis (DTW, medoid, DBA, silhouette, ARI, robust, dll.) |

## Menjalankan di laptop

```bash
pip install -r requirements.txt
streamlit run app.py
```

Distance matrix dihitung dalam ±0,3 detik (kompilasi numba ±3 detik saat pertama dibuka) lalu di-cache. K-Means DTW memerlukan sekitar 30 detik per run (n_init = 3, max_iter = 20, sama dengan analisis folder 02); karena itu silhouette per K dan uji robustness untuk K-Means tidak otomatis, tetapi lewat checkbox.

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
- Itakura (1975), Itakura parallelogram. https://doi.org/10.1109/TASSP.1975.1162641
- Jeong, Jeong & Omitaomu (2011), Weighted DTW. https://doi.org/10.1016/j.patcog.2010.09.022
- Kaufman & Rousseeuw (1990), K-Medoids. https://doi.org/10.1002/9780470316801
- Keogh & Pazzani (2001), Derivative DTW. https://doi.org/10.1137/1.9781611972719.1
- Krishnapuram, Joshi, Nasraoui & Yi (2001), Fuzzy C-Medoids. https://doi.org/10.1109/91.940971
- Petitjean, Ketterlin & Gançarski (2011), DTW Barycenter Averaging. https://doi.org/10.1016/j.patcog.2010.09.013
- Sakoe & Chiba (1978), Sakoe-Chiba band. https://doi.org/10.1109/TASSP.1978.1163055
- Syntetos, Boylan & Croston (2005), klasifikasi ADI–CV². https://doi.org/10.1057/palgrave.jors.2601841
- von Luxburg (2007), spectral clustering. https://doi.org/10.1007/s11222-007-9033-z
