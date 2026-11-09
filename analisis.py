"""Logika analisis: zero handling, preprocessing, DTW (Sakoe-Chiba band), dan metode clustering.

Semua fungsi murni (tanpa Streamlit) sehingga bisa diuji di luar aplikasi.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse.csgraph import laplacian
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN, HDBSCAN, SpectralClustering
from sklearn.metrics import silhouette_samples

MINGGU = [f"W{i}" for i in range(52)]
ADI_BATAS, CV2_BATAS = 1.32, 0.49  # Syntetos dkk. (2005), sesuai idclass paket R tsintermittent

# metode -> (pendekatan, keluarga)
METODE = {
    "K-Means (DTW + DBA)": ("Non-hierarchical", "centroid-based"),
    "K-Medoids": ("Non-hierarchical", "medoid-based"),
    "Fuzzy C-Medoids": ("Non-hierarchical", "fuzzy"),
    "Spectral clustering": ("Non-hierarchical", "graph-based"),
    "DBSCAN": ("Non-hierarchical", "density-based"),
    "HDBSCAN": ("Non-hierarchical", "density-based"),
    "Complete linkage": ("Hierarchical", "agglomerative"),
    "Average linkage": ("Hierarchical", "agglomerative"),
    "Single linkage": ("Hierarchical", "agglomerative"),
}
PAKAI_K = {m for m in METODE if m not in ("DBSCAN", "HDBSCAN")}
PAKAI_SEED = {"K-Means (DTW + DBA)", "K-Medoids", "Fuzzy C-Medoids"}
HIERARKI = {m for m, (p, _) in METODE.items() if p == "Hierarchical"}


# ---------- 1. Zero handling ----------
def adi_cv2(x):
    posisi = np.where(x > 0)[0] + 1
    if len(posisi) == 0:
        return np.nan, np.nan
    adi = np.diff(np.r_[0, posisi]).mean()
    jumlah = x[x > 0]
    cv2 = (jumlah.std(ddof=1) / jumlah.mean()) ** 2 if len(jumlah) > 1 else np.nan
    return adi, cv2


def segmentasi(df, batas_nol=4):
    """Lifecycle segment (leading/trailing zeros) lalu demand pattern ADI-CV2."""
    X = df[MINGGU].values
    ada = X > 0
    awal = np.where(ada.any(1), ada.argmax(1), 52)
    akhir = np.where(ada.any(1), ada[:, ::-1].argmax(1), 52)
    out = df[["Product_Code"]].copy()
    out["nol_awal"], out["nol_akhir"] = awal, akhir
    out["minggu_nol"] = (X == 0).sum(1)
    out["ADI"], out["CV2"] = zip(*[adi_cv2(x) for x in X])
    out["pola"] = np.select(
        [(out.ADI < ADI_BATAS) & (out.CV2 < CV2_BATAS), out.ADI < ADI_BATAS, out.CV2 < CV2_BATAS],
        ["smooth", "erratic", "intermittent"], default="lumpy")
    out["segmen"] = np.select(
        [(awal >= batas_nol) & (akhir >= batas_nol), awal >= batas_nol, akhir >= batas_nol,
         out.pola.isin(["smooth", "erratic"])],
        ["Short lifecycle", "New product", "Discontinued", "Regular"], default="Intermittent")
    return out


# ---------- 2. Preprocessing ----------
def praproses(R, jendela=4):
    """Moving average lalu z-normalization per produk."""
    halus = pd.DataFrame(R).T.rolling(jendela, min_periods=1).mean().T.values
    sd = halus.std(1, keepdims=True)
    return (halus - halus.mean(1, keepdims=True)) / np.where(sd == 0, 1, sd)


# ---------- 3. Jenis DTW ----------
# Semua jenis dihitung dengan satu kernel: area warping yang diizinkan (mask) + bobot per time shift |i-j|.
DTW_JENIS = {
    "DTW + Sakoe-Chiba band": "Time shift dibatasi ±r minggu di sepanjang deret (Sakoe & Chiba, 1978). Default analisis.",
    "DTW + Itakura parallelogram": "Area warping berbentuk jajaran genjang: sempit di awal/akhir deret, lebar di tengah; kemiringan maksimum s (Itakura, 1975).",
    "DTW klasik (unconstrained)": "Time shift bebas tanpa batas; rawan pathological warping.",
    "Derivative DTW (DDTW)": "DTW pada turunan (kemiringan) deret, sehingga yang dicocokkan adalah arah naik/turun (Keogh & Pazzani, 2001). Memakai Sakoe-Chiba band r.",
    "Weighted DTW (WDTW)": "Tanpa batas keras; time shift besar diberi penalti logistik dengan konstanta g (Jeong dkk., 2011).",
    "Euclidean (tanpa warping)": "Baseline: minggu ke-t hanya dibandingkan dengan minggu ke-t (setara Sakoe-Chiba r = 0).",
}


def turunan(Z):
    """Estimasi turunan Keogh & Pazzani (2001): rata-rata kemiringan kiri dan kemiringan dua langkah."""
    d = np.empty_like(Z)
    d[:, 1:-1] = ((Z[:, 1:-1] - Z[:, :-2]) + (Z[:, 2:] - Z[:, :-2]) / 2) / 2
    d[:, 0], d[:, -1] = d[:, 1], d[:, -2]
    return d


def aturan_dtw(jenis, T=52, radius=4, slope=2.0, g=0.05):
    """Mengembalikan (mask boolean T x T area yang diizinkan, bobot per |i-j|)."""
    i, j = np.indices((T, T))
    bobot = np.ones(T)
    if jenis == "Euclidean (tanpa warping)":
        boleh = i == j
    elif jenis in ("DTW + Sakoe-Chiba band", "Derivative DTW (DDTW)"):
        boleh = np.abs(i - j) <= radius
    elif jenis == "DTW + Itakura parallelogram":
        from tslearn.metrics import itakura_mask
        boleh = itakura_mask(T, T, max_slope=slope).astype(bool)  # True = sel yang diizinkan
    else:  # klasik dan WDTW: tanpa batas keras
        boleh = np.ones((T, T), bool)
    if jenis == "Weighted DTW (WDTW)":
        bobot = 1 / (1 + np.exp(-g * (np.arange(T) - T / 2)))  # w_max = 1
    return boleh, bobot


def seri_dtw(Z, jenis):
    return turunan(Z) if jenis == "Derivative DTW (DDTW)" else Z


def _kernel():
    """Kernel numba di-compile sekali saat pertama dipakai."""
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def semua_pasangan(Z, boleh, bobot):
        n, T = Z.shape
        D = np.zeros((n, n))
        for p in prange(n):
            g = np.empty((T + 1, T + 1))
            for q in range(p + 1, n):
                g[:, :] = np.inf
                g[0, 0] = 0.0
                for i in range(1, T + 1):
                    for j in range(1, T + 1):
                        if boleh[i - 1, j - 1]:
                            m = min(g[i - 1, j - 1], g[i - 1, j], g[i, j - 1])
                            g[i, j] = bobot[abs(i - j)] * (Z[p, i - 1] - Z[q, j - 1]) ** 2 + m
                D[p, q] = np.sqrt(g[T, T])
                D[q, p] = D[p, q]
        return D

    return semua_pasangan


_semua_pasangan = None


def matriks_dtw(Z, jenis="DTW + Sakoe-Chiba band", radius=4, slope=2.0, g=0.05):
    """Matriks jarak semua pasangan untuk jenis DTW yang dipilih.
    Sakoe-Chiba, Itakura, dan klasik identik dengan tslearn.metrics.cdist_dtw."""
    global _semua_pasangan
    if _semua_pasangan is None:
        _semua_pasangan = _kernel()
    boleh, bobot = aturan_dtw(jenis, Z.shape[1], radius, slope, g)
    return _semua_pasangan(np.ascontiguousarray(seri_dtw(Z, jenis)), boleh, bobot)


def jalur_dtw(a, b, boleh, bobot):
    """Jarak dan warping path satu pasangan deret untuk mask dan bobot tertentu."""
    T = len(a)
    g = np.full((T + 1, T + 1), np.inf)
    g[0, 0] = 0
    for i in range(1, T + 1):
        for j in range(1, T + 1):
            if boleh[i - 1, j - 1]:
                g[i, j] = bobot[abs(i - j)] * (a[i - 1] - b[j - 1]) ** 2 + min(g[i - 1, j - 1], g[i - 1, j], g[i, j - 1])
    jalur, i, j = [], T, T
    while i > 0 and j > 0:
        jalur.append((i - 1, j - 1))
        langkah = np.argmin([g[i - 1, j - 1], g[i - 1, j], g[i, j - 1]])
        i, j = (i - 1, j - 1) if langkah == 0 else (i - 1, j) if langkah == 1 else (i, j - 1)
    return float(np.sqrt(g[T, T])), np.array(jalur[::-1])


# ---------- 4. Metode clustering ----------
def kmedoids(D, k, n_init=10, seed=42):
    """K-Medoids (alternating / Voronoi iteration; Kaufman & Rousseeuw, 1990)."""
    rng = np.random.default_rng(seed)
    terbaik = (np.inf, None, None)
    for _ in range(n_init):
        med = rng.choice(len(D), k, replace=False)
        for _ in range(100):
            label = D[:, med].argmin(1)
            baru = med.copy()
            for c in range(k):
                anggota = np.where(label == c)[0]
                if len(anggota):
                    baru[c] = anggota[D[np.ix_(anggota, anggota)].sum(1).argmin()]
            if np.array_equal(baru, med):
                break
            med = baru
        label = D[:, med].argmin(1)
        biaya = D[np.arange(len(D)), med[label]].sum()
        if biaya < terbaik[0]:
            terbaik = (biaya, med, label)
    return terbaik[1], terbaik[2]


def kmeans_dtw(Z, k, dtw=None, seed=42):
    """K-Means dengan DTW; centroid = DTW Barycenter Averaging (Petitjean dkk., 2011).
    Untuk Euclidean memakai K-Means biasa; WDTW tidak didukung DBA."""
    from tslearn.clustering import TimeSeriesKMeans  # impor lambat: numba JIT cukup sekali
    dtw = dtw or {"jenis": "DTW + Sakoe-Chiba band", "radius": 4}
    jenis = dtw["jenis"]
    if jenis == "Weighted DTW (WDTW)":
        raise ValueError("K-Means (DBA) tidak tersedia untuk Weighted DTW: DBA pada tslearn tidak mendukung bobot time shift.")
    metric, mp = "dtw", {}
    if jenis == "Euclidean (tanpa warping)":
        metric = "euclidean"
    elif jenis in ("DTW + Sakoe-Chiba band", "Derivative DTW (DDTW)"):
        mp = {"global_constraint": "sakoe_chiba", "sakoe_chiba_radius": dtw.get("radius", 4)}
    elif jenis == "DTW + Itakura parallelogram":
        mp = {"global_constraint": "itakura", "itakura_max_slope": dtw.get("slope", 2.0)}
    km = TimeSeriesKMeans(k, metric=metric, metric_params=mp or None, n_init=3, max_iter=20,
                          random_state=seed).fit(seri_dtw(Z, jenis)[:, :, None])
    return km.labels_, km.cluster_centers_[:, :, 0]


def fcmdd(D, k, m=1.3, n_init=10, seed=42):
    """Fuzzy C-Medoids (Krishnapuram dkk., 2001)."""
    rng = np.random.default_rng(seed)

    def membership(med):
        d = np.maximum(D[:, med], 1e-10)
        return 1 / ((d[:, :, None] / d[:, None, :]) ** (2 / (m - 1))).sum(2)

    terbaik = (np.inf, None, None)
    for _ in range(n_init):
        med = rng.choice(len(D), k, replace=False)
        for _ in range(100):
            U = membership(med)
            baru = np.array([(D * (U[:, c] ** m)[:, None]).sum(0).argmin() for c in range(k)])
            if set(baru) == set(med):
                break
            med = baru
        U = membership(med)
        J = ((U ** m) * D[:, med]).sum()
        if J < terbaik[0]:
            terbaik = (J, med, U)
    return terbaik[1], terbaik[2]


def afinitas(D, faktor_sigma=1.0):
    sigma = faktor_sigma * np.median(D[np.triu_indices(len(D), 1)])
    return np.exp(-D ** 2 / (2 * sigma ** 2)), sigma


def nilai_eigen(D, faktor_sigma=1.0, banyak=8):
    A, _ = afinitas(D, faktor_sigma)
    np.fill_diagonal(A, 0)
    return np.sort(np.linalg.eigvalsh(laplacian(A, normed=True)))[:banyak]


def pohon(D, metode):
    """Linkage matrix untuk hierarchical clustering (complete/average/single)."""
    return linkage(squareform(D, checks=False), metode.split()[0].lower())


def tinggi_potong(Zl, k):
    """Tinggi cut pada dendrogram yang menghasilkan k cluster."""
    return float((Zl[-k, 2] + Zl[-k + 1, 2]) / 2) if k > 1 else float(Zl[-1, 2] * 1.05)


def jalankan(D, metode, Z=None, dtw=None, k=2, m=1.3, faktor_sigma=1.0, eps=4.5, min_samples=5,
             min_cluster_size=5, seed=42):
    """Mengembalikan label (0..K-1, noise = -1) dan info tambahan per metode."""
    info = {}
    if metode == "K-Means (DTW + DBA)":
        label, info["centroid"] = kmeans_dtw(Z, k, dtw, seed)
    elif metode == "K-Medoids":
        info["medoid"], label = kmedoids(D, k, seed=seed)
    elif metode == "Fuzzy C-Medoids":
        info["medoid"], U = fcmdd(D, k, m=m, seed=seed)
        label = U.argmax(1)
        info.update(U=U, PC=float((U ** 2).sum(1).mean()))
    elif metode == "Spectral clustering":
        A, info["sigma"] = afinitas(D, faktor_sigma)
        label = SpectralClustering(k, affinity="precomputed", assign_labels="cluster_qr",
                                   random_state=seed).fit_predict(A)
    elif metode in HIERARKI:
        label = fcluster(pohon(D, metode), k, "maxclust") - 1
    elif metode == "DBSCAN":
        label = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit_predict(D)
    else:
        label = HDBSCAN(min_cluster_size=min_cluster_size, metric="precomputed").fit_predict(D.copy())
    return np.asarray(label), info


def silhouette(D, label):
    """Silhouette tanpa noise; (None, None) bila cluster < 2."""
    m = label >= 0
    if len(set(label[m])) < 2:
        return None, None
    s = np.full(len(label), np.nan)
    s[m] = silhouette_samples(D[np.ix_(m, m)], label[m], metric="precomputed")
    return float(np.nanmean(s)), s


def urutkan_label(label, R):
    """Cluster 1 = rata-rata volume terbesar, agar penomoran konsisten antarmetode."""
    sah = sorted(set(label[label >= 0]), key=lambda c: -R[label == c].mean())
    peta = {c: i + 1 for i, c in enumerate(sah)}
    return np.array([peta.get(c, 0) for c in label])  # 0 = noise


def wakil(D, label):
    """Medoid tiap cluster = anggota dengan total jarak DTW terkecil ke anggota lain."""
    hasil = {}
    for c in sorted(set(label[label > 0])):
        idx = np.where(label == c)[0]
        hasil[c] = idx[D[np.ix_(idx, idx)].sum(1).argmin()]
    return hasil


def mds_klasik(D, dim=2):
    """Classical MDS dari matriks jarak; hanya alat bantu visual."""
    n = len(D)
    J = np.eye(n) - 1 / n
    B = -0.5 * J @ (D ** 2) @ J
    val, vec = np.linalg.eigh(B)
    urut = np.argsort(val)[::-1][:dim]
    return vec[:, urut] * np.sqrt(np.maximum(val[urut], 0)), val[urut].sum() / val[val > 0].sum()


def skor_per_k(D, metode, k_list=range(2, 7), **param):
    baris = []
    for k in k_list:
        label, _ = jalankan(D, metode, k=k, **param)
        s, _ = silhouette(D, label)
        baris.append({"K": k, "silhouette": s, "cluster_terkecil": int(np.bincount(label[label >= 0]).min())})
    return pd.DataFrame(baris)
