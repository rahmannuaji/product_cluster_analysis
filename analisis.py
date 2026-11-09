"""Logika analisis: penanganan nol, praproses, DTW Sakoe-Chiba, dan enam metode klaster.

Semua fungsi murni (tanpa Streamlit) sehingga bisa diuji di luar aplikasi.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse.csgraph import laplacian
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN, HDBSCAN, SpectralClustering
from sklearn.metrics import silhouette_samples, silhouette_score

MINGGU = [f"W{i}" for i in range(52)]
ADI_BATAS, CV2_BATAS = 1.32, 0.49  # Syntetos dkk. (2005), sesuai idclass paket R tsintermittent

METODE = {
    "Spectral clustering": "graf",
    "Fuzzy C-Medoids": "fuzzy",
    "Complete linkage": "hierarki",
    "Average linkage": "hierarki",
    "DBSCAN": "kepadatan",
    "HDBSCAN": "kepadatan",
}
PAKAI_K = {"Spectral clustering", "Fuzzy C-Medoids", "Complete linkage", "Average linkage"}


# ---------- 1. Penanganan nilai nol ----------
def adi_cv2(x):
    posisi = np.where(x > 0)[0] + 1
    if len(posisi) == 0:
        return np.nan, np.nan
    adi = np.diff(np.r_[0, posisi]).mean()
    jumlah = x[x > 0]
    cv2 = (jumlah.std(ddof=1) / jumlah.mean()) ** 2 if len(jumlah) > 1 else np.nan
    return adi, cv2


def segmentasi(df, batas_nol=4):
    """Segmen siklus hidup (nol beruntun di awal/akhir) lalu pola permintaan ADI-CV2."""
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
        ["Siklus pendek", "Mulai terjual", "Berhenti terjual", "Reguler"], default="Intermiten")
    return out


# ---------- 2. Praproses ----------
def praproses(R, jendela=4):
    halus = pd.DataFrame(R).T.rolling(jendela, min_periods=1).mean().T.values
    sd = halus.std(1, keepdims=True)
    return (halus - halus.mean(1, keepdims=True)) / np.where(sd == 0, 1, sd)


# ---------- 3. DTW dengan pita Sakoe-Chiba ----------
def matriks_dtw(Z, radius=4):
    """Matriks jarak DTW semua pasangan. Divektorkan per pasangan: 52 x (2r+1) operasi array."""
    n, T = Z.shape
    i_idx, j_idx = np.triu_indices(n, 1)
    a, b = Z[i_idx], Z[j_idx]
    sebelum = np.full((T, len(i_idx)), np.inf)
    for i in range(T):
        kini = np.full((T, len(i_idx)), np.inf)
        for j in range(max(0, i - radius), min(T, i + radius + 1)):
            biaya = (a[:, i] - b[:, j]) ** 2
            if i == 0 and j == 0:
                kini[j] = biaya
                continue
            terkecil = sebelum[j]
            if j > 0:
                terkecil = np.minimum(terkecil, np.minimum(sebelum[j - 1], kini[j - 1]))
            kini[j] = biaya + terkecil
        sebelum = kini
    D = np.zeros((n, n))
    D[i_idx, j_idx] = np.sqrt(sebelum[T - 1])
    return D + D.T


def jalur_dtw(a, b, radius=None):
    """Jarak DTW dan jalur warping untuk satu pasangan deret (radius None = tanpa batas)."""
    T = len(a)
    r = T if radius is None else radius
    g = np.full((T + 1, T + 1), np.inf)
    g[0, 0] = 0
    for i in range(1, T + 1):
        for j in range(max(1, i - r), min(T, i + r) + 1):
            g[i, j] = (a[i - 1] - b[j - 1]) ** 2 + min(g[i - 1, j - 1], g[i - 1, j], g[i, j - 1])
    jalur, i, j = [], T, T
    while i > 0 and j > 0:
        jalur.append((i - 1, j - 1))
        langkah = np.argmin([g[i - 1, j - 1], g[i - 1, j], g[i, j - 1]])
        i, j = (i - 1, j - 1) if langkah == 0 else (i - 1, j) if langkah == 1 else (i, j - 1)
    return float(np.sqrt(g[T, T])), np.array(jalur[::-1])


# ---------- 4. Metode klaster ----------
def fcmdd(D, k, m=1.3, n_init=10, seed=42):
    """Fuzzy C-Medoids (Krishnapuram dkk., 2001)."""
    rng = np.random.default_rng(seed)

    def keanggotaan(med):
        d = np.maximum(D[:, med], 1e-10)
        return 1 / ((d[:, :, None] / d[:, None, :]) ** (2 / (m - 1))).sum(2)

    terbaik = (np.inf, None, None)
    for _ in range(n_init):
        med = rng.choice(len(D), k, replace=False)
        for _ in range(100):
            U = keanggotaan(med)
            baru = np.array([(D * (U[:, c] ** m)[:, None]).sum(0).argmin() for c in range(k)])
            if set(baru) == set(med):
                break
            med = baru
        U = keanggotaan(med)
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


def jalankan(D, metode, k=2, m=1.3, faktor_sigma=1.0, eps=4.5, min_samples=5, min_cluster_size=5, seed=42):
    """Mengembalikan label (0..K-1, noise = -1) dan info tambahan per metode."""
    info = {}
    if metode == "Spectral clustering":
        A, info["sigma"] = afinitas(D, faktor_sigma)
        label = SpectralClustering(k, affinity="precomputed", assign_labels="cluster_qr",
                                   random_state=seed).fit_predict(A)
    elif metode == "Fuzzy C-Medoids":
        med, U = fcmdd(D, k, m=m, seed=seed)
        label = U.argmax(1)
        info.update(U=U, medoid=med, PC=float((U ** 2).sum(1).mean()))
    elif metode in ("Complete linkage", "Average linkage"):
        Zl = linkage(squareform(D, checks=False), metode.split()[0].lower())
        label = fcluster(Zl, k, "maxclust") - 1
    elif metode == "DBSCAN":
        label = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit_predict(D)
    else:
        label = HDBSCAN(min_cluster_size=min_cluster_size, metric="precomputed").fit_predict(D.copy())
    return np.asarray(label), info


def silhouette(D, label):
    """Silhouette tanpa noise; None bila klaster < 2."""
    m = label >= 0
    if len(set(label[m])) < 2:
        return None, None
    s = np.full(len(label), np.nan)
    s[m] = silhouette_samples(D[np.ix_(m, m)], label[m], metric="precomputed")
    return float(np.nanmean(s)), s


def urutkan_label(label, R):
    """Klaster 1 = rata-rata volume terbesar, agar penomoran stabil dan mudah dibaca."""
    sah = sorted(set(label[label >= 0]), key=lambda c: -R[label == c].mean())
    peta = {c: i + 1 for i, c in enumerate(sah)}
    return np.array([peta.get(c, 0) for c in label])  # 0 = noise


def wakil(D, label):
    """Produk wakil = anggota dengan total jarak DTW terkecil ke anggota lain (medoid)."""
    hasil = {}
    for c in sorted(set(label[label > 0])):
        idx = np.where(label == c)[0]
        hasil[c] = idx[D[np.ix_(idx, idx)].sum(1).argmin()]
    return hasil


def mds_klasik(D, dim=2):
    """Proyeksi 2D dari matriks jarak (MDS klasik); hanya alat bantu visual."""
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
        baris.append({"K": k, "silhouette": s, "klaster_terkecil": int(np.bincount(label[label >= 0]).min())})
    return pd.DataFrame(baris)
