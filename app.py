from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import cophenet, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

import analisis as an

st.set_page_config(page_title="DTW Clustering Penjualan Mingguan", layout="wide", initial_sidebar_state="auto")

# ---------- gaya ----------
WARNA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # urutan tetap, divalidasi
NOISE = "#9a9a96"
TINTA, TINTA_2, GRID = "#18181b", "#52514e", "#ececea"
SIMBOL = ["circle", "square", "diamond", "triangle-up", "cross", "star"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, .stApp, .stApp *:not([data-testid="stIconMaterial"]):not(code) { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
h1, h2, h3 { letter-spacing: -0.02em; font-weight: 600; }
h1 { font-size: 1.75rem !important; }
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-variant-numeric: tabular-nums; }
[data-testid="stMetricLabel"] p { color: #52514e; font-size: 0.85rem; }
[data-testid="stMetric"] { border-top: 1px solid #e4e4e0; padding-top: 0.6rem; }
[data-testid="stDataFrame"] { font-variant-numeric: tabular-nums; }
::selection { background: #cde2fb; color: #18181b; }
.catatan { color: #52514e; font-size: 0.92rem; max-width: 78ch; line-height: 1.55; }
.tebal { font-weight: 600; color: #18181b; }
</style>
""", unsafe_allow_html=True)


def tata(fig, tinggi=360, **kw):
    ada_judul = "title" in kw
    if ada_judul:
        kw["title"].update(x=0, xanchor="left", y=0.98, yanchor="top", yref="container")
    fig.update_layout(height=tinggi, template="plotly_white", font=dict(family="IBM Plex Sans, system-ui, sans-serif", size=13, color=TINTA),
                      margin=dict(l=10, r=10, t=78 if ada_judul else 40, b=10), hoverlabel=dict(font_family="IBM Plex Mono"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0), **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor="#d4d4d0")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="#d4d4d0")
    return fig


def warna(c):
    return NOISE if c == 0 else WARNA[(c - 1) % len(WARNA)]


def nama(c):
    return "Noise" if c == 0 else f"Cluster {c}"


def angka(x, d=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def catatan(teks):
    st.markdown(f"<p class='catatan'>{teks}</p>", unsafe_allow_html=True)


def grafik(fig):
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


# ---------- data dan komputasi (di-cache) ----------
DATA = Path(__file__).parent / "data" / "sales_transaction.csv"


@st.cache_data
def muat():
    return pd.read_csv(DATA)


@st.cache_data
def siapkan(batas_nol, jendela, radius):
    df = muat()
    seg = an.segmentasi(df, batas_nol)
    reg = seg[seg.segmen == "Regular"].index.values
    R = df.loc[reg, an.MINGGU].values.astype(float)
    Z = an.praproses(R, jendela)
    return seg, reg, R, Z, an.matriks_dtw(Z, radius)


@st.cache_data(show_spinner=False)
def klaster(_D, _Z, kunci, metode, param):
    label, info = an.jalankan(_D, metode, Z=_Z, radius=kunci[2], **param)
    s_rata, s_produk = an.silhouette(_D, label)
    return label, info, s_rata, s_produk


@st.cache_data(show_spinner=False)
def skor_k(_D, _Z, kunci, metode, param_tanpa_k):
    return an.skor_per_k(_D, metode, Z=_Z, radius=kunci[2], **param_tanpa_k)


@st.cache_data(show_spinner=False)
def robustness(_D, _Z, kunci, metode, param):
    """ARI antarpasangan hasil: 5 seed (metode acak) atau 5 nilai σ (spectral)."""
    if metode == "Spectral clustering":
        variasi = [dict(param, faktor_sigma=f) for f in (0.5, 0.75, 1.0, 1.5, 2.0)]
    else:
        variasi = [dict(param, seed=s) for s in range(5)]
    hasil = [an.jalankan(_D, metode, Z=_Z, radius=kunci[2], **v)[0] for v in variasi]
    ari = [adjusted_rand_score(hasil[i], hasil[j]) for i in range(5) for j in range(i + 1, 5)]
    return float(np.mean(ari)), float(np.min(ari))


@st.cache_data(show_spinner=False)
def peta_2d(_D, kunci):
    return an.mds_klasik(_D)


# ---------- sidebar: input ----------
PENJELASAN = {
    "K-Means (DTW + DBA)": "Centroid-based. Centroid dihitung dengan DTW Barycenter Averaging (DBA), yaitu rata-rata yang mengikuti pasangan minggu hasil DTW. Hasil bergantung pada random seed.",
    "K-Medoids": "Medoid-based. Pusat cluster adalah produk nyata (medoid) dengan total jarak DTW terkecil ke anggotanya. Bekerja langsung dari matriks DTW.",
    "Fuzzy C-Medoids": "Fuzzy clustering. Setiap produk punya membership (0–1) di semua cluster; label akhir = membership tertinggi.",
    "Spectral clustering": "Graph-based. Matriks DTW diubah menjadi similarity graph, lalu graph dipotong di bagian yang koneksinya paling lemah. Rekomendasi dari analisis (folder 03).",
    "DBSCAN": "Density-based. Cluster = area yang padat; produk di area renggang menjadi noise. K tidak diinput.",
    "HDBSCAN": "Density-based (hierarchical density). Memilih cluster yang paling stabil di semua tingkat kepadatan. K tidak diinput.",
    "Complete linkage": "Jarak antar-cluster = pasangan anggota terjauh. Menghasilkan cluster yang compact.",
    "Average linkage": "Jarak antar-cluster = rata-rata jarak semua pasangan anggota. Sensitif terhadap outlier.",
    "Single linkage": "Jarak antar-cluster = pasangan anggota terdekat. Rawan chaining effect.",
}

with st.sidebar:
    st.header("Pengaturan")
    pendekatan = st.radio("1 · Clustering approach", ["Non-hierarchical", "Hierarchical"], horizontal=True,
                          help="Non-hierarchical: langsung membagi produk menjadi K cluster. Hierarchical: menggabungkan produk bertahap menjadi dendrogram, lalu dipotong menjadi K cluster.")
    pilihan = [m for m, (p, _) in an.METODE.items() if p == pendekatan]
    bawaan = "Spectral clustering" if pendekatan == "Non-hierarchical" else "Complete linkage"
    metode = st.selectbox("2 · Method", pilihan, index=pilihan.index(bawaan))
    st.caption(PENJELASAN[metode])

    param = {}
    if metode in an.PAKAI_K:
        param["k"] = st.slider("3 · Jumlah cluster (K)", 2, 6, 2)
    if metode == "Spectral clustering":
        param["faktor_sigma"] = st.select_slider("Kernel bandwidth σ (× median DTW distance)", [0.5, 0.75, 1.0, 1.5, 2.0], value=1.0,
                                                 help="σ besar: lebih banyak pasangan produk dianggap mirip. Label diberikan dengan cluster_qr (tanpa K-Means).")
    elif metode == "Fuzzy C-Medoids":
        param["m"] = st.slider("Fuzziness (m)", 1.1, 2.0, 1.3, 0.1, help="m mendekati 1 = crisp clustering; m = 2 = nilai default FCM.")
    elif metode == "DBSCAN":
        param["eps"] = st.slider("eps (radius neighborhood)", 2.5, 6.0, 4.5, 0.1)
        param["min_samples"] = st.slider("min_samples", 3, 15, 5)
    elif metode == "HDBSCAN":
        param["min_cluster_size"] = st.slider("min_cluster_size", 3, 40, 5)
    if metode in an.PAKAI_SEED:
        param["seed"] = int(st.number_input("Random seed", 0, 999, 42, help="Ganti seed untuk melihat apakah hasilnya robust."))

    with st.expander("Preprocessing & DTW"):
        batas_nol = st.select_slider("Lifecycle threshold (minggu nol beruntun)", [2, 4, 8], value=4,
                                     help="Produk dengan leading/trailing zeros sepanjang ini dianggap New product / Discontinued.")
        jendela = st.select_slider("Moving average window (minggu)", [1, 2, 4, 8], value=4, help="1 = tanpa smoothing.")
        radius = st.slider("Sakoe-Chiba band radius (minggu)", 0, 8, 4, help="0 = setara Euclidean distance (tanpa time shift).")
    st.caption("Data: Sales Transactions Dataset Weekly (Tan & Lau, 2014), 811 produk × 52 minggu.")

kunci = (batas_nol, jendela, radius)
with st.spinner("Menghitung DTW distance matrix…"):
    seg, reg, R, Z, D = siapkan(*kunci)
with st.spinner("Menjalankan K-Means DTW (± 30 detik pada run pertama)…" if metode.startswith("K-Means") else "Menjalankan clustering…"):
    label_mentah, info, sil, sil_produk = klaster(D, Z, kunci, metode, param)
label = an.urutkan_label(label_mentah, R)  # 1..K, 0 = noise
klasters = sorted(set(label[label > 0]))
wakil = an.wakil(D, label)
kode = seg.Product_Code.values[reg]
df = muat()
total_unit = df[an.MINGGU].values.sum()
Zl = an.pohon(D, metode) if metode in an.HIERARKI else None

# ---------- ringkasan ----------
st.title("DTW clustering pola penjualan mingguan")
teks_k = f" · K = {param['k']}" if "k" in param else ""
catatan(f"{len(reg)} produk Regular dibandingkan bentuk polanya dengan DTW (Sakoe-Chiba band r = {radius}), setelah moving average "
        f"{jendela} minggu dan z-normalization. Approach: <span class='tebal'>{pendekatan}</span> · Method: "
        f"<span class='tebal'>{metode}</span>{teks_k}.")

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Produk di-cluster (dari {len(seg)})", len(reg))
c2.metric("Cluster terbentuk", len(klasters))
c3.metric("Silhouette", angka(sil), help="−1 sampai 1. > 0,5 strong; 0,25–0,5 moderate; < 0,25 weak. Noise tidak dihitung.")
if Zl is not None:
    c4.metric("Cophenetic correlation", angka(float(cophenet(Zl, squareform(D, checks=False))[0])),
              help="Seberapa setia dendrogram mempertahankan DTW distance asli (mendekati 1 = setia).")
else:
    c4.metric("Noise", f"{(label == 0).sum()} produk")

ukuran = np.bincount(label, minlength=max(klasters, default=0) + 1)[1:] if klasters else np.array([])
if len(klasters) < 2:
    st.warning("Hanya satu cluster terbentuk. Produk Regular membentuk satu continuum tanpa density gap, "
               "sehingga density-based method tidak dapat memisahkannya pada parameter ini.")
elif ukuran.min() < 0.05 * len(reg):
    st.warning(f"Cluster terkecil hanya {ukuran.min()} produk (< 5%). Metode ini cenderung memisahkan outlier, bukan pola.")
elif sil is not None and sil < 0.25:
    st.info("Silhouette < 0,25 (weak structure): batas antar-cluster tidak tegas. Baca cluster sebagai kecenderungan pola, bukan kelompok yang terpisah jelas.")

TAMPILAN = ["Hasil cluster", "Validasi & pilihan K", "Zero handling", "DTW explorer", "Tabel & unduh", "Glossary"]
tampilan = st.segmented_control("Tampilan", TAMPILAN, default=TAMPILAN[0], key="tampilan", label_visibility="collapsed") or TAMPILAN[0]

# ---------- tab 1: hasil ----------
if tampilan == TAMPILAN[0]:
    if Zl is not None:
        st.subheader("Dendrogram")
        p = 40
        d = dendrogram(Zl, truncate_mode="lastp", p=p, no_plot=True)
        fig = go.Figure()
        for xs, ys in zip(d["icoord"], d["dcoord"]):
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#6b6b66", width=1.3), hoverinfo="skip", showlegend=False))
        potong = an.tinggi_potong(Zl, param["k"])
        fig.add_hline(y=potong, line=dict(color=WARNA[0], dash="dash", width=1.6),
                      annotation_text=f"cut → K = {param['k']}", annotation_position="top left")
        tata(fig, 340)
        fig.update_xaxes(tickvals=[5 + 10 * i for i in range(len(d["ivl"]))], ticktext=d["ivl"], tickangle=-90, tickfont=dict(size=10),
                         title_text=f"{p} cabang terakhir; angka dalam kurung = jumlah produk")
        fig.update_yaxes(title_text="DTW distance (merge height)")
        grafik(fig)
        catatan("Setiap sambungan menunjukkan dua kelompok yang digabung; makin tinggi sambungan, makin berbeda kedua kelompok itu. "
                "Garis putus-putus adalah cut yang menghasilkan K cluster.")

    panel = klasters + ([0] if (label == 0).any() else [])
    if panel:
        kolom = min(3, len(panel))
        baris = int(np.ceil(len(panel) / kolom))
        judul = [f"{nama(c)} · {(label == c).sum()} produk · {angka(R[label == c].sum() / total_unit * 100, 1)}% unit" for c in panel]
        fig = make_subplots(baris, kolom, subplot_titles=judul, shared_yaxes=True, horizontal_spacing=0.04, vertical_spacing=0.14)
        x = np.arange(52)
        centroid = info.get("centroid")
        for n, c in enumerate(panel):
            r, k = n // kolom + 1, n % kolom + 1
            anggota = Z[label == c]
            xs = np.tile(np.r_[x, np.nan], len(anggota)).astype(np.float32)  # float32: separuh ukuran data ke browser
            ys = np.c_[anggota, np.full(len(anggota), np.nan)].ravel().astype(np.float32)
            fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", line=dict(color="rgba(120,120,115,0.09)", width=1),
                                       hoverinfo="skip", showlegend=False), r, k)
            pusat = centroid[label_mentah[label == c][0]] if centroid is not None and c > 0 else anggota.mean(0)
            fig.add_trace(go.Scatter(x=x, y=pusat, mode="lines", line=dict(color=warna(c), width=3.2), showlegend=False,
                                     hovertemplate="Minggu %{x}<br>z = %{y:.2f}<extra></extra>"), r, k)
            if c in wakil:
                fig.add_trace(go.Scatter(x=x, y=Z[wakil[c]], mode="lines", line=dict(color=TINTA, width=1.2, dash="dot"), showlegend=False,
                                         hovertemplate=f"Medoid {kode[wakil[c]]}<br>Minggu %{{x}}: %{{y:.2f}}<extra></extra>"), r, k)
        fig.update_xaxes(title_text="Minggu ke-", row=baris)
        tata(fig, tinggi=300 * baris)
        st.subheader("Pola setiap cluster")
        garis = "DBA centroid" if centroid is not None else "rata-rata cluster"
        catatan(f"Garis abu-abu: anggota (z-normalized). Garis berwarna: {garis}. "
                "Garis titik-titik: medoid, yaitu produk nyata dengan total DTW distance terkecil ke anggota lain.")
        grafik(fig)

    if klasters:
        ka, kb = st.columns(2)
        f1, f2 = go.Figure(), go.Figure()
        for c in klasters:
            m = label == c
            f1.add_trace(go.Scatter(x=np.arange(52), y=Z[m].mean(0), name=nama(c), mode="lines", line=dict(color=warna(c), width=2),
                                    hovertemplate=f"{nama(c)}<br>Minggu %{{x}}: %{{y:.2f}}<extra></extra>"))
            f2.add_trace(go.Scatter(x=np.arange(52), y=R[m].sum(0), name=nama(c), mode="lines", line=dict(color=warna(c), width=2),
                                    hovertemplate=f"{nama(c)}<br>Minggu %{{x}}: %{{y:,.0f}} unit<extra></extra>"))
        for f, t, y in [(f1, "Shape: rata-rata pola (z-score)", "z-score"), (f2, "Volume: total unit asli", "Unit")]:
            tata(f, 330, title=dict(text=t, font=dict(size=14)))
            f.update_xaxes(title_text="Minggu ke-"); f.update_yaxes(title_text=y)
        with ka: grafik(f1)
        with kb: grafik(f2)

        st.subheader("Cluster profile")
        profil = []
        for c in klasters:
            m = label == c
            rata_z = Z[m].mean(0)
            h1, h2 = R[m][:, :25].mean(1), R[m][:, 25:].mean(1)
            perubahan = np.median((h2 / np.where(h1 == 0, np.nan, h1) - 1) * 100)
            profil.append({"Cluster": nama(c), "Produk": int(m.sum()), "Share unit (%)": round(R[m].sum() / total_unit * 100, 1),
                           "Unit/minggu": round(R[m].mean(), 1), "Peak week": int(rata_z.argmax()), "Trough week": int(rata_z.argmin()),
                           "Median change H2 vs H1 (%)": round(perubahan, 1),
                           "Silhouette": None if sil_produk is None else round(float(np.nanmean(sil_produk[m])), 3),
                           "Medoid": kode[wakil[c]]})
        st.dataframe(pd.DataFrame(profil), hide_index=True, width="stretch")
        catatan("H1 = minggu 0–24, H2 = minggu 25–51. Peak/trough week dihitung dari rata-rata pola z-score.")
        for p in profil:
            arah = "turun" if p["Median change H2 vs H1 (%)"] < 0 else "naik"
            catatan(f"<span class='tebal'>{p['Cluster']}</span>: pola rata-rata mencapai peak pada minggu ke-{p['Peak week']} "
                    f"dan trough pada minggu ke-{p['Trough week']}. Penjualan H2 {arah} "
                    f"{angka(abs(p['Median change H2 vs H1 (%)']), 1)}% (median) dibanding H1.")

# ---------- tab 2: validasi ----------
if tampilan == TAMPILAN[1]:
    if metode in an.PAKAI_K:
        lambat = metode.startswith("K-Means")
        hitung = st.checkbox("Hitung silhouette untuk K = 2 sampai 6" + (" (± 2 menit untuk K-Means)" if lambat else ""), value=not lambat)
        if hitung:
            with st.spinner("Menghitung silhouette untuk K = 2 sampai 6…"):
                skor = skor_k(D, Z, kunci, metode, {k: v for k, v in param.items() if k != "k"})
            ka, kb = st.columns(2)
            f = go.Figure(go.Scatter(x=skor.K, y=skor.silhouette, mode="lines+markers", line=dict(color=WARNA[0], width=2),
                                     marker=dict(size=9), hovertemplate="K = %{x}<br>Silhouette %{y:.3f}<extra></extra>"))
            f.add_trace(go.Scatter(x=[param["k"]], y=skor.loc[skor.K == param["k"], "silhouette"], mode="markers",
                                   marker=dict(size=15, color="rgba(0,0,0,0)", line=dict(color=TINTA, width=2)), hoverinfo="skip"))
            tata(f, 300, title=dict(text="Silhouette per K", font=dict(size=14)), showlegend=False)
            f.update_xaxes(title_text="K", dtick=1)
            g = go.Figure(go.Bar(x=skor.K, y=skor.cluster_terkecil, marker=dict(color=WARNA[0], cornerradius=4),
                                 hovertemplate="K = %{x}<br>%{y} produk<extra></extra>"))
            tata(g, 300, title=dict(text="Ukuran cluster terkecil", font=dict(size=14)))
            g.update_xaxes(title_text="K", dtick=1); g.update_yaxes(title_text="Produk")
            with ka: grafik(f)
            with kb: grafik(g)
            terbaik = skor.loc[skor.silhouette.idxmax()]
            catatan(f"Silhouette tertinggi pada K = {int(terbaik.K)} ({angka(terbaik.silhouette)}). Cek juga ukuran cluster terkecil: "
                    "silhouette tinggi dengan cluster berisi 1–2 produk berarti metode hanya memisahkan outlier.")

    st.markdown("**Robustness**")
    if metode in an.HIERARKI:
        catatan("Hierarchical clustering bersifat deterministic: input yang sama selalu memberi dendrogram yang sama, jadi tidak ada uji seed.")
    elif metode in an.PAKAI_SEED or metode == "Spectral clustering":
        uji = "5 random seed (0–4)" if metode in an.PAKAI_SEED else "5 nilai σ (0,5–2 × median)"
        jalan = st.checkbox(f"Uji robustness: bandingkan hasil dari {uji} dengan ARI" + (" (± 2 menit)" if metode.startswith("K-Means") else ""),
                            value=not metode.startswith("K-Means"))
        if jalan:
            with st.spinner("Menguji robustness…"):
                rata, minimum = robustness(D, Z, kunci, metode, {k: v for k, v in param.items() if k != "seed"})
            r1, r2 = st.columns(2)
            r1.metric("Mean ARI", angka(rata)); r2.metric("Min ARI", angka(minimum))
            tingkat = "robust" if rata >= 0.8 else "cukup robust" if rata >= 0.6 else "kurang robust"
            catatan(f"ARI (Adjusted Rand Index) = 1 berarti label identik; ≈ 0 berarti sama seperti acak. "
                    f"Mean ARI {angka(rata)} → hasil metode ini <span class='tebal'>{tingkat}</span> terhadap {uji}.")
    else:
        catatan("DBSCAN dan HDBSCAN deterministic untuk parameter yang sama; robustness dilihat dengan menggeser parameter di sidebar.")

    ka, kb = st.columns(2)
    with ka:
        st.markdown("**Silhouette per produk**")
        if sil_produk is not None:
            fig, awal = go.Figure(), 0
            for c in klasters:
                s = np.sort(sil_produk[label == c])[::-1]
                fig.add_trace(go.Bar(x=np.arange(awal, awal + len(s)), y=s, name=nama(c), marker=dict(color=warna(c), line=dict(width=0)),
                                     hovertemplate=f"{nama(c)}<br>s = %{{y:.3f}}<extra></extra>"))
                awal += len(s) + 12
            fig.add_hline(y=sil, line=dict(color=TINTA, dash="dot", width=1.2), annotation_text=f"mean {angka(sil)}")
            tata(fig, 340, bargap=0)
            fig.update_xaxes(showticklabels=False, title_text="Produk, diurutkan dalam cluster")
            fig.update_yaxes(title_text="s(i)")
            grafik(fig)
            catatan(f"{int((sil_produk < 0).sum())} produk dengan silhouette negatif: lebih dekat ke cluster lain daripada ke cluster-nya sendiri (misclassified/borderline).")
        else:
            catatan("Silhouette tidak dapat dihitung karena hanya ada satu cluster berlabel. Ubah parameter di sidebar, atau pilih metode yang memakai K.")
    with kb:
        P, porsi = peta_2d(D, kunci)
        st.markdown("**2D map DTW distance (classical MDS)**")
        tampil = klasters + ([0] if (label == 0).any() else [])
        if len(klasters) <= 3:
            fig = go.Figure()
            for c in tampil:
                m = label == c
                fig.add_trace(go.Scattergl(x=P[m, 0], y=P[m, 1], mode="markers", name=nama(c), text=kode[m],
                                           marker=dict(size=8 if c else 6, color=warna(c), symbol=SIMBOL[(c - 1) % 6] if c else "x",
                                                       line=dict(color="white", width=1 if c else 0), opacity=0.85),
                                           hovertemplate="%{text}<extra>" + nama(c) + "</extra>"))
            tata(fig, 340)
        else:
            kolom = 3
            fig = make_subplots(int(np.ceil(len(tampil) / kolom)), kolom, subplot_titles=[nama(c) for c in tampil],
                                shared_xaxes=True, shared_yaxes=True, horizontal_spacing=0.03, vertical_spacing=0.1)
            for n, c in enumerate(tampil):
                r, k = n // kolom + 1, n % kolom + 1
                m = label == c
                fig.add_trace(go.Scattergl(x=P[~m, 0], y=P[~m, 1], mode="markers", marker=dict(size=4, color="#dcdcd8"),
                                           hoverinfo="skip", showlegend=False), r, k)
                fig.add_trace(go.Scattergl(x=P[m, 0], y=P[m, 1], mode="markers", text=kode[m], showlegend=False,
                                           marker=dict(size=6, color=warna(c)), hovertemplate="%{text}<extra>" + nama(c) + "</extra>"), r, k)
            tata(fig, 200 * int(np.ceil(len(tampil) / kolom)) + 60)
        fig.update_xaxes(showticklabels=False); fig.update_yaxes(showticklabels=False)
        grafik(fig)
        catatan(f"Dua sumbu MDS hanya menangkap {porsi:.0%} variasi DTW distance, jadi overlap di peta tidak selalu berarti overlap sebenarnya.")

    if metode == "Spectral clustering":
        eig = an.nilai_eigen(D, param["faktor_sigma"])
        fig = go.Figure(go.Scatter(x=np.arange(1, 9), y=eig, mode="lines+markers", line=dict(color=WARNA[0], width=2), marker=dict(size=9),
                                   hovertemplate="λ%{x} = %{y:.3f}<extra></extra>"))
        tata(fig, 280, title=dict(text=f"Eigenvalues normalized Laplacian (σ = {angka(info['sigma'], 2)})", font=dict(size=14)))
        fig.update_xaxes(title_text="Urutan eigenvalue", dtick=1)
        grafik(fig)
        gap = np.diff(eig)
        catatan(f"Eigengap terbesar setelah λ1 ({angka(gap[0])}) → similarity graph sangat terhubung (satu continuum). "
                f"Di antara gap berikutnya, yang terbesar ada setelah λ{int(gap[1:].argmax()) + 2} → petunjuk K = {int(gap[1:].argmax()) + 2}.")
    elif metode == "Fuzzy C-Medoids":
        U = info["U"]
        fig = go.Figure(go.Histogram(x=U.max(1), nbinsx=25, marker=dict(color=WARNA[0], line=dict(color="white", width=1)),
                                     hovertemplate="max membership %{x}<br>%{y} produk<extra></extra>"))
        tata(fig, 280, title=dict(text=f"Max membership tiap produk · Partition Coefficient (PC) = {angka(info['PC'])}", font=dict(size=14)))
        fig.update_xaxes(title_text="Max membership", range=[1 / param["k"], 1]); fig.update_yaxes(title_text="Produk")
        grafik(fig)
        catatan(f"PC = 1 berarti crisp partition; PC = 1/K = {angka(1 / param['k'])} berarti fully fuzzy. "
                f"{(U.max(1) > 0.7).mean():.0%} produk punya membership dominan (> 0,7).")
    elif metode == "DBSCAN":
        kd = np.sort(np.sort(D, 1)[:, param["min_samples"]])
        fig = go.Figure(go.Scatter(y=kd, mode="lines", line=dict(color=WARNA[0], width=2), hovertemplate="%{y:.2f}<extra></extra>"))
        fig.add_hline(y=param["eps"], line=dict(color=TINTA, dash="dot", width=1.2), annotation_text=f"eps = {angka(param['eps'], 1)}")
        tata(fig, 280, title=dict(text=f"k-distance plot (neighbor ke-{param['min_samples']})", font=dict(size=14)))
        fig.update_xaxes(title_text="Produk (diurutkan)"); fig.update_yaxes(title_text="DTW distance")
        grafik(fig)
        catatan("Kurva naik landai tanpa elbow di tengah: tidak ada density gap yang memisahkan kelompok. "
                "Elbow hanya muncul di ujung kanan, pada beberapa outlier.")
    elif metode == "HDBSCAN":
        catatan(f"HDBSCAN hanya mengakui cluster yang stabil di semua tingkat density. {(label == 0).mean():.0%} produk menjadi noise; "
                "silhouette di atas hanya dihitung pada produk berlabel.")
    elif metode.startswith("K-Means"):
        catatan("Catatan metodologis: centroid K-Means biasa adalah rata-rata per minggu, yang tidak konsisten dengan DTW. "
                "DBA menghitung centroid dari pasangan minggu hasil DTW, sehingga K-Means tetap sah dipakai bersama DTW.")

# ---------- tab 3: zero handling ----------
if tampilan == TAMPILAN[2]:
    X = df[an.MINGGU].values
    ringkas = seg.assign(unit=X.mean(1)).groupby("segmen").agg(Produk=("Product_Code", "size"), **{"Unit/minggu": ("unit", "mean"),
                                                                "Minggu bernilai 0": ("minggu_nol", "mean")}).round(2)
    ringkas["Share unit (%)"] = (pd.Series(X.sum(1), index=seg.index).groupby(seg.segmen).sum() / total_unit * 100).round(1)
    ringkas["Treatment"] = np.where(ringkas.index == "Regular", "DTW clustering", "Segmen tersendiri")
    ringkas.index.name = "Segment"
    ka, kb = st.columns([1.15, 1])
    with ka:
        st.markdown("**Segmentasi hasil zero handling**")
        st.dataframe(ringkas.sort_values("Produk", ascending=False), width="stretch")
        catatan(f"Tidak ada missing value: minggu yang tampak hilang bernilai 0 ({angka((X == 0).mean() * 100, 1)}% sel). "
                "Tidak dilakukan imputation karena 0 berarti memang tidak ada pembelian. Produk dipilah menurut posisi zero "
                "(leading/trailing = lifecycle) dan demand pattern ADI–CV² (Syntetos dkk., 2005). Hanya produk Regular "
                "yang shape-nya layak dibandingkan dengan DTW.")
    with kb:
        aktif = seg[seg.segmen.isin(["Regular", "Intermittent"])]
        fig = go.Figure()
        for (pola, sim), w, kel in [(("smooth", "circle"), WARNA[0], "Regular"), (("erratic", "diamond"), WARNA[0], "Regular"),
                                    (("intermittent", "square"), WARNA[1], "Intermittent"), (("lumpy", "triangle-up"), WARNA[1], "Intermittent")]:
            s = aktif[aktif.pola == pola]
            fig.add_trace(go.Scattergl(x=s.ADI, y=s.CV2, mode="markers", name=f"{pola} ({len(s)}) → {kel}", text=s.Product_Code,
                                       marker=dict(size=7, color=w, symbol=sim, opacity=0.75, line=dict(color="white", width=0.8)),
                                       hovertemplate="%{text}<br>ADI %{x:.2f} · CV² %{y:.2f}<extra></extra>"))
        fig.add_vline(x=an.ADI_BATAS, line=dict(color=TINTA_2, dash="dot", width=1))
        fig.add_hline(y=an.CV2_BATAS, line=dict(color=TINTA_2, dash="dot", width=1))
        tata(fig, 360, title=dict(text="Demand pattern classification (ADI–CV²)", font=dict(size=14)))
        fig.update_xaxes(type="log", title_text="ADI (average demand interval, log scale)"); fig.update_yaxes(title_text="CV²")
        grafik(fig)

    urut = np.lexsort(((X == 0).sum(1), seg.nol_awal.values))
    fig = go.Figure(go.Heatmap(z=(X[urut] > 0).astype(int), y=seg.Product_Code.values[urut], colorscale=[[0, "#ffffff"], [1, "#256abf"]],
                               showscale=False, hovertemplate="%{y} · minggu %{x}<br>%{z:.0f} = terjual<extra></extra>"))
    tata(fig, 420, title=dict(text="Sales presence map: biru = terjual, putih = 0 (diurutkan menurut minggu penjualan pertama)", font=dict(size=14)))
    fig.update_xaxes(title_text="Minggu ke-"); fig.update_yaxes(showticklabels=False, title_text="Produk", autorange="reversed")
    grafik(fig)

# ---------- tab 4: DTW explorer ----------
if tampilan == TAMPILAN[3]:
    catatan("DTW mencocokkan minggu secara fleksibel (time warping) agar pola yang sama tetapi bergeser tetap terbaca mirip. "
            "Sakoe-Chiba band membatasi time shift paling jauh r minggu.")
    pilihan_produk = list(kode)
    ka, kb = st.columns(2)
    p1 = ka.selectbox("Produk A", pilihan_produk, index=pilihan_produk.index("P1") if "P1" in pilihan_produk else 0)
    p2 = kb.selectbox("Produk B", pilihan_produk, index=pilihan_produk.index("P6") if "P6" in pilihan_produk else 1)
    a, b = Z[pilihan_produk.index(p1)], Z[pilihan_produk.index(p2)]
    d_bebas, jalur_bebas = an.jalur_dtw(a, b)
    d_pita, jalur_pita = an.jalur_dtw(a, b, radius)
    m1, m2, m3 = st.columns(3)
    m1.metric("Euclidean distance", angka(float(np.linalg.norm(a - b))))
    m2.metric("DTW (unconstrained)", angka(d_bebas))
    m3.metric(f"DTW (Sakoe-Chiba r = {radius})", angka(d_pita))

    ka, kb = st.columns([3, 2])
    with ka:
        geser = 4.5
        fig = go.Figure()
        for i, j in jalur_pita:
            fig.add_trace(go.Scatter(x=[i, j], y=[a[i] + geser, b[j]], mode="lines", line=dict(color="#c9c9c4", width=0.8),
                                     hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=np.arange(52), y=a + geser, name=p1, mode="lines", line=dict(color=WARNA[0], width=2.2)))
        fig.add_trace(go.Scatter(x=np.arange(52), y=b, name=p2, mode="lines", line=dict(color=WARNA[1], width=2.2)))
        tata(fig, 380, title=dict(text=f"Week alignment (DTW, r = {radius})", font=dict(size=14)))
        fig.update_yaxes(showticklabels=False); fig.update_xaxes(title_text="Minggu ke-")
        grafik(fig)
    with kb:
        fig = go.Figure()
        t = np.arange(52)
        fig.add_trace(go.Scatter(x=np.r_[t, t[::-1]], y=np.r_[t - radius, (t + radius)[::-1]], fill="toself", fillcolor="rgba(42,120,214,0.10)",
                                 line=dict(width=0), name=f"Band ±{radius}", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=jalur_bebas[:, 0], y=jalur_bebas[:, 1], mode="lines", name="Unconstrained", line=dict(color=WARNA[1], width=2)))
        fig.add_trace(go.Scatter(x=jalur_pita[:, 0], y=jalur_pita[:, 1], mode="lines", name="Sakoe-Chiba", line=dict(color=TINTA, width=2, dash="dot")))
        tata(fig, 380, title=dict(text="Warping path", font=dict(size=14)))
        fig.update_xaxes(title_text=f"Minggu {p1}", range=[0, 51]); fig.update_yaxes(title_text=f"Minggu {p2}", range=[0, 51])
        grafik(fig)
    geser_maks = int(np.abs(jalur_bebas[:, 0] - jalur_bebas[:, 1]).max())
    catatan(f"Tanpa band, DTW menggeser sampai {geser_maks} minggu untuk memaksakan kemiripan (pathological warping). "
            f"Band r = {radius} menahan time shift itu, sehingga jaraknya berada di antara Euclidean dan DTW unconstrained.")

# ---------- tab 5: tabel ----------
if tampilan == TAMPILAN[4]:
    hasil = seg[["Product_Code", "ADI", "CV2", "pola", "segmen"]].rename(columns={"pola": "demand_pattern", "segmen": "segment"})
    hasil["cluster"] = ""
    hasil.loc[reg, "cluster"] = [nama(c) for c in label]
    if sil_produk is not None:
        hasil.loc[reg, "silhouette"] = np.round(sil_produk, 3)
    hasil["final_label"] = np.where(hasil.segment == "Regular", "Regular - " + hasil.cluster, hasil.segment)
    st.dataframe(hasil, hide_index=True, width="stretch", height=420)
    nama_file = f"label_{metode.split()[0].lower()}_{'k' + str(param['k']) if 'k' in param else 'auto'}.csv"
    st.download_button("Unduh label (CSV)", hasil.to_csv(index=False).encode("utf-8"), file_name=nama_file, mime="text/csv")

# ---------- tab 6: glossary ----------
if tampilan == TAMPILAN[5]:
    istilah = [
        ("DTW (Dynamic Time Warping)", "Jarak antar dua time series yang boleh menggeser waktu agar pola yang sama tetapi terlambat/lebih cepat tetap dinilai mirip."),
        ("Sakoe-Chiba band", "Batas time shift maksimum (r minggu) pada DTW, mencegah pathological warping."),
        ("Pathological warping", "Pencocokan minggu yang terlalu jauh, sehingga dua pola berbeda tampak mirip."),
        ("Z-normalization", "Mengubah tiap deret menjadi mean 0 dan standard deviation 1, agar yang dibandingkan adalah shape, bukan volume."),
        ("Moving average", "Smoothing: tiap minggu diganti rata-rata beberapa minggu terakhir untuk meredam fluktuasi acak."),
        ("Leading / trailing zeros", "Minggu bernilai 0 sebelum penjualan pertama / setelah penjualan terakhir (indikasi lifecycle produk)."),
        ("ADI / CV²", "ADI = average demand interval (rata-rata jarak antar pembelian). CV² = squared coefficient of variation ukuran pembelian."),
        ("Hierarchical vs non-hierarchical", "Hierarchical membangun dendrogram dari penggabungan bertahap; non-hierarchical langsung membagi data menjadi K cluster."),
        ("Centroid / medoid", "Centroid = titik rata-rata cluster (bisa deret buatan). Medoid = produk nyata yang paling sentral di cluster."),
        ("DBA (DTW Barycenter Averaging)", "Cara menghitung centroid yang konsisten dengan DTW, dipakai K-Means DTW."),
        ("Linkage", "Aturan jarak antar-cluster pada hierarchical clustering: complete (terjauh), average (rata-rata), single (terdekat)."),
        ("Chaining effect", "Kelemahan single linkage: cluster memanjang seperti rantai sehingga satu cluster menelan hampir semua produk."),
        ("Dendrogram / cut", "Pohon penggabungan cluster; memotong (cut) pada ketinggian tertentu menghasilkan K cluster."),
        ("Cophenetic correlation", "Korelasi antara jarak asli dan jarak di dendrogram; mendekati 1 = dendrogram setia pada data."),
        ("Density-based / noise", "Cluster = area padat; produk di area renggang tidak diberi label (noise)."),
        ("Graph-based / eigengap", "Spectral clustering memotong similarity graph; eigengap (lompatan eigenvalue) memberi petunjuk jumlah cluster."),
        ("Membership / fuzziness (m)", "Pada fuzzy clustering, membership = derajat keanggotaan 0–1; m mengatur seberapa fuzzy pembagiannya."),
        ("Partition Coefficient (PC)", "Ukuran crisp vs fuzzy: 1 = crisp, 1/K = fully fuzzy."),
        ("Silhouette", "Seberapa dekat produk ke cluster-nya dibanding cluster lain (−1 sampai 1). Negatif = borderline/misclassified."),
        ("ARI (Adjusted Rand Index)", "Kesamaan dua hasil clustering setelah dikoreksi faktor kebetulan (1 = identik, 0 ≈ acak)."),
        ("Robust", "Hasil tidak banyak berubah ketika seed atau parameter diubah sedikit."),
        ("Classical MDS", "Proyeksi matriks jarak ke 2 dimensi untuk visualisasi; hanya pendekatan."),
    ]
    st.markdown("| Istilah | Arti |\n|---|---|\n" + "\n".join(f"| **{t}** | {a} |" for t, a in istilah))
