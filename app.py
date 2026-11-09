from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import analisis as an

st.set_page_config(page_title="Klaster DTW Penjualan Mingguan", layout="wide", initial_sidebar_state="auto")

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
.label-klaster { font-weight: 600; }
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
    return "Noise" if c == 0 else f"Klaster {c}"


def angka(x, d=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


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
    reg = seg[seg.segmen == "Reguler"].index.values
    R = df.loc[reg, an.MINGGU].values.astype(float)
    Z = an.praproses(R, jendela)
    return seg, reg, R, Z, an.matriks_dtw(Z, radius)


@st.cache_data
def klaster(_D, kunci, metode, param):
    label, info = an.jalankan(_D, metode, **param)
    s_rata, s_produk = an.silhouette(_D, label)
    return label, info, s_rata, s_produk


@st.cache_data
def skor_k(_D, kunci, metode, param):
    p = {k: v for k, v in param.items() if k != "k"}
    return an.skor_per_k(_D, metode, **p)


@st.cache_data
def peta_2d(_D, kunci):
    return an.mds_klasik(_D)


# ---------- sidebar: input ----------
with st.sidebar:
    st.header("Pengaturan")
    metode = st.selectbox("Metode klaster", list(an.METODE), index=0,
                          help="Semua metode bekerja langsung dari matriks jarak DTW; K-Means tidak dipakai karena rata-ratanya tidak konsisten dengan DTW.")
    st.caption({"graf": "Berbasis graf: memotong graf kemiripan DTW di bagian yang paling lemah.",
                "fuzzy": "Fuzzy: tiap produk punya derajat keanggotaan di semua klaster; pusatnya produk nyata (medoid).",
                "hierarki": "Hierarki: menggabungkan produk/klaster terdekat bertahap, lalu dipotong menjadi K klaster.",
                "kepadatan": "Kepadatan: klaster = daerah padat; produk di daerah jarang menjadi noise. K tidak diinput."}[an.METODE[metode]])
    param = {}
    if metode in an.PAKAI_K:
        param["k"] = st.slider("Jumlah klaster (K)", 2, 6, 2)
    if metode == "Spectral clustering":
        param["faktor_sigma"] = st.select_slider("Skala kernel σ (× median jarak)", [0.5, 0.75, 1.0, 1.5, 2.0], value=1.0,
                                                 help="σ besar: lebih banyak pasangan dianggap mirip. Label diberikan dengan cluster_qr, tanpa K-Means.")
    elif metode == "Fuzzy C-Medoids":
        param["m"] = st.slider("Derajat kekaburan m", 1.1, 2.0, 1.3, 0.1,
                               help="m mendekati 1 = klaster tegas; m = 2 = nilai baku FCM.")
    elif metode == "DBSCAN":
        param["eps"] = st.slider("eps (radius tetangga)", 2.5, 6.0, 4.5, 0.1)
        param["min_samples"] = st.slider("min_samples", 3, 15, 5)
    elif metode == "HDBSCAN":
        param["min_cluster_size"] = st.slider("min_cluster_size", 3, 40, 5)

    with st.expander("Praproses dan DTW"):
        batas_nol = st.select_slider("Batas nol beruntun awal/akhir (minggu)", [2, 4, 8], value=4,
                                     help="Produk dengan nol beruntun sepanjang ini di awal/akhir deret dianggap belum atau sudah berhenti dijual.")
        jendela = st.select_slider("Rata-rata bergerak (minggu)", [1, 2, 4, 8], value=4, help="1 = tanpa penghalusan.")
        radius = st.slider("Radius pita Sakoe-Chiba (minggu)", 0, 8, 4, help="0 = setara jarak Euclidean (tanpa pergeseran waktu).")
    st.caption("Data: Sales Transactions Dataset Weekly (Tan & Lau, 2014), 811 produk × 52 minggu.")

kunci = (batas_nol, jendela, radius)
with st.spinner("Menghitung matriks jarak DTW…"):
    seg, reg, R, Z, D = siapkan(*kunci)
label_mentah, info, sil, sil_produk = klaster(D, kunci, metode, param)
label = an.urutkan_label(label_mentah, R)  # 1..K, 0 = noise
klasters = sorted(set(label[label > 0]))
wakil = an.wakil(D, label)
kode = seg.Product_Code.values[reg]
df = muat()
total_unit = df[an.MINGGU].values.sum()

# ---------- ringkasan ----------
st.title("Klaster pola penjualan mingguan dengan DTW")
st.markdown(f"<p class='catatan'>{len(reg)} produk reguler dibandingkan bentuk polanya dengan DTW berpita Sakoe-Chiba r = {radius}, "
            f"setelah rata-rata bergerak {jendela} minggu dan z-normalisasi. Metode aktif: <b>{metode}</b>"
            f"{f', K = {param['k']}' if 'k' in param else ''}.</p>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Produk diklaster (dari {len(seg)})", len(reg))
c2.metric("Klaster terbentuk", len(klasters))
c3.metric("Silhouette", angka(sil), help="−1 sampai 1. Di atas 0,5 kuat; 0,25–0,5 sedang; di bawah 0,25 lemah. Noise tidak dihitung.")
c4.metric("Noise (tanpa label)", f"{(label == 0).sum()} produk")

ukuran = np.bincount(label, minlength=max(klasters, default=0) + 1)[1:] if klasters else np.array([])
if len(klasters) < 2:
    st.warning("Hanya satu klaster terbentuk. Produk reguler membentuk satu awan tanpa celah kepadatan; "
               "metode berbasis kepadatan tidak dapat memisahkannya pada parameter ini.")
elif ukuran.min() < 0.05 * len(reg):
    st.warning(f"Klaster terkecil hanya {ukuran.min()} produk (< 5%). Metode ini cenderung memisahkan pencilan, bukan pola.")
elif sil is not None and sil < 0.25:
    st.info("Silhouette di bawah 0,25: batas antarklaster tidak tegas. Baca klaster sebagai kecenderungan pola, bukan kelompok terpisah.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Hasil klaster", "Kualitas dan pilihan K", "Penanganan nilai nol", "DTW dua produk", "Tabel dan unduh"])

# ---------- tab 1: hasil ----------
with tab1:
    panel = klasters + ([0] if (label == 0).any() else [])
    if panel:
        kolom = min(3, len(panel))
        baris = int(np.ceil(len(panel) / kolom))
        judul = [f"{nama(c)} · {(label == c).sum()} produk · {R[label == c].sum() / total_unit:.1%} unit".replace(".", ",", 1)
                 for c in panel]
        fig = make_subplots(baris, kolom, subplot_titles=judul, shared_yaxes=True, horizontal_spacing=0.04, vertical_spacing=0.14)
        x = np.arange(52)
        for n, c in enumerate(panel):
            r, k = n // kolom + 1, n % kolom + 1
            anggota = Z[label == c]
            xs = np.tile(np.r_[x, np.nan], len(anggota))
            ys = np.c_[anggota, np.full(len(anggota), np.nan)].ravel()
            fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", line=dict(color="rgba(120,120,115,0.09)", width=1),
                                       hoverinfo="skip", showlegend=False), r, k)
            fig.add_trace(go.Scatter(x=x, y=anggota.mean(0), mode="lines", line=dict(color=warna(c), width=3.2),
                                     name=f"Rata-rata {nama(c)}", showlegend=False,
                                     hovertemplate="Minggu %{x}<br>Rata-rata z: %{y:.2f}<extra></extra>"), r, k)
            if c in wakil:
                fig.add_trace(go.Scatter(x=x, y=Z[wakil[c]], mode="lines", line=dict(color=TINTA, width=1.2, dash="dot"),
                                         name=f"Wakil {kode[wakil[c]]}", showlegend=False,
                                         hovertemplate=f"{kode[wakil[c]]}<br>Minggu %{{x}}: %{{y:.2f}}<extra></extra>"), r, k)
        fig.update_xaxes(title_text="Minggu ke-", row=baris)
        tata(fig, tinggi=300 * baris)
        st.subheader("Pola setiap klaster")
        st.markdown("<p class='catatan'>Garis abu-abu: anggota (deret ternormalisasi). Garis berwarna: rata-rata klaster. "
                    "Garis titik-titik: produk wakil, yaitu anggota dengan total jarak DTW terkecil ke anggota lain.</p>", unsafe_allow_html=True)
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
        for f, t, y in [(f1, "Bentuk pola (rata-rata z)", "z"), (f2, "Volume asli (total unit)", "Unit")]:
            tata(f, 330, title=dict(text=t, font=dict(size=14)))
            f.update_xaxes(title_text="Minggu ke-"); f.update_yaxes(title_text=y)
        with ka: grafik(f1)
        with kb: grafik(f2)

        st.subheader("Profil klaster")
        profil = []
        for c in klasters:
            m = label == c
            rata_z = Z[m].mean(0)
            h1, h2 = R[m][:, :25].mean(1), R[m][:, 25:].mean(1)
            perubahan = np.median((h2 / np.where(h1 == 0, np.nan, h1) - 1) * 100)
            profil.append({"Klaster": nama(c), "Produk": int(m.sum()), "Porsi unit (%)": round(R[m].sum() / total_unit * 100, 1),
                           "Unit/minggu": round(R[m].mean(), 1), "Minggu puncak": int(rata_z.argmax()), "Minggu terendah": int(rata_z.argmin()),
                           "Median perubahan W25–51 vs W0–24 (%)": round(perubahan, 1),
                           "Silhouette": None if sil_produk is None else round(float(np.nanmean(sil_produk[m])), 3),
                           "Produk wakil": kode[wakil[c]]})
        st.dataframe(pd.DataFrame(profil), hide_index=True, width="stretch")
        for p in profil:
            arah = "turun" if p["Median perubahan W25–51 vs W0–24 (%)"] < 0 else "naik"
            st.markdown(f"<p class='catatan'><span class='label-klaster'>{p['Klaster']}</span>: rata-rata pola memuncak pada minggu ke-{p['Minggu puncak']} "
                        f"dan paling rendah pada minggu ke-{p['Minggu terendah']}. Penjualan paruh kedua {arah} "
                        f"{angka(abs(p['Median perubahan W25–51 vs W0–24 (%)']), 1)}% (median) dibanding paruh pertama.</p>", unsafe_allow_html=True)

# ---------- tab 2: kualitas ----------
with tab2:
    if metode in an.PAKAI_K:
        with st.spinner("Menghitung silhouette untuk K = 2 sampai 6…"):
            skor = skor_k(D, kunci, metode, param)
        ka, kb = st.columns(2)
        f = go.Figure(go.Scatter(x=skor.K, y=skor.silhouette, mode="lines+markers", line=dict(color=WARNA[0], width=2),
                                 marker=dict(size=9), hovertemplate="K = %{x}<br>Silhouette %{y:.3f}<extra></extra>"))
        f.add_trace(go.Scatter(x=[param["k"]], y=skor.loc[skor.K == param["k"], "silhouette"], mode="markers",
                               marker=dict(size=15, color="rgba(0,0,0,0)", line=dict(color=TINTA, width=2)), name="K terpilih", hoverinfo="skip"))
        tata(f, 300, title=dict(text="Silhouette menurut K", font=dict(size=14)), showlegend=False)
        f.update_xaxes(title_text="K", dtick=1)
        g = go.Figure(go.Bar(x=skor.K, y=skor.klaster_terkecil, marker=dict(color=WARNA[0], cornerradius=4),
                             hovertemplate="K = %{x}<br>%{y} produk<extra></extra>"))
        tata(g, 300, title=dict(text="Ukuran klaster terkecil", font=dict(size=14)))
        g.update_xaxes(title_text="K", dtick=1); g.update_yaxes(title_text="Produk")
        with ka: grafik(f)
        with kb: grafik(g)
        terbaik = skor.loc[skor.silhouette.idxmax()]
        st.markdown(f"<p class='catatan'>Silhouette tertinggi pada K = {int(terbaik.K)} ({angka(terbaik.silhouette)}). "
                    "Periksa juga ukuran klaster terkecil: silhouette tinggi dengan klaster berisi 1–2 produk berarti metode hanya memisahkan pencilan.</p>",
                    unsafe_allow_html=True)

    ka, kb = st.columns(2)
    with ka:
        if sil_produk is not None:
            st.markdown("**Silhouette per produk**")
            fig, awal = go.Figure(), 0
            for c in klasters:
                s = np.sort(sil_produk[label == c])[::-1]
                fig.add_trace(go.Bar(x=np.arange(awal, awal + len(s)), y=s, name=nama(c), marker=dict(color=warna(c), line=dict(width=0)),
                                     hovertemplate=f"{nama(c)}<br>s = %{{y:.3f}}<extra></extra>"))
                awal += len(s) + 12  # celah antarklaster
            fig.add_hline(y=sil, line=dict(color=TINTA, dash="dot", width=1.2), annotation_text=f"rata-rata {angka(sil)}")
            tata(fig, 340, bargap=0)
            fig.update_xaxes(showticklabels=False, title_text="Produk, diurutkan dalam klaster")
            fig.update_yaxes(title_text="s(i)")
            grafik(fig)
            st.markdown(f"<p class='catatan'>{int((sil_produk < 0).sum())} produk bersilhouette negatif: lebih dekat ke klaster lain daripada ke klasternya sendiri.</p>",
                        unsafe_allow_html=True)
        else:
            st.markdown("**Silhouette per produk**")
            st.markdown("<p class='catatan'>Silhouette tidak dapat dihitung karena hanya ada satu klaster berlabel. "
                        "Ubah parameter di sidebar, atau pilih metode yang memakai K.</p>", unsafe_allow_html=True)
    with kb:
        P, porsi = peta_2d(D, kunci)
        st.markdown("**Peta 2D jarak DTW**")
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
        st.markdown(f"<p class='catatan'>Proyeksi MDS klasik dari matriks DTW. Dua sumbu ini hanya menangkap {porsi:.0%} variasi jarak, "
                    "jadi tumpang tindih di peta tidak selalu berarti tumpang tindih sebenarnya.</p>", unsafe_allow_html=True)

    if metode == "Spectral clustering":
        eig = an.nilai_eigen(D, param["faktor_sigma"])
        fig = go.Figure(go.Scatter(x=np.arange(1, 9), y=eig, mode="lines+markers", line=dict(color=WARNA[0], width=2), marker=dict(size=9),
                                   hovertemplate="λ%{x} = %{y:.3f}<extra></extra>"))
        tata(fig, 280, title=dict(text=f"Nilai eigen Laplacian (σ = {angka(info['sigma'], 2)})", font=dict(size=14)))
        fig.update_xaxes(title_text="Urutan", dtick=1)
        grafik(fig)
        gap = np.diff(eig)
        st.markdown(f"<p class='catatan'>Lompatan terbesar setelah λ1 ({angka(gap[0])}) menandakan graf sangat terhubung. "
                    f"Di antara lompatan berikutnya, yang terbesar ada setelah λ{int(gap[1:].argmax()) + 2} → petunjuk K = {int(gap[1:].argmax()) + 2}.</p>",
                    unsafe_allow_html=True)
    elif metode == "Fuzzy C-Medoids":
        U = info["U"]
        fig = go.Figure(go.Histogram(x=U.max(1), nbinsx=25, marker=dict(color=WARNA[0], line=dict(color="white", width=1)),
                                     hovertemplate="u maks %{x}<br>%{y} produk<extra></extra>"))
        tata(fig, 280, title=dict(text=f"Keanggotaan tertinggi tiap produk · PC = {angka(info['PC'])}", font=dict(size=14)))
        fig.update_xaxes(title_text="u maks", range=[1 / param["k"], 1]); fig.update_yaxes(title_text="Produk")
        grafik(fig)
        st.markdown(f"<p class='catatan'>PC = 1 berarti partisi tegas; PC = 1/K = {angka(1 / param['k'])} berarti kabur total. "
                    f"{(U.max(1) > 0.7).mean():.0%} produk punya keanggotaan dominan (> 0,7).</p>", unsafe_allow_html=True)
    elif metode == "DBSCAN":
        kd = np.sort(np.sort(D, 1)[:, param["min_samples"]])
        fig = go.Figure(go.Scatter(y=kd, mode="lines", line=dict(color=WARNA[0], width=2), hovertemplate="%{y:.2f}<extra></extra>"))
        fig.add_hline(y=param["eps"], line=dict(color=TINTA, dash="dot", width=1.2), annotation_text=f"eps = {angka(param['eps'], 1)}")
        tata(fig, 280, title=dict(text=f"k-distance plot (tetangga ke-{param['min_samples']})", font=dict(size=14)))
        fig.update_xaxes(title_text="Produk (diurutkan)"); fig.update_yaxes(title_text="Jarak DTW")
        grafik(fig)
        st.markdown("<p class='catatan'>Kurva naik landai tanpa siku di tengah: tidak ada daerah jarang yang memisahkan kelompok. "
                    "Siku hanya muncul di ujung kanan, pada beberapa produk pencilan.</p>", unsafe_allow_html=True)
    elif metode == "HDBSCAN":
        st.markdown(f"<p class='catatan'>HDBSCAN hanya mengakui kelompok yang stabil di sepanjang rentang kepadatan. "
                    f"{(label == 0).mean():.0%} produk tidak memperoleh label (noise); silhouette di atas hanya dihitung pada produk berlabel.</p>",
                    unsafe_allow_html=True)

# ---------- tab 3: nilai nol ----------
with tab3:
    X = df[an.MINGGU].values
    ringkas = seg.assign(unit=X.mean(1)).groupby("segmen").agg(Produk=("Product_Code", "size"), **{"Unit/minggu": ("unit", "mean"),
                                                                "Minggu bernilai nol": ("minggu_nol", "mean")}).round(2)
    ringkas["Porsi unit (%)"] = (pd.Series(X.sum(1), index=seg.index).groupby(seg.segmen).sum() / total_unit * 100).round(1)
    ringkas["Perlakuan"] = np.where(ringkas.index == "Reguler", "Klaster DTW", "Segmen tersendiri")
    ka, kb = st.columns([1.15, 1])
    with ka:
        st.markdown("**Segmen hasil penanganan nol**")
        st.dataframe(ringkas.sort_values("Produk", ascending=False), width="stretch")
        st.markdown(f"<p class='catatan'>Tidak ada sel kosong: minggu yang tampak hilang bernilai 0 ({angka((X == 0).mean() * 100, 1)}% sel). "
                    "Nol tidak diimputasi karena berarti memang tidak ada pembelian. Produk dipilah menurut posisi nol (awal/akhir = siklus hidup) "
                    "dan pola permintaan ADI–CV² (Syntetos dkk., 2005); hanya produk reguler yang bentuk deretnya layak dibandingkan dengan DTW.</p>",
                    unsafe_allow_html=True)
    with kb:
        aktif = seg[seg.segmen.isin(["Reguler", "Intermiten"])]
        fig = go.Figure()
        for (pola, sim), w, kel in [(("smooth", "circle"), WARNA[0], "Reguler"), (("erratic", "diamond"), WARNA[0], "Reguler"),
                                    (("intermittent", "square"), WARNA[1], "Intermiten"), (("lumpy", "triangle-up"), WARNA[1], "Intermiten")]:
            s = aktif[aktif.pola == pola]
            fig.add_trace(go.Scattergl(x=s.ADI, y=s.CV2, mode="markers", name=f"{pola} ({len(s)}) → {kel.lower()}", text=s.Product_Code,
                                       marker=dict(size=7, color=w, symbol=sim, opacity=0.75, line=dict(color="white", width=0.8)),
                                       hovertemplate="%{text}<br>ADI %{x:.2f} · CV² %{y:.2f}<extra></extra>"))
        fig.add_vline(x=an.ADI_BATAS, line=dict(color=TINTA_2, dash="dot", width=1))
        fig.add_hline(y=an.CV2_BATAS, line=dict(color=TINTA_2, dash="dot", width=1))
        tata(fig, 360, title=dict(text="Klasifikasi ADI–CV² produk aktif setahun", font=dict(size=14)))
        fig.update_xaxes(type="log", title_text="ADI (skala log)"); fig.update_yaxes(title_text="CV²")
        grafik(fig)

    urut = np.lexsort(((X == 0).sum(1), seg.nol_awal.values))
    fig = go.Figure(go.Heatmap(z=(X[urut] > 0).astype(int), colorscale=[[0, "#ffffff"], [1, "#256abf"]], showscale=False,
                               customdata=np.repeat(seg.Product_Code.values[urut][:, None], 52, 1),
                               hovertemplate="%{customdata} · minggu %{x}<br>%{z:.0f} = terjual<extra></extra>"))
    tata(fig, 420, title=dict(text="Peta penjualan: biru = terjual, putih = 0 (produk diurutkan menurut minggu penjualan pertama)", font=dict(size=14)))
    fig.update_xaxes(title_text="Minggu ke-"); fig.update_yaxes(showticklabels=False, title_text="Produk", autorange="reversed")
    grafik(fig)

# ---------- tab 4: DTW dua produk ----------
with tab4:
    st.markdown("<p class='catatan'>DTW memasangkan minggu secara lentur agar pola yang sama tetapi bergeser tetap terbaca mirip. "
                "Pita Sakoe-Chiba membatasi pergeseran paling jauh r minggu.</p>", unsafe_allow_html=True)
    pilihan = list(kode)
    ka, kb = st.columns(2)
    p1 = ka.selectbox("Produk A", pilihan, index=pilihan.index("P1") if "P1" in pilihan else 0)
    p2 = kb.selectbox("Produk B", pilihan, index=pilihan.index("P6") if "P6" in pilihan else 1)
    a, b = Z[pilihan.index(p1)], Z[pilihan.index(p2)]
    d_bebas, jalur_bebas = an.jalur_dtw(a, b)
    d_pita, jalur_pita = an.jalur_dtw(a, b, radius)
    m1, m2, m3 = st.columns(3)
    m1.metric("Euclidean", angka(float(np.linalg.norm(a - b))))
    m2.metric("DTW tanpa batas", angka(d_bebas))
    m3.metric(f"DTW pita r = {radius}", angka(d_pita))

    ka, kb = st.columns([3, 2])
    with ka:
        geser = 4.5
        fig = go.Figure()
        for i, j in jalur_pita[::1]:
            fig.add_trace(go.Scatter(x=[i, j], y=[a[i] + geser, b[j]], mode="lines", line=dict(color="#c9c9c4", width=0.8),
                                     hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=np.arange(52), y=a + geser, name=p1, mode="lines", line=dict(color=WARNA[0], width=2.2)))
        fig.add_trace(go.Scatter(x=np.arange(52), y=b, name=p2, mode="lines", line=dict(color=WARNA[1], width=2.2)))
        tata(fig, 380, title=dict(text=f"Pasangan minggu menurut DTW pita r = {radius}", font=dict(size=14)))
        fig.update_yaxes(showticklabels=False); fig.update_xaxes(title_text="Minggu ke-")
        grafik(fig)
    with kb:
        fig = go.Figure()
        t = np.arange(52)
        fig.add_trace(go.Scatter(x=np.r_[t, t[::-1]], y=np.r_[t - radius, (t + radius)[::-1]], fill="toself", fillcolor="rgba(42,120,214,0.10)",
                                 line=dict(width=0), name=f"Pita ±{radius}", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=jalur_bebas[:, 0], y=jalur_bebas[:, 1], mode="lines", name="Tanpa batas", line=dict(color=WARNA[1], width=2)))
        fig.add_trace(go.Scatter(x=jalur_pita[:, 0], y=jalur_pita[:, 1], mode="lines", name="Dengan pita", line=dict(color=TINTA, width=2, dash="dot")))
        tata(fig, 380, title=dict(text="Jalur warping", font=dict(size=14)))
        fig.update_xaxes(title_text=f"Minggu {p1}", range=[0, 51]); fig.update_yaxes(title_text=f"Minggu {p2}", range=[0, 51])
        grafik(fig)
    geser_maks = int(np.abs(jalur_bebas[:, 0] - jalur_bebas[:, 1]).max())
    st.markdown(f"<p class='catatan'>Tanpa batas, DTW menggeser hingga {geser_maks} minggu untuk memaksakan kemiripan. "
                f"Pita r = {radius} menahan pergeseran itu, sehingga jaraknya berada di antara Euclidean dan DTW tanpa batas.</p>",
                unsafe_allow_html=True)

# ---------- tab 5: tabel ----------
with tab5:
    hasil = seg[["Product_Code", "ADI", "CV2", "pola", "segmen"]].copy()
    hasil["klaster"] = ""
    hasil.loc[reg, "klaster"] = [nama(c) for c in label]
    if sil_produk is not None:
        hasil.loc[reg, "silhouette"] = np.round(sil_produk, 3)
    hasil["hasil_akhir"] = np.where(hasil.segmen == "Reguler", "Reguler - " + hasil.klaster, hasil.segmen)
    st.dataframe(hasil, hide_index=True, width="stretch", height=420)
    st.download_button("Unduh label (CSV)", hasil.to_csv(index=False).encode("utf-8"),
                       file_name=f"label_{metode.split()[0].lower()}_{'k' + str(param['k']) if 'k' in param else 'auto'}.csv", mime="text/csv")
