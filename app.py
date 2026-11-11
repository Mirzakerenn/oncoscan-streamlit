# =============================================================================
# OncoScan - Aplikasi Streamlit Deteksi Kanker Payudara (Breast Cancer Wisconsin)
# Praktik Bab 1-2 "Implementasi Algoritma ML Berbasis WEB dengan Streamlit"
# Jalankan:  streamlit run app.py
# =============================================================================
import json
import re
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

# -----------------------------------------------------------------------------
# BAGIAN 1 (Bab 1 & 2.1): Konfigurasi halaman
# -----------------------------------------------------------------------------
st.set_page_config(page_title="OncoScan - Deteksi Kanker Payudara", page_icon="🩺", layout="wide")

# -----------------------------------------------------------------------------
# DESAIN (mengikuti DESIGN.md gaya Apple): CSS kustom + animasi
# -----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
:root{--primary:#0066cc;--primary-focus:#0071e3;--sky:#2997ff;--ink:#1d1d1f;--muted:#7a7a7a;
--hairline:#e0e0e0;--parchment:#f5f5f7;--tile:#272729;--tile3:#252527;
--font:"SF Pro Text",system-ui,-apple-system,BlinkMacSystemFont,"Inter",sans-serif;
--ease:cubic-bezier(.22,.61,.36,1)}
.stApp{background:#fff;color:var(--ink);font-family:var(--font)}
.stApp :is(h1,h2,h3,h4,h5,p,li,label,input,textarea,button,td,th){font-family:var(--font)!important}
#MainMenu,footer{visibility:hidden}
[data-testid="stMainBlockContainer"]{max-width:1080px;padding:2.5rem 1.5rem 4rem}
.stApp h1{font-size:40px!important;font-weight:600;line-height:1.1;letter-spacing:0;padding:0}
.stApp h2{font-size:34px!important;font-weight:600;line-height:1.2;letter-spacing:-.374px}
.stApp h3{font-size:21px!important;font-weight:600;line-height:1.19;letter-spacing:.231px}
.stApp p,.stApp li{font-size:17px;line-height:1.47;letter-spacing:-.374px}
[data-testid="stCaptionContainer"] p{font-size:14px;color:var(--muted);letter-spacing:-.224px}

/* sidebar */
[data-testid="stSidebar"]{background:var(--parchment);border-right:1px solid var(--hairline)}
.brand{font-size:21px;font-weight:600;letter-spacing:.231px;margin-top:.5rem}
.brand-sub{font-size:12px;color:var(--muted);letter-spacing:-.12px;margin-bottom:1.5rem}
[data-testid="stSidebar"] [data-testid="stRadio"] label{padding:10px 18px;border-radius:9999px;margin-bottom:4px;width:100%;
transition:background .25s var(--ease),transform .15s var(--ease);cursor:pointer}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{background:rgba(0,0,0,.05)}
[data-testid="stSidebar"] [data-testid="stRadio"] label:active{transform:scale(.97)}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked){background:var(--primary)}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p{color:#fff!important;font-weight:600}
[data-testid="stSidebar"] [data-testid="stRadio"] label p{font-size:17px;margin:0}

/* metric cards */
[data-testid="stMetric"]{background:#fff;border:1px solid var(--hairline);border-radius:18px;padding:24px;
transition:border-color .3s var(--ease),transform .3s var(--ease);animation:fadeUp .7s var(--ease) both}
[data-testid="stMetric"]:hover{border-color:var(--primary);transform:translateY(-3px)}
[data-testid="stMetricLabel"] p{font-size:14px;font-weight:600;color:#333;letter-spacing:-.224px}
[data-testid="stMetricValue"]{font-size:34px;font-weight:600;color:var(--primary);letter-spacing:-.374px}
[data-testid="stColumn"]:nth-child(2) [data-testid="stMetric"]{animation-delay:.08s}
[data-testid="stColumn"]:nth-child(3) [data-testid="stMetric"]{animation-delay:.16s}
[data-testid="stColumn"]:nth-child(4) [data-testid="stMetric"]{animation-delay:.24s}
[data-testid="stColumn"]:nth-child(5) [data-testid="stMetric"]{animation-delay:.32s}

/* buttons */
[data-testid^="stBaseButton-"]{border-radius:9999px;padding:11px 28px;min-height:44px;font-weight:400;
transition:transform .15s var(--ease),background .2s}
[data-testid^="stBaseButton-"]:active{transform:scale(.95)}
[data-testid^="stBaseButton-primary"]{background:var(--primary);border:none;color:#fff}
[data-testid^="stBaseButton-primary"] p{color:#fff}
[data-testid^="stBaseButton-primary"]:hover{background:var(--primary-focus)}
[data-testid^="stBaseButton-primary"]:focus-visible{outline:2px solid var(--primary-focus);outline-offset:2px}
[data-testid="stBaseButton-secondary"]{background:transparent;border:1px solid var(--primary);color:var(--primary)}

/* inputs, form, containers */
[data-baseweb="input"],[data-baseweb="select"]>div{border-radius:9999px!important;border:1px solid rgba(0,0,0,.08)!important;background:#fff!important}
[data-baseweb="input"] input{padding-left:18px!important}
[data-testid="stTextInputRootElement"]{border-radius:9999px!important;border:1px solid rgba(0,0,0,.08)!important;background:#fff!important;overflow:hidden}
[data-testid="stAppDeployButton"]{display:none}
[data-testid="stForm"]{background:var(--parchment);border:none;border-radius:18px;padding:28px;animation:fadeUp .6s var(--ease) both}
[data-testid="stFileUploaderDropzone"]{background:#fff;border:1px dashed var(--hairline);border-radius:18px;
transition:border-color .25s var(--ease),background .25s var(--ease)}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--primary);background:#fafafc}
[data-testid="stExpander"]{border:1px solid var(--hairline);border-radius:18px;overflow:hidden;animation:fadeUp .6s var(--ease) both}
[data-testid="stDataFrame"],[data-testid="stTable"]{border:1px solid var(--hairline);border-radius:18px;overflow:hidden;animation:fadeUp .6s var(--ease) both}
[data-testid="stVegaLiteChart"],[data-testid="stArrowVegaLiteChart"]{animation:fadeUp .8s var(--ease) both}
[data-testid="stAlert"]{border-radius:18px}
[data-baseweb="tab-highlight"]{background:var(--primary)!important}
[data-baseweb="tab"]{font-weight:600}

/* hero & result tiles */
.stApp .hero,.stApp .result{background:var(--tile);color:#fff;border-radius:18px;text-align:center}
.stApp .hero{padding:72px 32px 40px;margin-bottom:32px;animation:fadeUp .8s var(--ease) both}
.stApp .hero h1{color:#fff;font-size:56px!important;line-height:1.07;letter-spacing:-.28px;margin:0 0 12px}
.stApp .hero .lead{font-size:24px;font-weight:300;line-height:1.5;color:#ccc;max-width:640px;margin:0 auto 28px;letter-spacing:0}
.eyebrow{font-size:14px;letter-spacing:.5px;text-transform:uppercase;color:#ccc;margin-bottom:8px}
.pill{display:inline-block;padding:11px 22px;border-radius:9999px;font-size:17px;margin:6px}
.pill.fill{background:var(--primary);color:#fff}
.pill.ghost{border:1px solid var(--sky);color:var(--sky)}
.roc{max-width:420px;width:100%;margin-top:28px;filter:drop-shadow(rgba(0,0,0,.22) 3px 5px 30px)}
.roc-line{stroke-dasharray:1;stroke-dashoffset:1;animation:draw 2.2s var(--ease) .4s forwards}
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:17px;margin:8px 0 32px}
.step{background:var(--parchment);border-radius:18px;padding:24px;animation:fadeUp .7s var(--ease) both;transition:transform .3s var(--ease)}
.step:nth-child(2){animation-delay:.1s}.step:nth-child(3){animation-delay:.2s}
.step:hover{transform:translateY(-3px)}
.step b{display:block;font-size:34px;font-weight:600;color:var(--primary);letter-spacing:-.374px}
.step span{font-size:17px;color:#333}
.stApp .result{padding:48px 24px;margin:24px 0;animation:fadeUp .6s var(--ease) both}
.badge{display:inline-block;padding:10px 28px;border-radius:9999px;font-size:34px;font-weight:600;margin:16px 0;animation:pop .6s var(--ease) both}
.badge.jinak{background:var(--primary)}
.badge.ganas{background:var(--tile3);border:1px solid #ccc}
.stApp .result .sub{font-size:21px;color:#ccc;margin:0;letter-spacing:.231px}
.bar{max-width:520px;height:10px;margin:24px auto 8px;background:rgba(255,255,255,.15);border-radius:9999px;overflow:hidden}
.fill{height:100%;background:var(--sky);border-radius:9999px;width:var(--w);animation:grow 1.2s var(--ease) both}
.cells{display:grid;grid-template-columns:repeat(3,1fr);gap:17px;max-width:640px;margin:32px auto 0}
.cell{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);border-radius:18px;padding:17px}
.cell .k{font-size:12px;color:#ccc}
.cell .v{font-size:21px;font-weight:600;margin-top:4px}
.stApp .fine{font-size:12px;color:#ccc;max-width:640px;margin:32px auto 0;line-height:1.3;letter-spacing:-.12px;text-align:center}
.foot{text-align:center;font-size:12px;color:var(--muted);margin-top:64px;padding-top:24px;border-top:1px solid var(--hairline)}

/* count-up angka */
@property --ip{syntax:'<integer>';inherits:false;initial-value:0}
@property --dp{syntax:'<integer>';inherits:false;initial-value:0}
.count{--ip:var(--ti);--dp:var(--td);animation:count 1.4s var(--ease) both;counter-reset:ip var(--ip) dp var(--dp)}
.count::after{content:counter(ip) "." counter(dp) "%"}

@keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:none}}
@keyframes pop{0%{opacity:0;transform:scale(.8)}60%{transform:scale(1.05)}100%{opacity:1;transform:scale(1)}}
@keyframes grow{from{width:0}}
@keyframes draw{to{stroke-dashoffset:0}}
@keyframes count{from{--ip:0;--dp:0}to{--ip:var(--ti);--dp:var(--td)}}
@media (max-width:833px){.stApp .hero h1{font-size:40px!important}.steps,.cells{grid-template-columns:1fr}}
@media (prefers-reduced-motion:reduce){*{animation-duration:.01ms!important;animation-delay:0s!important;transition:none!important}.roc-line{stroke-dashoffset:0}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def html(block: str) -> None:
    """Render HTML tanpa baris kosong/indentasi (supaya tidak dianggap code block markdown)."""
    st.markdown(" ".join(line.strip() for line in block.splitlines()), unsafe_allow_html=True)


def style_chart(chart):
    """Gaya grafik Altair: bersih, satu warna aksen."""
    return (chart.configure_view(strokeWidth=0)
            .configure_axis(gridColor="#f0f0f0", domainColor="#7a7a7a", tickColor="#7a7a7a",
                            labelColor="#7a7a7a", titleColor="#333333")
            .configure_legend(labelColor="#333333", titleColor="#333333", orient="bottom"))


def roc_svg(fpr, tpr) -> str:
    x0, y0, x1, y1 = 40, 20, 380, 220
    pts = " L ".join(f"{x0 + f * (x1 - x0):.1f},{y1 - t * (y1 - y0):.1f}" for f, t in zip(fpr, tpr))
    return (f'<svg class="roc" viewBox="0 0 420 260" xmlns="http://www.w3.org/2000/svg">'
            f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#7a7a7a"/>'
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#7a7a7a"/>'
            f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y0}" stroke="#7a7a7a" stroke-dasharray="4 4"/>'
            f'<path class="roc-line" pathLength="1" d="M {pts}" fill="none" stroke="#2997ff" stroke-width="2.5"/>'
            f'<text x="210" y="250" font-size="12" fill="#cccccc" text-anchor="middle">False Positive Rate</text>'
            f'<text x="18" y="120" font-size="12" fill="#cccccc" text-anchor="middle" transform="rotate(-90 18 120)">True Positive Rate</text>'
            f'</svg>')


def count_html(pct: float) -> str:
    """Angka persen dengan animasi count-up (CSS murni)."""
    tenths = int(round(pct * 10))
    return f'<span class="count" style="--ti:{tenths // 10};--td:{tenths % 10}" aria-label="{pct:.1f}%"></span>'


MEAN_FEATURES = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry",
    "mean fractal dimension",
]
LABELS = {0: "Ganas (Malignant)", 1: "Jinak (Benign)"}  # target sklearn: 0 = ganas, 1 = jinak


# -----------------------------------------------------------------------------
# Parser file hasil lab (CSV / JSON) -> dict {nama_fitur: nilai}
# -----------------------------------------------------------------------------
def _norm(k) -> str:
    return re.sub(r"[^a-z0-9]", "", str(k).lower())


def _build_aliases() -> dict:
    alias = {}
    for feat in MEAN_FEATURES:
        base = feat.replace("mean ", "")              # "radius", "concave points", ...
        for name in (feat, f"{base} mean", base):     # "mean radius", "radius mean", "radius"
            alias[_norm(name)] = feat
    alias["concave"] = "mean concave points"
    return alias


ALIASES = _build_aliases()


def map_columns(keys) -> dict:
    """Petakan nama kolom bebas -> nama fitur standar (hanya yang dikenali)."""
    out = {}
    for k in keys:
        feat = ALIASES.get(_norm(k))
        if feat and feat not in out:
            out[feat] = k
    return out


def read_lab_file(uploaded) -> pd.DataFrame:
    """Baca CSV/JSON menjadi DataFrame berisi 10 kolom mean (baris = sampel)."""
    name = uploaded.name.lower()
    if name.endswith(".json"):
        data = json.load(uploaded)
        if isinstance(data, dict) and isinstance(data.get("features"), (dict, list)):
            data = data["features"]
        raw = pd.DataFrame(data if isinstance(data, list) else [data])
    else:
        raw = pd.read_csv(uploaded, sep=None, engine="python")  # deteksi , atau ;
        if not any(map_columns(raw.columns)):                    # tanpa header -> posisi kolom
            uploaded.seek(0)
            raw = pd.read_csv(uploaded, header=None, sep=None, engine="python")
            if raw.shape[1] >= len(MEAN_FEATURES):
                raw = raw.iloc[:, :len(MEAN_FEATURES)]
                raw.columns = MEAN_FEATURES
    mapping = map_columns(raw.columns)
    missing = [f for f in MEAN_FEATURES if f not in mapping]
    if missing:
        raise ValueError("Parameter tidak ditemukan: " + ", ".join(missing))
    out = raw[[mapping[f] for f in MEAN_FEATURES]].copy()
    out.columns = MEAN_FEATURES
    return out.apply(pd.to_numeric, errors="raise")


# -----------------------------------------------------------------------------
# BAGIAN 2 (Bab 2.3): Data & model (di-cache agar tidak dilatih ulang tiap klik)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    csv = Path(__file__).parent / "breast_cancer.csv"   # dibuat oleh script Breast Cancer Wisconsin.py
    if csv.exists():
        return pd.read_csv(csv)
    data = load_breast_cancer()
    return pd.DataFrame(data.data, columns=data.feature_names).assign(target=data.target)


@st.cache_resource
def train_model():
    df = load_data()
    X, y = df[MEAN_FEATURES], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)       # split 80:20 berstrata
    model = Pipeline([
        ("scaler", RobustScaler()),                              # tahan outlier medis
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=8,
                                       random_state=42, class_weight="balanced")),
    ])
    model.fit(X_train, y_train)
    return model, X_test, y_test


df = load_data()
model, X_test, y_test = train_model()

# -----------------------------------------------------------------------------
# BAGIAN 4 (Bab 2.7): Sidebar untuk navigasi menu
# -----------------------------------------------------------------------------
st.sidebar.markdown('<div class="brand">OncoScan</div><div class="brand-sub">Breast Cancer Detection</div>',
                    unsafe_allow_html=True)
page = st.sidebar.radio("Menu", ["Beranda", "Dataset", "Prediksi", "Evaluasi Model"],
                        label_visibility="collapsed")
st.sidebar.caption("Hanya untuk edukasi, bukan pengganti diagnosis medis.")

# kurva ROC asli dari data uji (dipakai di hero & halaman evaluasi)
_proba_jinak = model.predict_proba(X_test)[:, 1]
_fpr, _tpr, _ = roc_curve(y_test, _proba_jinak)
_auc = roc_auc_score(y_test, _proba_jinak)


# =============================================================================
# HALAMAN: BERANDA  (Bab 2.2 - elemen teks)
# =============================================================================
if page == "Beranda":
    html(f"""
    <div class="hero">
      <div class="eyebrow">Breast Cancer Wisconsin (Diagnostic)</div>
      <h1>Deteksi Kanker Payudara.</h1>
      <p class="lead">Klasifikasi tumor Jinak vs Ganas dari 10 fitur sel, dengan model Random Forest.</p>
      <span class="pill fill">Coba di menu Prediksi</span><span class="pill ghost">ROC-AUC {_auc:.4f}</span>
      <div>{roc_svg(_fpr, _tpr)}</div>
    </div>""")
    st.markdown(
        "Aplikasi ini mengklasifikasikan tumor payudara sebagai **Jinak** atau **Ganas** "
        "dari **10 fitur rata-rata (mean)** hasil pengukuran sel, memakai pipeline "
        "`RobustScaler` + `RandomForestClassifier`."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah sampel", len(df))
    c2.metric("Fitur dipakai", len(MEAN_FEATURES))
    c3.metric("Kelas", "Jinak / Ganas")
    st.write("")
    html("""
    <div class="steps">
      <div class="step"><b>1</b><span>Upload hasil lab atau atur parameter secara manual.</span></div>
      <div class="step"><b>2</b><span>Tekan Mulai Analisis untuk menjalankan model.</span></div>
      <div class="step"><b>3</b><span>Baca prediksi dan probabilitasnya.</span></div>
    </div>""")
    with st.expander("Tentang dataset"):
        st.write(
            f"Dataset berisi {len(df)} observasi dan 30 fitur numerik + 1 target biner "
            "(0 = ganas, 1 = jinak). Form prediksi hanya memakai 10 fitur *mean*."
        )

# =============================================================================
# HALAMAN: DATASET  (Bab 2.3 & 2.4 - data & visualisasi)
# =============================================================================
elif page == "Dataset":
    st.header("Dataset Breast Cancer Wisconsin")
    tab1, tab2, tab3 = st.tabs(["Tabel Data", "Statistik", "Visualisasi"])

    with tab1:
        show_all = st.checkbox("Tampilkan semua 30 fitur", value=False)
        cols = list(df.columns) if show_all else MEAN_FEATURES + ["target"]
        st.dataframe(df[cols], width="stretch", height=380)

    with tab2:
        st.write("Ringkasan statistik 10 fitur mean:")
        st.table(df[MEAN_FEATURES].describe().T[["mean", "std", "min", "max"]].round(3))

    with tab3:
        left, right = st.columns(2)
        with left:
            st.subheader("Distribusi kelas")
            counts = df["target"].map(LABELS).value_counts()
            st.bar_chart(counts, color="#0066cc")
        with right:
            st.subheader("Distribusi fitur per kelas")
            feat = st.selectbox("Pilih fitur", MEAN_FEATURES)
            plot_df = pd.DataFrame({feat: df[feat], "Kelas": df["target"].map(LABELS)})
            chart = (alt.Chart(plot_df).mark_bar(opacity=0.7)
                     .encode(x=alt.X(f"{feat}:Q", bin=alt.Bin(maxbins=30)),
                             y=alt.Y("count()", stack=None),
                             color=alt.Color("Kelas:N", scale=alt.Scale(
                                 domain=[LABELS[0], LABELS[1]], range=["#1d1d1f", "#0066cc"]))))
            st.altair_chart(style_chart(chart), width="stretch", theme=None)

# =============================================================================
# HALAMAN: PREDIKSI  (Bab 2.5 - widget input & Bab 2.8 - control flow)
# =============================================================================
elif page == "Prediksi":
    st.header("Parameter Klinis")
    st.write("Atur 10 fitur mean, lalu tekan **Mulai Analisis**.")

    # Sumber nilai awal form: file hasil lab (prioritas) atau contoh/median data
    uploaded = st.file_uploader("Upload hasil lab (.csv / .json)", type=["csv", "json"],
                                help="Harus memuat 10 parameter mean, mis. 'mean radius' atau 'radius_mean'.")
    lab_values, src_tag = None, "manual"
    if uploaded is not None:
        try:
            lab_df = read_lab_file(uploaded)
            row = 0
            if len(lab_df) > 1:
                row = st.selectbox("File berisi beberapa sampel, pilih baris", range(len(lab_df)),
                                   format_func=lambda i: f"Baris {i + 1}")
            lab_values = lab_df.iloc[row]
            src_tag = f"{uploaded.name}|{uploaded.size}|{row}"
            st.success(f"Parameter dari **{uploaded.name}** (baris {row + 1}) sudah dimuat ke form.")
        except Exception as e:
            st.error(f"File tidak bisa dibaca: {e}")

    contoh = st.selectbox("Isi otomatis dengan", ["Median seluruh data", "Contoh Jinak", "Contoh Ganas"],
                          disabled=lab_values is not None)
    if lab_values is not None:
        defaults = lab_values
    elif contoh == "Contoh Jinak":
        defaults = df[df["target"] == 1][MEAN_FEATURES].median()
    elif contoh == "Contoh Ganas":
        defaults = df[df["target"] == 0][MEAN_FEATURES].median()
    else:
        defaults = df[MEAN_FEATURES].median()
    tag = src_tag if lab_values is not None else contoh

    with st.form("form_klinis"):
        nama = st.text_input("ID / nama sampel (opsional)", placeholder="contoh: Sampel-001")
        values, clipped = {}, []
        col_a, col_b = st.columns(2)
        for i, feat in enumerate(MEAN_FEATURES):
            lo, hi = float(df[feat].min()), float(df[feat].max())
            step = float(10 ** np.floor(np.log10((hi - lo) / 200)))   # langkah halus sesuai skala fitur
            target_col = col_a if i % 2 == 0 else col_b
            init = float(np.clip(defaults[feat], lo, hi))
            if init != float(defaults[feat]):
                clipped.append(feat)
            values[feat] = target_col.slider(
                feat.title(), min_value=lo, max_value=hi,
                value=init, step=step, key=f"{feat}|{tag}")
        if clipped:
            st.warning("Di luar rentang dataset, dibatasi ke min/max: " + ", ".join(clipped))
        submitted = st.form_submit_button("Mulai Analisis", type="primary")

    # Control flow: hasil hanya tampil setelah tombol ditekan
    if submitted:
        x_new = pd.DataFrame([values])[MEAN_FEATURES]
        proba = model.predict_proba(x_new)[0]          # urutan kelas: [ganas, jinak]
        p_ganas, p_jinak = float(proba[0]), float(proba[1])
        pred = int(model.predict(x_new)[0])
        label_cls, label_txt = ("ganas", "Ganas") if pred == 0 else ("jinak", "Jinak")
        judul = f"Hasil Prediksi - {nama}" if nama else "Hasil Prediksi"

        html(f"""
        <div class="result">
          <div class="eyebrow">{judul}</div>
          <div class="badge {label_cls}">{label_txt}</div>
          <p class="sub">Probabilitas ganas: {count_html(p_ganas * 100)}</p>
          <div class="bar"><div class="fill" style="--w:{p_ganas * 100:.1f}%"></div></div>
          <div class="cells">
            <div class="cell"><div class="k">Kelas Prediksi</div><div class="v">{label_txt}</div></div>
            <div class="cell"><div class="k">Probabilitas Jinak</div><div class="v">{count_html(p_jinak * 100)}</div></div>
            <div class="cell"><div class="k">Probabilitas Ganas</div><div class="v">{count_html(p_ganas * 100)}</div></div>
          </div>
          <p class="fine">Keluaran model edukasi dan tidak menggantikan diagnosis medis profesional.
          Konsultasikan hasil dengan dokter spesialis.</p>
        </div>""")
    else:
        st.info("Belum ada prediksi. Atur parameter lalu tekan tombol.")

# =============================================================================
# HALAMAN: EVALUASI  (metrik dihitung langsung dari model & data uji)
# =============================================================================
else:
    st.header("Performansi Model")
    st.caption(f"Dihitung pada data uji (N = {len(y_test)}), kelas positif = Ganas.")

    pred = model.predict(X_test)

    cols = st.columns(5)
    cols[0].metric("Akurasi", f"{accuracy_score(y_test, pred):.3f}")
    cols[1].metric("Presisi", f"{precision_score(y_test, pred, pos_label=0):.3f}")
    cols[2].metric("Recall", f"{recall_score(y_test, pred, pos_label=0):.3f}")
    cols[3].metric("F1-Score", f"{f1_score(y_test, pred, pos_label=0):.3f}")
    cols[4].metric("ROC-AUC", f"{_auc:.4f}")
    st.write("")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        st.dataframe(pd.DataFrame(cm, index=["Aktual Ganas", "Aktual Jinak"],
                                  columns=["Pred Ganas", "Pred Jinak"]))
    with right:
        st.subheader("Kurva ROC")
        roc_df = pd.DataFrame({"FPR": _fpr, "TPR": _tpr})
        line = alt.Chart(roc_df).mark_line(color="#0066cc", strokeWidth=2.5).encode(x="FPR:Q", y="TPR:Q")
        diag = alt.Chart(pd.DataFrame({"FPR": [0, 1], "TPR": [0, 1]})).mark_line(
            strokeDash=[4, 4], color="#cccccc").encode(x="FPR:Q", y="TPR:Q")
        st.altair_chart(style_chart(line + diag), width="stretch", theme=None)

    with st.expander("Classification report lengkap"):
        report = classification_report(y_test, pred, target_names=[LABELS[0], LABELS[1]], output_dict=True)
        st.dataframe(pd.DataFrame(report).T.round(3))

html("""<div class="foot">© 2026 OncoScan. Hanya untuk tujuan edukasi; bukan pengganti diagnosis medis.</div>""")
