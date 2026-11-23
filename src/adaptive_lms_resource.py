"""
=====================================================================
 ADAPTIVE RESOURCE ALLOCATION UNTUK LMS (Predictive Auto-Scaling)
 vCPU, RAM, Storage, GPU  ->  efisiensi biaya berbasis trafik pengguna
=====================================================================

Pertanyaan yang dijawab modul ini:
  "Berapa vCPU, RAM, storage, dan GPU yang perlu disediakan untuk LMS
   pada jam/hari berikutnya, berdasarkan trafik pengguna yang
   DIPREDIKSI, agar biaya minimum tetapi layanan tidak overload?"

Alur:
  1. TRAFIK   : log klik OULAD (studentVle) -> trafik LMS harian riil
                (jumlah klik & pengguna aktif) -> dipecah per jam memakai
                pola diurnal (ASUMSI, bisa diganti log server sendiri).
  2. PREDIKSI : model ML (Gradient Boosting, regresi kuantil) memprediksi
                beban (request/detik) H jam ke depan: median (p50) dan
                batas atas (p90) -> margin aman yang adaptif.
                Fitur khas LMS: jam, hari, minggu akademik, jarak ke
                deadline tugas, jumlah kelas aktif, lag trafik.
  3. ALOKASI  : beban -> kebutuhan vCPU & RAM -> jumlah node (instance).
                Storage: tiering SSD (data panas) + object storage (log lama).
                GPU   : hanya disewa bila lebih murah dari CPU untuk job
                        pelatihan model, dijadwalkan di jam trafik terendah.
  4. PENGUJIAN: simulasi biaya & SLA 4 kebijakan pada periode uji:
                 - static     : disediakan untuk puncak historis (cara umum)
                 - reactive   : ikut beban jam sebelumnya + headroom
                 - predictive : ikut prediksi p90 (MODEL USULAN)
                 - oracle     : batas bawah teoretis (tahu masa depan)
  5. ESTIMASI : rencana 24 jam ke depan (node, vCPU, RAM, biaya),
                proyeksi storage 90 hari, keputusan CPU vs GPU.

Cara pakai:
  python adaptive_lms_resource.py --data_dir ./OULAD --out_dir ./hasil_resource
  python adaptive_lms_resource.py --demo --out_dir ./hasil_resource_demo
  # skala ke LMS kampus Anda, mis. 3x populasi OULAD:
  python adaptive_lms_resource.py --data_dir ./OULAD --scale_users 3

CATATAN JUJUR: OULAD hanya menyimpan klik HARIAN dan tidak memuat metrik
server (CPU%, RAM). Pola per jam dan konversi klik->vCPU/RAM adalah
PARAMETER ASUMSI (lihat ResourceConfig) yang sebaiknya dikalibrasi dengan
uji beban (load test) / monitoring LMS sungguhan.
"""

import argparse
import math
import os
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
DEMAND_COLOR, TEXT_PRIMARY, TEXT_MUTED, GRID = "#1f1f1e", "#1f1f1e", "#6b6a63", "#e4e3dd"

# Tanggal mulai presentasi OULAD (B = Februari, J = Oktober). Tanggal persis
# tidak dipublikasikan; tanggal 1 dipakai sebagai pendekatan.
PRES_START = {"2013B": "2013-02-01", "2013J": "2013-10-01",
              "2014B": "2014-02-01", "2014J": "2014-10-01"}

# Pola akses LMS per jam (bobot relatif, jam 00..23). ASUMSI pola
# mahasiswa: puncak pagi-siang (kuliah) dan malam (mengerjakan tugas).
DIURNAL = np.array([1.5, 0.8, 0.5, 0.4, 0.5, 1.0, 2.0, 3.5, 5.0, 6.0, 6.5, 6.0,
                    4.5, 5.5, 6.0, 5.5, 5.0, 4.0, 4.0, 6.0, 7.5, 8.0, 6.5, 4.0])


@dataclass
class ResourceConfig:
    # ---------------- Trafik (pengguna -> beban) ----------------
    scale_users: float = 1.0        # pengali populasi (LMS Anda / OULAD ~32 rb mhs)
    req_per_click: float = 8.0      # HTTP request per 1 klik (halaman+aset+API)
    burst_factor: float = 1.5       # puncak dalam 1 jam / rata-rata jam itu
    session_hours: float = 0.5      # rata-rata lama online per pengguna aktif per hari
    hourly_noise: float = 0.15      # variasi acak per jam (log-normal sigma)
    # ---------------- Kapasitas (beban -> vCPU & RAM) ----------------
    rps_per_vcpu: float = 10.0      # request dinamis/detik per 1 vCPU (Moodle/PHP; kalibrasi via load test)
    target_cpu_util: float = 0.65   # target utilisasi saat scaling (sisa = buffer)
    max_safe_util: float = 0.90     # di atas ini dianggap overload (SLA dilanggar)
    ram_per_user_mb: float = 12.0   # RAM per pengguna online (sesi, worker PHP)
    ram_base_per_node_gb: float = 1.5  # OS + web server per node
    # --- diisi otomatis bila --calib kalibrasi_dari_trace.json dipakai (server riil) ---
    ram_model: str = "users"        # "users" = per pengguna (asumsi) | "trace" = dari server riil
    ram_per_vcpu_gb: float = 0.0    # tambahan RAM per vCPU terpakai (regresi jejak server)
    ram_base_ratio: float = 0.0     # RAM dasar per vCPU beban puncak (jejak server)
    ram_base_gb_cluster: float = 0.0  # dihitung saat run = ram_base_ratio x vCPU puncak LMS
    overprovision_real: float = 0.0  # rasio disediakan/terpakai-puncak di server riil
    # ---------------- Node / instance compute ----------------
    node_vcpu: int = 2              # contoh AWS m6i.large: 2 vCPU, 8 GiB
    node_ram_gb: float = 8.0
    node_price_hour: float = 0.1235  # USD/jam on-demand m6i.large, region Jakarta
    min_nodes: int = 2              # high availability
    max_nodes: int = 60
    # ---------------- Kebijakan scaling ----------------
    horizon_hours: int = 1          # jeda provisioning: prediksi H jam ke depan
    forecast_quantile: float = 0.90  # kuantil batas atas prediksi (margin adaptif)
    reactive_headroom: float = 0.20  # headroom kebijakan reactive
    static_headroom: float = 0.20   # headroom kebijakan static di atas puncak historis
    max_scale_in_step: int = 1      # maksimum node dilepas per jam (anti flapping)
    # ---------------- Storage ----------------
    content_gb: float = 200.0       # materi kuliah, video kecil, file tugas (dasar)
    bytes_per_click_log: float = 600.0  # ukuran 1 baris log aktivitas + indeks
    hot_days: int = 30              # log < N hari di SSD, sisanya ke object storage
    storage_headroom: float = 0.20
    ssd_price_gb_month: float = 0.096   # AWS EBS gp3, Jakarta (AWS Pricing Calculator, 24 Nov 2026)
    object_price_gb_month: float = 0.025  # AWS S3 Standard, Jakarta (AWS Pricing Calculator, 24 Nov 2026)
    # ---------------- GPU (job ML terjadwal) ----------------
    gpu_price_hour: float = 0.736   # AWS g4dn.xlarge (1x T4), Singapura on-demand (job batch, boleh beda region)
    gpu_startup_hours: float = 0.15  # waktu boot + memuat data (ikut dibayar)
    retrain_every_days: int = 7
    ml_jobs: list = field(default_factory=lambda: [
        # (nama, jam-CPU per job pada 1 node, percepatan bila pakai GPU)
        ("Retrain model risiko mahasiswa (GradientBoosting/sklearn)", 0.5, 1.0),
        ("Retrain Deep Knowledge Tracing (LSTM/Transformer)", 6.0, 8.0),
    ])


# =====================================================================
# 1. TRAFIK LMS PER JAM DARI OULAD
# =====================================================================
def build_hourly_traffic(data_dir, cfg):
    base = pd.Timestamp("2013-02-01")
    offset = {k: (pd.Timestamp(v) - base).days for k, v in PRES_START.items()}
    print(f"[1] Membaca log penggunaan LMS: {data_dir}")
    sv = pd.read_csv(os.path.join(data_dir, "studentVle.csv"),
                     usecols=["code_presentation", "id_student", "date", "sum_click"],
                     dtype={"code_presentation": "category", "id_student": "int32",
                            "date": "int16", "sum_click": "int32"})
    sv["abs_day"] = sv["code_presentation"].map(offset).astype("int32") + sv["date"]
    daily = sv.groupby("abs_day").agg(clicks=("sum_click", "sum"),
                                      users=("id_student", "nunique"))
    days = np.arange(daily.index.min(), daily.index.max() + 1)
    daily = daily.reindex(days, fill_value=0)
    daily.index = base + pd.to_timedelta(daily.index, unit="D")
    daily.index.name = "date"
    daily[["clicks", "users"]] *= cfg.scale_users

    # kalender akademik: deadline tugas & jumlah kelas aktif per hari
    ass = pd.read_csv(os.path.join(data_dir, "assessments.csv"), na_values=["?", ""])
    ass = ass.dropna(subset=["date"])
    ass_day = base + pd.to_timedelta(ass["code_presentation"].map(offset) + ass["date"], unit="D")
    deadlines = np.sort(ass_day.values)
    courses_p = os.path.join(data_dir, "courses.csv")
    if os.path.exists(courses_p):
        crs = pd.read_csv(courses_p)
    else:
        info = pd.read_csv(os.path.join(data_dir, "studentInfo.csv"),
                           usecols=["code_module", "code_presentation"]).drop_duplicates()
        crs = info.assign(module_presentation_length=269)
    d_idx = daily.index.values
    nxt = np.searchsorted(deadlines, d_idx)
    days_to = np.full(len(d_idx), 30.0)
    ok = nxt < len(deadlines)
    days_to[ok] = (deadlines[nxt[ok]] - d_idx[ok]) / np.timedelta64(1, "D")
    daily["days_to_deadline"] = np.clip(days_to, 0, 30)
    lo = np.searchsorted(deadlines, d_idx)
    hi = np.searchsorted(deadlines, d_idx + np.timedelta64(3, "D"), side="right")
    daily["deadlines_next3d"] = hi - lo
    act = np.zeros(len(d_idx))
    for _, r in crs.iterrows():
        s = base + pd.Timedelta(days=offset.get(r["code_presentation"], 0))
        e = s + pd.Timedelta(days=int(r["module_presentation_length"]))
        act += (daily.index >= s) & (daily.index <= e)
    daily["active_courses"] = act

    # pecah ke per jam
    rng = np.random.default_rng(RANDOM_STATE)
    share = DIURNAL / DIURNAL.sum()
    h = daily.loc[daily.index.repeat(24)].copy()
    h.index = h.index + pd.to_timedelta(np.tile(np.arange(24), len(daily)), unit="h")
    noise = rng.lognormal(0, cfg.hourly_noise, len(h))
    sh = np.tile(share, len(daily))
    h["clicks_h"] = h["clicks"] * sh * noise
    # pengguna online pada jam h = pengguna aktif harian x lama sesi x porsi jam h
    h["users_online"] = h["users"] * cfg.session_hours * sh * noise
    h["rps"] = h["clicks_h"] * cfg.req_per_click / 3600 * cfg.burst_factor  # beban puncak/jam
    h["hour"], h["dow"] = h.index.hour, h.index.dayofweek
    h["month"] = h.index.month
    print(f"    {len(daily)} hari ({daily.index.min().date()} s.d. {daily.index.max().date()}), "
          f"puncak {h['rps'].max():.1f} req/s, maks {h['users_online'].max():,.0f} pengguna online")
    return h, daily


# =====================================================================
# 2. MODEL PREDIKSI BEBAN (ADAPTIF TERHADAP TRAFIK)
# =====================================================================
CAL_FEATS = ["hour", "dow", "month", "days_to_deadline", "deadlines_next3d", "active_courses"]


def make_features(h, H):
    """Fitur yang tersedia H jam sebelum waktu target (tanpa kebocoran)."""
    y = h["rps"]
    X = h[CAL_FEATS].copy()
    s24, s168 = 24 * math.ceil(H / 24), 168 * math.ceil(H / 168)
    for L in sorted({H, H + 1, H + 2, s24, s24 + 24, s168}):
        X[f"lag_{L}"] = y.shift(L)
    X["roll_mean_24"] = y.shift(H).rolling(24).mean()
    X["roll_max_24"] = y.shift(H).rolling(24).max()
    X["roll_mean_168"] = y.shift(H).rolling(168).mean()
    X["users_lag"] = h["users_online"].shift(H)
    return X


def fit_quantile_models(X, y, q_hi):
    yl = np.log1p(y)
    p50 = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, max_iter=400,
                                        learning_rate=0.05, random_state=RANDOM_STATE).fit(X, yl)
    phi = HistGradientBoostingRegressor(loss="quantile", quantile=q_hi, max_iter=400,
                                        learning_rate=0.05, random_state=RANDOM_STATE).fit(X, yl)
    return p50, phi


def predict_q(m, X):
    return np.expm1(m.predict(X)).clip(min=0)


def forecast_metrics(y, pred, name, hi=None):
    mask = y > np.percentile(y, 10)
    d = dict(model=name, MAE=mean_absolute_error(y, pred),
             RMSE=math.sqrt(mean_squared_error(y, pred)),
             MAPE_pct=float(np.mean(np.abs((y[mask] - pred[mask]) / y[mask])) * 100),
             R2=r2_score(y, pred))
    if hi is not None:
        d["coverage_p_hi_pct"] = float(np.mean(y <= hi) * 100)
    return d


# =====================================================================
# 3. ALOKASI SUMBER DAYA
# =====================================================================
def ram_demand(rps, users, cfg):
    """Kebutuhan RAM (GB). Mode 'trace': RAM = dasar + b x vCPU terpakai,
    koefisien dari regresi jejak server riil (Bitbrains/Alibaba)."""
    if cfg.ram_model == "trace":
        return cfg.ram_base_gb_cluster + cfg.ram_per_vcpu_gb * (rps / cfg.rps_per_vcpu)
    return users * cfg.ram_per_user_mb / 1024


def nodes_needed(rps, users, cfg, util=None):
    util = util or cfg.target_cpu_util
    vcpu = rps / (cfg.rps_per_vcpu * util)
    ram = ram_demand(rps, users, cfg)
    n_cpu = vcpu / cfg.node_vcpu
    n_ram = ram / (cfg.node_ram_gb - cfg.ram_base_per_node_gb)
    n = np.ceil(np.maximum(n_cpu, n_ram))
    return np.clip(n, cfg.min_nodes, cfg.max_nodes).astype(int), vcpu, ram


def limit_scale_in(nodes, step):
    out = nodes.copy()
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1] - step)
    return out


def simulate_policies(test, pred_hi, train, cfg):
    rps, users = test["rps"].values, test["users_online"].values
    upr = np.median(train["users_online"] / train["rps"].replace(0, np.nan))
    pol = {}
    peak_n, _, _ = nodes_needed(np.array([train["rps"].max() * (1 + cfg.static_headroom)]),
                                np.array([train["users_online"].max() * (1 + cfg.static_headroom)]), cfg)
    pol["static"] = np.full(len(test), peak_n[0])
    prev_rps = np.r_[train["rps"].values[-1], rps[:-1]]
    prev_users = np.r_[train["users_online"].values[-1], users[:-1]]
    pol["reactive"] = nodes_needed(prev_rps * (1 + cfg.reactive_headroom),
                                   prev_users * (1 + cfg.reactive_headroom), cfg)[0]
    pol["predictive"] = limit_scale_in(nodes_needed(pred_hi, pred_hi * upr, cfg)[0],
                                       cfg.max_scale_in_step)
    pol["oracle"] = nodes_needed(rps, users, cfg)[0]
    if cfg.overprovision_real > 0:
        # praktik nyata: rasio disediakan/terpakai yang teramati di server riil
        base_n = nodes_needed(np.array([train["rps"].max()]), np.array([train["users_online"].max()]),
                              cfg, util=1.0)[0][0]
        pol["static_praktik_riil"] = np.full(len(test), int(math.ceil(base_n * cfg.overprovision_real)))

    cap_rps_per_node = cfg.node_vcpu * cfg.rps_per_vcpu * cfg.max_safe_util
    ram_cap_per_node = cfg.node_ram_gb - cfg.ram_base_per_node_gb
    ram_d = ram_demand(rps, users, cfg)
    rows, alloc = [], pd.DataFrame(index=test.index)
    alloc["demand_vcpu"] = rps / cfg.rps_per_vcpu
    alloc["demand_ram_gb"] = ram_d
    for name, n in pol.items():
        overload = (rps > n * cap_rps_per_node) | (ram_d > n * ram_cap_per_node)
        cost = float(n.sum() * cfg.node_price_hour)
        rows.append(dict(kebijakan=name, node_jam=int(n.sum()), rata2_node=n.mean(),
                         maks_node=int(n.max()), biaya_compute_usd=cost,
                         utilisasi_cpu_rata2_pct=float(np.mean(rps / (n * cfg.node_vcpu * cfg.rps_per_vcpu)) * 100),
                         jam_overload=int(overload.sum()),
                         overload_pct=float(overload.mean() * 100)))
        alloc[f"vcpu_{name}"] = n * cfg.node_vcpu
        alloc[f"ram_gb_{name}"] = n * cfg.node_ram_gb
        alloc[f"node_{name}"] = n
    res = pd.DataFrame(rows)
    static_cost = res.loc[res.kebijakan == "static", "biaya_compute_usd"].iat[0]
    res["hemat_vs_static_pct"] = (1 - res["biaya_compute_usd"] / static_cost) * 100
    return res, alloc


def simulate_storage(daily, test_start, cfg):
    d = daily.copy()
    d["log_gb"] = d["clicks"] * cfg.bytes_per_click_log / 1e9
    d["cum_log_gb"] = d["log_gb"].cumsum()
    d["hot_log_gb"] = d["log_gb"].rolling(cfg.hot_days, min_periods=1).sum()
    d["cold_log_gb"] = d["cum_log_gb"] - d["hot_log_gb"]
    t = d[d.index >= test_start.normalize()].copy()
    # static: disk SSD besar disiapkan di awal untuk seluruh data + pertumbuhan
    static_gb = (cfg.content_gb + d["cum_log_gb"].iat[-1]) * 1.5
    t["static_ssd_gb"] = static_gb
    # adaptif: SSD hanya data panas, dibulatkan 50 GB, tidak bisa mengecil (sifat EBS)
    need = np.ceil((cfg.content_gb + t["hot_log_gb"]) * (1 + cfg.storage_headroom) / 50) * 50
    t["adaptive_ssd_gb"] = np.maximum.accumulate(need.values)
    t["adaptive_object_gb"] = t["cold_log_gb"]
    day_ssd, day_obj = cfg.ssd_price_gb_month / 30, cfg.object_price_gb_month / 30
    cost_static = float((t["static_ssd_gb"] * day_ssd).sum())
    cost_adapt = float((t["adaptive_ssd_gb"] * day_ssd + t["adaptive_object_gb"] * day_obj).sum())
    summary = pd.DataFrame([
        dict(kebijakan="static (SSD besar di awal)", rata2_ssd_gb=t["static_ssd_gb"].mean(),
             rata2_object_gb=0.0, biaya_storage_usd=cost_static),
        dict(kebijakan="adaptive (tiering SSD + object)", rata2_ssd_gb=t["adaptive_ssd_gb"].mean(),
             rata2_object_gb=t["adaptive_object_gb"].mean(), biaya_storage_usd=cost_adapt)])
    summary["hemat_vs_static_pct"] = (1 - summary["biaya_storage_usd"] / cost_static) * 100
    # proyeksi 90 hari: laju log rata-rata 28 hari terakhir
    rate = d["log_gb"].tail(28).mean()
    proj = pd.DataFrame({"hari_ke": [30, 60, 90]})
    proj["total_log_gb"] = d["cum_log_gb"].iat[-1] + rate * proj["hari_ke"]
    proj["ssd_gb_disarankan"] = np.ceil((cfg.content_gb + rate * cfg.hot_days) *
                                        (1 + cfg.storage_headroom) / 50) * 50
    proj["object_gb"] = proj["total_log_gb"] - rate * cfg.hot_days
    proj["biaya_bulanan_usd"] = (proj["ssd_gb_disarankan"] * cfg.ssd_price_gb_month +
                                 proj["object_gb"] * cfg.object_price_gb_month)
    return summary, proj


def gpu_decisions(cfg, trough_hour):
    rows = []
    for name, cpu_h, speedup in cfg.ml_jobs:
        cpu_cost = cpu_h * cfg.node_price_hour
        gpu_h = cpu_h / speedup + cfg.gpu_startup_hours
        gpu_cost = gpu_h * cfg.gpu_price_hour
        use_gpu = gpu_cost < cpu_cost
        per_month = (30 / cfg.retrain_every_days) * min(cpu_cost, gpu_cost)
        rows.append(dict(job=name, durasi_cpu_jam=cpu_h, biaya_cpu_usd=round(cpu_cost, 3),
                         durasi_gpu_jam=round(gpu_h, 2), biaya_gpu_usd=round(gpu_cost, 3),
                         keputusan="SEWA GPU sementara" if use_gpu else "pakai CPU (GPU tidak efisien)",
                         jadwal=f"setiap {cfg.retrain_every_days} hari, mulai jam {trough_hour:02d}:00 "
                                "(prediksi trafik terendah)",
                         biaya_per_bulan_usd=round(per_month, 2)))
    return pd.DataFrame(rows)


# =====================================================================
# 4. VISUALISASI
# =====================================================================
def _style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", fontsize=12, color=TEXT_PRIMARY, pad=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_MUTED, fontsize=9)


def plot_forecast(test, p50, phi, q, path, hours=24 * 14):
    t = test.index[:hours]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.fill_between(t, p50[:hours], phi[:hours], color=SERIES_COLORS[0], alpha=0.18,
                    linewidth=0, label=f"rentang p50–p{int(q * 100)}")
    ax.plot(t, test["rps"].values[:hours], color=DEMAND_COLOR, lw=1.4, label="aktual")
    ax.plot(t, p50[:hours], color=SERIES_COLORS[0], lw=2, label="prediksi p50")
    _style(ax, "Prediksi beban LMS (2 minggu pertama periode uji)", None, "request / detik")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_allocation(alloc, path, hours=24 * 14):
    a = alloc.iloc[:hours]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(a.index, a["demand_vcpu"], color=DEMAND_COLOR, lw=1.4, label="kebutuhan vCPU aktual")
    for c, name in zip(SERIES_COLORS[1:], ["static", "reactive", "predictive"]):
        ax.step(a.index, a[f"vcpu_{name}"], where="post", color=c, lw=2, label=name)
    _style(ax, "vCPU yang disediakan tiap kebijakan vs kebutuhan", None, "vCPU")
    ax.set_ylim(0, a[[c for c in a.columns if c.startswith("vcpu_")]].values.max() * 1.3)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=4)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cost(res, path):
    r = res[res.kebijakan != "oracle"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(r["kebijakan"], r["biaya_compute_usd"], color=SERIES_COLORS[0], width=0.55)
    for b, (_, row) in zip(bars, r.iterrows()):
        ax.annotate(f"${row['biaya_compute_usd']:,.0f}\noverload {row['overload_pct']:.1f}%",
                    (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8, color=TEXT_PRIMARY)
    _style(ax, "Biaya compute pada periode uji", None, "USD")
    ax.set_ylim(0, r["biaya_compute_usd"].max() * 1.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =====================================================================
# 5. PIPELINE
# =====================================================================
def run(args, cfg):
    os.makedirs(args.out_dir, exist_ok=True)
    pd.Series(asdict(cfg)).to_csv(os.path.join(args.out_dir, "parameter_dipakai.csv"),
                                  header=["nilai"])
    h, daily = build_hourly_traffic(args.data_dir, cfg)
    # data hanya pada periode LMS aktif (abaikan hari tanpa trafik sama sekali)
    H = cfg.horizon_hours
    X = make_features(h, H)
    valid = X.notna().all(axis=1) & (h["rps"] > 0)
    Xv, yv, hv = X[valid], h.loc[valid, "rps"], h[valid]
    cut = int(len(Xv) * (1 - args.test_frac))
    Xtr, Xte, ytr, yte = Xv.iloc[:cut], Xv.iloc[cut:], yv.iloc[:cut], yv.iloc[cut:]
    print(f"[2] Melatih prediktor beban (H={H} jam, latih {len(Xtr):,} jam, uji {len(Xte):,} jam)")
    m50, mhi = fit_quantile_models(Xtr, ytr, cfg.forecast_quantile)
    p50, phi = predict_q(m50, Xte), predict_q(mhi, Xte)
    s24 = 24 * math.ceil(H / 24)
    met = pd.DataFrame([
        forecast_metrics(yte.values, p50, f"GradientBoosting kuantil (usulan)", phi),
        forecast_metrics(yte.values, Xte[f"lag_{s24}"].values, "Baseline: jam sama kemarin"),
        forecast_metrics(yte.values, Xte["lag_168"].values, "Baseline: jam sama minggu lalu"),
        forecast_metrics(yte.values, Xte[f"lag_{H}"].values, f"Baseline: nilai {H} jam lalu"),
    ])
    met.to_csv(os.path.join(args.out_dir, "akurasi_prediksi_beban.csv"), index=False)
    plot_forecast(hv.iloc[cut:], p50, phi, cfg.forecast_quantile,
                  os.path.join(args.out_dir, "prediksi_beban.png"))

    if cfg.ram_model == "trace":
        cfg.ram_base_gb_cluster = cfg.ram_base_ratio * hv["rps"].iloc[:cut].max() / cfg.rps_per_vcpu
        pd.Series(asdict(cfg)).to_csv(os.path.join(args.out_dir, "parameter_dipakai.csv"),
                                      header=["nilai"])
    print("[3] Simulasi kebijakan alokasi vCPU/RAM pada periode uji")
    res, alloc = simulate_policies(hv.iloc[cut:], phi, hv.iloc[:cut], cfg)
    res.to_csv(os.path.join(args.out_dir, "perbandingan_kebijakan_compute.csv"), index=False)
    alloc.to_csv(os.path.join(args.out_dir, "alokasi_per_jam.csv"))
    plot_allocation(alloc, os.path.join(args.out_dir, "alokasi_vcpu.png"))
    plot_cost(res, os.path.join(args.out_dir, "biaya_compute.png"))

    print("[4] Simulasi storage adaptif (tiering) + proyeksi 90 hari")
    st, proj = simulate_storage(daily, hv.index[cut], cfg)
    st.to_csv(os.path.join(args.out_dir, "perbandingan_storage.csv"), index=False)
    proj.to_csv(os.path.join(args.out_dir, "proyeksi_storage_90hari.csv"), index=False)

    print("[5] Estimasi kondisi 24 jam ke depan (model day-ahead)")
    X24 = make_features(h, 24)
    v24 = X24.notna().all(axis=1) & (h["rps"] > 0)
    d50, dhi = fit_quantile_models(X24[v24], h.loc[v24, "rps"], cfg.forecast_quantile)
    last = h.index[-1]
    fut_idx = pd.date_range(last + pd.Timedelta(hours=1), periods=24, freq="h")
    ext = pd.concat([h, pd.DataFrame(index=fut_idx)])
    ext["hour"], ext["dow"], ext["month"] = ext.index.hour, ext.index.dayofweek, ext.index.month
    for c in ["days_to_deadline", "deadlines_next3d", "active_courses"]:
        ext[c] = ext[c].ffill()
    Xf = make_features(ext, 24).loc[fut_idx]
    f50, fhi = predict_q(d50, Xf), predict_q(dhi, Xf)
    upr = np.median(h["users_online"] / h["rps"].replace(0, np.nan))
    n, vcpu, ram = nodes_needed(fhi, fhi * upr, cfg)
    plan = pd.DataFrame({"waktu": fut_idx, "rps_p50": f50.round(2),
                         f"rps_p{int(cfg.forecast_quantile * 100)}": fhi.round(2),
                         "pengguna_online_est": (f50 * upr).round(0),
                         "vcpu_dibutuhkan": vcpu.round(2), "ram_dibutuhkan_gb": ram.round(2),
                         "node_disediakan": n, "vcpu_disediakan": n * cfg.node_vcpu,
                         "ram_disediakan_gb": n * cfg.node_ram_gb,
                         "biaya_usd": (n * cfg.node_price_hour).round(4)})
    plan.to_csv(os.path.join(args.out_dir, "rencana_24jam.csv"), index=False)
    trough = int(fut_idx[np.argmin(f50)].hour)
    gpu = gpu_decisions(cfg, trough)
    gpu.to_csv(os.path.join(args.out_dir, "keputusan_gpu.csv"), index=False)

    # --- ringkasan ---
    pd.set_option("display.width", 160)
    print("\n================ RINGKASAN ================")
    print("Akurasi prediksi beban (periode uji):")
    print(met.round(3).to_string(index=False))
    print("\nKebijakan compute (vCPU+RAM):")
    print(res.round(2).to_string(index=False))
    print("\nStorage:")
    print(st.round(2).to_string(index=False))
    print(f"\nRencana 24 jam ke depan: node {plan.node_disediakan.min()}–{plan.node_disediakan.max()}, "
          f"estimasi biaya compute ${plan.biaya_usd.sum():.2f}/hari "
          f"(static: ${res.loc[res.kebijakan=='static','maks_node'].iat[0]*24*cfg.node_price_hour:.2f}/hari)")
    print("\nKeputusan GPU:")
    print(gpu[["job", "biaya_cpu_usd", "biaya_gpu_usd", "keputusan", "jadwal"]].to_string(index=False))
    print(f"\nHasil tersimpan di: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    cfg = ResourceConfig()
    ap = argparse.ArgumentParser(description="Adaptive resource allocation LMS")
    ap.add_argument("--data_dir", default="./data/OULAD")
    ap.add_argument("--out_dir", default="./hasil/resource")
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--calib", default=None,
                    help="kalibrasi_dari_trace.json hasil server_trace_experiment.py (data server riil)")
    # semua field numerik ResourceConfig bisa diubah dari command line
    for k, v in asdict(cfg).items():
        if isinstance(v, (int, float)):
            ap.add_argument(f"--{k}", type=type(v), default=v)
    args = ap.parse_args()
    for k in asdict(cfg):
        if hasattr(args, k):
            setattr(cfg, k, getattr(args, k))
    if args.calib:
        import json
        with open(args.calib) as f:
            cal = json.load(f)
        print(f"[*] Kalibrasi dari server riil: {cal['sumber']}")
        for name, old, new in [("burst_factor", cfg.burst_factor, cal["burst_factor"]),
                               ("ram_model", cfg.ram_model, "trace"),
                               ("ram_per_vcpu_gb", "-", cal["ram_per_vcpu_gb"]),
                               ("ram_base_ratio", "-", cal["ram_base_ratio"]),
                               ("overprovision_real", "-", cal["overprovision_cpu"])]:
            print(f"    {name:<20}: {old} -> {new}")
        cfg.burst_factor = cal["burst_factor"]
        cfg.ram_model = "trace"
        cfg.ram_per_vcpu_gb = cal["ram_per_vcpu_gb"]
        cfg.ram_base_ratio = cal["ram_base_ratio"]
        cfg.ram_base_per_node_gb = 0.0   # RAM dasar sudah tercakup di ram_base_gb_cluster
        cfg.overprovision_real = cal["overprovision_cpu"]
    if args.demo:
        from adaptive_lms_model import make_demo_oulad
        args.data_dir = os.path.join(args.out_dir, "data_demo")
        make_demo_oulad(args.data_dir)
    run(args, cfg)
