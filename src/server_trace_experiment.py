"""
=====================================================================
 EKSPERIMEN DATA SERVER RIIL (vCPU & RAM) UNTUK AUTOSCALING ADAPTIF
 Dataset jejak beban cloud publik:
   - Bitbrains GWA-T-12 (UTAMA): 1.250 VM (fastStorage, 1 bulan) /
     500 VM (Rnd, 3 bulan), sampel tiap 5 menit, per VM tercatat:
     jumlah core, kapasitas & pemakaian CPU (MHz), kapasitas &
     pemakaian memori (KB).  -> vCPU & RAM riil (disediakan vs terpakai)
   - Alibaba cluster-trace-v2018 (ALTERNATIF): ~4.000 mesin, 8 hari,
     CPU% dan memori% per mesin (memori dinormalisasi, GB = asumsi).
=====================================================================

Peran dalam penelitian (dipasangkan dengan OULAD):
  OULAD          -> POLA PENGGUNA LMS (kapan & seberapa ramai)
  Jejak server   -> PERILAKU vCPU & RAM RIIL (burst, rasio RAM/CPU,
                    over-provisioning nyata, akurasi prediksi di server riil)

Eksperimen A (file ini):
  prediksi beban vCPU & RAM pada data server riil + simulasi autoscaling
  (static riil vs static puncak vs reactive vs predictive vs oracle).
  Keluaran juga berisi kalibrasi_dari_trace.json.

Eksperimen B (adaptive_lms_resource.py --calib kalibrasi_dari_trace.json):
  trafik LMS dari OULAD, dengan parameter server diganti nilai dari
  jejak server riil (bukan asumsi).

Cara pakai:
  # Bitbrains: ekstrak fastStorage.zip -> ./fastStorage/2013-8/*.csv
  python server_trace_experiment.py --trace bitbrains --trace_dir ./fastStorage
  # Alibaba: machine_meta.csv & machine_usage.csv di satu folder
  python server_trace_experiment.py --trace alibaba --trace_dir ./alibaba2018 --max_machines 300
  # uji coba tanpa unduh (data sintetis berformat sama):
  python server_trace_experiment.py --demo bitbrains
"""

import argparse
import glob
import json
import math
import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adaptive_lms_resource import (DEMAND_COLOR, SERIES_COLORS, ResourceConfig, _style,
                                   fit_quantile_models, forecast_metrics, limit_scale_in,
                                   predict_q)

warnings.filterwarnings("ignore")
BIN = 300  # 5 menit


# =====================================================================
# 1. LOADER JEJAK SERVER
# =====================================================================
def _col(cols, *keys):
    for c in cols:
        if all(k in c.lower() for k in keys):
            return c
    raise KeyError(f"Kolom dengan kata kunci {keys} tidak ditemukan di {list(cols)}")


def _natural(p):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p)]


def load_bitbrains(trace_dir, max_vms=None):
    """Format GWA-T-12: satu file CSV per VM, pemisah ';' (diikuti tab).
    Kolom: Timestamp, CPU cores, CPU capacity provisioned [MHZ], CPU usage [MHZ],
    CPU usage [%], Memory capacity provisioned [KB], Memory usage [KB], disk, network."""
    files = sorted(glob.glob(os.path.join(trace_dir, "**", "*.csv"), recursive=True), key=_natural)
    if not files:
        raise FileNotFoundError(f"Tidak ada file .csv di {trace_dir}")
    if max_vms:
        files = files[:max_vms]
    total = None
    for i, f in enumerate(files, 1):
        df = pd.read_csv(f, sep=";", skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        ts = pd.to_numeric(df[_col(df.columns, "timestamp")], errors="coerce")
        if ts.median() > 1e11:      # jaga-jaga bila benar-benar milidetik
            ts = ts / 1000
        cores = pd.to_numeric(df[_col(df.columns, "cpu cores")], errors="coerce")
        cap = pd.to_numeric(df[_col(df.columns, "cpu capacity")], errors="coerce")
        use = pd.to_numeric(df[_col(df.columns, "cpu usage", "mhz")], errors="coerce")
        mcap = pd.to_numeric(df[_col(df.columns, "memory capacity")], errors="coerce")
        muse = pd.to_numeric(df[_col(df.columns, "memory usage")], errors="coerce")
        mhz_per_core = cap / cores.replace(0, np.nan)
        vcpu = np.minimum((use / mhz_per_core).clip(lower=0), cores).fillna(0)
        part = pd.DataFrame({"bin": (ts // BIN * BIN), "vcpu_used": vcpu,
                             "ram_used_gb": muse / 1024 ** 2, "vcpu_prov": cores,
                             "ram_prov_gb": mcap / 1024 ** 2}).dropna(subset=["bin"])
        part = part.groupby(part["bin"].astype("int64")).mean().drop(columns="bin", errors="ignore")
        part["n_vm"] = 1
        total = part if total is None else total.add(part, fill_value=0)
        if i % 200 == 0:
            print(f"    {i}/{len(files)} VM dibaca")
    total.index = pd.to_datetime(total.index, unit="s")
    return total.sort_index(), f"Bitbrains GWA-T-12 ({len(files)} VM)"


def load_alibaba(trace_dir, max_machines=300, machine_ram_gb=256.0, chunksize=5_000_000):
    """machine_meta.csv & machine_usage.csv (tanpa header, skema v2018).
    Memori Alibaba dinormalisasi 0-100, jadi GB = mem% x machine_ram_gb (ASUMSI)."""
    meta = pd.read_csv(os.path.join(trace_dir, "machine_meta.csv"), header=None,
                       names=["machine_id", "time_stamp", "fd1", "fd2", "cpu_num", "mem_size", "status"])
    cpu_num = meta.groupby("machine_id")["cpu_num"].max()
    ids = set(sorted(cpu_num.index, key=_natural)[:max_machines])
    parts = []
    reader = pd.read_csv(os.path.join(trace_dir, "machine_usage.csv"), header=None,
                         usecols=[0, 1, 2, 3], names=["machine_id", "ts", "cpu", "mem"],
                         chunksize=chunksize)
    for k, ch in enumerate(reader, 1):
        ch = ch[ch["machine_id"].isin(ids) & ch["cpu"].between(0, 100) & ch["mem"].between(0, 100)]
        ch = ch.assign(bin=(ch["ts"] // BIN * BIN).astype("int64"))
        parts.append(ch.groupby(["machine_id", "bin"])[["cpu", "mem"]].agg(["sum", "count"]))
        print(f"    chunk {k} dibaca")
    g = pd.concat(parts).groupby(level=[0, 1]).sum()
    df = pd.DataFrame({"cpu": g[("cpu", "sum")] / g[("cpu", "count")],
                       "mem": g[("mem", "sum")] / g[("mem", "count")]}).reset_index()
    df["cores"] = df["machine_id"].map(cpu_num)
    df["vcpu_used"] = df["cpu"] / 100 * df["cores"]
    df["ram_used_gb"] = df["mem"] / 100 * machine_ram_gb
    total = df.groupby("bin").agg(vcpu_used=("vcpu_used", "sum"), ram_used_gb=("ram_used_gb", "sum"),
                                  vcpu_prov=("cores", "sum"), n_vm=("machine_id", "nunique"))
    total["ram_prov_gb"] = total["n_vm"] * machine_ram_gb
    # timestamp Alibaba relatif (detik sejak awal periode); tanggal awal tidak diumumkan
    total.index = pd.Timestamp("2018-01-01") + pd.to_timedelta(total.index, unit="s")
    return total.sort_index(), f"Alibaba cluster-trace-v2018 ({len(ids)} mesin)"


def to_hourly(t):
    t = t[t["n_vm"] >= 0.9 * t["n_vm"].max()]   # buang bin yang VM-nya tidak lengkap
    r = t.resample("h")
    h = pd.DataFrame({"vcpu_mean": r["vcpu_used"].mean(), "vcpu_peak": r["vcpu_used"].max(),
                      "ram_mean": r["ram_used_gb"].mean(), "ram_peak": r["ram_used_gb"].max(),
                      "vcpu_prov": r["vcpu_prov"].mean(), "ram_prov_gb": r["ram_prov_gb"].mean()})
    return h.interpolate(limit=3).dropna()


# =====================================================================
# 2. PREDIKSI vCPU & RAM
# =====================================================================
def trace_features(h, target, H, use_week):
    y = h[target]
    X = pd.DataFrame({"hour": h.index.hour, "dow": h.index.dayofweek}, index=h.index)
    lags = {H, H + 1, H + 2, 24 * math.ceil(H / 24), 48}
    if use_week:
        lags.add(168)
    for L in sorted(lags):
        X[f"lag_{L}"] = y.shift(L)
    X["roll_mean_24"] = y.shift(H).rolling(24).mean()
    X["roll_max_24"] = y.shift(H).rolling(24).max()
    other = "ram_peak" if target.startswith("vcpu") else "vcpu_peak"
    X["other_lag"] = h[other].shift(H)
    return X


# =====================================================================
# 3. SIMULASI KEBIJAKAN
# =====================================================================
def nodes_for(vcpu, ram, cfg, cpu_util, mem_util=0.85):
    n = np.ceil(np.maximum(vcpu / (cfg.node_vcpu * cpu_util), ram / (cfg.node_ram_gb * mem_util)))
    return np.maximum(n, cfg.min_nodes).astype(int)


def simulate(test, train, pv, pr, cfg):
    v, r = test["vcpu_peak"].values, test["ram_peak"].values
    pol = {}
    pol["static_riil"] = nodes_for(np.full(len(test), train["vcpu_prov"].mean()),
                                   np.full(len(test), train["ram_prov_gb"].mean()), cfg, 1.0, 1.0)
    pk = nodes_for(np.array([train["vcpu_peak"].max() * (1 + cfg.static_headroom)]),
                   np.array([train["ram_peak"].max() * (1 + cfg.static_headroom)]), cfg,
                   cfg.target_cpu_util)[0]
    pol["static_puncak"] = np.full(len(test), pk)
    pv0 = np.r_[train["vcpu_peak"].values[-1], v[:-1]]
    pr0 = np.r_[train["ram_peak"].values[-1], r[:-1]]
    pol["reactive"] = nodes_for(pv0 * (1 + cfg.reactive_headroom), pr0 * (1 + cfg.reactive_headroom),
                                cfg, cfg.target_cpu_util)
    pol["predictive"] = limit_scale_in(nodes_for(pv, pr, cfg, cfg.target_cpu_util), cfg.max_scale_in_step)
    pol["oracle"] = nodes_for(v, r, cfg, cfg.target_cpu_util)
    rows, alloc = [], pd.DataFrame({"demand_vcpu": v, "demand_ram_gb": r}, index=test.index)
    for name, n in pol.items():
        over = (v > n * cfg.node_vcpu * cfg.max_safe_util) | (r > n * cfg.node_ram_gb * 0.95)
        rows.append(dict(kebijakan=name, rata2_node=n.mean(), maks_node=int(n.max()),
                         biaya_usd=float(n.sum() * cfg.node_price_hour),
                         util_cpu_pct=float(np.mean(v / (n * cfg.node_vcpu)) * 100),
                         util_ram_pct=float(np.mean(r / (n * cfg.node_ram_gb)) * 100),
                         jam_overload=int(over.sum()), overload_pct=float(over.mean() * 100)))
        alloc[f"vcpu_{name}"] = n * cfg.node_vcpu
        alloc[f"ram_gb_{name}"] = n * cfg.node_ram_gb
    res = pd.DataFrame(rows)
    base = res.loc[res.kebijakan == "static_riil", "biaya_usd"].iat[0]
    res["hemat_vs_static_riil_pct"] = (1 - res["biaya_usd"] / base) * 100
    return res, alloc


# =====================================================================
# 4. GRAFIK
# =====================================================================
def plot_pred(test, p50, phi, col, unit, path, hours=24 * 5):
    t = test.index[:hours]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(t, p50[:hours], phi[:hours], color=SERIES_COLORS[0], alpha=0.18, linewidth=0,
                    label="rentang p50–p90")
    ax.plot(t, test[col].values[:hours], color=DEMAND_COLOR, lw=1.4, label="aktual (server riil)")
    ax.plot(t, p50[:hours], color=SERIES_COLORS[0], lw=2, label="prediksi p50")
    _style(ax, f"Prediksi {unit} puncak per jam pada jejak server riil", None, unit)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
    ax.set_ylim(0, max(test[col].values[:hours].max(), phi[:hours].max()) * 1.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_alloc(alloc, path, hours=24 * 5):
    a = alloc.iloc[:hours]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(a.index, a["demand_vcpu"], color=DEMAND_COLOR, lw=1.4, label="vCPU terpakai (riil)")
    for c, name in zip(SERIES_COLORS, ["static_puncak", "reactive", "predictive"]):
        ax.step(a.index, a[f"vcpu_{name}"], where="post", color=c, lw=2, label=name)
    _style(ax, "vCPU disediakan vs vCPU terpakai (server riil)", None, "vCPU")
    ax.set_ylim(0, a[["vcpu_static_puncak", "vcpu_reactive", "vcpu_predictive"]].values.max() * 1.3)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=4)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cost(res, path):
    r = res[res.kebijakan != "oracle"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(r["kebijakan"], r["biaya_usd"], color=SERIES_COLORS[0], width=0.55)
    for b, (_, row) in zip(bars, r.iterrows()):
        ax.annotate(f"${row['biaya_usd']:,.0f}\noverload {row['overload_pct']:.1f}%",
                    (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8)
    _style(ax, "Biaya compute periode uji (server riil)", None, "USD")
    ax.set_ylim(0, r["biaya_usd"].max() * 1.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =====================================================================
# 5. PIPELINE EKSPERIMEN A + KALIBRASI
# =====================================================================
def run(args, cfg):
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[1] Membaca jejak server: {args.trace} @ {args.trace_dir}")
    if args.trace == "bitbrains":
        raw, name = load_bitbrains(args.trace_dir, args.max_vms)
    else:
        raw, name = load_alibaba(args.trace_dir, args.max_machines, args.machine_ram_gb)
    h = to_hourly(raw)
    h.to_csv(os.path.join(args.out_dir, "trace_per_jam.csv"))
    use_week = len(h) > 24 * 21
    print(f"    {name}: {len(h)} jam, vCPU terpakai puncak {h.vcpu_peak.max():,.1f} "
          f"dari {h.vcpu_prov.mean():,.0f} disediakan; RAM puncak {h.ram_peak.max():,.1f} GB "
          f"dari {h.ram_prov_gb.mean():,.0f} GB")

    H = cfg.horizon_hours
    preds, mets = {}, []
    for tgt, unit in [("vcpu_peak", "vCPU"), ("ram_peak", "RAM (GB)")]:
        X = trace_features(h, tgt, H, use_week)
        ok = X.notna().all(axis=1)
        Xv, yv = X[ok], h.loc[ok, tgt]
        cut = int(len(Xv) * (1 - args.test_frac))
        m50, mhi = fit_quantile_models(Xv.iloc[:cut], yv.iloc[:cut], cfg.forecast_quantile)
        p50, phi = predict_q(m50, Xv.iloc[cut:]), predict_q(mhi, Xv.iloc[cut:])
        yt = yv.iloc[cut:].values
        mets += [dict(target=unit, **forecast_metrics(yt, p50, "GradientBoosting kuantil (usulan)", phi)),
                 dict(target=unit, **forecast_metrics(yt, Xv.iloc[cut:]["lag_24"].values, "Baseline: jam sama kemarin")),
                 dict(target=unit, **forecast_metrics(yt, Xv.iloc[cut:][f"lag_{H}"].values, f"Baseline: {H} jam lalu"))]
        preds[tgt] = (Xv.index[cut:], p50, phi)
        plot_pred(h.loc[Xv.index[cut:]], p50, phi, tgt, unit,
                  os.path.join(args.out_dir, f"prediksi_{tgt}.png"))
    met = pd.DataFrame(mets)
    met.to_csv(os.path.join(args.out_dir, "akurasi_prediksi_trace.csv"), index=False)

    idx = preds["vcpu_peak"][0].intersection(preds["ram_peak"][0])
    sel = lambda k: pd.Series(preds[k][2], index=preds[k][0]).loc[idx].values
    test, train = h.loc[idx], h.loc[h.index < idx[0]]
    print("[2] Simulasi autoscaling pada vCPU & RAM riil")
    res, alloc = simulate(test, train, sel("vcpu_peak"), sel("ram_peak"), cfg)
    res.to_csv(os.path.join(args.out_dir, "perbandingan_kebijakan_trace.csv"), index=False)
    alloc.to_csv(os.path.join(args.out_dir, "alokasi_trace_per_jam.csv"))
    plot_alloc(alloc, os.path.join(args.out_dir, "alokasi_vcpu_trace.png"))
    plot_cost(res, os.path.join(args.out_dir, "biaya_trace.png"))

    print("[3] Kalibrasi parameter dari server riil")
    burst = (h["vcpu_peak"] / h["vcpu_mean"]).replace([np.inf, -np.inf], np.nan).dropna()
    b, a = np.polyfit(h["vcpu_mean"], h["ram_mean"], 1)
    b = max(float(b), 0.0)
    a = float(h["ram_mean"].mean() - b * h["vcpu_mean"].mean()) if b == 0 else float(a)
    calib = {
        "sumber": name,
        "jumlah_jam": int(len(h)),
        "burst_factor": round(float(burst.median()), 4),
        "ram_per_vcpu_gb": round(b, 4),
        "ram_base_ratio": round(a / float(h["vcpu_mean"].max()), 4),
        "overprovision_cpu": round(float(h["vcpu_prov"].mean() / h["vcpu_peak"].max()), 3),
        "overprovision_ram": round(float(h["ram_prov_gb"].mean() / h["ram_peak"].max()), 3),
        "util_cpu_rata2_pct": round(float((h["vcpu_mean"] / h["vcpu_prov"]).mean() * 100), 2),
        "util_ram_rata2_pct": round(float((h["ram_mean"] / h["ram_prov_gb"]).mean() * 100), 2),
        "korelasi_cpu_ram_per_jam": round(float(h["vcpu_mean"].corr(h["ram_mean"])), 3),
        "catatan": ("ram_base_ratio = RAM dasar (GB) per vCPU beban puncak; ram_per_vcpu_gb = "
                    "tambahan RAM per vCPU terpakai; burst_factor = puncak 5-menit / rata-rata jam"),
    }
    with open(os.path.join(args.out_dir, "kalibrasi_dari_trace.json"), "w") as f:
        json.dump(calib, f, indent=2, ensure_ascii=False)

    pd.set_option("display.width", 170)
    print("\n================ RINGKASAN EKSPERIMEN A (server riil) ================")
    print(met.round(3).to_string(index=False))
    print()
    print(res.round(2).to_string(index=False))
    print("\nKalibrasi (dipakai Eksperimen B / OULAD):")
    for k, v in calib.items():
        if k != "catatan":
            print(f"  {k:<26}: {v}")
    print(f"\nHasil tersimpan di: {os.path.abspath(args.out_dir)}")


# =====================================================================
# 6. DATA DEMO (format identik, HANYA untuk uji kode)
# =====================================================================
def make_demo_bitbrains(out_dir, n_vms=80, days=30, seed=3):
    rng = np.random.default_rng(seed)
    d = os.path.join(out_dir, "2013-8")
    os.makedirs(d, exist_ok=True)
    t0 = 1375315200  # 2013-08-01
    ts = t0 + np.arange(0, days * 86400, BIN)
    hod = ((ts - t0) % 86400) / 3600
    dow = ((ts - t0) // 86400 + 3) % 7
    daily = 0.5 + 0.5 * np.exp(-((hod - 14) ** 2) / 18)
    week = np.where(dow >= 5, 0.6, 1.0)
    header = ";\t".join(["Timestamp [ms]", "CPU cores", "CPU capacity provisioned [MHZ]",
                         "CPU usage [MHZ]", "CPU usage [%]", "Memory capacity provisioned [KB]",
                         "Memory usage [KB]", "Disk read throughput [KB/s]",
                         "Disk write throughput [KB/s]", "Network received throughput [KB/s]",
                         "Network transmitted throughput [KB/s]"])
    for v in range(1, n_vms + 1):
        cores = int(rng.choice([1, 2, 4, 8], p=[.2, .35, .3, .15]))
        cap = cores * 2600.0
        memcap = float(rng.choice([2, 4, 8, 16])) * 1024 ** 2
        base = rng.beta(1.2, 9)
        amp = rng.uniform(0, 1.2)
        util = base * (1 + amp * (daily - 0.5)) * week * rng.lognormal(0, 0.35, len(ts))
        util = np.clip(util + (rng.random(len(ts)) < 0.005) * rng.uniform(.3, .8, len(ts)), 0, 1)
        mem = np.clip(memcap * (0.08 + 0.25 * base + 0.15 * util + rng.normal(0, 0.01, len(ts))), 0, memcap)
        lines = [header] + [f"{t};\t{cores};\t{cap:.3f};\t{u * cap:.3f};\t{u * 100:.3f};\t{memcap:.1f};\t{m:.1f};\t0;\t0;\t0;\t0"
                            for t, u, m in zip(ts, util, mem)]
        with open(os.path.join(d, f"{v}.csv"), "w") as f:
            f.write("\n".join(lines))
    print(f"[0] Demo Bitbrains: {n_vms} VM x {days} hari di {d}")


def make_demo_alibaba(out_dir, n_machines=40, days=8, seed=5):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"m": [f"m_{i}" for i in range(1, n_machines + 1)], "t": 0, "f1": 1, "f2": "A",
                  "c": 96, "s": 100, "st": "USING"}).to_csv(
        os.path.join(out_dir, "machine_meta.csv"), header=False, index=False)
    ts = np.arange(0, days * 86400, 60)
    hod = (ts % 86400) / 3600
    rows = []
    for i in range(1, n_machines + 1):
        base = rng.uniform(20, 45)
        cpu = np.clip(base + 15 * np.sin((hod - 8) / 24 * 2 * np.pi) + rng.normal(0, 6, len(ts)), 0, 100)
        mem = np.clip(base + 35 + 0.2 * cpu + rng.normal(0, 2, len(ts)), 0, 100)
        rows.append(pd.DataFrame({"m": f"m_{i}", "t": ts, "c": cpu.round(), "mm": mem.round(),
                                  "g": 0, "k": 0, "ni": 0, "no": 0, "d": 0}))
    pd.concat(rows).sort_values("t").to_csv(os.path.join(out_dir, "machine_usage.csv"),
                                            header=False, index=False)
    print(f"[0] Demo Alibaba: {n_machines} mesin x {days} hari di {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Eksperimen autoscaling pada jejak server riil")
    ap.add_argument("--trace", choices=["bitbrains", "alibaba"], default="bitbrains")
    ap.add_argument("--trace_dir", default="./data/bitbrains/fastStorage")
    ap.add_argument("--out_dir", default="./hasil/trace")
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--max_vms", type=int, default=None, help="Bitbrains: batasi jumlah VM")
    ap.add_argument("--max_machines", type=int, default=300, help="Alibaba: jumlah mesin")
    ap.add_argument("--machine_ram_gb", type=float, default=256.0,
                    help="Alibaba: RAM per mesin (GB), ASUMSI karena memori dinormalisasi")
    ap.add_argument("--node_vcpu", type=int, default=4)
    ap.add_argument("--node_ram_gb", type=float, default=16.0)
    ap.add_argument("--node_price_hour", type=float, default=0.247,
                    help="contoh m6i.xlarge (4 vCPU, 16 GiB) ~2x harga m6i.large Jakarta")
    ap.add_argument("--demo", choices=["bitbrains", "alibaba"], default=None)
    args = ap.parse_args()
    cfg = ResourceConfig(node_vcpu=args.node_vcpu, node_ram_gb=args.node_ram_gb,
                         node_price_hour=args.node_price_hour, max_nodes=10 ** 6)
    if args.demo:
        args.trace = args.demo
        args.trace_dir = os.path.join(args.out_dir, f"data_demo_{args.demo}")
        (make_demo_bitbrains if args.demo == "bitbrains" else make_demo_alibaba)(args.trace_dir)
    run(args, cfg)
