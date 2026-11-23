"""
=====================================================================
 MODEL LMS ADAPTIF BERBASIS MACHINE LEARNING
 Judul: Pengujian Model LMS yang Adaptif Menggunakan Dataset
        Terkait Penggunaan Aplikasi LMS
 Dataset utama: OULAD (Open University Learning Analytics Dataset)
                https://analyse.kmi.open.ac.uk/open_dataset
=====================================================================

Alur model (ringkas):
  1. Memuat log penggunaan LMS (klik per materi/aktivitas per hari),
     nilai tugas, dan data demografis mahasiswa.
  2. Membangun fitur "sampai minggu ke-w" (early prediction): model hanya
     melihat data yang sudah tersedia pada minggu tersebut.
  3. Melatih & menguji beberapa model ML (Logistic Regression, Random
     Forest, Gradient Boosting, MLP) untuk memprediksi mahasiswa berisiko
     (final_result = Fail / Withdrawn).
  4. Mesin adaptif:
       a. Level risiko (Rendah / Sedang / Tinggi) dari probabilitas model
       b. Profil belajar (K-Means pada pola penggunaan aktivitas LMS)
       c. Rekomendasi aktivitas: membandingkan pola aktivitas mahasiswa
          dengan pola mahasiswa yang lulus pada modul yang sama
       d. Jalur belajar adaptif (remedial / terbimbing / pengayaan)
  5. Pengujian: metrik per model, metrik per minggu (seberapa dini model
     bisa mendeteksi), perkembangan risiko tiap mahasiswa dari minggu ke
     minggu, dan validasi offline rekomendasi.

Cara pakai:
  # dengan dataset OULAD asli (ekstrak zip ke folder ./OULAD)
  python adaptive_lms_model.py --data_dir ./OULAD --out_dir ./hasil

  # mode demo (membuat data sintetis berstruktur sama dengan OULAD)
  python adaptive_lms_model.py --demo --out_dir ./hasil_demo

Kebutuhan: python>=3.9, pandas, numpy, scikit-learn>=1.2, matplotlib
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

KEY = ["code_module", "code_presentation", "id_student"]
RANDOM_STATE = 42

# Deskripsi tipe aktivitas pada VLE OULAD (untuk teks rekomendasi)
ACTIVITY_DESC = {
    "oucontent": "materi inti modul",
    "resource": "bahan bacaan / file materi",
    "subpage": "sub-halaman materi",
    "homepage": "beranda kursus (pengumuman)",
    "url": "tautan sumber eksternal",
    "forumng": "forum diskusi",
    "quiz": "kuis latihan",
    "externalquiz": "kuis eksternal",
    "ouwiki": "wiki kolaboratif",
    "oucollaborate": "kelas online / webinar",
    "ouelluminate": "sesi kelas virtual",
    "glossary": "glosarium istilah",
    "dataplus": "aktivitas data",
    "htmlactivity": "aktivitas interaktif",
    "page": "halaman informasi",
    "questionnaire": "kuesioner",
    "repeatactivity": "aktivitas pengulangan",
    "sharedsubpage": "sub-halaman bersama",
    "dualpane": "materi dua panel",
    "folder": "folder materi",
}

# Palet kategorikal (urutan tetap) untuk grafik
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT_PRIMARY, TEXT_MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"


# =====================================================================
# 1. MEMUAT DATA
# =====================================================================
def load_oulad(data_dir):
    """Membaca file CSV OULAD. File studentVle.csv besar (~430 MB),
    jadi tipe data diatur agar hemat memori."""
    na = ["?", ""]
    p = lambda f: os.path.join(data_dir, f)
    print(f"[1] Memuat dataset dari: {data_dir}")
    info = pd.read_csv(p("studentInfo.csv"), na_values=na)
    reg = pd.read_csv(p("studentRegistration.csv"), na_values=na)
    assess = pd.read_csv(p("assessments.csv"), na_values=na)
    sassess = pd.read_csv(p("studentAssessment.csv"), na_values=na)
    vle = pd.read_csv(p("vle.csv"), na_values=na)
    svle = pd.read_csv(
        p("studentVle.csv"),
        usecols=["code_module", "code_presentation", "id_student",
                 "id_site", "date", "sum_click"],
        dtype={"code_module": "category", "code_presentation": "category",
               "id_student": "int32", "id_site": "int32",
               "date": "int16", "sum_click": "int32"},
    )
    site2type = vle.drop_duplicates("id_site").set_index("id_site")["activity_type"]
    svle["activity_type"] = svle["id_site"].map(site2type).astype("category")
    activity_types = sorted(vle["activity_type"].dropna().unique())
    print(f"    mahasiswa-modul: {len(info):,} | log interaksi: {len(svle):,} "
          f"| tipe aktivitas: {len(activity_types)}")
    return dict(info=info, reg=reg, assess=assess, sassess=sassess,
                svle=svle, activity_types=activity_types)


def _keys_to_str(df):
    df = df.reset_index()
    df["code_module"] = df["code_module"].astype(str)
    df["code_presentation"] = df["code_presentation"].astype(str)
    df["id_student"] = df["id_student"].astype("int64")
    return df


# =====================================================================
# 2. FEATURE ENGINEERING (data sampai minggu ke-w)
# =====================================================================
def build_features(data, week):
    """Fitur per (modul, presentasi, mahasiswa) memakai data s.d. hari week*7.
    Mahasiswa yang sudah keluar (unregister) sebelum batas tidak diprediksi,
    karena statusnya sudah diketahui (menghindari kebocoran label)."""
    cutoff = week * 7
    info = data["info"].merge(data["reg"], on=KEY, how="left")
    still_active = info["date_unregistration"].isna() | (info["date_unregistration"] > cutoff)
    base = info[still_active].copy()
    base["id_student"] = base["id_student"].astype("int64")

    # --- a. Penggunaan LMS per tipe aktivitas ---
    sv = data["svle"][data["svle"]["date"] <= cutoff]
    act = (sv.groupby(KEY + ["activity_type"], observed=True)["sum_click"]
             .sum().unstack(fill_value=0))
    act = act.reindex(columns=data["activity_types"], fill_value=0)
    act.columns = [f"clk_{c}" for c in act.columns]
    act = _keys_to_str(act)

    eng = sv.groupby(KEY, observed=True).agg(
        total_clicks=("sum_click", "sum"),
        active_days=("date", "nunique"),
        n_materi=("id_site", "nunique"),
    )
    eng = _keys_to_str(eng)
    last = _keys_to_str(sv[sv["date"] > cutoff - 7].groupby(KEY, observed=True)["sum_click"]
                        .sum().rename("clicks_last_week"))
    prev = _keys_to_str(sv[(sv["date"] > cutoff - 14) & (sv["date"] <= cutoff - 7)]
                        .groupby(KEY, observed=True)["sum_click"].sum().rename("clicks_prev_week"))

    # --- b. Penilaian (tugas yang sudah dikumpulkan s.d. batas) ---
    a = data["sassess"].merge(data["assess"], on="id_assessment", how="left")
    a = a[(a["assessment_type"] != "Exam") & (a["date_submitted"] <= cutoff)].copy()
    a["late"] = (a["date_submitted"] > a["date"]).astype(int)
    a["w_score"] = a["score"] * a["weight"]
    sc = a.groupby(KEY).agg(mean_score=("score", "mean"), n_submitted=("score", "size"),
                            n_late=("late", "sum"), w_score=("w_score", "sum"),
                            w_sum=("weight", "sum")).reset_index()
    sc["weighted_score"] = np.where(sc["w_sum"] > 0, sc["w_score"] / sc["w_sum"].replace(0, np.nan),
                                    sc["mean_score"])
    sc = sc.drop(columns=["w_score", "w_sum"])
    due = (data["assess"][(data["assess"]["assessment_type"] != "Exam") &
                          (data["assess"]["date"] <= cutoff)]
           .groupby(["code_module", "code_presentation"]).size().rename("n_due").reset_index())

    # --- c. Gabungkan ---
    df = base
    for part in (act, eng, last, prev, sc):
        df = df.merge(part, on=KEY, how="left")
    df = df.merge(due, on=["code_module", "code_presentation"], how="left")

    clk_cols = [c for c in df.columns if c.startswith("clk_")]
    fill0 = clk_cols + ["total_clicks", "active_days", "n_materi", "clicks_last_week",
                        "clicks_prev_week", "n_submitted", "n_late", "n_due"]
    df[fill0] = df[fill0].fillna(0)
    df["has_score"] = df["mean_score"].notna().astype(int)
    df[["mean_score", "weighted_score"]] = df[["mean_score", "weighted_score"]].fillna(0)
    df["n_missed"] = (df["n_due"] - df["n_submitted"]).clip(lower=0)
    df["clicks_trend"] = df["clicks_last_week"] - df["clicks_prev_week"]
    df["clicks_per_day"] = df["total_clicks"] / df["active_days"].replace(0, np.nan)
    df["clicks_per_day"] = df["clicks_per_day"].fillna(0)
    # keterlibatan relatif terhadap teman sekelas (modul & presentasi sama)
    med = df.groupby(["code_module", "code_presentation"])["total_clicks"].transform("median")
    df["rel_engagement"] = df["total_clicks"] / med.replace(0, np.nan)
    df["rel_engagement"] = df["rel_engagement"].fillna(0)
    df["date_registration"] = df["date_registration"].fillna(df["date_registration"].median())
    df["imd_band"] = df["imd_band"].fillna("Unknown")
    for c in clk_cols:  # skala log mengurangi pengaruh outlier klik
        df[c] = np.log1p(df[c])

    df["at_risk"] = df["final_result"].isin(["Fail", "Withdrawn"]).astype(int)

    cat_cols = ["code_module", "gender", "highest_education", "imd_band",
                "age_band", "disability"]
    dummies = pd.get_dummies(df[cat_cols], prefix=cat_cols, dtype=int)
    num_cols = (clk_cols + ["total_clicks", "active_days", "n_materi", "clicks_last_week",
                            "clicks_trend", "clicks_per_day", "rel_engagement",
                            "mean_score", "weighted_score", "has_score", "n_submitted",
                            "n_late", "n_missed", "num_of_prev_attempts",
                            "studied_credits", "date_registration"])
    X = pd.concat([df[num_cols].astype(float), dummies], axis=1)
    X["total_clicks"] = np.log1p(X["total_clicks"])
    meta = df[KEY + ["final_result", "at_risk"]].reset_index(drop=True)
    X = X.reset_index(drop=True)
    return X, meta


# =====================================================================
# 3. MODEL MACHINE LEARNING
# =====================================================================
def get_models():
    return {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced")),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE),
        "GradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, class_weight="balanced",
            random_state=RANDOM_STATE),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=400,
                          early_stopping=True, random_state=RANDOM_STATE)),
    }


def split_data(meta, mode, test_presentation):
    """temporal: latih pada semester lama, uji pada semester terbaru
    (paling mirip kondisi nyata). random: pembagian acak per mahasiswa."""
    if mode == "temporal":
        test = (meta["code_presentation"] == test_presentation).values
        if test.sum() == 0:
            raise ValueError(f"Presentasi {test_presentation} tidak ditemukan.")
        return np.where(~test)[0], np.where(test)[0]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    return next(gss.split(meta, meta["at_risk"], groups=meta["id_student"]))


def evaluate(y_true, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    return dict(accuracy=accuracy_score(y_true, pred),
                precision=precision_score(y_true, pred, zero_division=0),
                recall=recall_score(y_true, pred, zero_division=0),
                f1=f1_score(y_true, pred, zero_division=0),
                f1_macro=f1_score(y_true, pred, average="macro", zero_division=0),
                roc_auc=roc_auc_score(y_true, prob))


# =====================================================================
# 4. MESIN ADAPTIF
# =====================================================================
class AdaptiveLMSEngine:
    """Mengubah prediksi model menjadi keputusan adaptif per mahasiswa."""

    def __init__(self, model, low=0.4, high=0.7, n_profiles=4):
        self.model, self.low, self.high, self.k = model, low, high, n_profiles

    @staticmethod
    def _activity_share(X):
        clk = X[[c for c in X.columns if c.startswith("clk_")]]
        raw = np.expm1(clk)
        share = raw.div(raw.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        share.columns = [c.replace("clk_", "") for c in share.columns]
        return share

    def fit(self, X_tr, meta_tr):
        self.model.fit(X_tr, meta_tr["at_risk"])
        share = self._activity_share(X_tr)
        # pola aktivitas mahasiswa yang berhasil, per modul
        ok = meta_tr["at_risk"].values == 0
        self.success_profile = share[ok].groupby(meta_tr.loc[ok, "code_module"].values).mean()
        self.global_success = share[ok].mean()
        # profil belajar (clustering perilaku)
        Z = np.column_stack([share.values, X_tr["total_clicks"].values / 10.0])
        self.km = KMeans(n_clusters=self.k, n_init=10, random_state=RANDOM_STATE).fit(Z)
        cent = pd.DataFrame(self.km.cluster_centers_[:, :-1], columns=share.columns)
        lift = cent / (share.mean() + 1e-9)
        names = {}
        for i in range(self.k):
            top = lift.iloc[i].sort_values(ascending=False).index[:2]
            names[i] = "Dominan " + " & ".join(ACTIVITY_DESC.get(t, t) for t in top)
        self.profile_names = names
        lab = self.km.labels_
        self.profile_summary = pd.DataFrame({
            "profil": [names[i] for i in range(self.k)],
            "jumlah_mahasiswa": [int((lab == i).sum()) for i in range(self.k)],
            "rata2_log_klik": [float(X_tr["total_clicks"].values[lab == i].mean()) for i in range(self.k)],
            "proporsi_berisiko": [float(meta_tr["at_risk"].values[lab == i].mean()) for i in range(self.k)],
        })
        return self

    def risk_level(self, prob):
        return np.select([prob >= self.high, prob >= self.low], ["Tinggi", "Sedang"], "Rendah")

    def recommend(self, X, meta, top_n=3):
        prob = self.model.predict_proba(X)[:, 1]
        level = self.risk_level(prob)
        share = self._activity_share(X)
        Z = np.column_stack([share.values, X["total_clicks"].values / 10.0])
        prof = self.km.predict(Z)
        rows = []
        for i in range(len(X)):
            mod = meta["code_module"].iat[i]
            target = (self.success_profile.loc[mod] if mod in self.success_profile.index
                      else self.global_success)
            s = share.iloc[i]
            gap = (target - s)[target >= 0.03].sort_values(ascending=False)
            gap = gap[gap > 0.02].head(top_n)
            alignment = 1 - 0.5 * float(np.abs(target - s).sum())  # 1 = identik
            acts = [f"{ACTIVITY_DESC.get(a, a)} ({a})" for a in gap.index]
            notes = []
            xr = X.iloc[i]
            if xr["rel_engagement"] < 0.5:
                notes.append(f"aktivitas hanya {xr['rel_engagement']:.0%} dari median kelas, "
                             "tingkatkan frekuensi login")
            if xr["n_missed"] > 0:
                notes.append(f"{int(xr['n_missed'])} tugas belum dikumpulkan, prioritaskan")
            if xr["has_score"] and xr["mean_score"] < 40:
                notes.append("nilai rata-rata < 40, ikuti materi remedial")
            if level[i] == "Tinggi":
                path = "Remedial: materi dipecah per sub-topik, kesulitan diturunkan, intervensi tutor"
            elif level[i] == "Sedang":
                path = "Terbimbing: kuis formatif mingguan + pengingat otomatis"
            else:
                path = "Pengayaan: materi lanjutan & tantangan, belajar mandiri"
                if xr["has_score"] and xr["mean_score"] >= 70:
                    notes.append("performa baik, buka materi pengayaan")
            rows.append(dict(prob_risiko=round(float(prob[i]), 3), level_risiko=level[i],
                             profil_belajar=self.profile_names[prof[i]], jalur_belajar=path,
                             rekomendasi_aktivitas="; ".join(acts) if acts else "pertahankan pola belajar",
                             catatan="; ".join(notes), skor_kesesuaian_pola=round(alignment, 3)))
        out = pd.concat([meta.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
        return out


# =====================================================================
# 5. VISUALISASI
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


def plot_metric_by_week(dfw, metric, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c, (name, g) in zip(SERIES_COLORS, dfw.groupby("model", sort=False)):
        ax.plot(g["minggu"], g[metric], color=c, lw=2, marker="o", ms=6,
                markeredgecolor="white", markeredgewidth=1.5, label=name)
        ax.annotate(name, (g["minggu"].iloc[-1], g[metric].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=TEXT_PRIMARY)
    _style(ax, f"{metric.upper()} per minggu (data uji)", "Minggu ke-", metric)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xticks(sorted(dfw["minggu"].unique()))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, path, title):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    labels = ["Tidak berisiko", "Berisiko"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else TEXT_PRIMARY, fontsize=11)
    ax.set_xlabel("Prediksi", color=TEXT_MUTED)
    ax.set_ylabel("Aktual", color=TEXT_MUTED)
    ax.set_title(title, loc="left", fontsize=11, color=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(imp, path, top=15):
    imp = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index, imp.values, color=SERIES_COLORS[0], height=0.6)
    _style(ax, f"{top} fitur paling berpengaruh (permutation importance)",
           "Penurunan ROC-AUC bila fitur diacak")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =====================================================================
# 6. PIPELINE PENGUJIAN
# =====================================================================
def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    data = load_oulad(args.data_dir)
    weeks = sorted(args.weeks)
    main_week = args.main_week if args.main_week in weeks else weeks[len(weeks) // 2]

    # --- (a) Pengujian semua model di setiap minggu ---
    print(f"[2] Pengujian {len(get_models())} model pada minggu {weeks} "
          f"(split={args.split})")
    records, risk_track, cache = [], [], {}
    for w in weeks:
        X, meta = build_features(data, w)
        tr, te = split_data(meta, args.split, args.test_presentation)
        cache[w] = (X, meta, tr, te)
        for name, model in get_models().items():
            model.fit(X.iloc[tr], meta["at_risk"].iloc[tr])
            prob = model.predict_proba(X.iloc[te])[:, 1]
            m = evaluate(meta["at_risk"].iloc[te].values, prob)
            records.append(dict(minggu=w, model=name, n_train=len(tr), n_test=len(te), **m))
            print(f"    minggu {w:>2} | {name:<18} | F1={m['f1']:.3f} "
                  f"AUC={m['roc_auc']:.3f} Recall={m['recall']:.3f}")
    dfw = pd.DataFrame(records)
    dfw.to_csv(os.path.join(args.out_dir, "metrik_per_minggu.csv"), index=False)
    plot_metric_by_week(dfw, "f1", os.path.join(args.out_dir, "f1_per_minggu.png"))
    plot_metric_by_week(dfw, "roc_auc", os.path.join(args.out_dir, "auc_per_minggu.png"))
    main_tbl = dfw[dfw["minggu"] == main_week].sort_values("f1", ascending=False)
    main_tbl.to_csv(os.path.join(args.out_dir, f"perbandingan_model_minggu{main_week}.csv"),
                    index=False)

    # --- (b) Mesin adaptif pada minggu utama ---
    print(f"[3] Mesin adaptif (model={args.adaptive_model}, minggu={main_week})")
    X, meta, tr, te = cache[main_week]
    engine = AdaptiveLMSEngine(get_models()[args.adaptive_model]).fit(X.iloc[tr], meta.iloc[tr])
    rec = engine.recommend(X.iloc[te], meta.iloc[te])
    rec.to_csv(os.path.join(args.out_dir, "rekomendasi_mahasiswa.csv"), index=False)
    engine.profile_summary.to_csv(os.path.join(args.out_dir, "profil_belajar.csv"), index=False)

    pred = (rec["prob_risiko"] >= 0.5).astype(int)
    cm = confusion_matrix(rec["at_risk"], pred)
    plot_confusion(cm, os.path.join(args.out_dir, "confusion_matrix.png"),
                   f"{args.adaptive_model}, minggu {main_week}")

    # validasi level risiko: proporsi aktual berisiko per level
    lvl = (rec.groupby("level_risiko")["at_risk"].agg(["count", "mean"])
              .rename(columns={"count": "jumlah", "mean": "proporsi_aktual_berisiko"})
              .reindex(["Rendah", "Sedang", "Tinggi"]))
    lvl.to_csv(os.path.join(args.out_dir, "validasi_level_risiko.csv"))
    # validasi offline rekomendasi: makin mirip pola mahasiswa sukses,
    # makin kecil proporsi berisiko? (dasar logis rekomendasi)
    rec["kuartil_kesesuaian"] = pd.qcut(rec["skor_kesesuaian_pola"].rank(method="first"), 4,
                                        labels=["Q1 (paling beda)", "Q2", "Q3", "Q4 (paling mirip)"])
    align = rec.groupby("kuartil_kesesuaian", observed=True)["at_risk"].mean().rename(
        "proporsi_aktual_berisiko")
    align.to_csv(os.path.join(args.out_dir, "validasi_rekomendasi.csv"))

    # fitur penting
    sub = np.random.RandomState(RANDOM_STATE).choice(te, size=min(3000, len(te)), replace=False)
    pi = permutation_importance(engine.model, X.iloc[sub], meta["at_risk"].iloc[sub],
                                scoring="roc_auc", n_repeats=5, random_state=RANDOM_STATE,
                                n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
    imp.to_csv(os.path.join(args.out_dir, "feature_importance.csv"), header=["importance"])
    plot_importance(imp, os.path.join(args.out_dir, "feature_importance.png"))

    # --- (c) Adaptasi dari minggu ke minggu (mahasiswa uji yang sama) ---
    print("[4] Melacak perubahan level risiko tiap minggu")
    for w in weeks:
        Xw, mw, trw, tew = cache[w]
        mdl = get_models()[args.adaptive_model].fit(Xw.iloc[trw], mw["at_risk"].iloc[trw])
        p = mdl.predict_proba(Xw.iloc[tew])[:, 1]
        t = mw.iloc[tew][KEY].copy()
        t[f"risiko_m{w}"] = engine.risk_level(p)
        risk_track.append(t.set_index(KEY))
    track = pd.concat(risk_track, axis=1)  # NaN = sudah mengundurkan diri
    track = track.join(meta.set_index(KEY)[["final_result"]], how="left")
    track.reset_index().to_csv(os.path.join(args.out_dir, "perkembangan_risiko.csv"), index=False)

    # --- Ringkasan ---
    print("\n================ RINGKASAN PENGUJIAN ================")
    print(f"Perbandingan model pada minggu {main_week}:")
    print(main_tbl[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]]
          .round(3).to_string(index=False))
    print("\nValidasi level risiko (semakin tinggi level -> semakin banyak yang benar berisiko):")
    print(lvl.round(3).to_string())
    print("\nValidasi rekomendasi (kesesuaian pola aktivitas vs pola mahasiswa lulus):")
    print(align.round(3).to_string())
    print("\nProfil belajar (K-Means):")
    print(engine.profile_summary.round(3).to_string(index=False))
    print("\nContoh rekomendasi adaptif:")
    cols = ["id_student", "code_module", "level_risiko", "jalur_belajar",
            "rekomendasi_aktivitas", "catatan"]
    ex = rec.sort_values("prob_risiko", ascending=False).groupby("level_risiko").head(1)
    for _, r in ex[cols].iterrows():
        print(f" - {r['id_student']} [{r['code_module']}] risiko {r['level_risiko']}\n"
              f"     jalur : {r['jalur_belajar']}\n"
              f"     materi: {r['rekomendasi_aktivitas']}\n"
              f"     catatan: {r['catatan'] or '-'}")
    print(f"\nSemua hasil tersimpan di: {os.path.abspath(args.out_dir)}")


# =====================================================================
# 7. DATA DEMO (sintetis, struktur identik dengan OULAD)
# =====================================================================
def make_demo_oulad(out_dir, n_per_pres=600, seed=7):
    """Hanya untuk mencoba kode tanpa mengunduh OULAD. JANGAN dipakai
    sebagai hasil penelitian."""
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    modules, presentations, length = ["AAA", "BBB", "CCC"], ["2013B", "2013J", "2014B", "2014J"], 240
    types = ["oucontent", "resource", "subpage", "homepage", "url", "forumng", "quiz", "ouwiki",
             "oucollaborate", "glossary"]
    vle, assess, info, reg, sv, sa = [], [], [], [], [], []
    site_id, ass_id, stu_id = 1, 1, 100000
    for m in modules:
        for pr in presentations:
            sites = {}
            for t in types:
                k = rng.integers(2, 8)
                sites[t] = list(range(site_id, site_id + k))
                vle += [dict(id_site=s, code_module=m, code_presentation=pr, activity_type=t,
                             week_from=np.nan, week_to=np.nan) for s in sites[t]]
                site_id += k
            ass_list = []
            for d, typ in zip([25, 55, 90, 130, 180, 235], ["CMA", "TMA", "TMA", "CMA", "TMA", "Exam"]):
                assess.append(dict(code_module=m, code_presentation=pr, id_assessment=ass_id,
                                   assessment_type=typ, date=d, weight=100 if typ == "Exam" else 20))
                ass_list.append((ass_id, d, typ))
                ass_id += 1
            for _ in range(n_per_pres):
                stu_id += 1
                eng = rng.beta(2, 3)              # motivasi / keterlibatan
                abil = rng.normal(0, 1)           # kemampuan
                pref = rng.dirichlet(np.ones(len(types)) * 0.8)
                good_pref = np.array([.30, .15, .10, .08, .03, .12, .14, .03, .03, .02])
                mix = 0.5 * pref + 0.5 * good_pref * (0.5 + eng)
                mix /= mix.sum()
                latent = 1.6 * eng + 0.6 * abil + 0.4 * mix[[0, 5, 6]].sum() + rng.normal(0, .35)
                if latent < 0.15:
                    res = "Withdrawn"
                elif latent < 0.55:
                    res = "Fail"
                elif latent < 1.6:
                    res = "Pass"
                else:
                    res = "Distinction"
                unreg = int(rng.integers(-5, 200)) if res == "Withdrawn" else np.nan
                end = unreg if res == "Withdrawn" else length
                info.append(dict(code_module=m, code_presentation=pr, id_student=stu_id,
                                 gender=rng.choice(["M", "F"]), region="Region",
                                 highest_education=rng.choice(["A Level or Equivalent",
                                                               "Lower Than A Level", "HE Qualification"]),
                                 imd_band=rng.choice(["0-10%", "10-20", "20-30%", "30-40%", "?"]),
                                 age_band=rng.choice(["0-35", "35-55", "55<="], p=[.7, .28, .02]),
                                 num_of_prev_attempts=int(rng.random() < .12),
                                 studied_credits=int(rng.choice([30, 60, 90, 120])),
                                 disability=rng.choice(["N", "Y"], p=[.9, .1]), final_result=res))
                reg.append(dict(code_module=m, code_presentation=pr, id_student=stu_id,
                                date_registration=int(rng.integers(-100, 0)),
                                date_unregistration=unreg))
                days = np.arange(-10, max(end, -9))
                if len(days):
                    decay = 1.0 if res != "Fail" else np.linspace(1, .4, len(days))
                    act_days = days[rng.random(len(days)) < (0.08 + 0.45 * eng) * decay]
                    for d in act_days:
                        for t in rng.choice(types, size=rng.integers(1, 4), p=mix):
                            sv.append((m, pr, stu_id, int(rng.choice(sites[t])), int(d),
                                       int(rng.integers(1, 6 + int(10 * eng)))))
                for aid, d, typ in ass_list:
                    if typ == "Exam" or (res == "Withdrawn" and d > unreg):
                        continue
                    if rng.random() < 0.55 + 0.4 * eng:
                        sc = float(np.clip(60 + 15 * abil + 15 * eng + rng.normal(0, 10), 0, 100))
                        sa.append(dict(id_assessment=aid, id_student=stu_id,
                                       date_submitted=d + int(rng.integers(-5, 4)), is_banked=0,
                                       score=round(sc)))
    pd.DataFrame(vle).to_csv(os.path.join(out_dir, "vle.csv"), index=False)
    pd.DataFrame(assess).to_csv(os.path.join(out_dir, "assessments.csv"), index=False)
    pd.DataFrame(info).to_csv(os.path.join(out_dir, "studentInfo.csv"), index=False)
    pd.DataFrame(reg).to_csv(os.path.join(out_dir, "studentRegistration.csv"), index=False)
    pd.DataFrame(sa).to_csv(os.path.join(out_dir, "studentAssessment.csv"), index=False)
    pd.DataFrame(sv, columns=["code_module", "code_presentation", "id_student", "id_site",
                              "date", "sum_click"]).to_csv(os.path.join(out_dir, "studentVle.csv"),
                                                           index=False)
    print(f"[0] Data demo dibuat di {out_dir} ({len(info):,} mahasiswa, {len(sv):,} log)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pengujian model LMS adaptif (OULAD)")
    ap.add_argument("--data_dir", default="./data/OULAD", help="folder berisi CSV OULAD")
    ap.add_argument("--out_dir", default="./hasil/pedagogis")
    ap.add_argument("--weeks", type=int, nargs="+", default=[2, 4, 6, 8, 12, 16, 20],
                    help="minggu-minggu pengujian early prediction")
    ap.add_argument("--main_week", type=int, default=8,
                    help="minggu untuk rekomendasi adaptif utama")
    ap.add_argument("--split", choices=["temporal", "random"], default="temporal")
    ap.add_argument("--test_presentation", default="2014J",
                    help="semester yang dijadikan data uji (split temporal)")
    ap.add_argument("--adaptive_model", default="GradientBoosting",
                    choices=list(get_models().keys()))
    ap.add_argument("--demo", action="store_true", help="pakai data sintetis untuk uji coba")
    args = ap.parse_args()
    if args.demo:
        args.data_dir = os.path.join(args.out_dir, "data_demo")
        make_demo_oulad(args.data_dir)
    run(args)
