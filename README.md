# TA-18121010 — Pengujian Model LMS yang Adaptif

Kode tugas akhir **"Pengujian Model LMS yang Adaptif Menggunakan Dataset Terkait Penggunaan Aplikasi LMS"**.

Model ini memprediksi beban LMS dari pola penggunaan mahasiswa (dataset OULAD) dengan *gradient boosting* berbasis regresi kuantil, lalu menyesuaikan alokasi **vCPU, RAM, storage, dan GPU** secara otomatis untuk menekan biaya tanpa membuat server *overload*. Perilaku vCPU dan RAM dikalibrasi dari jejak server riil Bitbrains GWA-T-12.

Kebijakan yang dibandingkan: **static** (kapasitas tetap dari beban puncak), **reactive** (*autoscaling* berdasarkan beban saat ini), **predictive** (model usulan), dan **oracle** (batas ideal).

## Struktur repositori

```
├── src/
│   ├── adaptive_lms_resource.py    # model adaptif sumber daya (vCPU, RAM, storage, GPU)
│   ├── server_trace_experiment.py  # eksperimen pada jejak server riil + kalibrasi
│   └── adaptive_lms_model.py       # pendukung: prediksi risiko mahasiswa (beban kerja ML yang ikut dijadwalkan)
├── data/README.md                  # cara mengunduh dan menata dataset
├── docs/URUTAN_PENGERJAAN.md       # checklist langkah pengerjaan TA
├── docs/KEPUTUSAN_BIMBINGAN.md     # catatan keputusan dengan dosen pembimbing
├── hasil/                          # keluaran eksperimen (tidak di-commit)
└── requirements.txt
```

## Cara menjalankan

Semua perintah dijalankan dari folder utama repositori.

```bash
pip install -r requirements.txt
```

### Uji coba tanpa dataset (data sintetis, bukan untuk hasil skripsi)

```bash
python src/server_trace_experiment.py --demo bitbrains --out_dir hasil/trace_demo
python src/adaptive_lms_resource.py --demo --scale_users 40 --calib hasil/trace_demo/kalibrasi_dari_trace.json --out_dir hasil/resource_demo
```

### Dengan data asli

Letakkan dataset sesuai [`data/README.md`](data/README.md), lalu:

```bash
# Eksperimen A: prediksi dan autoscaling langsung pada vCPU/RAM server riil, menghasilkan file kalibrasi
python src/server_trace_experiment.py --trace bitbrains --trace_dir data/bitbrains/fastStorage --out_dir hasil/trace

# Eksperimen B: trafik LMS (OULAD) dengan parameter server dari jejak riil
python src/adaptive_lms_resource.py --data_dir data/OULAD --calib hasil/trace/kalibrasi_dari_trace.json --out_dir hasil/resource_kalibrasi

```

Parameter `ResourceConfig` bisa diubah dari baris perintah, misalnya `--rps_per_vcpu 12 --target_cpu_util 0.6 --node_price_hour 0.1235`. Daftar lengkapnya: `python src/adaptive_lms_resource.py --help`.

## Alur model

```
Log klik OULAD (harian) ──► trafik per jam (pola diurnal) ──► request/detik dan pengguna online
        ▼
Prediktor beban: gradient boosting kuantil (p50 dan p90), fitur jam/hari/tenggat/lag
        ▼
Alokasi: vCPU dan RAM → jumlah node (dibatasi min/max node dan laju pelepasan node)
         Storage: SSD untuk data panas + object storage untuk log lama
         GPU: disewa hanya jika lebih murah dari CPU, dijadwalkan di jam trafik terendah
        ▼
Pengujian: static vs reactive vs predictive vs oracle → biaya, utilisasi, jam overload
```

## File hasil utama

| File | Isi | Dipakai untuk |
|---|---|---|
| `akurasi_prediksi_beban.csv` | MAE, RMSE, MAPE, cakupan p90, dibandingkan baseline | O1, H1 |
| `perbandingan_kebijakan_compute.csv` | biaya, utilisasi CPU, jam overload per kebijakan | O2, O6, H2, H3 |
| `alokasi_per_jam.csv` | kebutuhan vs kapasitas vCPU dan RAM per jam | O2, O3 |
| `perbandingan_storage.csv` | biaya storage static vs tiering | O4 |
| `keputusan_gpu.csv` | CPU vs GPU per job ML dan jadwalnya | O5 |
| `rencana_24jam.csv` | estimasi kebutuhan 24 jam ke depan | pembahasan |
| `parameter_dipakai.csv` | semua parameter yang dipakai | lampiran skripsi |
| `kalibrasi_dari_trace.json` | parameter dari jejak server riil | BAB III/IV |

## Asumsi harga (dicek 24 November 2026, on-demand, tanpa pajak dan diskon)

| Parameter | Nilai | Sumber |
|---|---|---|
| `node_price_hour` | 0,1235 USD/jam | EC2 m6i.large (2 vCPU, 8 GiB), Asia Pacific (Jakarta) |
| `ssd_price_gb_month` | 0,096 USD/GB-bulan | EBS gp3, Jakarta, AWS Pricing Calculator |
| `object_price_gb_month` | 0,025 USD/GB-bulan | S3 Standard, Jakarta, AWS Pricing Calculator |
| `gpu_price_hour` | 0,736 USD/jam | EC2 g4dn.xlarge (1× T4), Singapura; job pelatihan batch |

Yang masih asumsi: `rps_per_vcpu` dan `req_per_click` (konversi request LMS ke beban CPU). Nilainya akan diukur lewat uji beban Moodle dengan Locust (tahap 3B di `docs/URUTAN_PENGERJAAN.md`).

## Sitasi dataset

- Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data, 4*, 170171. https://doi.org/10.1038/sdata.2017.171
- Shen, S., van Beek, V., & Iosup, A. (2015). Statistical characterization of business-critical workloads hosted in cloud datacenters. *CCGrid 2015* (Bitbrains GWA-T-12).

## Urutan pengerjaan

Lihat [`docs/URUTAN_PENGERJAAN.md`](docs/URUTAN_PENGERJAAN.md).
