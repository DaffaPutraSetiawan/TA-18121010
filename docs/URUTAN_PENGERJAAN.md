# Urutan Pengerjaan Tugas Akhir

Judul: **Pengujian Model LMS yang Adaptif Menggunakan Dataset Terkait Penggunaan Aplikasi LMS**

Fokus: adaptif terhadap sumber daya (vCPU, RAM, storage, GPU), berbasis simulasi dengan data OULAD dan jejak server Bitbrains.

Centang setiap langkah yang sudah selesai.

## Tahap 1 — Persiapan (minggu 1)

- [ ] Konsultasi dosen pembimbing dengan membawa draf BAB I, BAB III (DFD dan flowchart), dan kerangka BAB IV
- [ ] Sepakati: cukup simulasi atau perlu uji server nyata; jejak server yang dipakai (Bitbrains/Alibaba); asumsi harga (AWS Jakarta, on-demand)
- [ ] Lengkapi BAB I: tujuan, manfaat, dan batasan masalah
- [ ] Pasang Python 3.10+ dan jalankan `pip install -r requirements.txt`
- [ ] Coba jalankan mode demo (lihat README) untuk memastikan kode berjalan di laptop atau Colab

## Tahap 2 — Data asli (minggu 1–2)

- [ ] Unduh OULAD ke `data/OULAD/`
- [ ] Unduh Bitbrains GWA-T-12 `fastStorage` ke `data/bitbrains/fastStorage/`
- [ ] (Opsional) Unduh Alibaba cluster-trace-v2018 sebagai pembanding
- [ ] Catat sumber, versi, dan tanggal unduh untuk BAB III

## Tahap 3 — Eksperimen (minggu 2–3)

- [ ] Eksperimen A: `server_trace_experiment.py` pada Bitbrains → `hasil/trace/kalibrasi_dari_trace.json`
- [ ] Eksperimen B: `adaptive_lms_resource.py` dengan OULAD + `--calib` → `hasil/resource_kalibrasi/`
- [ ] Periksa kewajaran: nilai kalibrasi, jumlah node tidak selalu mentok batas, MAE model < baseline
- [ ] (Opsional, sangat disarankan) Uji beban Moodle di 1 VM dengan Locust/JMeter untuk mengukur `rps_per_vcpu`
- [ ] Analisis sensitivitas: ulangi dengan beberapa nilai `rps_per_vcpu`, `target_cpu_util`, dan harga node
- [ ] Uji statistik hipotesis: Diebold–Mariano/Wilcoxon (H1), Wilcoxon (H2), McNemar (H3)

## Tahap 4 — Penulisan (minggu 3–5)

- [ ] BAB II: landasan teori dan tabel penelitian terdahulu
- [ ] BAB III: metodologi (dataset, tahapan, skenario uji, metrik) + perancangan sistem yang sudah ada
- [ ] BAB IV: ganti semua grafik dan Tabel 4.1 dengan hasil data asli, hapus label "data demo"
- [ ] BAB V: jawab hipotesis satu per satu, kesimpulan, dan saran (termasuk uji server nyata)
- [ ] Abstrak (Indonesia dan Inggris)
- [ ] Cek ulang semua referensi (lihat "Catatan untuk Penulis" di dokumen)

## Tahap 5 — Revisi dan sidang (minggu 6+)

- [ ] Revisi sesuai masukan dosen
- [ ] Siapkan slide sidang
- [ ] Latihan menjawab pertanyaan penguji: alasan regresi kuantil, asal nilai constraint, kelemahan simulasi, dan kenapa `rps_per_vcpu` diasumsikan
