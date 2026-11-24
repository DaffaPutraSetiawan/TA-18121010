# Urutan Pengerjaan Tugas Akhir

Judul: **Pengujian Model LMS yang Adaptif Menggunakan Dataset Terkait Penggunaan Aplikasi LMS**

Fokus: adaptif terhadap sumber daya (vCPU, RAM, storage, GPU). Simulasi dengan data OULAD dan jejak server Bitbrains, lalu divalidasi pada server LMS nyata (Moodle di AWS Jakarta). Keputusan bimbingan: [`KEPUTUSAN_BIMBINGAN.md`](KEPUTUSAN_BIMBINGAN.md).

Centang setiap langkah yang sudah selesai.

## Tahap 1 — Persiapan (minggu 1)

- [ ] Konsultasi dosen pembimbing dengan membawa draf BAB I, BAB III (DFD dan flowchart), dan kerangka BAB IV
- [x] Sepakati: simulasi + uji server nyata; jejak server Bitbrains; asumsi harga AWS Jakarta on-demand; constraint sesuai Tabel 1.4
- [ ] Lengkapi BAB I: tujuan, manfaat, dan batasan masalah
- [ ] Pasang Python 3.10+ dan jalankan `pip install -r requirements.txt`
- [ ] Coba jalankan mode demo (lihat README) untuk memastikan kode berjalan di laptop atau Colab

## Tahap 2 — Data asli (minggu 1–2)

- [ ] Unduh OULAD ke `data/OULAD/`
- [ ] Unduh Bitbrains GWA-T-12 `fastStorage` ke `data/bitbrains/fastStorage/`
- [ ] Catat sumber, versi, dan tanggal unduh untuk BAB III

## Tahap 3A — Eksperimen simulasi (minggu 2–3)

- [ ] Eksperimen A: `server_trace_experiment.py` pada Bitbrains → `hasil/trace/kalibrasi_dari_trace.json`
- [ ] Eksperimen B: `adaptive_lms_resource.py` dengan OULAD + `--calib` → `hasil/resource_kalibrasi/`
- [ ] Periksa kewajaran: nilai kalibrasi, jumlah node tidak selalu mentok batas, MAE model < baseline
- [ ] Analisis sensitivitas: ulangi dengan beberapa nilai `rps_per_vcpu`, `target_cpu_util`, dan harga node
- [ ] Uji statistik hipotesis: Diebold–Mariano/Wilcoxon (H1), Wilcoxon (H2), McNemar (H3)

## Tahap 3B — Validasi di server nyata (minggu 3–5)

- [ ] Siapkan akun AWS dan anggaran; cek apakah ada kredit cloud gratis untuk mahasiswa
- [ ] Pasang Moodle di 1 VM m6i.large (Jakarta) dengan isi kursus contoh
- [ ] Uji beban dengan Locust: ukur berapa request/detik yang sanggup dilayani 1 vCPU → ganti `rps_per_vcpu` di simulasi
- [ ] Pisahkan kode prediktor dan eksekutor (eksekutor simulasi ↔ eksekutor AWS Auto Scaling)
- [ ] Jalankan Moodle di Auto Scaling Group, kirim trafik Locust yang meniru pola OULAD (diperkecil, beberapa jam)
- [ ] Bandingkan reactive vs predictive: CPU, RAM, waktu respons p95, error, dan tagihan nyata
- [ ] Bandingkan hasil server nyata dengan hasil simulasi pada skenario yang sama
- [ ] Matikan semua resource AWS setelah selesai supaya tidak terus ditagih

## Tahap 4 — Penulisan (minggu 3–5)

- [ ] BAB II: landasan teori dan tabel penelitian terdahulu
- [ ] BAB III: metodologi (dataset, tahapan, skenario uji, metrik) + perancangan sistem yang sudah ada
- [ ] BAB IV: ganti semua grafik dan Tabel 4.1 dengan hasil data asli, hapus label "data demo"
- [ ] BAB IV: tambah subbab validasi server nyata
- [ ] BAB V: jawab hipotesis satu per satu, kesimpulan, dan saran
- [ ] Abstrak (Indonesia dan Inggris)
- [ ] Cek ulang semua referensi (lihat "Catatan untuk Penulis" di dokumen)

## Tahap 5 — Revisi dan sidang (minggu 6+)

- [ ] Revisi sesuai masukan dosen
- [ ] Siapkan slide sidang
- [ ] Latihan menjawab pertanyaan penguji: alasan regresi kuantil, asal nilai constraint, kelemahan simulasi, dan kenapa `rps_per_vcpu` diasumsikan
