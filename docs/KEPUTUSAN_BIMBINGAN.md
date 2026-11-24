# Catatan Keputusan Bimbingan

Catat setiap keputusan di hari yang sama, lalu minta paraf di kartu bimbingan.

| Tanggal | Topik | Keputusan | Dampak |
|---|---|---|---|
| 24 Nov 2026 | Ruang lingkup "adaptif" | Adaptif terhadap sumber daya (infrastruktur) saja | Sisi mahasiswa (prediksi risiko) hanya pendukung, dicatat sebagai pengembangan lanjutan |
| 24 Nov 2026 | Bentuk pengujian | Simulasi berbasis dataset **dan** validasi pada server LMS nyata | Tambah tahap uji Moodle di AWS + Locust (lihat URUTAN_PENGERJAAN.md, tahap 3B) |
| 24 Nov 2026 | Jejak server riil | Bitbrains GWA-T-12 (`fastStorage`) | Alibaba tidak dipakai; Google 2019 dan Azure dibahas di BAB II/III sebagai dataset yang dipertimbangkan beserta alasan tidak dipilih |
| 24 Nov 2026 | Asumsi harga | AWS on-demand, region Jakarta (GPU: Singapura), dicek 24 Nov 2026 | Sudah menjadi nilai default di `src/adaptive_lms_resource.py` |
| 24 Nov 2026 | Nilai constraint | Sesuai Tabel 1.4 (utilisasi target 65%, batas aman 90%, overload ≤ 1%, dll.) | Tidak ada perubahan |
| 24 Nov 2026 | Batas waktu | Tidak ditetapkan | — |

## Alasan pemilihan dataset (untuk BAB III)

- **Bitbrains:** CPU (MHz) dan RAM (KB) dicatat dalam nilai absolut, baik yang disediakan maupun yang terpakai, tiap 5 menit; berisi VM aplikasi bisnis yang mirip server aplikasi web; ukuran cukup kecil untuk diolah di laptop.
- **Google 2019 (tidak dipilih):** sekitar 2,4 TiB terkompresi dan hanya tersedia lewat BigQuery; nilai CPU dan memori dinormalisasi; beban kerjanya campuran job batch dan layanan internal.
- **Azure Functions 2019 (tidak dipilih):** tidak memuat pemakaian CPU; memori hanya berupa distribusi per aplikasi; beban kerja serverless, bukan server/VM.
- **Alibaba 2018 (tidak dipilih):** memori dinormalisasi sehingga butuh asumsi RAM per mesin.
