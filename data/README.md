# Folder data

Dataset tidak disimpan di repositori karena ukurannya besar. Unduh sendiri, lalu letakkan dengan struktur berikut:

```
data/
├── OULAD/
│   ├── assessments.csv
│   ├── courses.csv
│   ├── studentAssessment.csv
│   ├── studentInfo.csv
│   ├── studentRegistration.csv
│   ├── studentVle.csv
│   └── vle.csv
└── bitbrains/
    └── fastStorage/
        ├── 1.csv
        ├── 2.csv
        └── ...
```

| Dataset | Sumber | Keterangan |
|---|---|---|
| OULAD | https://analyse.kmi.open.ac.uk/open_dataset (atau UCI ML Repository, dataset 349) | Pola penggunaan LMS (klik harian, tugas, hasil studi) |
| Bitbrains GWA-T-12 | Grid Workloads Archive, TU Delft (http://gwa.ewi.tudelft.nl) | Jejak vCPU dan RAM server riil, tiap 5 menit. Pakai set `fastStorage` |
| Alibaba cluster-trace-v2018 (opsional) | https://github.com/alibaba/clusterdata | Pembanding. Cukup `machine_meta` dan `machine_usage` |

Catat tanggal unduh dan versi dataset untuk BAB III.
