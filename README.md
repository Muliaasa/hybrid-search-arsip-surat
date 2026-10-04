# Pencarian Hibrida TF-IDF/BM25 dan IndoSBERT dengan Reciprocal Rank Fusion pada Arsip Surat Berbahasa Indonesia

Kode, data sintetis, dan penilaian relevansi untuk artikel studi kelayakan pencarian hibrida (TF-IDF/BM25 + IndoSBERT + RRF) dan eksplorasi topik (tahapan BERTopic: UMAP, HDBSCAN, c-TF-IDF) pada arsip surat korporasi jalan tol.

> Seluruh dokumen bersifat **sintetis**. Dua surat asli perusahaan hanya dipakai sebagai acuan format dan **tidak dipublikasikan**. Nomor dan tanggal surat bersifat fiktif.

## Isi repositori

| Berkas | Keterangan |
|---|---|
| `evaluasi_ir.py` | Seluruh pipeline: baseline TF-IDF, BM25, IndoSBERT, RRF, metrik, uji Wilcoxon, analisis klaster |
| `dataset_bersih.csv` | 150 surat sintetis (`id`, `nomor`, `perihal`, `ocr`; kolom `ocr` berisi ringkasan isi simulasi, bukan keluaran OCR) |
| `kueri_relevansi.csv` | 30 kueri uji. Kolom `id_relevan_kandidat` hanya daftar awal untuk membantu menyusun pool, **bukan** kunci jawaban |
| `anotasi.csv` | Kunci jawaban: penilaian relevansi biner (kolom `a1`) oleh satu penilai atas 515 pasangan kueri-dokumen hasil pooling. Pemisah kolom: titik koma. Kolom `a2` dan `final` kosong |

## Cara menjalankan

Diuji pada Google Colab (Python 3.13) dengan sentence-transformers 5.7.0, umap-learn 0.5.12, hdbscan 0.8.44, scikit-learn 1.6.1, rank-bm25 0.2.2, dan scipy 1.16.3. Model embedding `firqaaa/indo-sentence-bert-base` diunduh otomatis dari Hugging Face.

```bash
pip install sentence-transformers rank_bm25 umap-learn hdbscan scipy scikit-learn pandas Sastrawi

# analisis klaster (konfigurasi utama, sensitivitas, 10 seed, gambar UMAP)
python evaluasi_ir.py --semantic --only_clusters

# ablasi tanpa 21 dokumen bertema sistem
python evaluasi_ir.py --semantic --exclude_system --only_clusters --outdir hasil_tanpa_sistem

# evaluasi pencarian (butuh anotasi.csv)
python evaluasi_ir.py --semantic

# evaluasi pencarian tanpa 21 dokumen bertema sistem
python evaluasi_ir.py --semantic --exclude_system --outdir hasil_tanpa_sistem

# studi kasus satu kueri
python evaluasi_ir.py --semantic --show "perbaikan aspal jalan" --show_ids 139,116
```

Keluaran ditulis ke folder `hasil/` (atau folder pada `--outdir`). `random_state` UMAP bernilai 42 untuk konfigurasi utama.

## Catatan penting

- Hasil bersifat bukti kelayakan awal pada data sintetis dan tidak dapat digeneralisasi ke arsip nyata.
- Relevansi dinilai oleh satu penilai (penulis); kesepakatan antarpenilai tidak diukur.
- Pasangan kueri-dokumen di luar pool dianggap tidak relevan.

## Lisensi

MIT.
