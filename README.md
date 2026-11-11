# Aplikasi identifikasi penulis tulisan tangan

Aplikasi web Flask yang menebak penulis dari satu baris tulisan tangan, memakai
ResNet18 hasil UAS (50 penulis, akurasi uji 0,9808).

Kamera dipegang **browser** lewat `getUserMedia`, bukan server. Server hanya
menerima satu frame JPEG per permintaan. Preview video jalan sepenuhnya di sisi
klien, jadi tidak ada beban inferensi per frame.

## Pasang

```
pip install -r requirements.txt
```

Lalu salin bobot model dari Drive (`MyDrive/UAS_ML/hasil/`) ke folder `model/`:

```
model/resnet18_fold1.pt
model/resnet18_fold2.pt      (opsional)
model/resnet18_fold3.pt      (opsional)
model/resnet18_fold4.pt      (opsional)
model/resnet18_fold5.pt      (opsional)
```

Satu berkas sudah cukup. Kalau kelima ada, aplikasi memakai ensemble: softmax
dirata-ratakan dan jumlah model yang sepakat ditampilkan sebagai sinyal
keyakinan. Tanpa bobot, halaman tetap terbuka tetapi menampilkan peringatan.

Bobot harus dari notebook yang **baru** (50 kelas, input 128 x 1024).
`model_ResNet18.pt` dari arsip lama tidak kompatibel: 49 kelas, input 384 x 64.

## Jalankan

```
python app.py
```

Buka `http://localhost:5000`.

### Kamera dari ponsel

`getUserMedia` hanya jalan di secure context. `http://192.168.x.x:5000` dari
ponsel **tidak** akan membuka kamera. Pilih salah satu:

```
flask --app app run --host 0.0.0.0 --cert=adhoc
```

atau tunnel (`cloudflared tunnel --url http://localhost:5000`). Tab Unggah tetap
berfungsi di HTTP biasa.

### Produksi

```
gunicorn -w 1 --threads 4 -b 0.0.0.0:5000 app:app
```

Satu worker saja — tiap worker menduplikasi model di memori. Thread aman karena
operasi PyTorch melepas GIL. `torch.set_num_threads(2)` sudah diset di
`model_service.py`.

## Isi folder

| Berkas | Keterangan |
|---|---|
| `app.py` | Route Flask saja |
| `model_service.py` | Muat model, preprocessing, prediksi, Grad-CAM |
| `quality.py` | Gerbang mutu: ukuran, kontras, ketajaman, rasio tinta |
| `segment.py` | Potong baris dengan projection profile |
| `kalibrasi_ambang.py` | Menghitung ambang "tidak dikenal" dari test set |
| `meta.json` | 50 nama kelas urut `label_id`, ukuran input, normalisasi, ambang |
| `contoh/` | 3 crop per penulis untuk panel pembanding (820 KB) |
| `templates/`, `static/` | Halaman, CSS, JS kamera |
| `Klasifikasi_Tulisan_Tangan.ipynb` | Notebook UAS yang menghasilkan bobot model (berisi output) |

Notebook disertakan sebagai asal-usul model: di situlah split data, ekstraksi
GLCM, dan pelatihan ResNet18 serta SqueezeNet dilakukan. Datasetnya sendiri
(81 MB) tidak masuk repo.

## Endpoint

| Route | Method | Keterangan |
|---|---|---|
| `/` | GET | Halaman utama |
| `/api/predict` | POST | Gambar -> top-5, keyakinan, flag `tidak_dikenal` |
| `/api/verify` | POST | Gambar + `penulis` -> skor 1:1 |
| `/api/gradcam` | POST | Gambar (+ `label_id`) -> PNG overlay |
| `/api/writers` | GET | Daftar 50 penulis |
| `/api/health` | GET | Model termuat, device, ambang |
| `/contoh/<penulis>/<berkas>` | GET | Contoh tulisan penulis |

Gambar dikirim sebagai multipart `image`, atau JSON `{"image": "data:image/jpeg;base64,..."}`.
Batas unggahan 10 MB.

Grad-CAM dipisah dari `/api/predict` karena butuh backward pass. Kalau digabung,
setiap prediksi jadi 2-3 kali lebih lambat padahal heatmap jarang dibuka.

## Pipeline di server

1. Decode + `exif_transpose` (foto ponsel sering tersimpan berotasi)
2. Gerbang mutu — tolak dengan pesan spesifik, bukan "gagal"
3. Potong baris kalau terdeteksi lebih dari satu band tulisan
4. Resize tinggi 128 px (rasio dijaga), pad putih sampai 1024 px
5. 3 channel, normalisasi ImageNet
6. Softmax, dirata-ratakan antar fold
7. Di bawah `ambang_keyakinan` -> "tidak dikenal"

Langkah 4 dan 5 **harus** sama dengan sel 5.2 notebook. Kalau berbeda, akurasi
turun tanpa satu pun pesan error.

## Kalibrasi ambang

Nilai `ambang_keyakinan` di `meta.json` masih 0,55 — angka sementara, bukan hasil
pengukuran. Setelah bobot tersedia:

```
python kalibrasi_ambang.py --dataset "D:/dataset_clean" --split "D:/hasil/split_data.csv" --tulis
```

Skrip mengukur sebaran keyakinan pada 532 gambar uji, mengambil persentil bawah
dari prediksi yang benar, lalu melaporkan berapa prediksi salah yang tersaring.

## Batas yang perlu disebut di laporan

- **Closed set.** Model hanya tahu 50 penulis ini. Tulisan orang lain akan
  dipaksa masuk ke salah satunya — itulah sebabnya ambang penolakan ada, dan
  sebabnya ambang itu harus dikalibrasi, bukan dikira-kira.
- Model dilatih dari crop satu baris pada satu sesi penulisan. Foto dengan
  kertas, pena, atau pencahayaan berbeda belum teruji.
- Gerbang mutu menangkap buram, gelap, dan kosong, tetapi **meloloskan derau**
  yang tajam dan berkontras. Penyaring akhir untuk kasus seperti itu adalah
  ambang keyakinan.
- Aplikasi tidak membaca isi tulisan. Itu tugas OCR, model lain.
