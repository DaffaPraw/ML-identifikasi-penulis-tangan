# Rencana halaman

**1. Jenis:** software (page type `software`, site-kind `software`): sebuah alat AI yang dipakai
sendiri. Yang "dijual": identifikasi penulis tulisan tangan. Pembelinya: pemeriksa yang punya satu
baris tulisan dan 50 penulis terdaftar. Satu aksi utama: **Unggah tulisan** (ke `alat.html`).

**2. Hasil deal:** hero `collage` · palet `charcoal-room-sage` (bg #1c1c1c, aksen sage #8fa68e) ·
pasangan tipe `suse` (SUSE + JetBrains Mono) · radius `soft` · nav `nav-mega` · footer `footer-dark` ·
close `close-type` · ritme `space` · tombol `arrow` · mark `wordmark` · efek `gradient-mesh` di frame
seni hero (satu-satunya efek) · moment `pinned-chapters` (#cara) dan `counter-rolls` (baris angka
terakhir #data).
Kenapa paletnya cocok: tanah hampir hitam membuat crop tulisan tangan dan peta Grad-CAM jadi
satu-satunya yang terang di halaman, dan aksen sage hanya menandai batang kandidat teratas.

**Kartu suara**
```
Suara: technical-dry (copy-registers 2)
Pasangan: tepat, bukan dingin; pasti, bukan ramai
Orang: modelnya yang jadi subjek; "kamu" tidak dipakai
Kalimat: 8 sampai 16 kata; satu kalimat panjang per bagian
Humor: tidak ada
Kata rumah: penulis, crop, keyakinan, ambang, gerbang mutu, fold
Jangan: kata pujian tentang diri sendiri, angka yang tidak diukur
```

**Lembar fakta** (semua dari notebook UAS dan pengukuran lokal, tidak ada yang dikarang)
- 50 penulis terdaftar, 2.658 crop tulisan tangan.
- Pembagian: 20% uji tetap (532 gambar) + 5-fold StratifiedKFold, seed 42.
- Empat model pada test set, rata-rata 5 eksperimen: LDA + GLCM akurasi 0,8944 / F1 0,8930 / 0,07 s;
  Gaussian NB + GLCM 0,3338 / 0,2983 / 0,004 s; ResNet18 0,9808 / 0,9807 / 316 s;
  SqueezeNet 1.1 0,9398 / 0,9391 / 249 s.
- Bobot yang dipakai aplikasi (fold 1, dilatih ulang di CPU 32 core, 15 menit, epoch terbaik 10):
  presisi 0,9863 / recall 0,9848 / akurasi 0,9850 / F1 0,9848 pada 532 gambar uji.
- Ambang keyakinan 0,4023 (persentil 1 keyakinan prediksi benar): 10 dari 532 dijawab "tidak
  dikenal", 4 dari 8 prediksi salah tersaring.
- Masukan model 128 x 1024 piksel, skala abu-abu, normalisasi ImageNet.
- Recall terendah per penulis: A250040, 0,820.
- Batas: closed set 50 penulis; tulisan orang lain dipaksa masuk salah satunya.
- Cacat data yang ditemukan: A250002/535250002_1.jpg dan _2.jpg identik byte demi byte.

**3. Layar** (9 layar, 8 bagian)

| # | id | satu gagasan | gambar (jenis) | layar |
|---|---|---|---|---|
| 1 | hero | apa yang ditebak dan seberapa yakin | jendela aplikasi + kartu hasil (objek) | 1 |
| 2 | `#baca` | daftar prediksi di samping gambarnya | app-frame, jendela penuh (objek) | 1 |
| 3 | `#cara` | tiga langkah sebelum angka keluar | crop asli, kanvas praproses, Grad-CAM (keluaran) — **moment pinned-chapters**, karena tiap langkah mengubah gambarnya | 2 |
| 4 | `#hasil` | empat model pada 532 gambar uji | tabel (tanpa gambar, tenang) | 1 |
| 5 | `#data` | dari mana 2.658 crop itu | deretan crop asli (foto) + baris angka — **moment counter-rolls** | 1 |
| 6 | `#batas` | hanya 50 penulis ini | panel hasil dengan keyakinan turun (objek, ke-3 dan terakhir) | 1 |
| 7 | `#tanya` | pertanyaan terakhir | tanpa gambar, tenang | 1 |
| 8 | `footer.closing` | unggah tulisan | tombol besar `close-type` | 1 |

Ketukan: efek di hero, `pinned-chapters` di #cara, `counter-rolls` di #data — tidak ada dua yang
bertetangga. Objek utuh (jendela aplikasi) di hero, #baca dan #batas: tepat tiga.

Foto: semua milik sendiri (crop dari `contoh/`), jadi tidak ada baris kredit pustaka.
