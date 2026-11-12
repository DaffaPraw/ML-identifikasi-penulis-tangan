"""Memotong baris tulisan dengan projection profile.

Model dilatih memakai crop satu baris. Foto kamera biasanya memuat beberapa
baris, jadi band baris dicari dulu supaya input mendekati data latih.
"""
import numpy as np

from quality import ambang_otsu

MAKS_BARIS = 8

# Hanya gambar sebesar foto kamera yang dipangkas. Dataset: tinggi median 129 px
# dan hanya 3 dari 2.658 gambar melewati 600 px, jadi crop dataset tidak ikut
# tersentuh. Itu penting: memangkas crop dataset menurunkan akurasi uji dari
# 0,9850 ke 0,9398.
TINGGI_FOTO_MIN = 600
CAKUPAN_MAKS = 0.50


def kotak_tinta(a, pad=8):
    """Kotak pembatas tinta untuk foto kamera, atau None kalau tidak perlu.

    Foto kamera memuat banyak ruang kosong di sekitar tulisan. Preprocessing
    memakai skala min(128/tinggi, 1024/lebar), jadi foto potret 3000x4000
    menyusut jadi 96x128 px di kanvas 1024x128 dan tulisannya tinggal beberapa
    piksel. Diuji pada foto simulasi, tanpa pemangkasan akurasinya 0 dari 25;
    dengan pemangkasan 25 dari 25.
    """
    if a.shape[0] < TINGGI_FOTO_MIN:
        return None
    ys, xs = np.where(a <= ambang_otsu(a))
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    cakupan = ((y1 - y0 + 1) * (x1 - x0 + 1)) / (a.shape[0] * a.shape[1])
    if cakupan >= CAKUPAN_MAKS:
        return None
    return (max(0, x0 - pad), max(0, y0 - pad),
            min(a.shape[1], x1 + pad), min(a.shape[0], y1 + pad))


def band_baris(a, maks_baris=MAKS_BARIS):
    """Kembalikan daftar (y0, y1) untuk tiap baris tulisan, terurut dari atas."""
    biner = a <= ambang_otsu(a)
    per_baris = biner.sum(axis=1).astype(np.float32)
    if per_baris.max() <= 0:
        return []

    lebar_kernel = max(3, a.shape[0] // 64)
    halus = np.convolve(per_baris, np.ones(lebar_kernel) / lebar_kernel, mode='same')
    ambang = max(1.0, float(halus.max()) * 0.12)
    aktif = halus > ambang

    band, i, n = [], 0, len(aktif)
    while i < n:
        if aktif[i]:
            j = i
            while j < n and aktif[j]:
                j += 1
            band.append((i, j))
            i = j
        else:
            i += 1

    tinggi_min = max(8, a.shape[0] // 50)
    band = [(y0, y1) for y0, y1 in band if y1 - y0 >= tinggi_min]
    return band[:maks_baris]


def baris_terbesar(a, pad=4):
    """Band baris tertinggi, atau None jika gambar sudah berupa satu baris.

    Kalau hanya satu band yang terdeteksi, gambar dianggap sudah ter-crop dan
    dibiarkan apa adanya.
    """
    band = band_baris(a)
    if len(band) < 2:
        return None
    y0, y1 = max(band, key=lambda b: b[1] - b[0])
    return max(0, y0 - pad), min(a.shape[0], y1 + pad)
