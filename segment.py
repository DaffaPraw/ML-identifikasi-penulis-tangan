"""Memotong baris tulisan dengan projection profile.

Model dilatih memakai crop satu baris. Foto kamera biasanya memuat beberapa
baris, jadi band baris dicari dulu supaya input mendekati data latih.
"""
import numpy as np

from quality import ambang_otsu

MAKS_BARIS = 8


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
