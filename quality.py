"""Gerbang mutu: menolak gambar yang tidak layak sebelum masuk ke model.

Lebih baik mengembalikan "gambar terlalu buram" daripada menebak dengan yakin
dari foto yang tidak terbaca.
"""
import numpy as np

MIN_LEBAR = 120
MIN_TINGGI = 16
MIN_KETAJAMAN = 40.0
MIN_KONTRAS = 12.0
RASIO_TINTA_MIN = 0.004
RASIO_TINTA_MAX = 0.60


def _varians_laplacian(a):
    f = a.astype(np.float32)
    lap = (-4 * f[1:-1, 1:-1] + f[:-2, 1:-1] + f[2:, 1:-1]
           + f[1:-1, :-2] + f[1:-1, 2:])
    return float(lap.var())


def ambang_otsu(a):
    """Ambang Otsu dari histogram 256 bin."""
    hist = np.bincount(a.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128
    omega = np.cumsum(hist) / total
    mu = np.cumsum(hist * np.arange(256)) / total
    mu_total = mu[-1]
    penyebut = omega * (1.0 - omega)
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma = (mu_total * omega - mu) ** 2 / penyebut
    sigma[~np.isfinite(sigma)] = 0.0
    return int(np.argmax(sigma))


def rasio_tinta(a):
    ambang = ambang_otsu(a)
    return float((a <= ambang).mean()), ambang


def periksa(a):
    """a: array 2D uint8 grayscale. Kembalikan (lolos, alasan)."""
    if a.ndim != 2:
        return False, 'Gambar harus grayscale 2 dimensi.'
    tinggi, lebar = a.shape
    if lebar < MIN_LEBAR or tinggi < MIN_TINGGI:
        return False, f'Gambar terlalu kecil ({lebar} x {tinggi} piksel).'
    if float(a.std()) < MIN_KONTRAS:
        return False, 'Kontras terlalu rendah. Foto terlalu gelap atau terlalu terang.'
    if _varians_laplacian(a) < MIN_KETAJAMAN:
        return False, 'Gambar terlalu buram. Tahan kamera lebih stabil.'
    rasio, _ = rasio_tinta(a)
    if rasio < RASIO_TINTA_MIN:
        return False, 'Hampir tidak ada tulisan terdeteksi.'
    if rasio > RASIO_TINTA_MAX:
        return False, 'Tulisan tidak terpisah dari latar. Cari pencahayaan yang lebih rata.'
    return True, None
