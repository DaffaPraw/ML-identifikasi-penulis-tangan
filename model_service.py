"""Memuat model, preprocessing, prediksi, dan Grad-CAM.

Preprocessing di sini WAJIB sama dengan sel 5.2 notebook. Kalau berbeda,
akurasi turun tanpa satu pun pesan error.
"""
import glob
import io
import json
import os
import threading

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import models

torch.set_num_threads(2)

BASE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE, 'meta.json'), encoding='utf-8') as fh:
    META = json.load(fh)

PENULIS = META['penulis']
TINGGI = int(META['input']['tinggi'])
LEBAR = int(META['input']['lebar'])
MEAN = torch.tensor(META['normalisasi']['mean']).view(3, 1, 1)
STD = torch.tensor(META['normalisasi']['std']).view(3, 1, 1)
AMBANG = float(META['ambang_keyakinan'])
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

_model = []
_kunci = threading.Lock()


def bangun_resnet18():
    m = models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, len(PENULIS))
    return m


def muat_model():
    """Memuat semua bobot di folder model/ sekali saja (maksimal 5 fold)."""
    if _model:
        return _model
    with _kunci:
        if _model:
            return _model
        berkas = sorted(glob.glob(os.path.join(BASE, 'model', 'resnet18_fold*.pt')))
        if not berkas:
            berkas = sorted(glob.glob(os.path.join(BASE, 'model', '*.pt')))
        for b in berkas[:5]:
            bobot = torch.load(b, map_location=DEVICE, weights_only=True)
            if isinstance(bobot, dict) and 'state_dict' in bobot:
                bobot = bobot['state_dict']
            m = bangun_resnet18()
            m.load_state_dict(bobot)
            m.eval().to(DEVICE)
            _model.append((os.path.basename(b), m))
    return _model


def buka_gambar(data):
    """Bytes -> PIL grayscale, dengan koreksi orientasi EXIF dari ponsel."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    return img.convert('L')


def ubah_ukuran_dan_pad(img):
    """Tinggi jadi 128 px (rasio dijaga), sisa kanvas 1024 px diisi putih."""
    lebar, tinggi = img.size
    skala = min(TINGGI / tinggi, LEBAR / lebar)
    lebar_baru = max(1, round(lebar * skala))
    tinggi_baru = max(1, round(tinggi * skala))
    img = img.resize((lebar_baru, tinggi_baru))
    kanvas = Image.new('L', (LEBAR, TINGGI), 255)
    kanvas.paste(img, (0, (TINGGI - tinggi_baru) // 2))
    return kanvas


def ke_tensor(img_siap):
    a = torch.from_numpy(np.array(img_siap, dtype=np.uint8)).float().div(255)
    x = a.unsqueeze(0).repeat(3, 1, 1)
    return ((x - MEAN) / STD).unsqueeze(0)


def png_dari(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


@torch.no_grad()
def prediksi(img_l, topk=5):
    """Kembalikan (hasil dict, gambar 128x1024 yang masuk ke model)."""
    daftar = muat_model()
    if not daftar:
        raise RuntimeError('Belum ada bobot model di folder model/.')

    siap = ubah_ukuran_dan_pad(img_l)
    x = ke_tensor(siap).to(DEVICE)

    # _kunci: satu forward pass sekali waktu. Grad-CAM memasang hook di layer4 model
    # yang sama; tanpa kunci, permintaan lain yang lewat saat hook terpasang bikin
    # retain_grad() gagal di bawah no_grad (muat_model sudah dipanggil di atas, jadi
    # kunci ini tidak pernah bersarang).
    peluang, tebakan = [], []
    with _kunci:
        for _, m in daftar:
            p = F.softmax(m(x), dim=1)[0]
            peluang.append(p)
            tebakan.append(int(p.argmax()))

    rata = torch.stack(peluang).mean(0)
    urut = torch.argsort(rata, descending=True)[:topk]
    juara = int(urut[0])
    skor = float(rata[juara])

    hasil = {
        'penulis': PENULIS[juara],
        'label_id': juara,
        'keyakinan': round(skor, 4),
        'tidak_dikenal': skor < AMBANG,
        'ambang': AMBANG,
        'jumlah_model': len(daftar),
        'sepakat': sum(1 for t in tebakan if t == juara),
        'top': [{'penulis': PENULIS[int(i)], 'prob': round(float(rata[int(i)]), 4)}
                for i in urut],
    }
    return hasil, siap


@torch.no_grad()
def verifikasi(img_l, nama_penulis):
    """Mode 1:1 — seberapa yakin gambar ini milik penulis tertentu."""
    if nama_penulis not in PENULIS:
        raise ValueError('Penulis tidak terdaftar.')
    hasil, siap = prediksi(img_l, topk=len(PENULIS))
    idx = PENULIS.index(nama_penulis)
    skor = next(r['prob'] for r in hasil['top'] if r['penulis'] == nama_penulis)
    return {
        'penulis': nama_penulis,
        'skor': skor,
        'cocok': bool(skor >= AMBANG and hasil['label_id'] == idx),
        'ambang': AMBANG,
        'tebakan_teratas': hasil['penulis'],
        'jumlah_model': hasil['jumlah_model'],
    }, siap


def gradcam(img_l, label_id=None):
    """Peta Grad-CAM dari layer4 model pertama. Butuh backward pass."""
    daftar = muat_model()
    if not daftar:
        raise RuntimeError('Belum ada bobot model di folder model/.')
    _, m = daftar[0]

    siap = ubah_ukuran_dan_pad(img_l)
    x = ke_tensor(siap).to(DEVICE)

    simpanan = {}

    def hook(_modul, _masuk, keluar):
        if not keluar.requires_grad:
            return
        keluar.retain_grad()
        simpanan['a'] = keluar

    with _kunci:
        pegangan = m.layer4.register_forward_hook(hook)
        try:
            with torch.enable_grad():
                keluaran = m(x)
                if label_id is None:
                    label_id = int(keluaran.argmax())
                m.zero_grad(set_to_none=True)
                keluaran[0, label_id].backward()
                aktivasi = simpanan['a'][0]
                gradien = simpanan['a'].grad[0]
                bobot = gradien.mean(dim=(1, 2), keepdim=True)
                peta = torch.relu((bobot * aktivasi).sum(0))
        finally:
            pegangan.remove()
            m.zero_grad(set_to_none=True)

    peta = peta.detach().cpu().numpy()
    rentang = float(peta.max()) - float(peta.min())
    peta = (peta - peta.min()) / rentang if rentang > 0 else np.zeros_like(peta)
    return siap, peta, label_id


def gradcam_png(img_l, label_id=None):
    """Overlay hangat di atas gambar input, sebagai PNG."""
    siap, peta, label_id = gradcam(img_l, label_id)
    panas = Image.fromarray((peta * 255).astype(np.uint8)).resize(
        (LEBAR, TINGGI), Image.BILINEAR)
    bobot = np.asarray(panas, dtype=np.float32) / 255.0
    dasar = np.asarray(siap, dtype=np.float32) / 255.0
    merah = dasar * (1 - bobot) + bobot * 1.00
    hijau = dasar * (1 - bobot) + bobot * 0.45
    biru = dasar * (1 - bobot) + bobot * 0.10
    rgb = (np.stack([merah, hijau, biru], axis=-1) * 255).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format='PNG')
    return buf.getvalue(), label_id
