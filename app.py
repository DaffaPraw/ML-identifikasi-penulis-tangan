"""Aplikasi web identifikasi penulis tulisan tangan.

Kamera dipegang browser (getUserMedia), bukan server. Server hanya menerima
satu frame JPEG per permintaan, lalu mengembalikan JSON.
"""
import base64
import os
import time

import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_from_directory

import model_service as ms
import quality
import segment

BASE = os.path.dirname(os.path.abspath(__file__))
DIR_CONTOH = os.path.join(BASE, 'contoh')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024


def _ambil_gambar():
    """Dari multipart 'image' atau JSON {'image': dataURL}."""
    if 'image' in request.files:
        data = request.files['image'].read()
    else:
        muatan = request.get_json(silent=True) or {}
        dataurl = muatan.get('image') or ''
        if ',' in dataurl:
            dataurl = dataurl.split(',', 1)[1]
        if not dataurl:
            return None
        data = base64.b64decode(dataurl)
    if not data:
        return None
    return ms.buka_gambar(data)


def _minta_potong():
    """Pemotongan baris harus diminta eksplisit (checkbox 'halaman penuh')."""
    nilai = (request.form.get('potong')
             or (request.get_json(silent=True) or {}).get('potong'))
    return str(nilai).lower() in ('1', 'true', 'on', 'ya')


def _siapkan(img, potong=False):
    """Pangkas foto ke area tulisan, gerbang mutu, lalu potong baris bila diminta.

    Urutannya penting. Pemangkasan foto jalan lebih dulu supaya gerbang mutu
    menilai area tulisannya, bukan hamparan kertas kosong di sekelilingnya.

    Potong baris tetap opsional: model dilatih pada gambar utuh yang
    dimampatkan ke tinggi 128, bukan pada hasil potong satu baris. Diukur pada
    531 gambar uji, memotong menurunkan akurasi dari 0,9849 ke 0,9454.
    """
    a = np.asarray(img, dtype=np.uint8)
    catatan = None

    kotak = segment.kotak_tinta(a)
    if kotak:
        img = img.crop(kotak)
        a = np.asarray(img, dtype=np.uint8)
        catatan = 'foto dipangkas ke area tulisan'

    lolos, alasan = quality.periksa(a)
    if not lolos:
        return None, None, alasan

    if potong:
        band = segment.baris_terbesar(a)
        if band:
            img = img.crop((0, band[0], img.width, band[1]))
            catatan = 'dipotong ke baris terbesar'

    return img, catatan, None


@app.get('/')
def halaman():
    return render_template('index.html', jumlah_penulis=len(ms.PENULIS))


@app.get('/api/health')
def health():
    daftar = ms.muat_model()
    return jsonify(
        model_siap=bool(daftar),
        berkas_model=[nama for nama, _ in daftar],
        device=ms.DEVICE,
        jumlah_penulis=len(ms.PENULIS),
        ambang=ms.AMBANG,
    )


@app.get('/api/writers')
def writers():
    return jsonify(penulis=ms.PENULIS)


@app.post('/api/predict')
def predict():
    mulai = time.perf_counter()
    try:
        img = _ambil_gambar()
    except Exception:
        return jsonify(ok=False, alasan='Berkas tidak bisa dibaca sebagai gambar.'), 400
    if img is None:
        return jsonify(ok=False, alasan='Tidak ada gambar yang dikirim.'), 400

    img, catatan, alasan = _siapkan(img, _minta_potong())
    if alasan:
        return jsonify(ok=False, alasan=alasan)

    try:
        hasil, siap = ms.prediksi(img)
    except RuntimeError as galat:
        return jsonify(ok=False, alasan=str(galat)), 503

    hasil['ok'] = True
    hasil['catatan'] = catatan
    hasil['ms'] = round((time.perf_counter() - mulai) * 1000, 1)
    hasil['input_png'] = 'data:image/png;base64,' + base64.b64encode(
        ms.png_dari(siap)).decode()
    hasil['contoh'] = _contoh_penulis(hasil['penulis'])
    return jsonify(hasil)


@app.post('/api/verify')
def verify():
    nama = (request.form.get('penulis')
            or (request.get_json(silent=True) or {}).get('penulis') or '')
    try:
        img = _ambil_gambar()
    except Exception:
        return jsonify(ok=False, alasan='Berkas tidak bisa dibaca sebagai gambar.'), 400
    if img is None:
        return jsonify(ok=False, alasan='Tidak ada gambar yang dikirim.'), 400

    img, catatan, alasan = _siapkan(img, _minta_potong())
    if alasan:
        return jsonify(ok=False, alasan=alasan)

    try:
        hasil, siap = ms.verifikasi(img, nama)
    except ValueError as galat:
        return jsonify(ok=False, alasan=str(galat)), 400
    except RuntimeError as galat:
        return jsonify(ok=False, alasan=str(galat)), 503

    hasil['ok'] = True
    hasil['catatan'] = catatan
    hasil['input_png'] = 'data:image/png;base64,' + base64.b64encode(
        ms.png_dari(siap)).decode()
    hasil['contoh'] = _contoh_penulis(nama)
    return jsonify(hasil)


@app.post('/api/gradcam')
def gradcam():
    label = (request.form.get('label_id')
             or (request.get_json(silent=True) or {}).get('label_id'))
    try:
        img = _ambil_gambar()
    except Exception:
        return jsonify(ok=False, alasan='Berkas tidak bisa dibaca sebagai gambar.'), 400
    if img is None:
        return jsonify(ok=False, alasan='Tidak ada gambar yang dikirim.'), 400

    img, _, alasan = _siapkan(img, _minta_potong())
    if alasan:
        return jsonify(ok=False, alasan=alasan)

    try:
        png, label_id = ms.gradcam_png(img, int(label) if label not in (None, '') else None)
    except RuntimeError as galat:
        return jsonify(ok=False, alasan=str(galat)), 503

    return jsonify(ok=True, label_id=label_id,
                   png='data:image/png;base64,' + base64.b64encode(png).decode())


def _contoh_penulis(nama):
    """Daftar URL contoh tulisan penulis, kosong kalau folder contoh belum ada."""
    folder = os.path.join(DIR_CONTOH, nama)
    if nama not in ms.PENULIS or not os.path.isdir(folder):
        return []
    berkas = sorted(f for f in os.listdir(folder) if f.lower().endswith('.jpg'))
    return [f'/contoh/{nama}/{f}' for f in berkas[:3]]


@app.get('/contoh/<penulis>/<berkas>')
def contoh(penulis, berkas):
    if penulis not in ms.PENULIS or not berkas.lower().endswith('.jpg'):
        abort(404)
    return send_from_directory(os.path.join(DIR_CONTOH, penulis), berkas)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
