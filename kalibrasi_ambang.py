"""Mengkalibrasi ambang keyakinan memakai test set.

Ambang di meta.json awalnya hanya nilai sementara. Skrip ini mengukur sebaran
keyakinan pada data uji, lalu mengambil persentil bawah dari prediksi yang
benar sebagai ambang: di bawah nilai itu, aplikasi menjawab "tidak dikenal".

Contoh:
    python kalibrasi_ambang.py --dataset "D:/dataset_clean" --split hasil/split_data.csv
    python kalibrasi_ambang.py --dataset "D:/dataset_clean" --split hasil/split_data.csv --tulis
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import model_service as ms

# Kalibrasi adalah pekerjaan batch offline, bukan server: pakai semua core.
torch.set_num_threads(os.cpu_count() or 4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, help='Folder dataset_clean')
    p.add_argument('--split', required=True, help='split_data.csv dari notebook')
    p.add_argument('--persentil', type=float, default=1.0,
                   help='Persentil bawah dari prediksi benar (default 1)')
    p.add_argument('--tulis', action='store_true', help='Simpan hasil ke meta.json')
    arg = p.parse_args()

    daftar = ms.muat_model()
    if not daftar:
        raise SystemExit('Tidak ada bobot di folder model/.')
    print(f'{len(daftar)} model dimuat: {[n for n, _ in daftar]}')

    df = pd.read_csv(arg.split)
    df['split'] = df['split'].astype(str)
    uji = df[df['split'] == 'test'].copy()
    if uji.empty:
        raise SystemExit('Tidak ada baris split == test di CSV.')
    print(f'{len(uji)} gambar uji')

    skor, benar = [], []
    for _, baris in uji.iterrows():
        bagian = str(baris['path']).replace('\\', '/').split('/')[-2:]
        jalur = os.path.join(arg.dataset, *bagian)
        with Image.open(jalur) as im:
            siap = ms.ubah_ukuran_dan_pad(im.convert('L'))
        x = ms.ke_tensor(siap).to(ms.DEVICE)
        with torch.no_grad():
            rata = torch.stack([F.softmax(m(x), dim=1)[0] for _, m in daftar]).mean(0)
        tebak = int(rata.argmax())
        skor.append(float(rata[tebak]))
        benar.append(tebak == ms.PENULIS.index(baris['label']))

    skor = np.array(skor)
    benar = np.array(benar)
    print(f'Akurasi ensemble: {benar.mean():.4f}')
    print(f'Keyakinan prediksi benar  - rata {skor[benar].mean():.4f}, '
          f'min {skor[benar].min():.4f}')
    if (~benar).any():
        print(f'Keyakinan prediksi salah  - rata {skor[~benar].mean():.4f}, '
              f'maks {skor[~benar].max():.4f}')

    ambang = float(np.percentile(skor[benar], arg.persentil))
    ditolak = int((skor < ambang).sum())
    salah_tersaring = int(((skor < ambang) & ~benar).sum())
    print(f'\nAmbang persentil {arg.persentil} = {ambang:.4f}')
    print(f'  {ditolak} dari {len(skor)} gambar uji akan dijawab "tidak dikenal"')
    print(f'  {salah_tersaring} dari {int((~benar).sum())} prediksi salah tersaring')

    if arg.tulis:
        jalur_meta = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meta.json')
        with open(jalur_meta, encoding='utf-8') as fh:
            meta = json.load(fh)
        meta['ambang_keyakinan'] = round(ambang, 4)
        meta['catatan_ambang'] = (f'Persentil {arg.persentil} dari keyakinan prediksi benar '
                                  f'pada {len(skor)} gambar uji.')
        with open(jalur_meta, 'w', encoding='utf-8') as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        print(f'\nmeta.json diperbarui: ambang_keyakinan = {round(ambang, 4)}')


if __name__ == '__main__':
    main()
