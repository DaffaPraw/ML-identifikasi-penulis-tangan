"""Melatih ulang satu fold ResNet18 di mesin lokal (CPU).

Memakai split, preprocessing, dan hyperparameter yang sama persis dengan
notebook UAS, supaya bobot yang dihasilkan bisa langsung dipakai aplikasi.
Bobot tidak akan identik dengan hasil Colab (RNG dan hardware berbeda), tetapi
pembagian datanya identik karena seed-nya sama.

Contoh:
    python latih_lokal.py --dataset "../data_uas/dataset_clean"
    python latih_lokal.py --dataset "../data_uas/dataset_clean" --fold 2
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset

import model_service as ms

SEED = 42
TEST_SIZE = 0.20
N_FOLD = 5
BATCH = 32
LR = 1e-4
MAX_EPOCH = 30
PATIENCE = 5

BASE = os.path.dirname(os.path.abspath(__file__))


def daftar_gambar(dataset):
    baris = []
    for penulis in sorted(os.listdir(dataset)):
        folder = os.path.join(dataset, penulis)
        if not os.path.isdir(folder):
            continue
        for nama in sorted(os.listdir(folder)):
            baris.append({'path': os.path.join(folder, nama), 'label': penulis})
    df = pd.DataFrame(baris)
    kelas = sorted(df['label'].unique())
    df['label_id'] = df['label'].map({p: i for i, p in enumerate(kelas)})
    return df, kelas


def buat_split(df):
    """Sama persis dengan sel 1.4 notebook."""
    idx_sisa, idx_test = train_test_split(
        df.index, test_size=TEST_SIZE, stratify=df['label_id'], random_state=SEED)
    df.loc[idx_test, 'split'] = 'test'
    sisa = df.loc[idx_sisa]
    skf = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)
    for nomor, (_, idx_val) in enumerate(skf.split(sisa, sisa['label_id'])):
        df.loc[sisa.index[idx_val], 'split'] = str(nomor)
    return df


class DataTulisan(Dataset):
    def __init__(self, data, cache):
        self.path = data['path'].tolist()
        self.label = data['label_id'].tolist()
        self.cache = cache

    def __len__(self):
        return len(self.path)

    def __getitem__(self, i):
        gambar = Image.fromarray(self.cache[self.path[i]])
        return ms.ke_tensor(gambar)[0], self.label[i]


def evaluasi(model, loader, lossf, device):
    model.eval()
    total, asli, tebak = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            keluaran = model(x)
            total += lossf(keluaran, y).item() * len(y)
            asli += y.tolist()
            tebak += keluaran.argmax(dim=1).tolist()
    return total / len(asli), asli, tebak


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--fold', type=int, default=1, help='Nomor fold 1-5')
    p.add_argument('--epoch', type=int, default=MAX_EPOCH)
    arg = p.parse_args()

    inti = os.cpu_count() or 4
    torch.set_num_threads(inti)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device} | thread: {torch.get_num_threads()}', flush=True)

    df, kelas = daftar_gambar(arg.dataset)
    if kelas != ms.PENULIS:
        raise SystemExit('Daftar kelas dataset tidak cocok dengan meta.json.')
    print(f'{len(df)} gambar, {len(kelas)} penulis', flush=True)

    df = buat_split(df)
    folder_hasil = os.path.join(BASE, 'hasil_lokal')
    os.makedirs(folder_hasil, exist_ok=True)
    df.to_csv(os.path.join(folder_hasil, 'split_data.csv'), index=False)
    print(df['split'].value_counts().sort_index().to_string(), flush=True)

    mulai = time.perf_counter()
    cache = {p_: np.array(ms.ubah_ukuran_dan_pad(Image.open(p_).convert('L')))
             for p_ in df['path']}
    print(f'{len(cache)} gambar dimuat ke RAM dalam {time.perf_counter()-mulai:.1f} detik',
          flush=True)

    fold = str(arg.fold - 1)
    latih = df[(df['split'] != 'test') & (df['split'] != fold)]
    val = df[df['split'] == fold]
    uji = df[df['split'] == 'test']
    print(f'Fold {arg.fold} -> latih {len(latih)} | validasi {len(val)} | uji {len(uji)}',
          flush=True)

    loader_latih = DataLoader(DataTulisan(latih, cache), batch_size=BATCH, shuffle=True)
    loader_val = DataLoader(DataTulisan(val, cache), batch_size=BATCH)
    loader_uji = DataLoader(DataTulisan(uji, cache), batch_size=BATCH)

    torch.manual_seed(SEED + arg.fold - 1)
    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(kelas))
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()
    berkas = os.path.join(BASE, 'model', f'resnet18_fold{arg.fold}.pt')
    os.makedirs(os.path.dirname(berkas), exist_ok=True)

    f1_terbaik, epoch_terbaik, tanpa_naik = -1.0, 0, 0
    t0 = time.perf_counter()
    for epoch in range(1, arg.epoch + 1):
        model.train()
        total = 0.0
        for x, y in loader_latih:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            rugi = lossf(model(x), y)
            rugi.backward()
            opt.step()
            total += rugi.item() * len(y)
        rugi_val, asli, tebak = evaluasi(model, loader_val, lossf, device)
        f1_val = f1_score(asli, tebak, average='macro', zero_division=0)
        print(f'Epoch {epoch:2d} | loss latih {total/len(latih):.4f} | '
              f'loss val {rugi_val:.4f} | F1 val {f1_val:.4f} | '
              f'{(time.perf_counter()-t0)/60:.1f} menit', flush=True)

        if f1_val > f1_terbaik:
            f1_terbaik, epoch_terbaik, tanpa_naik = f1_val, epoch, 0
            torch.save(model.state_dict(), berkas)
        else:
            tanpa_naik += 1
            if tanpa_naik >= PATIENCE:
                print(f'Early stopping: F1 validasi tidak naik {PATIENCE} epoch', flush=True)
                break
    waktu = time.perf_counter() - t0

    model.load_state_dict(torch.load(berkas, map_location=device, weights_only=True))
    _, asli, tebak = evaluasi(model, loader_uji, lossf, device)
    print('\n=== Hasil pada TEST SET ===', flush=True)
    print(f'Precision : {precision_score(asli, tebak, average="macro", zero_division=0):.4f}')
    print(f'Recall    : {recall_score(asli, tebak, average="macro", zero_division=0):.4f}')
    print(f'Akurasi   : {accuracy_score(asli, tebak):.4f}')
    print(f'F1-Score  : {f1_score(asli, tebak, average="macro", zero_division=0):.4f}')
    print(f'Epoch terbaik: {epoch_terbaik} | waktu latih: {waktu/60:.1f} menit')
    print(f'Bobot disimpan: {berkas}')


if __name__ == '__main__':
    main()
