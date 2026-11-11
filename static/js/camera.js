const MAKS_LEBAR = 1024;

const el = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

let sedangKirim = false;
let blobTerakhir = null;
let labelTerakhir = null;
let arus = null;

function gantiTab(nama) {
  document.querySelectorAll('.tab').forEach((t) =>
    t.classList.toggle('aktif', t.dataset.tab === nama));
  ['kamera', 'unggah', 'verifikasi'].forEach((n) =>
    el('panel-' + n).classList.toggle('sembunyi', n !== nama));
}

document.querySelectorAll('.tab').forEach((t) =>
  t.addEventListener('click', () => gantiTab(t.dataset.tab)));

async function mulaiKamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    el('pesan-kamera').textContent =
      'Kamera hanya bisa dibuka lewat HTTPS atau localhost. Buka http://localhost:5000, '
      + 'jalankan dengan --cert=adhoc, atau pakai tab Unggah.';
    return;
  }
  try {
    arus = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 } },
      audio: false,
    });
    el('video').srcObject = arus;
    await el('video').play();
    el('btn-ambil').disabled = false;
    el('btn-mulai').textContent = 'Kamera aktif';
    el('pesan-kamera').textContent = '';
  } catch (galat) {
    el('pesan-kamera').textContent = 'Kamera tidak bisa dibuka: ' + galat.message;
  }
}

function ambilFrame() {
  const v = el('video');
  if (!v.videoWidth) return Promise.resolve(null);
  const skala = Math.min(1, MAKS_LEBAR / v.videoWidth);
  const kanvas = document.createElement('canvas');
  kanvas.width = Math.round(v.videoWidth * skala);
  kanvas.height = Math.round(v.videoHeight * skala);
  kanvas.getContext('2d').drawImage(v, 0, 0, kanvas.width, kanvas.height);
  return new Promise((res) => kanvas.toBlob(res, 'image/jpeg', 0.8));
}

async function kirim(blob, url, tambahan) {
  if (sedangKirim || !blob) return;
  sedangKirim = true;
  el('hasil').innerHTML = '<p class="kosong">Memproses...</p>';
  const data = new FormData();
  data.append('image', blob, 'frame.jpg');
  if (tambahan) {
    Object.entries(tambahan).forEach(([k, v]) => data.append(k, v));
  }
  try {
    const jawab = await fetch(url, { method: 'POST', body: data });
    const hasil = await jawab.json();
    blobTerakhir = blob;
    if (url.endsWith('/verify')) renderVerifikasi(hasil);
    else renderPrediksi(hasil);
  } catch (galat) {
    tolak('Gagal menghubungi server: ' + galat.message);
  } finally {
    sedangKirim = false;
  }
}

function tolak(alasan) {
  el('bawah').classList.add('sembunyi');
  el('hasil').innerHTML =
    '<div class="kartu warn"><p class="label">Gambar belum bisa dipakai</p>'
    + '<p style="margin:0">' + esc(alasan) + '</p></div>';
}

function barisTop(top) {
  const puncak = top.length ? top[0].prob : 1;
  return top.map((r, i) => {
    const lebar = Math.max(3, Math.round((r.prob / (puncak || 1)) * 150));
    return '<div class="baris"><span class="kode">' + esc(r.penulis) + '</span>'
      + '<span class="bar' + (i === 0 ? ' utama' : '') + '" style="width:' + lebar + 'px"></span>'
      + '<span class="nilai">' + (r.prob * 100).toFixed(1) + '%</span></div>';
  }).join('');
}

function renderPrediksi(h) {
  if (!h.ok) { tolak(h.alasan || 'Gambar ditolak.'); return; }
  labelTerakhir = h.label_id;

  const dikenal = !h.tidak_dikenal;
  const judul = dikenal ? h.penulis : 'Tidak dikenal';
  const pita = dikenal
    ? '<span class="pita">' + (h.keyakinan * 100).toFixed(1) + '% keyakinan</span>'
    : '<span class="pita netral">keyakinan ' + (h.keyakinan * 100).toFixed(1)
      + '%, di bawah ambang ' + (h.ambang * 100).toFixed(0) + '%</span>';

  el('hasil').innerHTML =
    '<div class="kartu' + (dikenal ? ' ok' : '') + '">'
    + '<p class="label">' + (dikenal ? 'Penulis' : 'Hasil') + '</p>'
    + '<p class="nama">' + esc(judul) + '</p>'
    + pita
    + '<p class="catatan" style="margin-top:8px">' + h.sepakat + ' dari ' + h.jumlah_model
    + ' model sepakat' + (h.dipotong ? ' &middot; baris dipotong otomatis' : '') + '</p>'
    + '</div>'
    + '<div class="kartu"><p class="label">'
    + (dikenal ? 'Kandidat lain' : 'Kandidat terdekat') + '</p>'
    + barisTop(dikenal ? h.top.slice(1) : h.top) + '</div>';

  el('waktu').textContent = h.ms + ' ms';
  tampilkanBawah(h);
}

function renderVerifikasi(h) {
  if (!h.ok) { tolak(h.alasan || 'Gambar ditolak.'); return; }
  labelTerakhir = null;
  el('hasil').innerHTML =
    '<div class="kartu' + (h.cocok ? ' ok' : '') + '">'
    + '<p class="label">Verifikasi ' + esc(h.penulis) + '</p>'
    + '<p class="nama">' + (h.cocok ? 'Cocok' : 'Tidak cocok') + '</p>'
    + '<span class="pita' + (h.cocok ? '' : ' netral') + '">skor '
    + (h.skor * 100).toFixed(1) + '%</span>'
    + '<p class="catatan" style="margin-top:8px">Tebakan teratas model: '
    + esc(h.tebakan_teratas) + '</p></div>';
  tampilkanBawah(h);
}

function tampilkanBawah(h) {
  el('bawah').classList.remove('sembunyi');
  el('input-model').src = h.input_png;
  const contoh = h.contoh || [];
  el('kotak-contoh').innerHTML = contoh.length
    ? contoh.map((u) => '<img src="' + esc(u) + '" alt="Contoh tulisan">').join('')
    : '<div class="kotak-kosong"></div><div class="kotak-kosong"></div><div class="kotak-kosong"></div>';
  el('kotak-gradcam').innerHTML = '<button id="btn-gradcam">Tampilkan</button>';
  el('btn-gradcam').addEventListener('click', tampilkanGradcam);
}

async function tampilkanGradcam() {
  if (!blobTerakhir) return;
  el('kotak-gradcam').innerHTML = '<p class="catatan">Menghitung...</p>';
  const data = new FormData();
  data.append('image', blobTerakhir, 'frame.jpg');
  if (labelTerakhir !== null) data.append('label_id', labelTerakhir);
  try {
    const jawab = await fetch('/api/gradcam', { method: 'POST', body: data });
    const hasil = await jawab.json();
    el('kotak-gradcam').innerHTML = hasil.ok
      ? '<img src="' + hasil.png + '" alt="Grad-CAM">'
      : '<p class="catatan peringatan">' + esc(hasil.alasan) + '</p>';
  } catch (galat) {
    el('kotak-gradcam').innerHTML = '<p class="catatan peringatan">' + esc(galat.message) + '</p>';
  }
}

function pasangUnggah(idBerkas, idPratinjau, url, ambilTambahan) {
  el(idBerkas).addEventListener('change', (ev) => {
    const berkas = ev.target.files[0];
    if (!berkas) return;
    el(idPratinjau).src = URL.createObjectURL(berkas);
    kirim(berkas, url, ambilTambahan ? ambilTambahan() : null);
  });
}

el('btn-mulai').addEventListener('click', mulaiKamera);
el('btn-ambil').addEventListener('click', async () => {
  const blob = await ambilFrame();
  if (blob) kirim(blob, '/api/predict');
});
pasangUnggah('berkas', 'pratinjau', '/api/predict');
pasangUnggah('berkas-verifikasi', 'pratinjau-verifikasi', '/api/verify',
  () => ({ penulis: el('pilih-penulis').value }));

(async function awal() {
  try {
    const penulis = await (await fetch('/api/writers')).json();
    el('pilih-penulis').innerHTML = penulis.penulis
      .map((p) => '<option value="' + esc(p) + '">' + esc(p) + '</option>').join('');
    const sehat = await (await fetch('/api/health')).json();
    if (!sehat.model_siap) {
      el('banner').classList.remove('sembunyi');
      el('banner').textContent =
        'Bobot model belum ada. Salin resnet18_fold5.pt dari Drive ke folder model/, '
        + 'lalu muat ulang halaman ini.';
    }
  } catch (galat) {
    /* biarkan halaman tetap tampil walau health gagal */
  }
}());
