/* alat.js - satu gambar masuk, satu jawaban keluar.
   Tidak ada kamera: berkas dipilih atau dijatuhkan, lalu dikirim sebagai multipart ke /api/predict
   atau /api/verify. Satu permintaan sekali jalan; peta Grad-CAM dihitung hanya kalau diminta,
   karena ia butuh backward pass dan membuat prediksi dua sampai tiga kali lebih lambat. */
(function () {
  'use strict';

  var el = function (id) { return document.getElementById(id); };
  var esc = function (s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  };
  var pct = function (v) { return (v * 100).toFixed(1).replace('.', ',') + '%'; };

  var sedangKirim = false;
  var berkasTerakhir = null;
  var labelTerakhir = null;

  /* ------------------------------------------------------------------ kabar */

  function kabar(teks) { el('kabar').textContent = teks; }

  /* ------------------------------------------------------------------ bingkai */

  function bingkai(kotak, judul, meta, chip, isi) {
    kotak.innerHTML =
      '<div class="ui ui-panel rs">'
      + '<div class="rs__head"><span class="ui-title">' + esc(judul) + '</span>'
      + (meta ? '<span class="ui-sep" aria-hidden="true"></span><span class="ui-meta">' + esc(meta) + '</span>' : '')
      + '<span class="ui-spacer"></span>'
      + (chip ? '<span class="ui-chip ui-chip--plain">' + esc(chip) + '</span>' : '')
      + '</div><div class="rs__body">' + isi + '</div></div>';
  }

  function tunggu(kotak, judul) {
    bingkai(kotak, judul, '', 'Memproses', '<p class="rs__wait">Model sedang menghitung peluang 50 penulis.</p>');
  }

  function tolak(kotak, judul, alasan) {
    bingkai(kotak, judul, '', 'Ditolak',
      '<div class="rs__note"><i>!</i><span>' + esc(alasan) + '</span></div>');
  }

  function baris(top) {
    var puncak = top.length ? top[0].prob : 1;
    return top.map(function (r) {
      var lebar = Math.max(4, Math.round((r.prob / (puncak || 1)) * 100));
      return '<div class="rs__row"><span>' + esc(r.penulis) + '</span>'
        + '<span class="rs__bar"><i style="--v: ' + lebar + '%"></i></span>'
        + '<span>' + pct(r.prob) + '</span></div>';
    }).join('');
  }

  /* ------------------------------------------------------------------ hasil */

  function tampilkanPrediksi(h) {
    var kotak = el('kotak-hasil');
    if (!h.ok) { tolak(kotak, berkasTerakhir ? berkasTerakhir.name : 'Gambar', h.alasan || 'Gambar ditolak.'); return; }
    labelTerakhir = h.label_id;

    var dikenal = !h.tidak_dikenal;
    var isi = '<div class="rs__shot"><img src="' + h.input_png + '" alt="Gambar seperti yang masuk ke model: tinggi 128 piksel di atas kanvas putih 1024 piksel"></div>'
      + '<div class="rs__top"><span class="rs__name">' + esc(dikenal ? h.penulis : 'Tidak dikenal') + '</span>'
      + '<span class="rs__pct">keyakinan ' + pct(h.keyakinan) + ', ambang ' + pct(h.ambang) + '</span></div>'
      + '<div class="rs__rows">' + baris(h.top) + '</div>'
      + '<div class="rs__note"><i>' + h.sepakat + '</i><span>' + h.sepakat + ' dari ' + h.jumlah_model
      + ' model sepakat' + (h.catatan ? ', ' + esc(h.catatan) : '') + ', selesai dalam ' + String(h.ms).replace('.', ',') + ' ms</span></div>';

    if (dikenal && h.contoh && h.contoh.length) {
      isi += '<p class="rs__label">Contoh tulisan ' + esc(h.penulis) + ' dari data latih</p>'
        + '<div class="rs__contoh">' + h.contoh.map(function (u) {
          return '<img src="' + esc(u) + '" alt="Baris tulisan tangan ' + esc(h.penulis) + ' dari data latih" loading="lazy">';
        }).join('') + '</div>';
    }

    isi += '<div class="rs__kam"><button class="ui-btn ui-btn--ghost" type="button" id="btn-gradcam">Tampilkan Grad-CAM</button></div>';

    bingkai(el('kotak-hasil'), berkasTerakhir.name,
      dikenal ? 'di atas ambang' : 'di bawah ambang',
      dikenal ? h.penulis : 'Tidak dikenal', isi);
    el('btn-gradcam').addEventListener('click', ambilGradcam);
  }

  function tampilkanVerifikasi(h, nama) {
    var kotak = el('kotak-periksa');
    if (!h.ok) { tolak(kotak, nama, h.alasan || 'Gambar ditolak.'); return; }
    var isi = '<div class="rs__shot"><img src="' + h.input_png + '" alt="Gambar seperti yang masuk ke model"></div>'
      + '<div class="rs__top"><span class="rs__name">' + (h.cocok ? 'Cocok' : 'Tidak cocok') + '</span>'
      + '<span class="rs__pct">skor ' + pct(h.skor) + ' untuk ' + esc(h.penulis) + '</span></div>'
      + '<div class="rs__rows"><div class="rs__row"><span>' + esc(h.penulis) + '</span>'
      + '<span class="rs__bar"><i style="--v: ' + Math.max(4, Math.round(h.skor * 100)) + '%"></i></span>'
      + '<span>' + pct(h.skor) + '</span></div></div>'
      + '<div class="rs__note"><i>i</i><span>Tebakan teratas model: <b>' + esc(h.tebakan_teratas) + '</b></span></div>';
    bingkai(kotak, 'Skor untuk ' + nama, '', h.cocok ? 'Cocok' : 'Tidak cocok', isi);
  }

  /* ------------------------------------------------------------------ kirim */

  function kirim(berkas, alamat, kotak, tambahan, render) {
    if (sedangKirim || !berkas) return;
    sedangKirim = true;
    berkasTerakhir = berkas;
    tunggu(kotak, berkas.name);
    kabar('Mengirim ' + berkas.name + '.');

    var data = new FormData();
    data.append('image', berkas, berkas.name);
    data.append('potong', el('potong').checked ? '1' : '0');
    if (tambahan) {
      Object.keys(tambahan).forEach(function (k) { data.append(k, tambahan[k]); });
    }

    fetch(alamat, { method: 'POST', body: data })
      .then(function (jawab) {
        if (jawab.status === 413) throw new Error('Gambarnya lebih dari 10 MB.');
        return jawab.json().catch(function () { throw new Error('Server menjawab bukan JSON (status ' + jawab.status + ').'); });
      })
      .then(function (hasil) {
        render(hasil);
        kabar(hasil.ok ? 'Selesai.' : 'Gambar belum bisa dipakai.');
      })
      .catch(function (galat) {
        tolak(kotak, berkas.name, galat.message);
        kabar('Gagal: ' + galat.message);
      })
      .then(function () { sedangKirim = false; });
  }

  function ambilGradcam() {
    if (!berkasTerakhir) return;
    var tombol = el('btn-gradcam');
    tombol.disabled = true;
    tombol.textContent = 'Menghitung';
    var data = new FormData();
    data.append('image', berkasTerakhir, berkasTerakhir.name);
    data.append('potong', el('potong').checked ? '1' : '0');
    if (labelTerakhir !== null) data.append('label_id', labelTerakhir);

    fetch('/api/gradcam', { method: 'POST', body: data })
      .then(function (jawab) { return jawab.json(); })
      .then(function (hasil) {
        var tempat = document.querySelector('.rs__kam');
        if (!tempat) return;
        tempat.innerHTML = hasil.ok
          ? '<div class="rs__shot"><img src="' + hasil.png + '" alt="Peta Grad-CAM: bagian baris yang paling menentukan tebakan"></div>'
          : '<div class="rs__note"><i>!</i><span>' + esc(hasil.alasan || 'Grad-CAM gagal.') + '</span></div>';
      })
      .catch(function (galat) {
        tombol.disabled = false;
        tombol.textContent = 'Tampilkan Grad-CAM';
        kabar('Grad-CAM gagal: ' + galat.message);
      });
  }

  /* ------------------------------------------------------------------ masukan */

  function pasangJatuh(kotakId, ambil) {
    var kotak = el(kotakId);
    ['dragenter', 'dragover'].forEach(function (n) {
      kotak.addEventListener(n, function (ev) { ev.preventDefault(); kotak.classList.add('is-over'); });
    });
    ['dragleave', 'drop'].forEach(function (n) {
      kotak.addEventListener(n, function () { kotak.classList.remove('is-over'); });
    });
    kotak.addEventListener('drop', function (ev) {
      ev.preventDefault();
      var berkas = ev.dataTransfer && ev.dataTransfer.files[0];
      if (berkas) ambil(berkas);
    });
  }

  function prediksiBerkas(berkas) {
    kirim(berkas, '/api/predict', el('kotak-hasil'), null, tampilkanPrediksi);
  }

  function periksaBerkas(berkas) {
    var nama = el('penulis').value;
    if (!nama) { kabar('Pilih dulu kode penulisnya.'); return; }
    kirim(berkas, '/api/verify', el('kotak-periksa'), { penulis: nama }, function (h) {
      tampilkanVerifikasi(h, nama);
    });
  }

  el('berkas').addEventListener('change', function (ev) {
    if (ev.target.files[0]) prediksiBerkas(ev.target.files[0]);
  });
  el('berkas-periksa').addEventListener('change', function (ev) {
    if (ev.target.files[0]) periksaBerkas(ev.target.files[0]);
  });
  pasangJatuh('drop', prediksiBerkas);
  pasangJatuh('drop-periksa', periksaBerkas);

  /* ------------------------------------------------------------------ awal */

  fetch('/api/writers')
    .then(function (j) { return j.json(); })
    .then(function (d) {
      el('penulis').innerHTML = d.penulis.map(function (p) {
        return '<option value="' + esc(p) + '">' + esc(p) + '</option>';
      }).join('');
    })
    .catch(function () { /* halaman tetap terbuka tanpa daftar penulis */ });

  fetch('/api/health')
    .then(function (j) { return j.json(); })
    .then(function (d) {
      if (!d.model_siap) {
        kabar('Belum ada bobot model di folder model/, jadi gambar belum bisa diproses.');
      } else {
        kabar(d.berkas_model.length + ' bobot termuat di ' + d.device
          + ', ambang ' + String(d.ambang).replace('.', ',') + '. Gambar diproses di memori dan tidak disimpan.');
      }
    })
    .catch(function () { /* biarkan kabar bawaan */ });
}());
