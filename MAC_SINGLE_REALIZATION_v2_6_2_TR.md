# MacBook Air tek-tohum üretim kılavuzu — v2.6.2

Bu kılavuz yalnız `seed=601`, `structural_seed=160601` tasarımı içindir.
11.686 analiz görevi, tekilleştirme sonrası 3.610 gerçek simülasyon çalışır.
Model denklemleri ve biyolojik parametreler bulut sürümüyle aynıdır.

## 1. Sistem gereksinimi

- macOS, 16 GB RAM;
- CPython 3.12.x;
- en az 30 GB boş disk;
- çalışma sırasında güç adaptörü ve havalandırılan sert bir yüzey.

Sürümü kontrol edin:

```bash
python3 --version
uname -m
```

Python 3.12 yoksa Homebrew bulunan bir Mac'te:

```bash
brew install python@3.12
```

## 2. Paketi açma ve ortamı kurma

İndirilen dosyanın Downloads klasöründe olduğu varsayımıyla:

```bash
cd ~/Downloads
tar -xzf CPG_v2_6_2_SINGLE_REALIZATION_GCP_3VM_RELEASE.tar.gz
cd cpg_v2_6_2_single_realization_release
python3.12 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-lock.txt
python3 -m unittest -v test_single_realization_v2_6_2.py
```

Testlerin `7/7` geçmesi gerekir. Bir test başarısızsa üretim başlamaz.

## 3. Arka planda başlatma

16 GB MacBook Air için varsayılan değer 6 worker'dır:

```bash
chmod +x run_mac_single_realization_v2_6_2.sh
nohup ./run_mac_single_realization_v2_6_2.sh \
  single_realization_results_v2_6_2 \
  > run_single_realization_v2_6_2.log 2>&1 &
echo $!
```

`nohup` terminal kapansa da süreci korur; betik içindeki `caffeinate` Mac'in
uykuya geçmesini engeller. Ekran kapatılabilir fakat bilgisayar kapatılmamalı,
yeniden başlatılmamalı ve güç adaptöründen çıkarılmamalıdır.

İlerlemeyi görmek için:

```bash
tail -f run_single_realization_v2_6_2.log
```

Takibi bırakmak için `Control-C` yalnız `tail` komutunu kapatır; simülasyonu
durdurmaz.

## 4. Güvenli devam

Mac kapanır veya süreç kesilirse aynı paket ve aynı sonuç dizininde aynı komut
yeniden çalıştırılır. Hash-doğrulanmış tamamlanmış checkpointler atlanır:

```bash
source .venv/bin/activate
nohup ./run_mac_single_realization_v2_6_2.sh \
  single_realization_results_v2_6_2 \
  >> run_single_realization_v2_6_2.log 2>&1 &
```

Sonuç klasörü, plan JSON'u veya checkpointler elle düzenlenmez. Aynı anda iki
koşucu başlatılırsa PID kilidi ikinci süreci reddeder.

## 5. Bitiş ölçütü

Logun sonunda şu satır bulunmalıdır:

```text
single_realization_run_preflight_analysis=PASS
```

Ayrıca:

- `postrun_preflight_single_realization_v2_6_2.json` içinde
  `all_checks_pass=true`;
- `single_realization_results_v2_6_2.json`;
- `single_realization_contrasts_v2_6_2.csv`;
- tam 3.610 doğrulanmış checkpoint bulunmalıdır.

Sonuç yalnız dondurulmuş tek ağ gerçekleşimi için koşullu mekanistik sonuçtur;
tohumlar arası stokastik genelleme değildir.
