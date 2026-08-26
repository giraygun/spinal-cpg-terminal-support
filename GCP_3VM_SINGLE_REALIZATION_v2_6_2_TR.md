# Google Cloud üç-VM tek-gerçekleşim protokolü — v2.6.2

## Kilitli koşu

- Biyolojik model: değişmemiş v2.6.1 çekirdeği
- Model SHA-256: `a0dc8a7338ab1619874135b1a3e8809f4eaa22394cb65dfd951544df5b62f47a`
- Seed: `601`; yapısal seed: `160601`
- A–H analiz görevi: `11686`
- Benzersiz simülasyon: `3610`
- Üç shard: sırasıyla `1203`, `1204`, `1203` simülasyon
- Shard iş birimleri: `501120000`, `501200000`, `500800000`
- Shard atama SHA-256: `a5b74691019f44f0f950c663b94a268d3996f22e1639b2deeb3e32bfedf7f654`

Bu tasarım tek ağ gerçekleşimindeki mekanizma sınamasıdır. Tohumlar arası
istatistiksel genelleme üretmez.

## Google Cloud yapılandırması

Mevcut 32-vCPU genel ve 24-vCPU C4D-aile kotasına uygun plan üç adet
`c4d-highcpu-8` Spot VM'dir. Her VM Ubuntu 24.04 LTS, en az 100 GB balanced
persistent disk ve kesilmede `STOP` ayarı kullanır. Her shard için varsayılan
işçi sayısı 7'dir; kısa bilimsel-olmayan ölçüm bellek yeterliliğini gösterirse
8 yapılabilir.

## Komut sırası

Paketi her üç VM'de aynı konuma açtıktan sonra:

```bash
bash cloud_v2_6_2/bootstrap_vm.sh
```

Manifest yalnız koordinatör VM'de ve sonuçlar görülmeden önce oluşturulur:

```bash
sudo mkdir -p /opt/cpg/production
sudo chown -R "$USER":"$USER" /opt/cpg/production
.venv/bin/python distributed_single_realization_v2_6_2.py plan \
  --out /opt/cpg/production/SINGLE_REALIZATION_SHARDS_v2_6_2.json \
  --production-root /opt/cpg/production \
  --shards 3
```

Aynı manifest byte-byte değiştirilmeden diğer VM'lere kopyalanır. Sonra her
VM kendi indeksini çalıştırır:

```bash
bash cloud_v2_6_2/run_shard.sh 0 /opt/cpg/production/SINGLE_REALIZATION_SHARDS_v2_6_2.json 7
bash cloud_v2_6_2/run_shard.sh 1 /opt/cpg/production/SINGLE_REALIZATION_SHARDS_v2_6_2.json 7
bash cloud_v2_6_2/run_shard.sh 2 /opt/cpg/production/SINGLE_REALIZATION_SHARDS_v2_6_2.json 7
```

Spot kesintisinde aynı diskli VM yeniden başlatılır ve aynı komut verilir;
hash-doğrulanmış checkpoint'ler yeniden hesaplanmaz. Üç shard tamamlanıp
koordinatör VM'nin `/opt/cpg/production/shards/` klasöründe toplandıktan sonra:

```bash
bash cloud_v2_6_2/finalize_coordinator.sh \
  /opt/cpg/production/SINGLE_REALIZATION_SHARDS_v2_6_2.json
```

Birleştirme komutu checkpoint bütünlüğü, seed, yapısal seed, model SHA'sı,
11.686 görev indeksi ve 3.610 benzersiz simülasyonu fail-closed denetler; sonra
on betimsel mekanistik kontrastı üretir.

## Yasaklar

- Model dosyası, manifest, seed veya görev matrisi üretim başladıktan sonra
  değiştirilemez.
- Rota, hücre veya bağlam satırları bağımsız tekrar gibi kullanılamaz.
- p-değeri veya güven aralığı eklenemez.
- Preflight PASS olmadan sonuç dosyası yorumlanamaz.
- Spot VM'ler koşu ve dosya aktarımı tamamlanınca durdurulmadan bırakılmaz.
