# v2.6.2 makale analiz protokolü — kilitli sürüm

**Durum:** KİLİTLİ  
**Tarih:** 2026-08-26  
**Kapsam:** Dondurulmuş tek ağ gerçekleşiminin makale için post-run tanımlanmış betimsel ve mekanistik analizleri  
**Bağımsız stokastik gerçekleşim sayısı:** 1 (`seed=601`, `structural_seed=160601`)

Bu protokol, mevcut makalenin beş bilimsel sorusunu A–H deney aşamalarına ve dondurulmuş çıktı alanlarına bağlar. Üretim koşusu tamamlandıktan sonra, geniş A–H sonuç katmanı hesaplanmadan önce ölçüt, eşleştirme, toplulaştırma ve şekil seçimini sabitler. Önceden üretilmiş on-kontrast özeti bilinen bir önkayıt çıktısıdır; yönü ve kapsamı değiştirilmez. Daha önce oluşturulmuş `statistical_analysis_v2_6_2/` ve `outputs/.../CPG_v2_6_2_Statistical_Analysis.xlsx` dosyaları bu yeni ana analizin girdisi değildir; eski on-kontrast odağının arşiv çıktıları olarak korunur.

**Teknik düzeltme 1 (aynı tarih):** Kaynak şema denetiminde, dört saniyenin ham toparlanma olayının izlem ufku değil, yalnız kilitli Family-9 yükünün üst sınırı olduğu doğrulandı. Ana toparlanma analizi bu nedenle dondurulmuş ham `recovery_endpoint_eligible`, `recovery_event_observed`, `recovery_time_s`, `recovery_time_or_censor_s` ve `recovery_censor_time_s` alanlarını kullanır. Dört saniyelik `min(time-or-censor, 4 s)` dönüşümü yalnız önkayıt/duyarlılık kapsülünde kalır. Bu düzeltme karşılaştırma yönüne veya gözlenen etkiye göre yapılmamıştır; kod ve veri sözlüğündeki alan anlamlarını birbirinden ayırır.

## 1. Bilimsel yönelim

### 1.1. Genel hipotez

Ayrıntılı spinal lokomotor ağ modeli; hız, mekanik yük ve bozucu darbe bağlamları boyunca sağ–sol (L–R) ve fleksör–ekstansör (F–E) koordinasyonunu sürdürebilir. Pertürbasyon sonrası toparlanma ve faz kararlılığı, belirli model popülasyonlarına, presinaptik rotalara ve bunların bağlama bağlı etkileşimlerine dayanır.

### 1.2. Mekanistik alt hipotez

Terminal etkinliğine bağlı, mikrotübülle ilişkili fenomenolojik yavaş yenilenme desteği ağ kararlılığına ve toparlanmaya katkıda bulunur. Bu katkı yalnız ortalama destek düzeyine değil, desteğin doğru terminalde ve doğal etkinlikle doğru zamanda eşleşmesine bağlıdır ve hızlı KCa kolundan ayrıştırılabilir.

Modeldeki terminal destek değişkenleri mikrotübülün doğrudan ölçümü değildir. Çalışma mikrotübül biyolojisinin omurilik terminallerinde varlığını kanıtlamaz; açık bir mekanizma varsayımının dondurulmuş ağ içindeki sonuçlarını sınar.

### 1.3. Beş makale sorusu

1. Model, farklı hız ve yük koşullarında kararlı L–R ve F–E koordinasyonu üretebiliyor mu?
2. Uyarıcı ve baskılayıcı darbelerden sonra faz kararlılığı ve ritmik toparlanma nasıl değişiyor?
3. Tekli ve çiftli popülasyon/afferent müdahalelerinin etkileri hız, yük ve darbe türüne göre nasıl değişiyor?
4. Presinaptik rota bozulmaları ve sınıf × rota birliktelikleri hangi devre bağımlılıklarını gösteriyor?
5. Terminal-lokal fenomenolojik destek ağ kararlılığına katkıda bulunuyor mu; katkı doğru terminale, doğru zamana ve KCa'dan bağımsız bir yavaş kola bağlı mı?

## 2. Dondurulmuş veri kaynakları

Kaynak dizin:

`single_realization_results_v2_6_2/` (repository root; read-only)

| Dosya | Rol | Satır/kimlik sözleşmesi | SHA-256 |
|---|---|---|---|
| `analysis_task_index.csv` | Tasarım eksenleri ve görev–simülasyon eşlemesi | 11.686 benzersiz `task_id`; 3.610 `simulation_id` | `d61f30873a1a5c0c3150c58cd97ca0842e2340be281697a7ae447055d94fc095` |
| `metrics.csv` | Görev düzeyinde birleşik metrikler | 11.686 satır; yeniden kullanılan simülasyonlar replika değildir | `aac42c01231cd8b3057ceaba806ca8bd68e207c94c8b500ec302acfe425e72ab` |
| `unique_simulation_metrics.csv` | Fiziksel simülasyon/QC envanteri | 3.610 benzersiz `simulation_id` | `67889dd949276991ea79bca10a67268b7a430a393687b42bfff4a36b19b84ee4` |
| `long_epoch_metrics.csv` | G aşamasının dönem sonuçları | 31 simülasyon × 24 dönem = 744 benzersiz `(simulation_id, epoch)` | `8edeceaf70f143ae5d3c9b6efbfb29f743af76b987a3163f16ae0d6420b83755` |
| `experiment_plan_single_realization_v2_6_2.json` | Dondurulmuş deney kapsamı | A–H aşamaları | `62a3bc1d5cf94e3369c5f3a32f30486960f96b8606580599eb9fb45924889875` |
| `single_realization_contrasts_v2_6_2.csv` | Değiştirilmeden korunacak önkayıt kapsülü | On koşullu MT kontrastı | `b732a60e4ccf715f62c3705d9602b177292bb9fe1a7a53019a18354ffdfabec4` |

İncelenecek makale sürümü: `Makale_Turkce_Govde_English_Tables_Figures_v1_8.docx`, SHA-256 `8e456ebc61ce3076c001b2138675bb0887f3cbc270b995e28a3047634b56d903`.

## 3. Çıkarım sınırı ve analiz birimi

- Bağımsız ağ/gürültü gerçekleşimi sayısı birdir.
- Görev, bağlam, rota, sınıf, hücre, çevrim veya aktarım olayı bağımsız biyolojik ya da stokastik tekrar sayılmaz.
- p-değeri, güven aralığı, standart hata, serbestlik derecesi, bootstrap güven aralığı veya tohumlar arası etki büyüklüğü hesaplanmaz.
- `simulation_id`, teknik tamamlanma ve fiziksel simülasyon sayımı için kullanılır. F aşamasında yeniden kullanılan hücreler faktöriyel tasarımı kurmak için `task_id` ile korunur; bunlar tekrar sayılmaz.
- Atomik bilimsel çıktı, aynı dondurulmuş ağda ve aynı hız × yük × darbe bağlamında eşlenmiş koşul farkıdır.
- Medyan, IQR, minimum–maksimum ve yön sayıları yalnız önceden belirlenmiş tasarım ızgarasının betimsel sıkıştırmasıdır; örnekleme belirsizliği değildir.
- Hiçbir sonuç tek bir global faz puanına veya genel geçme–kalma kararına indirgenmez.

## 4. Sonlanım hiyerarşisi

### 4.1. Ana ağ-kararlılığı sonlanımları

1. `lr_phase_error_mean_abs_deg`
2. `fe_phase_error_mean_abs_deg`
3. L–R ve F–E faz kaymaları: ham `slip_count / cycle_count` çiftleri; oranlar yalnız bu sayılarla birlikte verilir
4. `rhythmic_failure`
5. Darbe sonrası toparlanma: `recovery_endpoint_eligible`, olay gözlendi/gözlenmedi, gözlenen süre ve tam izlem sonundaki sağ-sansür bilgisi

Faz kayması eşiği modelde dondurulduğu biçimiyle `|faz hatası| > 45°` olarak korunur.

### 4.2. İkincil performans sonlanımları

- `frequency_hz` ve `rg_cycle_interval_cv_mean`
- `bilateral_amplitude_imbalance`
- PF ve MN missed/anchor çiftleri ile türetilen ağ-yayılım açığı
- `rg_pf_latency_mean_ms` ve `rg_mn_latency_mean_ms`

Yorumsal metinde “RG→MN aktarımı” yerine “RG boşalımı–MN çıktısı ağ yayılımı” denir; kod/alan adı olduğu yerde aynen korunur.

### 4.3. Mekanistik açıklayıcı sonlanımlar

- On model popülasyonunun `*_mean_rate_hz` alanları
- `mt_*_mean`, `rrp_*_mean` ve `replenishment_resource_*_mean`
- G aşamasındaki route-specific epoch durumları

Bu alanlar mekanizma bağını açıklamak için kullanılır; tek başına işlevsel başarı veya mikrotübül biyolojisi kanıtı sayılmaz.

### 4.4. Kullanılmayacak alanlar

`global_phase_error`, `left_right_phase_error` gibi legacy convenience alanları açık L–R/F–E ölçümlerinin yerine kullanılmaz. Boş alanlar sıfır yapılmaz.

## 5. Başarısızlık ve eksiklik kuralları

1. `scientific_valid=True` ve `technical_valid=1` önkoşuldur. Teknik geçersizlik dışlanır; sayısı ve gerekçesi ayrıca raporlanır.
2. `rhythmic_failure`, yetersiz darbe-öncesi çevrim ve faza uygun darbenin verilememesi biyolojik ağ sonucudur; teknik dışlama değildir.
3. Her eşlenmiş karşılaştırmada önce durum geçişi verilir:
   - ritmik → ritmik,
   - ritmik → başarısız,
   - başarısız → ritmik,
   - başarısız → başarısız.
4. Sürekli fark yalnız her iki koşulda ölçüm tanımlı ve ritim mevcutsa hesaplanır; satır `complete_pair=1` olarak işaretlenir.
5. Toparlanma iki parçalı raporlanır: dondurulmuş tam post-pulse izleminde olay gözlendi/gözlenmedi ve toparlananlarda gözlenen süre. Sağ sansürlü koşullarda `time_or_censor` ile `censor_time` birlikte taşınır; bunların bağlam hücreleri bağımsız survival örnekleri gibi analiz edilmez.
6. Sıfır anchor, ritmik başarısızlıkla birlikteyse biyolojik başarısızlık; aksi durumda tanımsız ölçüm olarak işaretlenir.
7. Önceden kilitlenen en-kötü-yük dönüşümleri yalnız duyarlılık/önkayıt kapsülünde kullanılır: faz 180°, CV veya imbalance 1, sıfır-denominatör olasılık yükü π/2 ve `min(time-or-censor, 4 s)` toparlanma yükü. Bunlar ana ham sonuçların yerine geçirilmez.

## 6. Ortak fark yönü ve özetleme

Ana performans ölçümleri “yüksek = kötü” biçiminde tutulur. Faz hatası, kayma oranı, CV, amplitüd dengesizliği ve missed-transfer oranı için:

`ΔY = Y_müdahale − Y_eşlenmiş_referans`

- `ΔY > 0`: dondurulmuş bağlam içinde bozulma
- `ΔY < 0`: dondurulmuş bağlam içinde iyileşme
- `|ΔY| ≤ 1e-12`: yalnız sayısal olarak nötr

Bu işaretler istatistiksel önem veya biyolojik eşik değildir. Hız × yük hücrelerine eşit ağırlık verilir; çevrim, anchor veya görev sayısıyla ana ağırlıklandırma yapılmaz. Pulse yönleri önce ayrı raporlanır. Anchor-pooled oranlar yalnız duyarlılık çıktısı olabilir.

On kilitli MT kontrastı kendi özgün `dynamic − static_matched` tanımını ve “negatif = önceden olumlu yön” sözleşmesini değiştirmeden korur. Bu özel sözleşme geniş lokomotor analizlere taşınmaz.

## 7. A–H analiz sözleşmesi

### A — Sağlam ağ ve aktif darbeler

- Darbesiz temel ağ: 3 hız × 3 yük = 9 hücre.
- Hız ve yük boyunca L–R/F–E faz hataları, kayma sayı/oranları, frekans, CV, motor dengesizliği, PF/MN ağ-yayılım açıkları ve ritmik başarısızlık ayrı gösterilir.
- Uyarıcı/baskılayıcı aktif darbe: aynı hız ve yükteki `pulse=none` koşulunun yönle eşleşen sham penceresine karşılaştırılır:
  - excitatory `post_pulse_*` − no-pulse `sham_excitatory_*`
  - inhibitory `post_pulse_*` − no-pulse `sham_inhibitory_*`
- Toparlanma uygunluğu, olay durumu ve süre/sansür ayrıca verilir.

### B — Tekli popülasyon ve afferent müdahaleleri

On müdahale (`V1Ia`, `V1Ren`, `V2b`, `V2a`, `V0D`, `V0V`, `V3`, `Ia`, `Ib`, `groupI`) aynı hız × yük × pulse A koşuluna eşlenir. Darbesiz temel kararlılık ile aktif-darbe toparlanması ayrı panellerde tutulur.

### C — Çiftli müdahaleler

Altı önceden tanımlanmış çift hem A intact koşuluna hem iki eşlenmiş tekli B koluna karşılaştırılır. Complete-pair ölçümlerde nonadditivite:

`N_AB = Y_AB − Y_A − Y_B + Y_0`

Pozitif değer additif beklentiden daha yüksek yük, negatif değer daha düşük yük anlamına gelir; popülasyon düzeyinde istatistiksel etkileşim iddiası değildir.

### D — Hıza bağlı katılım

Normal yük altında hız × pulse boyunca sınıf ateşleme profilleri gösterilir. İlgili tekli/çiftli müdahalenin intact etkisinin yüksek hız ile düşük hız arasındaki değişimi exact difference-in-differences olarak verilir. Ateşleme hızları sonuç açıklayıcıdır; hücreler bağımsız örnek sayılmaz.

### E — Presinaptik rota bozulmaları

On rota bozulması aynı hız × yük × pulse A koşuluna eşlenir. Ana faz/toparlanma sonuçları ile ağ-yayılım ve terminal durumları ayrı tutulur.

### F — Sınıf × rota 2×2 matrisi

Her sınıf `c`, rota `r` ve bağlam `k` için `A0M0`, `A1M0`, `A0M1`, `A1M1` kolları `task_id` ve label üzerinden kurulur:

`I(c,r,k) = Y11 − Y10 − Y01 + Y00`

Pozitif değer birleşik müdahalenin additif beklentiden daha kötü olduğunu gösterir. Diyagonal 10 hücre ve off-diyagonal 90 hücre tam olarak gösterilir; aralarındaki özet fark betimseldir ve inferans içermez. Aynı `simulation_id`'nin tasarım içinde yeniden görünmesi replika sayılmaz.

### G — Uzun bileşik nöral, sinaptik ve mekanik stres

“Yüksek vezikül talebi” tek başına kullanılmaz; protokol bileşik nöral, sinaptik ve mekanik stres olarak adlandırılır.

- Dönem 1: erken başlangıç/denge gösterimi, özet pencereye alınmaz.
- Dönem 2–6: dondurulmuş başlangıç penceresi.
- Dönem 7–12: challenge öncesi bileşik stres.
- Dönem 13–18: dışsal terminal-kaynak challenge'ı altında bileşik stres.
- Dönem 19: geçiş; gösterilir, özet pencereye alınmaz.
- Dönem 20–24: dondurulmuş toparlanma penceresi.

Her rota için challenge × impairment farkların farkı hesaplanır. Faz, kayma, başarısızlık, ağ yayılımı ve terminal durumları ayrı gösterilir. Raw anchor/missed/matched sayıları dönüştürülmüş yüklerden ayrı korunur.

### H — Terminal desteğinin zaman/konum özgüllüğü ve KCa ayrıştırması

Aynı fast/KCa modu ve pulse içinde MT kolları karşılaştırılır:

- dynamic − static_matched: çevrimiçi dinamiğin ortalama-eşlenmiş desteğe göre etkisi
- dynamic − time_yoked: doğal çevrimiçi zaman eşleşmesinin etkisi
- dynamic − spatial_shuffled: doğru terminal konum eşleşmesinin etkisi
- dynamic − impaired ve dynamic − off: destek kolunun büyüklük/varlık kontrolleri

Aynı MT modu ve pulse içinde `dynamic`, `static_mean`, `yoked`, `off` KCa kolları karşılaştırılır. Ardından MT kontrastlarının KCa modları arasındaki exact difference-in-differences değerleri hesaplanır. Doğru zaman veya doğru terminal özgüllüğü yalnız ilgili kontrolün dynamic kola göre işlevsel sonlanımı değiştirmesi halinde, bu tek gerçekleşime koşullu olarak yorumlanır.

### Önceden kilitlenen on-kontrast kapsülü

`single_realization_contrasts_v2_6_2.csv` hiçbir tanımı değiştirilmeden yeniden üretilir ve Supplementary Table S10'da tam verilir. Ana metinde yalnız kısa bir dürüst özet ve H/G mekanizma haritasıyla uyumu/uyumsuzluğu belirtilir. 10/10 genel başarı kuralı bütün modelin ölçütü yapılmaz; olumlu, olumsuz ve nötr sonuçlar aynen korunur.

## 8. Sonuçlar bölümü için kilitli sıra

1. **3.1.** Dondurulmuş deney matrisi, teknik tamamlanma ve biyolojik başarısızlık sınıflaması
2. **3.2.** Sağlam ağda hız ve yük boyunca lokomotor ritim ve faz kararlılığı
3. **3.3.** Uyarıcı ve baskılayıcı darbelerden sonra faz bozulması ve toparlanma
4. **3.4.** Tekli model-popülasyonu ve afferent müdahaleleri
5. **3.5.** Çiftli müdahaleler, devre yedekliliği ve nonadditivite
6. **3.6.** Hıza bağlı interneuron katılımı ve bağlam modifikasyonu
7. **3.7.** Presinaptik terminal-rotası bozulmalarının etkileri
8. **3.8.** Sınıf × rota faktöriyel devre bağımlılıkları
9. **3.9.** Uzun bileşik stres, terminal-kaynak challenge'ı ve toparlanma
10. **3.10.** Terminal desteğinin zaman/konum özgüllüğü, KCa'dan ayrışması ve önkayıt kapsülüyle ilişki

## 9. Şekil ve tablo sözleşmesi

Kavramsal Şekil 1–3 korunur. Yeni ana şekiller:

- **Şekil 4:** A — 3×3 hız/yük faz-kararlılığı haritaları
- **Şekil 5:** A — active pulse–sham eşlenmiş farkları ve toparlanma
- **Şekil 6:** B–D — tekli/çiftli müdahale ile hız-katılım haritaları
- **Şekil 7:** E–F — rota etkileri ve 10×10 sınıf–rota nonadditivite matrisi
- **Şekil 8:** G–H — dönem seyirleri ile zaman/konum/KCa kontrolleri

Ana sonuç tabloları:

- **R1:** A–H görev/benzersiz simülasyon envanteri, geçerlilik ve analiz kuralları
- **R2:** Sağlam ağın dokuz darbesiz hız×yük hücresi ve pulse özeti
- **R3:** Tekli ve çiftli müdahalelerin bağlam-eşlenmiş özetleri
- **R4:** On presinaptik rota etkisi
- **R5:** F sınıf/rota marjinal ve nonadditif özetleri
- **R6:** G–H mekanistik kontrol özeti

Ek materyal; tüm atomik bağlam hücrelerini, başarısızlık geçişlerini, D recruitment değerlerini, F'nin 100×27 matrisini, G'nin 744 dönem satırını, H'nin 72 koşulunu ve on-kontrast kapsülünü içerir.

## 10. Değişiklik yasağı ve doğrulama

Sonuçlar okunduktan sonra aşağıdakiler değiştirilemez:

- beş bilimsel sorunun sırası,
- ana/ikincil/mekanistik sonlanım ayrımı,
- eşlenmiş referanslar,
- pulse–sham eşleştirmesi,
- B/C/E/F/G/H kontrast formülleri,
- başarısızlık ve complete-pair kuralları,
- ana şekil ve tablo kapsamı,
- tek-gerçekleşim çıkarım sınırı.

Her türetilmiş dosya; kaynak hashlerini, script sürüm/hashini, üretim zamanını ve satır sayısını taşıyacaktır. Analiz sonunda şu denetimler zorunludur:

1. 11.686 görev ve 3.610 benzersiz simülasyon tam eşleşmesi.
2. 744 G dönem satırı ve `(simulation_id, epoch)` benzersizliği.
3. `matched + missed = anchor` eşitlikleri.
4. F'de her sınıf × rota × bağlam için dört kolun tamlığı.
5. H'de 4 KCa × 6 MT × 3 pulse = 72 koşulun tamlığı.
6. Üretilen hiçbir tabloda p/CI/SE/df alanı bulunmaması.
7. Aynı `simulation_id`nin replika sayılmaması.
8. Ana sonuç ile duyarlılık/önkayıt çıktılarının ayrı etiketlenmesi.
