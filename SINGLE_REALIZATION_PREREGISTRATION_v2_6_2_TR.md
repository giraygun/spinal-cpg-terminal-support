# Tek ağ gerçekleşimi için önkayıt — v2.6.2

## Değişmez bilimsel karar

Üretim koşusu yalnız `seed = 601` ve buna karşılık gelen yapısal
`seed = 160601` ile yapılacaktır. Bu seçim sonuçlar görülmeden önce yapılmıştır;
601, önceki tasarımda birincil üretim aralığının ilk ayrılmış tohumudur.

Model çekirdeği, denklemler, biyolojik parametreler, nöron ve afferent alt
sınıfları, MT rotaları, bağlantılar, müdahaleler, süreler ve integrasyon adımı
değiştirilmemiştir. Yalnız bağımsız stokastik gerçekleşim ekseni bire
indirilmiştir.

## Korunan kapsam

- A–H aşamalarının tamamı korunur.
- F aşamasında 10 sınıf × 10 presinaptik MT rotası × 4 faktöriyel kol × 27
  bağlam = 10.800 görev korunur.
- Toplam 11.686 analiz görevi vardır.
- Simulator girdileri tamamen aynı görevler bir kez çalıştırılır; bu yalnız
  hesaplama tekilleştirmesidir. Sonuçta 3.610 benzersiz simülasyon kalır.
- Karşılaştırılan koşullar aynı seed, aynı yapısal ağ ve tanımlı eşlenmiş
  rastgelelik/yoke kuralları altında değerlendirilir.

## Çıkarım sınırı

Bağımsız stokastik örneklem büyüklüğü birdir. Dolayısıyla:

- p-değeri, güven aralığı, standart hata, serbestlik derecesi veya tohumlar
  arası etki büyüklüğü hesaplanmayacaktır;
- rota, hücre, bağlam ya da görev satırları bağımsız tekrar sayılmayacaktır;
- sonuç, yalnız dondurulmuş tek ağ gerçekleşiminde koşullu mekanistik destek
  veya destek yokluğu olarak raporlanacaktır;
- popülasyon düzeyinde stokastik genelleme, sağlamlık olasılığı ya da farklı
  ağ realizasyonlarında tekrarlanabilirlik iddia edilmeyecektir.

Tek gerçekleşimde bütün on önceden tanımlı kontrastın beklenen yönde olması,
`dondurulmuş gerçekleşim içinde mekanistik destek` olarak adlandırılır; genel
hipotezin istatistiksel doğrulanması veya reddi olarak adlandırılmaz.

## Sonuca gömülmeyi önleyen kural

Seed seçimi veya denklemler, çıktı işaretine göre değiştirilemez. Başarısız,
nötr ya da ters yönlü geçerli sonuçlar aynen korunur. Teknik geçersizlik yalnız
önceden tanımlanmış checkpoint ve gözlenebilirlik kontrolleriyle belirlenir;
biyolojik başarısızlık teknik dışlama nedeni değildir.
