# TCGA COAD MSI vs MSS Prediction
## Sindirim Sistemi Kanseri MSI/MSS Tahmini

Hastalardan alınan, MSI/MSS etiketli doku örnekleri ile eğitilen projemiz; girdisi yapılan dokuyu test eder. Test sonucunda dokunun MSI mı MSS mi olduğu belirlenir. Bu şekilde hastalara doğru bir tedavi süreci geçekleştirmeye olanak sağlar.

1. [Veri seti](https://www.kaggle.com/purpleberrie/train-tcga-coad-msi-mss) temin edildi.
2. [Önişleme aşamaları](./project/preprocess.py) gerçekleştirildi.
3. CNN modeli [oluşturuldu](./project/main.py) ve [eğitildi](./project/main.py).

> Bu projede makine öğrenmesi kullanılmıştır.