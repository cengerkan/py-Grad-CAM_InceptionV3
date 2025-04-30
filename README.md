# Grad-CAM Görselleştirme (InceptionV3 ile)

Bu proje, önceden ImageNet üzerinde eğitilmiş olan InceptionV3 modelini kullanarak bir görüntüde modelin hangi bölgelere dikkat ettiğini Grad-CAM (Gradient-weighted Class Activation Mapping) yöntemi ile görselleştirir.

## 🔍 Amaç

Sinir ağlarının kararlarını daha şeffaf ve anlaşılır hale getirmek için, sınıflandırma işlemi sırasında görüntünün hangi bölümlerinin etkili olduğunu bir ısı haritası (heatmap) ile gözlemlemek.

## 📌 Özellikler

- TensorFlow ve Keras kullanılarak Grad-CAM implementasyonu
- InceptionV3 modelinin ara katmanlarından biri seçilerek ısı haritası oluşturulabilir (`mixed7`, `mixed10`, vs.)
- Görüntünün üstüne Grad-CAM ısı haritası bindirilerek dikkat bölgeleri gösterilir
- Sonuç, OpenCV ile görsel olarak gösterilir

## 🛠 Gereksinimler

- Python 3.x  
- TensorFlow  
- NumPy  
- OpenCV (cv2)

```bash
pip install tensorflow numpy opencv-python
