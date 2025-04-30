import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# InceptionV3 modelini yükleyin
base_model = InceptionV3(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('mixed10').output)
last_conv_layer_name = "block14_sepconv2_act"  # Kullanmak istediğiniz konvolüsyon katmanının adı

# Grad-CAM uygulama fonksiyonu
def apply_grad_cam(model, img_array, last_conv_layer_name):
    # Grad-CAM için kullanılacak konvolüsyon katmanının adı
    last_conv_layer = model.get_layer(last_conv_layer_name)

    # Grad-CAM modeli oluşturma
    grad_model = Model([model.inputs], [last_conv_layer.output, model.output])

    # Giriş görüntüsüne ve tahmin sınıfına ihtiyaç vardır
    with tf.GradientTape() as tape:
        last_conv_layer_output, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    # Gradyanları ve özellik haritasını elde etme
    grads = tape.gradient(loss, last_conv_layer_output)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]

    # Grad-CAM haritasını oluşturma
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Giriş görüntüsünü Grad-CAM ile ağırlıklı olarak birleştirme
    img = img_array[0]
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return superimposed_img

# Örnek bir görüntü yükleyin (uygun bir görüntü yolunu güncelleyin)
img_path = '../Car/train/car/105.jpg'
img = image.load_img(img_path, target_size=(299, 299))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

last_conv_layer_name = "mixed7"  # Denemek için bir mixed katman adı
result_img = apply_grad_cam(model, img_array, last_conv_layer_name)

# Sonucu görselleştir
cv2.imshow('Original Image', cv2.imread(img_path))
cv2.imshow('Grad-CAM Result', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
