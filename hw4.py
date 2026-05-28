import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from PIL import Image

def main():
    # Загружаем картинку и сразу переводим в чб
    img = Image.open('aaa.png').convert('L')
    img_arr = np.array(img)
    
    # Собираем координаты всех тёмных пикселей
    points = []
    h, w = img_arr.shape
    for i in range(h):
        for j in range(w):
            if img_arr[i, j] < 50:
                points.append([i, j])
                
    if len(points) == 0:
        print("Не нашёл чёрных пикселей")
        return
        
    points = np.array(points)

    # Разделяем цифры через DBSCAN
    # eps=15 подбирается под разрешение картинки
    dbscan = DBSCAN(eps=10, min_samples=5)
    dbscan.fit(points)
    labels = dbscan.labels_
    
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)  # убираем шум
        
    # 4. Вырезаем каждую цифру и приводим к формату 8x8
    digit_features = []
    # Сортируем кластеры слева направо, чтобы порядок цифр совпадал с картинкой
    sorted_labels = sorted(unique_labels, key=lambda l: np.min(points[labels == l, 1]))
    
    for label in sorted_labels:
        cluster = points[labels == label]
        y_min, y_max = int(np.min(cluster[:, 0])), int(np.max(cluster[:, 0]))
        x_min, x_max = int(np.min(cluster[:, 1])), int(np.max(cluster[:, 1]))
        
        # Вырезаем область с цифрой из исходного массива
        crop = img_arr[y_min:y_max+1, x_min:x_max+1]
        
        # Ресайзим до 8x8
        resized = Image.fromarray(crop).resize((8, 8))
        arr_8x8 = np.array(resized)
        
        # В датасете load_digits фон = 0, цифры = до 16
        # У нас фон = 255, цифры 0. Инвертируем и масштабируем
        flat_feature = ((255 - arr_8x8) / 16.0).flatten()
        digit_features.append(flat_feature)
        
    digit_features = np.array(digit_features)
    print(f"Найдено объектов: {len(digit_features)}")

    X, y = load_digits(return_X_y=True)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    # Делаем предсказание на наших вырезанных цифрах
    predictions = clf.predict(digit_features)
    print(f"Распознанные числа: {predictions}")
    
    # Визуализируем резы
    fig, axes = plt.subplots(1, len(predictions), figsize=(10, 2))
    if len(predictions) == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(digit_features[i].reshape(8, 8))
        ax.set_title(f"Распознано: {predictions[i]}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()