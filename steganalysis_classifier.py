import os
import numpy as np
import cv2
import glob
import warnings
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import random
import string
from stegano import lsb
from scipy.fftpack import dct, idct
from PIL import Image
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)


def fridrich_features(image_path):              #улучшенная функция извлечения признаков
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Не удалось загрузить изображение: {image_path}")
            return None

        features = []

        cropped = img[4:, 4:]                       #основные признаки Фридриха
        h, w = img.shape
        h_crop, w_crop = cropped.shape

        dct_orig = np.zeros((h, w))
        dct_crop = np.zeros((h_crop, w_crop))

        for i in range(0, h - 8, 8):
            for j in range(0, w - 8, 8):
                block = img[i:i + 8, j:j + 8].astype(np.float32) - 128
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                dct_orig[i:i + 8, j:j + 8] = dct_block

        for i in range(0, h_crop - 8, 8):
            for j in range(0, w_crop - 8, 8):
                block = cropped[i:i + 8, j:j + 8].astype(np.float32) - 128
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                dct_crop[i:i + 8, j:j + 8] = dct_block

        quantized_orig = np.round(dct_orig / 2.0) * 2.0
        quantized_crop = np.round(dct_crop / 2.0) * 2.0

        def calc_hist(dct_coeffs):
            ac_coeffs = dct_coeffs.copy()
            ac_coeffs[::8, ::8] = 0  # Удаляем DC-коэффициенты
            hist, _ = np.histogram(ac_coeffs.flatten(), bins=128, range=(-64, 64))
            return hist / hist.sum() if hist.sum() > 0 else hist

        hist_orig = calc_hist(quantized_orig)
        hist_crop = calc_hist(quantized_crop)
        features.extend(np.abs(hist_orig - hist_crop)[:23])


        features.append(np.mean(img))               #дополнительные статистические признаки
        features.append(np.std(img))
        features.append(np.median(img))

        fft = np.fft.fft2(img)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log(np.abs(fft_shift) + 1)
        features.append(np.mean(magnitude))
        features.append(np.std(magnitude))

        hist, _ = np.histogram(img.ravel(), 256, [0, 256])              #признаки энтропии
        hist_norm = hist / hist.sum()
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        features.append(entropy)

        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)    #признаки текстуры
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        features.append(np.mean(sobelx))
        features.append(np.mean(sobely))
        features.append(np.std(sobelx))
        features.append(np.std(sobely))

        return np.array(features)

    except Exception as e:
        print(f"Ошибка при извлечении признаков из {image_path}: {str(e)}")
        return None




def verify_stego_embedding(input_path, stego_path):             #проверка качества стегоизображений
    try:

        secret_message = lsb.reveal(stego_path)

        if secret_message is None:
            print(f"Ошибка: сообщение не найдено в {os.path.basename(stego_path)}")
            return False

        orig_img = Image.open(input_path)                   #проверка размеров изображений
        stego_img = Image.open(stego_path)

        if orig_img.size != stego_img.size:
            print(f"Размеры изображений различаются: оригинал {orig_img.size}, стего {stego_img.size}")
            return False

        orig_arr = np.array(orig_img.convert('L'))
        stego_arr = np.array(stego_img.convert('L'))

        diff = orig_arr.astype(np.int16) - stego_arr.astype(np.int16)
        diff_std = np.std(diff)

        if diff_std < 0.1:
            print(f"Слишком малая разница между изображениями: std={diff_std:.4f}")
            return False

        return True

    except Exception as e:
        print(f"Ошибка при проверке стегоизображения: {str(e)}")
        return False



def embed_message(input_path, output_path, embedding_rate=0.4):         #функция встраивания
    try:
        img = Image.open(input_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        width, height = img.size
        total_capacity_bits = width * height * 3
        used_capacity_bits = int(total_capacity_bits * embedding_rate)
        max_chars = used_capacity_bits // 8

        if max_chars <= 0:
            print(f"Изображение слишком маленькое для встраивания: {os.path.basename(input_path)}")
            return False

        message = ''.join(random.choices(
            string.ascii_letters + string.digits + string.punctuation + ' ',
            k=max_chars
        ))

        secret = lsb.hide(input_path, message)
        secret.save(output_path, format='PNG')
        return True

    except Exception as e:
        print(f"Ошибка при встраивании {os.path.basename(input_path)}: {str(e)}")
        return False



def create_stego_images(clean_dir, stego_dir, embedding_rate=0.4):          #создание стегоизображений с проверкой
    clean_images = glob.glob(os.path.join(clean_dir, "*.jpg")) + \
                   glob.glob(os.path.join(clean_dir, "*.jpeg")) + \
                   glob.glob(os.path.join(clean_dir, "*.png"))

    if not clean_images:
        print("Не найдено изображений в указанной папке")
        return False

    os.makedirs(stego_dir, exist_ok=True)
    success_count = 0
    verification_failed = 0
    embedding_failed = 0

    print(f"\nСоздание стегоизображений из {len(clean_images)} чистых изображений...")
    for img_path in clean_images:
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(stego_dir, f"{name}_stego.png")

        if embed_message(img_path, output_path, embedding_rate):
                                                            #проверяем качества встраивания
            if verify_stego_embedding(img_path, output_path):
                success_count += 1
            else:
                verification_failed += 1
                os.remove(output_path)                      #удаление некачественного стего
        else:
            embedding_failed += 1

    print(f"\nРезультаты создания стегоизображений:")
    print(f"Успешно создано: {success_count}/{len(clean_images)}")
    print(f"Провалено проверок: {verification_failed}")
    print(f"Не удалось создать: {embedding_failed}")

    return success_count > 0




def extract_features_and_labels(clean_dir, stego_dir):              #извлечение признаков и меток
    clean_images = glob.glob(os.path.join(clean_dir, "*.jpg")) + \
                   glob.glob(os.path.join(clean_dir, "*.jpeg")) + \
                   glob.glob(os.path.join(clean_dir, "*.png"))
    stego_images = glob.glob(os.path.join(stego_dir, "*_stego.png"))

    if not clean_images or not stego_images:
        print("Не найдены изображения в указанных папках")
        return None, None

    features, labels = [], []

    print("\nИзвлечение признаков для чистых изображений...")
    for img_path in clean_images:
        feats = fridrich_features(img_path)
        if feats is not None:
            features.append(feats)
            labels.append(0)

    print("Извлечение признаков для стегоизображений...")
    for img_path in stego_images:
        feats = fridrich_features(img_path)
        if feats is not None:
            features.append(feats)
            labels.append(1)

    if not features:
        print("Ошибка, потому что не удалось извлечь признаки ни из одного изображения")
        return None, None

    print(f"\nОбработано изображений: {len(labels)}")
    print(f"Чистые: {labels.count(0)}, Стего: {labels.count(1)}")
    return np.array(features), np.array(labels)



def train_model(features, labels, k=5):                     #обучение модели евклидово
    if len(features) == 0 or len(labels) == 0:
        print("Ошибка: недостаточно данных для обучения")
        return None, None, 0.0

    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nОбучение KNN (k={k}, metric=euclidean)...")
    model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\nРезультаты классификации:")
    print(classification_report(y_test, y_pred, target_names=['чистый', 'стего']))
    print(f"Точность модели: {accuracy:.4f}")


    cm = confusion_matrix(y_test, y_pred)                   #вывод матрицы ошибок
    print("\nМатрица ошибок:")
    print("               Предсказано")
    print("               Чистые  Стего")
    print(f"Истинные Чистые  {cm[0, 0]}      {cm[0, 1]}")
    print(f"         Стего    {cm[1, 0]}      {cm[1, 1]}")

    return model, scaler, accuracy



def load_model(model_path):         #загрузка модели
    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        print(f"Модель загружена (точность: {model_data['accuracy']:.4f}, k={model_data['k']})")
        return model_data
    except Exception as e:
        print(f"Ошибка при загрузке модели: {str(e)}")
        return None


def predict_image(model_data, image_path):          #предсказание для одного изображения
    try:
        feats = fridrich_features(image_path)
        if feats is None:
            return None

        scaler = model_data["scaler"]
        features_scaled = scaler.transform([feats])
        model = model_data["model"]
        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]

        return prediction, proba
    except Exception as e:
        print(f"Ошибка при предсказании: {str(e)}")
        return None



def run_classifier_experiments(clean_dir, stego_dir, model_dir):                #эксперимент по оценке эффективности при разных k и метрик качества
    print("\n[Эксперимент: оценка эффективности при разных k и метриках]")

    embedding_rate = 0.4
    stego_subdir = os.path.join(stego_dir, f"embedding_{embedding_rate}")
    os.makedirs(stego_subdir, exist_ok=True)

    if not create_stego_images(clean_dir, stego_subdir, embedding_rate):
        return

    features, labels = extract_features_and_labels(clean_dir, stego_subdir)
    if features is None:
        print("Не удалось извлечь признаки. Эксперимент прерван.")
        return

    k_values = [1, 3, 5, 7, 9, 11]
    metrics = ['euclidean', 'manhattan', 'chebyshev']
    results = []

    print("\nПроведение экспериментов с разными k и метриками...")
    for metric in metrics:
        for k in k_values:
            print(f"\nОбучение модели с k={k}, metric={metric}")

            scaler = StandardScaler()
            X = scaler.fit_transform(features)
            y = labels

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            model = KNeighborsClassifier(n_neighbors=k, metric=metric)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)


            results.append({
                'k': k,
                'metric': metric,
                'accuracy': accuracy
            })
            print(f"Точность: {accuracy:.4f}")

    print("\nРезультаты эксперимента:")
    df = pd.DataFrame(results)
    print(df)

    results_path = os.path.join(model_dir, "experiment_results.csv")                #сохранение результатов в файл
    df.to_csv(results_path, index=False)
    print(f"\nРезультаты сохранены в: {results_path}")

    return df







def main():
    clean_dir = input("Введите путь к папке с чистыми изображениями: ").strip()
    stego_dir = input("Введите путь к папке для стегоизображений: ").strip()
    model_dir = input("Введите путь к папке для сохранения моделей: ").strip()

    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(stego_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    model_exists = any(fname.endswith('.pkl') for fname in os.listdir(model_dir))

    while True:

        print("1. Создать стегоизображения")
        print("2. Обучить новую модель KNN")
        if model_exists:
            print("3. Загрузить существующую модель")
            print("4. Протестировать модель на изображении")
        print("5. Эксперимент: оценка разных k и метрик")
        print("6. Выход")

        choice = input("\nВыберите действие: ").strip()

        if choice == "1":
            print("\n[Создание стегоизображений]")
            embedding_rate = float(input("Введите степень встраивания (0.1-0.5): ") or "0.4")
            if create_stego_images(clean_dir, stego_dir, embedding_rate):
                print("Стегоизображения успешно созданы!")
            else:
                print("Ошибка при создании стегоизображений")

        elif choice == "2":
            print("\n[Обучение новой модели KNN]")
            k = int(input("Введите количество соседей (k): ") or "5")
            features, labels = extract_features_and_labels(clean_dir, stego_dir)

            if features is None or labels is None:
                print("Ошибка: не удалось извлечь признаки!")
                continue

            model, scaler, accuracy = train_model(features, labels, k)
            if model is None: continue

            model_name = input("Введите имя для сохранения модели (без расширения): ").strip() or "knn_stego_model"
            model_path = os.path.join(model_dir, f"{model_name}.pkl")

            model_data = {
                "model": model,
                "scaler": scaler,
                "accuracy": accuracy,
                "k": k
            }

            with open(model_path, "wb") as f:
                pickle.dump(model_data, f)

            print(f"\nМодель KNN (k={k}, metric=euclidean) сохранена в: {model_path}")
            model_exists = True

        elif choice == "3" and model_exists:
            print("\n[Загрузка модели]")
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
            if not model_files:
                print("Модели не найдены")
                continue

            for i, fname in enumerate(model_files, 1):
                print(f"{i}. {fname}")

            model_idx = int(input("Выберите номер модели: ").strip()) - 1
            if model_idx < 0 or model_idx >= len(model_files):
                print("Неверный выбор")
                continue

            model_path = os.path.join(model_dir, model_files[model_idx])
            model_data = load_model(model_path)

        elif choice == "4" and model_exists:
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
            if not model_files: continue

            for i, fname in enumerate(model_files, 1):
                print(f"{i}. {fname}")

            model_idx = int(input("Выберите номер модели: ").strip()) - 1
            if model_idx < 0 or model_idx >= len(model_files): continue

            model_path = os.path.join(model_dir, model_files[model_idx])
            model_data = load_model(model_path)
            if model_data is None: continue

            image_path = input("Введите путь к тестовому изображению: ").strip()
            result = predict_image(model_data, image_path)

            if result:
                prediction, proba = result
                class_name = "стегоизображение" if prediction == 1 else "чистое изображение"
                print(f"\nРезультат предсказания: {class_name}")
                print(f"Вероятность (чистое): {proba[0]:.4f}")
                print(f"Вероятность (стего): {proba[1]:.4f}")
            else:
                print("Не удалось выполнить предсказание")

        elif choice == "5":
            print("\nЭксперимент: оценка разных k и метрик")
            run_classifier_experiments(clean_dir, stego_dir, model_dir)

        elif choice == "6":
            print("\nЗавершение работы программы.")
            break

        else:
            print("\nНеверный выбор. Пожалуйста, попробуйте снова.")











if __name__ == "__main__":
    main()