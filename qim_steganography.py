import numpy as np
from PIL import Image
import math
import os
import random
import matplotlib
import matplotlib.pyplot as plt
from PIL import ImageFilter, ImageEnhance
import tempfile

matplotlib.use('TkAgg')     #используем бэкенд, поддерживающий многопоточность

def text_to_bits(text):
    byte_array = text.encode('utf-8')           #преобразует текст в бинарную строку (UTF-8 кодировка)
    return ''.join(f"{byte:08b}" for byte in byte_array)


def bits_to_text(bits):
    try:
        padded = bits.ljust((len(bits) + 7) // 8 * 8, '0')               #преобразует бинарную строку в текст
        bytes_list = [int(padded[i:i + 8], 2) for i in range(0, len(padded), 8)]
        return bytes(bytes_list).decode('utf-8', errors='replace')
    except Exception as e:
        return f"[Ошибка декодирования] {str(e)}"


def plot_histograms(original, stego, q):
    def _plot(img, title, fig_num):
        plt.figure(fig_num, figsize=(10, 5)) #размеры окон в дюймах
        colors = ('red', 'green', 'blue')
        for ch in range(3):         #перебор цветовых каналов
            plt.hist(img[:, :, ch].ravel(), bins=256,   #bins 256 столбцов по количеству возможных значений яркости 0-255
                     color=colors[ch], alpha=0.5, label=f'Канал {ch}') #альфа 0.5 полупрозрачность для наложения
        plt.title(title)
        plt.xlabel('Яркость')
        plt.ylabel('Частота')
        plt.legend()
        plt.grid(True)
        plt.show(block=False) #неблокирующий показ

    _plot(original, f"Оригинал (q={q})", fig_num=1)     #два отдельных окна одновременно
    _plot(stego, f"Стего (q={q})", fig_num=2)
    plt.show()
    plt.close('all')








def qim_embed(image_path, message_bits, q=4):     #встраивание
    try:
        valid_q = {2, 4, 8, 16, 32, 64}
        if q not in valid_q:            #проверка на допустимые значения
            raise ValueError(f"допустимые значения q: {sorted(valid_q)}")

        full_msg = f"{len(message_bits):032b}" + message_bits  #добавляем 32-битный заголовок с длиной сообщения перед самими битами сообщения

        img = Image.open(image_path).convert('RGB')     #загружаем изображение
        arr = np.array(img)
        h, w, c = arr.shape              #получаем размеры высоты ширины и количество каналов

        max_bits = h * w * 3                     #максимальное количество бит
        if len(full_msg) > max_bits:
            raise ValueError(f"сообщение слишком длинное, максимум можно: {max_bits - 32} бит") #проверка на макс кол-во бит

        modified = []           #лист для хранения обнов пикселей
        idx = 0                             #индекс текущего бита сообщения
        for pixel in arr.reshape(-1):
            if idx < len(full_msg):
                bit = int(full_msg[idx])
                quant = q * (pixel // q)        #вычисляем квантованное значение пикселя
                new_val = quant + (q // 2) * bit     #новое знач пикселя
                modified.append(np.clip(new_val, 0, 255))   #обрезка
                idx += 1
            else:
                modified.append(pixel)

        stego_arr = np.array(modified, dtype=np.uint8).reshape(h, w, c)   #преобразуем список обратно в массив с исходными размерами

        mse = np.mean((arr - stego_arr) ** 2)
        psnr = 20 * math.log10(255 / math.sqrt(mse)) if mse else float('inf')       #расчет метрик

        return Image.fromarray(stego_arr, 'RGB'), psnr, arr, stego_arr

    except Exception as e:
        raise RuntimeError(f"Ошибка встраивания: {str(e)}")






def safe_subtract(a, b):                #вычисление разницы между значениями пикселей
    return abs(int(a) - int(b))






def qim_extract(stego_image, q):                #извлечение сообщения из стегоизображения
    try:
        valid_q = {2, 4, 8, 16, 32, 64}
        if q not in valid_q:
            raise ValueError(f"Допустимые значения q: {sorted(valid_q)}")

        arr = np.array(stego_image)     #преобразование изображения в одномерный массив для последовательной обработки
        flat = arr.reshape(-1)

        len_bits = []
        for v in flat[:32]:
            q0 = min(q * (v // q), 255)         #извлекаем длину сообщения
            q1 = min(q0 + q // 2, 255)
            len_bits.append('0' if safe_subtract(v, q0) < safe_subtract(v, q1) else '1')   #добавляем бит '0' или '1' в список len_bits в зависимости от ближайшего значения квантования.

        msg_len = int(''.join(len_bits), 2)         #опр длины сообщения

        message = []                #извлечение основного сообщения
        for v in flat[32:32 + msg_len]:
            q0 = min(q * (v // q), 255)
            q1 = min(q0 + q // 2, 255)
            message.append('0' if safe_subtract(v, q0) < safe_subtract(v, q1) else '1')

        return ''.join(message)

    except Exception as e:
        raise RuntimeError(f"Ошибка извлечения: {str(e)}")









def get_valid_q_input(prompt):         #ввод правильного шага квантования
    valid_q = {2, 4, 8, 16, 32, 64}
    while True:
        try:
            q_input = input(prompt).strip()
            q = int(q_input) if q_input else 4
            if q in valid_q:
                return q
            print("Ошибка! Допустимые значения: 2, 4, 8, 16, 32, 64")
        except ValueError:
            print("Ошибка, введи число из списка.")





def run_max_capacity_experiment():              #Эксперимент с максимальной загрузкой картинки
    try:

        path = input("Путь к изображению: ").strip().strip("'\"")
        if not os.path.exists(path):        #проверка существования файла
            print(" Файл не найден!")
            return

        q = get_valid_q_input("Шаг квантования (2/4/8/16/32/64) [по умолчанию 4]: ")

        img = Image.open(path).convert('RGB')                   #открытие изображения в формате RGB
        arr = np.array(img)
        h, w, c = arr.shape                      #получение размеров
        max_bits = h * w * 3 - 32           #макс емкость

        print(f" Рассчитанная максимальная емкость: {max_bits} бит ({max_bits // 8 // 1024} КБ)")


        print("Генерация тестовых данных")
        random_bits = ''.join(random.choice('01') for _ in range(max_bits))


        print("Встраивание данных")
        stego_img, psnr, orig_arr, stego_arr = qim_embed(path, random_bits, q)

        save_path = input("\nКуда сохранить стего (или Enter для пропуска): ").strip().strip("'\"")
        if save_path:
            if not save_path.lower().endswith(('.png', '.bmp')):            #проверка расширения
                save_path += ".png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            stego_img.save(save_path)                                   #сохранение
            print(f" стего сохранено: {os.path.abspath(save_path)}")

        print("Проверка целостности данных")
        extracted_bits = qim_extract(stego_img, q)

        errors = sum(1 for a, b in zip(random_bits, extracted_bits) if a != b)              #расчет ошибок

        ber = (errors / len(random_bits)) * 100                                     #расчет BER

        print(" Результаты эксперимента:")
        print(f" Размер изображения: {w}x{h} пикселей")
        print(f" Шаг квантования (q): {q}")
        print(f" Встроено бит: {len(random_bits)}")
        print(f" PSNR: {psnr:.2f} dB")
        print(f" Ошибок при извлечении: {errors} ({ber:.4f}%)")

        if input("Показать гистограммы? (да/нет): ").lower() == 'да':
            plot_histograms(orig_arr, stego_arr, q)

    except Exception as e:
        print(f" Ошибка: {e}")







def run_variable_capacity_experiment():                 #Эксперимент с разным объемом встраиваемых данных
    try:
        path = input("Путь к изображению: ").strip().strip("'\"")
        if not os.path.exists(path):            #проверка существования пути
            print("Файл не найден!")
            return

        q = get_valid_q_input("Шаг квантования (2/4/8/16/32/64) [по умолчанию 4]: ")
        img = Image.open(path).convert('RGB')
        arr = np.array(img)
        h, w, c = arr.shape
        max_bits = h * w * 3 - 32

        percentages = [0.25, 0.50, 0.75, 1.0]              #проценты нагрузки
        results = []

        print("Результаты эксперимента:")
        print("Загрузка   PSNR (dB) MSE")

        for perc in percentages:
            msg_length = int(max_bits * perc)

            random_bits = ''.join(random.choice('01') for _ in range(msg_length))         #генерация тестовых данных

            stego_img, psnr, orig_arr, stego_arr = qim_embed(path, random_bits, q)         #встраивание

            mse = np.mean((orig_arr - stego_arr) ** 2)                  #расчет MSE

            results.append({            #сохранение результатов
                'load_percent': perc,
                'psnr': psnr,
                'mse': mse,
                'orig': orig_arr,
                'stego': stego_arr
            })

            print(f"{perc * 100:6.1f}%   {psnr:8.2f}    {mse:6.2f}")

            save_path = f"stego_q{q}_load{int(perc * 100)}percent.png"                  # сохранение изображения
            stego_img.save(save_path)



        plt.figure(figsize=(10, 5))                 #построение графика
        plt.plot([r['load_percent'] * 100 for r in results],
                 [r['psnr'] for r in results],
                 'bo-')
        plt.title(f"Зависимость PSNR от нагрузки (q={q})")
        plt.xlabel("Загрузка (%)")
        plt.ylabel("PSNR (dB)")
        plt.grid(True)
        plt.show()

        if input("Показать гистограммы для 25% и 100%? (да/нет): ").lower() == 'да':
            plt.ion()                                                           #включение интерактивного режима
            plot_histograms(results[0]['orig'], results[0]['stego'], q)
            plot_histograms(results[-1]['orig'], results[-1]['stego'], q)
            plt.ioff()                                                              #отключение интерактивного режима
            plt.show()                                                      #оставить окна открытыми

    except Exception as e:
        print(f"Ошибка: {e}")









def apply_jpeg_compression(img, quality=50):            #Сжатие JPEG с качеством 50
    with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp:         #создание временного файла в формате JPEG, который автоматически удаляется после выхода из with

        img.save(tmp.name, "JPEG", quality=quality)
        return Image.open(tmp.name).convert("RGB")                          #открывает сжатое изображение из временного файла и конверт в RGB



def add_gaussian_noise(img, intensity=10):          #Добавление гауссова шума
    arr = np.array(img).astype(np.float32)                                     #преобразование изображения в массив
    noise = np.random.normal(0, intensity, arr.shape)           #генерация шума
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)                       #добавление шума к изображению и обрезка значений до диапазона от 0 до 255
    return Image.fromarray(noisy)



def adjust_brightness(img, factor=1.2):         #изменение яркости на 20%
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)



def scale_image(img, scale_factor=0.75):
    w, h = img.size
    scaled = img.resize((int(w*scale_factor), int(h*scale_factor)), Image.BILINEAR)         #уменьшение изображения с коэффициентом scale_factor и сглаживанием
    return scaled.resize((w, h), Image.NEAREST)  #возвращаем Image объект



def run_robustness_experiment():                #эксперимент по робастности
    try:
        path = input("Путь к изображению: ").strip().strip("'\"")
        text = input("Текст для встраивания: ").strip()
        q = get_valid_q_input("Шаг квантования (2/4/8/16/32/64) [по умолчанию 4]: ")

        bits = text_to_bits(text)                                                   #встраивание сообщения
        stego_img, psnr, _, _ = qim_embed(path, bits, q)

        tests = [
            ("Без обработки", lambda x: x),
            ("Сжатие JPEG (quality=80)", lambda img: apply_jpeg_compression(img, 80)),
            ("Сжатие JPEG (quality=50)", lambda img: apply_jpeg_compression(img, 50)),
            ("Гауссов шум (σ=15)", lambda img: add_gaussian_noise(img, 15)),
            ("Яркость +30%", lambda img: adjust_brightness(img, 1.3)),
            ("Масштабирование 75%", lambda img: scale_image(img, 0.75)),
        ]

        print("Результаты эксперимента:")
        print(f"{'Обработка':<25}  {'BER (%)':<8}  {'Текст совпадает'}")

        for name, processor in tests:        #тестирование каждой обработки
            try:
                processed = processor(stego_img)                 #применяем обработку

                extracted_bits = qim_extract(processed, q)                  #извлекаем сообщение
                extracted = extracted_bits[:len(bits)]

                errors = sum(1 for a, b in zip(bits, extracted) if a != b)                  #считаем ошибки
                ber = (errors / len(bits)) * 100
                match = "Да" if bits == extracted else "Нет"

                print(f"{name:<25}  {ber:>7.2f}%  {match}")
            except Exception as e:
                print(f"{name:<25}  ОШИБКА  {str(e)}")

    except Exception as e:
        print(f"ошибка в эксперименте: {str(e)}")









def main():
    menu_items = [
        "1 - Встраивание",
        "2 - Извлечение",
        "3 - Максимальная загрузка",
        "4 - Эксперимент с разной загрузкой",
        "5 - Эксперимент по робастности"
    ]
    print("\n".join(menu_items))

    mode = input("Выбор: ").strip()

    if mode == '1':
        path = input("Путь к изображению: ").strip().strip("'\"")
        if not os.path.exists(path):
            print(" Файл не найден!")
            return

        text = input("Текст для встраивания: ").strip()
        bits = text_to_bits(text)

        q = get_valid_q_input("Шаг квантования (2/4/8/16/32/64) [по умолчанию 4]: ")

        try:
            stego_img, psnr, orig_arr, stego_arr = qim_embed(path, bits, q)
            save_path = input("Куда сохранить: ").strip().strip("'\"")

            if save_path:
                if not save_path.lower().endswith(('.png', '.bmp')):
                    save_path += ".png"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                stego_img.save(save_path)
                print(f" Сохранено: {os.path.abspath(save_path)}")

            print("Результаты:")
            print(f"Длина сообщения: {len(bits)} бит")
            print(f"Качество PSNR: {psnr:.2f} dB")

            if input("Показать гистограммы? (да/нет): ").lower() == 'да':
                plot_histograms(orig_arr, stego_arr, q)

        except Exception as e:
            print(f" Ошибка: {e}")

    elif mode == '2':
        path = input("Путь к стегоизображению: ").strip().strip("'\"")
        if not os.path.exists(path):
            print(" Файл не найден!")
            return

        try:
            q = get_valid_q_input("Шаг квантования (2/4/8/16/32/64): ")
            img = Image.open(path)
            bits = qim_extract(img, q)
            text = bits_to_text(bits)

            print("Результаты извлечения:")
            print(f"Извлечено бит: {len(bits)}")
            print(f"Первые 128 бит: {bits[:128]}...")
            print(f"Декодированный текст:\n{text}")

        except Exception as e:
            print(f" Ошибка: {e}")

    elif mode == '3':
        run_max_capacity_experiment()

    elif mode == '4':
        run_variable_capacity_experiment()

    elif mode == '5':
        run_robustness_experiment()

    else:
        print("Некорректный выбор. Введи число от 1 до 5")












if __name__ == "__main__":
    main()
