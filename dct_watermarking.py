import numpy as np
from scipy.fftpack import dct, idct
from math import log10, sqrt
import os
from PIL import Image
import io
import cv2


class program:
    def __init__(self, Z=2, T=80, K=12, alpha=0.1, block_size=8, arnold_iter=1, target_channel=0):                      #альфа сила встраивания
        self.alpha = alpha
        self.block_size = block_size
        self.target_channel = target_channel
        self.Z = Z
        self.T = T
        self.K = K
        self.arnold_iter = arnold_iter


    def preprocess_image(self, img):
        img = img.convert('RGB')            #конвертация в RGB
        img_cv = np.array(img)              #конвертация в массив np
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        return cv2.resize(img_cv, (512, 512))           #масштабирование

    def postprocess_image(self, img_cv):
                                                                #Конвертация обратно в PIL Image
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)


    def calculate_psnr(self, original, watermarked):
        if original.shape != watermarked.shape:     #изменяет размер оригинального изображения
            original = cv2.resize(original, (watermarked.shape[1], watermarked.shape[0]))

        mse = np.mean((original - watermarked) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse))
        return psnr




    def calculate_ncc(self, original_wm, extracted_wm):
        original = original_wm.convert('L').resize((64, 64))                #изменение размера
        extracted = extracted_wm.convert('L').resize((64, 64))

        original_array = np.array(original).astype(np.float64)                 #преобразование на np массив для точных вычислений
        extracted_array = np.array(extracted).astype(np.float64)

        numerator = np.sum(original_array * extracted_array)            #числитель
        denominator = np.sqrt(np.sum(original_array ** 2) * np.sum(extracted_array ** 2))           #произведение матриц

        return numerator / denominator if denominator != 0 else 0.0





    def arnold_transform(self, img, inverse=False):
        M = img.shape[0]
        transformed = np.copy(img)
        for _ in range(self.arnold_iter):
            temp = np.zeros_like(transformed)
            for i in range(M):          #перебор всех пикселей
                for j in range(M):
                    if inverse:
                        i_orig = (2 * i - j) % M            #обратное преобразование
                        j_orig = (-i + j) % M
                        temp[i_orig, j_orig] = transformed[i, j]
                    else:
                        i_new = (i + j) % M             #прямое
                        j_new = (i + 2 * j) % M
                        temp[i_new, j_new] = transformed[i, j]
            transformed = temp
        return transformed


    def embed(self, host_img, watermark):           #встраивание
        original_host = np.array(host_img.convert('RGB'))                          #сохранение орига для PSNR

        host_cv = self.preprocess_image(host_img)
        wm = np.array(watermark.convert('L'))
        wm = cv2.resize(wm, (64, 64))

        ycrcb = cv2.cvtColor(host_cv, cv2.COLOR_BGR2YCrCb)                  #в цветовое пространство ycrcb
        y_channel = ycrcb[:, :, self.target_channel].astype(np.float32) / 255.0


        for i in range(0, 512, self.block_size):                        #обработка каждого блока, встраивание цвз
            for j in range(0, 512, self.block_size):
                block = y_channel[i:i + self.block_size, j:j + self.block_size]
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

                if i // self.block_size < 64 and j // self.block_size < 64:
                    wm_value = wm[i // self.block_size, j // self.block_size] / 255.0               #значение пикселя
                    dct_block[3, 3] += self.alpha * wm_value                #сила встраивания

                modified_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')          #обратное преобразование
                y_channel[i:i + self.block_size, j:j + self.block_size] = modified_block

        ycrcb[:, :, self.target_channel] = np.clip(y_channel * 255, 0, 255).astype(np.uint8)            #обрезка значений за пределы [0,255]
        result_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        watermarked_pil = self.postprocess_image(result_bgr)


        watermarked_array = np.array(watermarked_pil)           #расчет pnsr
        psnr = self.calculate_psnr(original_host, watermarked_array)

        return watermarked_pil, psnr







    def extract(self, watermarked_img, original_img, original_wm=None):                 #извлечение цвз

        watermarked_cv = self.preprocess_image(watermarked_img)         #предобработка изобр
        original_cv = self.preprocess_image(original_img)

        ycrcb_w = cv2.cvtColor(watermarked_cv, cv2.COLOR_BGR2YCrCb)             #конвертация в YCrCb
        y_w = ycrcb_w[:, :, self.target_channel].astype(np.float32) / 255.0

        ycrcb_o = cv2.cvtColor(original_cv, cv2.COLOR_BGR2YCrCb)                #ориг
        y_o = ycrcb_o[:, :, self.target_channel].astype(np.float32) / 255.0

        extracted_wm = np.zeros((64, 64), dtype=np.float32)         #создание пустой матрицы
        for i in range(0, 512, self.block_size):            #циклы по блокам изобр
            for j in range(0, 512, self.block_size):
                idx_i = i // self.block_size
                idx_j = j // self.block_size

                if idx_i < 64 and idx_j < 64:                               #проверка границ

                    block_w = y_w[i:i + self.block_size, j:j + self.block_size]         #извлечение блоков
                    block_o = y_o[i:i + self.block_size, j:j + self.block_size]

                    dct_w = dct(dct(block_w.T, norm='ortho').T, norm='ortho')
                    dct_o = dct(dct(block_o.T, norm='ortho').T, norm='ortho')

                    extracted_value = (dct_w[3, 3] - dct_o[3, 3]) / self.alpha          #извлеч знач цвз
                    extracted_wm[idx_i, idx_j] = extracted_value * 255                                      #сохранение знач

        extracted_wm = np.clip(extracted_wm, 0, 255).astype(np.uint8)
        extracted_img = Image.fromarray(extracted_wm)                           #создание изображения

        ncc = None
        if original_wm is not None:                         #расчет ncc
            ncc = self.calculate_ncc(original_wm, extracted_img)

        return extracted_img, ncc









def apply_jpeg_compression(img, quality):
    buffer = io.BytesIO()               #выделение оперативки
    img.save(buffer, format='JPEG', quality=quality, subsampling=0)         #сохранение с jpeg сжатием
    buffer.seek(0)                                  #сброс указателя для чтения сначала
    return Image.open(buffer)


def apply_gaussian_noise(img, noise_level=25):
    img_array = np.array(img).astype(np.float32)            #в массив np
    noise = np.random.normal(0, noise_level, img_array.shape)               #рандомно генерирует числа
    return Image.fromarray(np.clip(img_array + noise, 0, 255).astype(np.uint8))



def apply_brightness_change(img, delta=30):
    img_array = np.array(img).astype(np.float32)
    return Image.fromarray(np.clip(img_array + delta, 0, 255).astype(np.uint8))




def run_robustness_experiment(host_path, wm_path):
    parameters = {
        'attacks': [
            {'type': 'none', 'name': 'Без атаки'},              #параметры
            {'type': 'jpeg', 'quality': 90},
            {'type': 'jpeg', 'quality': 75},
            {'type': 'jpeg', 'quality': 50},
            {'type': 'noise', 'level': 15},
            {'type': 'noise', 'level': 30},
            {'type': 'brightness', 'delta': 30},
            {'type': 'brightness', 'delta': -30}
        ]
    }

    results = []

    try:
        host = Image.open(host_path)
        wm = Image.open(wm_path)
        algo = program(alpha=0.1)           #встраивание, извлечение и сохранение

        watermarked, original_psnr = algo.embed(host, wm)
        original_watermarked_array = np.array(watermarked)

        original_host = host.copy()

        for attack in parameters['attacks']:                    #циклы тестирования каждой атаки
            attack_name = "энэн"

            try:
                attack_type = attack['type']
                attack_params = {}
                if attack_type == 'jpeg':
                    attack_params = {'quality': attack['quality']}
                    attack_name = f"JPEG Q{attack['quality']}"
                elif attack_type == 'noise':
                    attack_params = {'noise_level': attack['level']}
                    attack_name = f"Шум {attack['level']}"
                elif attack_type == 'brightness':
                    attack_params = {'delta': attack['delta']}
                    attack_name = f"Яркость {attack['delta']:+}"
                elif attack_type == 'none':
                    attack_name = "Без атаки"

                if attack_type == 'none':       #применение атак
                    attacked = watermarked.copy()
                elif attack_type == 'jpeg':
                    attacked = apply_jpeg_compression(watermarked, **attack_params)
                elif attack_type == 'noise':
                    attacked = apply_gaussian_noise(watermarked, **attack_params)
                elif attack_type == 'brightness':
                    attacked = apply_brightness_change(watermarked, **attack_params)
                else:
                    continue

                                                                                #расчет PSNR после атаки
                attacked_array = np.array(attacked)
                psnr = algo.calculate_psnr(original_watermarked_array, attacked_array)
                psnr_value = f"{psnr:.2f} dB"

                                                                                        #извлечение ЦВЗ
                extracted, ncc = algo.extract(attacked, original_host, wm)
                ncc_value = f"{ncc:.4f}" if ncc is not None else "N/A"

            except Exception as e:
                print(f"Ошибка при обработке атаки {attack}: {str(e)}")
                psnr_value = "Ошибка"
                ncc_value = "Ошибка"

            results.append({
                'Атака': attack_name,           #сохранение и вывод
                'PSNR': psnr_value,
                'NCC': ncc_value
            })

    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")
        return

    print(f"{'Атака':<20} | {'PSNR':<12} | {'NCC':<6}")         #вывод в консоль
    for res in results:
        print(f"{res['Атака']:<20} | {res['PSNR']:<12} | {res['NCC']:<6}")
















def main():
    print("1 - Встраивание ЦВЗ\n2 - Извлечение ЦВЗ\n3 - Эксперимент на робастность")
    choice = input("Выбери режим: ").strip()

    if choice == '1':
        host_path = input("Путь к изображению: ").strip().strip("'\"")
        wm_path = input("Путь к ЦВЗ: ").strip().strip("'\"")
        output_path = input("Куда сохранить результат: ").strip().strip("'\"")

        if not os.path.splitext(output_path)[1]:            #проверка расширения
            output_path += ".png"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            host = Image.open(host_path)
            wm = Image.open(wm_path)
            algo = program(alpha=0.1)           #сила встраивания
            watermarked_img, psnr = algo.embed(host, wm)
            watermarked_img.save(output_path)
            print(f"PSNR: {psnr:.2f} dB")
            print(f"Сохранено в: {output_path}")
        except Exception as e:
            print(f"Ошибка: {str(e)}")


    elif choice == '2':
        wm_path = input("Путь к ЦВЗ: ").strip().strip("'\"")
        original_path = input("Путь к оригиналу (без ЦВЗ): ").strip().strip("'\"")
        watermarked_path = input("Путь к стегоизображению (с ЦВЗ): ").strip().strip("'\"")
        output_path = input("Куда сохранить извлеченный ЦВЗ: ").strip().strip("'\"")

        if not os.path.splitext(output_path)[1]:
            output_path += ".png"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            watermarked_img = Image.open(watermarked_path)
            original_img = Image.open(original_path)
            original_wm = Image.open(wm_path)
            algo = program(alpha=0.1)
            extracted_wm, ncc = algo.extract(watermarked_img, original_img, original_wm)
            extracted_wm.save(output_path)
            print(f"NCC: {ncc:.4f}" if ncc is not None else "NCC не получилось рассчитать")
            print(f"ЦВЗ сохранен в: {output_path}")
        except Exception as e:
            print(f"ошибка при извлечении: {str(e)}")


    elif choice == '3':
        host_path = input("Путь к исходнику: ").strip().strip("'\"")
        wm_path = input("Путь к ЦВЗ: ").strip().strip("'\"")
        run_robustness_experiment(
            host_path=host_path,
            wm_path=wm_path
        )
    else:
        print("Некорректный выбор, введи 1, 2 или 3.")








if __name__ == "__main__":
    main()