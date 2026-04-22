# Суреттегі нысандарды анықтау және классификациялау (CIFAR-10)
### Image Classification using CNN — CIFAR-10 Dataset

> **Курстық жоба** · «Нейрондық желілерді бағдарламалау» пәні  
> Жәңгір хан атындағы БҚАТУ · ИС-24 тобы  
> Орындаған: **Мурат Данат**  
> Жетекші: Б. Анаргуль

---

## Жоба туралы / About

Python тілінде жазылған, **CIFAR-10** деректер жиынында оқытылған **CNN** нейрондық желісі арқылы суреттерді 10 класстың біріне автоматты классификациялайтын қолданба.

Бағдарлама **қазақша/орысша** интерфейспен, офлайн режімде жұмыс жасайды және **.exe** орындалатын файлға айналдырылған.

---

## 10 Класс / Classes

| № | Ағылшынша | Қазақша | Орысша |
|---|-----------|---------|--------|
| 0 | airplane | Ұшақ | Самолёт |
| 1 | automobile | Автомобиль | Автомобиль |
| 2 | bird | Құс | Птица |
| 3 | cat | Мысық | Кошка |
| 4 | deer | Бұғы | Олень |
| 5 | dog | Ит | Собака |
| 6 | frog | Бақа | Лягушка |
| 7 | horse | Жылқы | Лошадь |
| 8 | ship | Кеме | Корабль |
| 9 | truck | Жүк көлігі | Грузовик |

---

## Нейрондық желі архитектурасы / Architecture

```
Input (32×32×3)
    ↓
Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout(0.2)
    ↓
Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout(0.3)
    ↓
Conv2D(128) → BatchNorm → MaxPool → Dropout(0.4)
    ↓
Flatten → Dense(256) → BatchNorm → Dropout(0.5)
    ↓
Dense(10, Softmax) → Нәтиже / Result
```

**Оқыту нәтижесі / Training results:**
- Train accuracy: ~83%
- Val accuracy: ~76%
- Эпохалар / Epochs: 20
- Batch size: 64
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy

---

## Орнату / Installation

### Талаптар / Requirements
- Python 3.9 немесе 3.10
- RAM: 4 ГБ минимум
- Диск: 4 ГБ

### 1. Репозиторийді клондау

```bash
git clone https://github.com/YOUR_USERNAME/cifar10-classification.git
cd cifar10-classification
```

### 2. Кітапханаларды орнату

```bash
pip install -r requirements.txt
```

### 3. Қолданбаны іске қосу

```bash
python app.py
```

> Бірінші іске қосуда CIFAR-10 деректер жиыны автоматты жүктеледі (~163 МБ) және модель оқытылады (~5–10 минут). Кейінгі іске қосуларда модель бірден жүктеледі.

---

## .exe файлын жасау / Build executable

### Windows
```bash
pyinstaller --onefile --windowed --name CIFAR10_Classifier ^
  --add-data="cifar10_cnn.h5;." ^
  --add-data="cifar10_norm.npy;." ^
  --exclude-module PySide6 ^
  --exclude-module PyQt5 app.py
```

### macOS
```bash
pyinstaller --onefile --windowed --name CIFAR10_Classifier \
  --add-data="cifar10_cnn.h5:." \
  --add-data="cifar10_norm.npy:." \
  --exclude-module PySide6 \
  --exclude-module PyQt5 app.py
```

Нәтиже / Result: `dist/CIFAR10_Classifier.exe` (Windows) немесе `dist/CIFAR10_Classifier.app` (macOS)

---

## Файлдар құрылымы / Project structure

```
cifar10-classification/
├── app.py               — Негізгі қолданба (CNN + GUI)
├── requirements.txt     — Python кітапханалары
├── build_exe.py         — .exe жасау скрипті
├── README.md            — Осы файл
├── cifar10_cnn.h5       — Оқытылған модель (автожасалады)
└── cifar10_norm.npy     — Нормализация параметрлері (автожасалады)
```

---

## Қолданылған технологиялар / Tech stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange)
![Keras](https://img.shields.io/badge/Keras-API-red)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-green)

| Технология | Нұсқасы | Мақсаты |
|------------|---------|---------|
| Python | 3.9–3.10 | Негізгі тіл |
| TensorFlow/Keras | 2.10+ | CNN жасау және оқыту |
| NumPy | 1.23+ | Массивтер |
| Pillow | 9.0+ | Суреттерді өңдеу |
| Tkinter | Стандартты | Графикалық интерфейс |
| PyInstaller | 5.0+ | .exe жасау |

---

## Лицензия / License

MIT License — еркін қолдануға болады.

---

*Жәңгір хан атындағы Батыс Қазақстан аграрлық-техникалық университеті · 2026*
