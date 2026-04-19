"""
╔══════════════════════════════════════════════════════════════════╗
║   Суреттегі нысандарды анықтау және классификациялау (CIFAR-10)  ║
║   Обнаружение и классификация объектов на изображениях           ║
║   Автор: Курмангазиева Нурдана                                    ║
╚══════════════════════════════════════════════════════════════════╝

Зависимости:
    pip install tensorflow pillow numpy tkinter

Запуск:
    python app.py
"""

import tkinter as tk
from tkinter import filedialog, ttk
import threading
import os
import sys
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFilter

# ──────────────────────────────────────────────────────────────────
#  ЦВЕТА И КОНСТАНТЫ
# ──────────────────────────────────────────────────────────────────
BG_DARK     = "#0A0E1A"
BG_CARD     = "#111827"
BG_CARD2    = "#1A2235"
ACCENT_BLUE = "#3B82F6"
ACCENT_GRN  = "#22C55E"
ACCENT_RED  = "#EF4444"
ACCENT_YELL = "#F59E0B"
TEXT_WHITE  = "#F1F5F9"
TEXT_GRAY   = "#64748B"
TEXT_LIGHT  = "#94A3B8"

CLASSES_KZ = {
    "airplane":    "✈️  Ұшақ (Самолёт)",
    "automobile":  "🚗  Автомобиль",
    "bird":        "🐦  Құс (Птица)",
    "cat":         "🐱  Мысық (Кошка)",
    "deer":        "🦌  Бұғы (Олень)",
    "dog":         "🐶  Ит (Собака)",
    "frog":        "🐸  Бақа (Лягушка)",
    "horse":       "🐴  Жылқы (Лошадь)",
    "ship":        "🚢  Кеме (Корабль)",
    "truck":       "🚛  Жүк көлігі (Грузовик)",
}
CLASSES_LIST = list(CLASSES_KZ.keys())

BAR_COLORS = [
    "#3B82F6", "#22C55E", "#F59E0B", "#EF4444", "#8B5CF6",
    "#EC4899", "#14B8A6", "#F97316", "#6366F1", "#84CC16",
]

# ──────────────────────────────────────────────────────────────────
#  МОДЕЛЬ CNN
# ──────────────────────────────────────────────────────────────────
MODEL_PATH = "cifar10_cnn.h5"

def build_model():
    """Строим простую CNN для CIFAR-10."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Conv2D, MaxPooling2D, BatchNormalization,
                                          Dropout, Flatten, Dense)
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
        BatchNormalization(),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        Dropout(0.2),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        Dropout(0.3),

        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.4),

        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_and_save(progress_callback=None):
    """Обучаем CNN на CIFAR-10 и сохраняем модель."""
    from tensorflow.keras.datasets import cifar10
    from tensorflow.keras.callbacks import Callback

    class ProgressCB(Callback):
        def on_epoch_end(self, epoch, logs=None):
            if progress_callback:
                acc = logs.get('val_accuracy', 0) * 100
                progress_callback(epoch + 1, acc)

    if progress_callback:
        progress_callback(0, 0, status="Датасет жүктелуде... (Загрузка датасета)")

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32')  / 255.0

    # Нормализация
    mean = x_train.mean(axis=(0,1,2))
    std  = x_train.std(axis=(0,1,2)) + 1e-7
    x_train = (x_train - mean) / std
    x_test  = (x_test  - mean) / std

    if progress_callback:
        progress_callback(0, 0, status="Модель жасалуда... (Создание модели)")

    model = build_model()

    if progress_callback:
        progress_callback(0, 0, status="Оқыту басталды... (Обучение начато) — ~5-10 мин")

    model.fit(
        x_train, y_train,
        epochs=20,
        batch_size=64,
        validation_data=(x_test, y_test),
        callbacks=[ProgressCB()],
        verbose=0,
    )

    model.save(MODEL_PATH)
    # Сохраняем нормализацию
    np.save("cifar10_norm.npy", np.array([mean, std]))

    if progress_callback:
        progress_callback(20, 0, status="✅ Модель сақталды! (Модель сохранена)")

    return model, mean, std


def load_model_and_norm():
    """Загружаем сохранённую модель."""
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH)
    norm  = np.load("cifar10_norm.npy")
    return model, norm[0], norm[1]


def predict_image(model, mean, std, img_path):
    """Предсказание для одного изображения."""
    img = Image.open(img_path).convert("RGB").resize((32, 32))
    arr = np.array(img).astype('float32') / 255.0
    arr = (arr - mean) / (std + 1e-7)
    arr = np.expand_dims(arr, 0)
    preds = model.predict(arr, verbose=0)[0]
    top_idx  = int(np.argmax(preds))
    top_conf = float(preds[top_idx])
    return CLASSES_LIST[top_idx], top_conf, preds


# ──────────────────────────────────────────────────────────────────
#  ВСПОМОГАТЕЛЬНЫЕ UI-ФУНКЦИИ
# ──────────────────────────────────────────────────────────────────
def rounded_rect_image(w, h, r, color):
    """Создаёт PIL-изображение скруглённого прямоугольника."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d   = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, w-1, h-1], radius=r, fill=color)
    return img


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


# ──────────────────────────────────────────────────────────────────
#  ГЛАВНОЕ ОКНО
# ──────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Суреттегі нысандарды анықтау — CIFAR-10")
        self.geometry("820x700")
        self.minsize(700, 600)
        self.configure(bg=BG_DARK)
        self.resizable(True, True)

        self.model  = None
        self.mean   = None
        self.std    = None
        self.img_path = None
        self._photo = None   # держим ссылку чтобы не удалился GC

        self._build_ui()
        self._check_model_on_start()

    # ── ПОСТРОЕНИЕ ИНТЕРФЕЙСА ─────────────────────────────────────
    def _build_ui(self):
        # ── Заголовок
        hdr = tk.Frame(self, bg=BG_CARD, height=70)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="🔍  Суреттегі нысандарды анықтау",
                 font=("Segoe UI", 16, "bold"),
                 bg=BG_CARD, fg=TEXT_WHITE).pack(side="left", padx=24, pady=18)

        self.status_lbl = tk.Label(hdr, text="● Дайын (Готов)",
                                   font=("Segoe UI", 11),
                                   bg=BG_CARD, fg=ACCENT_GRN)
        self.status_lbl.pack(side="right", padx=24)

        # ── Основная область
        main = tk.Frame(self, bg=BG_DARK)
        main.pack(fill="both", expand=True, padx=20, pady=16)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # ── Левая колонка: загрузка + превью
        left = tk.Frame(main, bg=BG_DARK)
        left.grid(row=0, column=0, sticky="nsew", padx=(0,10))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        # Кнопка выбора файла
        self.upload_btn = tk.Button(
            left,
            text="📂   Суретті таңдаңыз\n(Выбрать изображение)",
            font=("Segoe UI", 13, "bold"),
            bg=ACCENT_BLUE, fg="white",
            activebackground="#2563EB", activeforeground="white",
            relief="flat", cursor="hand2",
            bd=0, padx=16, pady=14,
            command=self._choose_image,
        )
        self.upload_btn.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        # Превью изображения
        preview_frame = tk.Frame(left, bg=BG_CARD, bd=0)
        preview_frame.grid(row=1, column=0, sticky="nsew")

        self.canvas_img = tk.Canvas(preview_frame,
                                    bg=BG_CARD2, highlightthickness=0)
        self.canvas_img.pack(fill="both", expand=True, padx=2, pady=2)
        self._draw_placeholder()

        # Кнопка классифицировать
        self.classify_btn = tk.Button(
            left,
            text="🧠   Классификациялау  (Классифицировать)",
            font=("Segoe UI", 12, "bold"),
            bg=ACCENT_GRN, fg="white",
            activebackground="#16A34A", activeforeground="white",
            relief="flat", cursor="hand2",
            bd=0, padx=16, pady=12,
            state="disabled",
            command=self._run_predict,
        )
        self.classify_btn.grid(row=2, column=0, sticky="ew", pady=(12, 0))

        # ── Правая колонка: результаты
        right = tk.Frame(main, bg=BG_DARK)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # Заголовок результата
        tk.Label(right, text="Нәтиже / Результат",
                 font=("Segoe UI", 13, "bold"),
                 bg=BG_DARK, fg=TEXT_LIGHT).grid(row=0, column=0, sticky="w", pady=(0,8))

        # Карточка результата
        self.result_frame = tk.Frame(right, bg=BG_CARD, bd=0)
        self.result_frame.grid(row=1, column=0, sticky="nsew")
        self._draw_empty_result()

        # ── Нижняя панель: прогресс обучения
        bot = tk.Frame(self, bg=BG_CARD, height=46)
        bot.pack(fill="x", side="bottom")
        bot.pack_propagate(False)

        self.progress_lbl = tk.Label(bot, text="",
                                     font=("Segoe UI", 10),
                                     bg=BG_CARD, fg=TEXT_GRAY)
        self.progress_lbl.pack(side="left", padx=16, pady=12)

        self.progress_bar = ttk.Progressbar(bot, length=200, mode="determinate")
        self.progress_bar.pack(side="right", padx=16, pady=14)

        # Стиль прогрессбара
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TProgressbar",
                        troughcolor=BG_CARD2,
                        background=ACCENT_BLUE,
                        thickness=6)

        # Обучить снова
        self.train_btn = tk.Button(
            bot, text="⚙ Қайта оқыту (Обучить заново)",
            font=("Segoe UI", 9),
            bg=BG_CARD2, fg=TEXT_GRAY,
            activebackground=BG_CARD, activeforeground=TEXT_WHITE,
            relief="flat", cursor="hand2", bd=0, padx=10,
            command=self._start_training,
        )
        self.train_btn.pack(side="right", padx=4, pady=10)

    # ── PLACEHOLDER ПРЕВЬЮ ───────────────────────────────────────
    def _draw_placeholder(self):
        self.canvas_img.delete("all")
        w = self.canvas_img.winfo_width()  or 340
        h = self.canvas_img.winfo_height() or 300
        self.canvas_img.create_text(
            w//2, h//2,
            text="🖼\n\nСуретті таңдаңыз\n(Выберите изображение)",
            font=("Segoe UI", 13),
            fill=TEXT_GRAY,
            justify="center",
        )

    # ── ПУСТОЙ РЕЗУЛЬТАТ ─────────────────────────────────────────
    def _draw_empty_result(self):
        for w in self.result_frame.winfo_children():
            w.destroy()
        tk.Label(self.result_frame,
                 text="\n\n🔍\n\nСуретті жүктеп,\nклассификациялау батырмасын\nбасыңыз\n\n"
                      "(Загрузите фото и нажмите\n«Классифицировать»)",
                 font=("Segoe UI", 12),
                 bg=BG_CARD, fg=TEXT_GRAY,
                 justify="center").pack(expand=True, fill="both")

    # ── ВЫБОР ИЗОБРАЖЕНИЯ ────────────────────────────────────────
    def _choose_image(self):
        path = filedialog.askopenfilename(
            title="Суретті таңдаңыз",
            filetypes=[("Суреттер / Изображения", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not path:
            return
        self.img_path = path
        self._show_preview(path)
        if self.model:
            self.classify_btn.config(state="normal")

    def _show_preview(self, path):
        self.canvas_img.update_idletasks()
        cw = max(self.canvas_img.winfo_width(),  300)
        ch = max(self.canvas_img.winfo_height(), 260)

        img = Image.open(path).convert("RGB")
        img.thumbnail((cw - 16, ch - 16), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)

        self.canvas_img.delete("all")
        self.canvas_img.create_image(cw//2, ch//2,
                                     anchor="center",
                                     image=self._photo)

    # ── КЛАССИФИКАЦИЯ ────────────────────────────────────────────
    def _run_predict(self):
        if not self.img_path or not self.model:
            return
        self.classify_btn.config(state="disabled")
        self.status_lbl.config(text="⏳ Анықталуда...", fg=ACCENT_YELL)
        self.update_idletasks()

        def task():
            cls, conf, preds = predict_image(
                self.model, self.mean, self.std, self.img_path)
            self.after(0, lambda: self._show_result(cls, conf, preds))

        threading.Thread(target=task, daemon=True).start()

    def _show_result(self, cls, conf, preds):
        self.status_lbl.config(text="● Дайын (Готов)", fg=ACCENT_GRN)
        self.classify_btn.config(state="normal")

        for w in self.result_frame.winfo_children():
            w.destroy()

        rf = self.result_frame
        rf.config(bg=BG_CARD)

        # Основной класс
        conf_pct = conf * 100
        conf_color = ACCENT_GRN if conf_pct >= 60 else (ACCENT_YELL if conf_pct >= 35 else ACCENT_RED)

        tk.Label(rf, text="Негізгі нәтиже / Главный результат",
                 font=("Segoe UI", 10), bg=BG_CARD, fg=TEXT_GRAY
                 ).pack(anchor="w", padx=14, pady=(14,2))

        top_frame = tk.Frame(rf, bg=BG_CARD2)
        top_frame.pack(fill="x", padx=12, pady=(0,10))

        tk.Label(top_frame, text=CLASSES_KZ[cls],
                 font=("Segoe UI", 15, "bold"),
                 bg=BG_CARD2, fg=TEXT_WHITE
                 ).pack(side="left", padx=14, pady=12)

        tk.Label(top_frame,
                 text=f"{conf_pct:.1f}%",
                 font=("Segoe UI", 20, "bold"),
                 bg=BG_CARD2, fg=conf_color
                 ).pack(side="right", padx=14, pady=12)

        # Разделитель
        tk.Frame(rf, bg=BG_CARD2, height=1).pack(fill="x", padx=12, pady=2)

        # Все 10 классов с барами
        tk.Label(rf, text="Барлық класстар / Все классы:",
                 font=("Segoe UI", 10), bg=BG_CARD, fg=TEXT_GRAY
                 ).pack(anchor="w", padx=14, pady=(8,4))

        bars_frame = tk.Frame(rf, bg=BG_CARD)
        bars_frame.pack(fill="both", expand=True, padx=12, pady=(0,12))

        sorted_idx = np.argsort(preds)[::-1]
        for rank, idx in enumerate(sorted_idx):
            c   = CLASSES_LIST[idx]
            pct = preds[idx] * 100
            color = BAR_COLORS[idx]
            is_top = (idx == int(np.argmax(preds)))

            row = tk.Frame(bars_frame, bg=BG_CARD)
            row.pack(fill="x", pady=2)

            # Иконка + название
            name_lbl = tk.Label(row,
                                 text=CLASSES_KZ[c],
                                 font=("Segoe UI", 9, "bold" if is_top else "normal"),
                                 bg=BG_CARD,
                                 fg=TEXT_WHITE if is_top else TEXT_LIGHT,
                                 width=22, anchor="w")
            name_lbl.pack(side="left")

            # Полоска прогресса
            bar_bg = tk.Frame(row, bg=BG_CARD2, height=14)
            bar_bg.pack(side="left", fill="x", expand=True, padx=(4,6))
            bar_bg.pack_propagate(False)

            bar_fill = tk.Frame(bar_bg, bg=color, height=14)
            bar_fill.place(x=0, y=0, relwidth=pct/100, height=14)

            # Процент
            tk.Label(row,
                     text=f"{pct:.1f}%",
                     font=("Segoe UI", 9, "bold" if is_top else "normal"),
                     bg=BG_CARD,
                     fg=color if is_top else TEXT_GRAY,
                     width=6, anchor="e"
                     ).pack(side="right")

    # ── ПРОВЕРКА МОДЕЛИ ПРИ ЗАПУСКЕ ──────────────────────────────
    def _check_model_on_start(self):
        if os.path.exists(MODEL_PATH) and os.path.exists("cifar10_norm.npy"):
            self.status_lbl.config(text="⏳ Модель жүктелуде...", fg=ACCENT_YELL)
            def load():
                try:
                    m, mn, st = load_model_and_norm()
                    self.after(0, lambda: self._on_model_ready(m, mn, st))
                except Exception as e:
                    self.after(0, lambda: self._on_model_error(str(e)))
            threading.Thread(target=load, daemon=True).start()
        else:
            self.status_lbl.config(text="⚠ Модель жоқ — оқыту керек", fg=ACCENT_YELL)
            self._start_training()

    def _on_model_ready(self, model, mean, std):
        self.model = model
        self.mean  = mean
        self.std   = std
        self.status_lbl.config(text="● Дайын (Готов)", fg=ACCENT_GRN)
        self.progress_lbl.config(text="✅ Модель жүктелді (Модель загружена)")
        if self.img_path:
            self.classify_btn.config(state="normal")

    def _on_model_error(self, err):
        self.status_lbl.config(text="❌ Қате (Ошибка)", fg=ACCENT_RED)
        self.progress_lbl.config(text=f"Ошибка: {err}")

    # ── ОБУЧЕНИЕ ─────────────────────────────────────────────────
    def _start_training(self):
        self.train_btn.config(state="disabled")
        self.classify_btn.config(state="disabled")
        self.status_lbl.config(text="🔄 Оқытылуда... (Обучение)", fg=ACCENT_YELL)
        self.progress_bar["value"] = 0

        def progress_cb(epoch, val_acc, status=None):
            if status:
                self.after(0, lambda s=status: self.progress_lbl.config(text=s))
            else:
                pct = int(epoch / 20 * 100)
                msg = f"Эпоха {epoch}/20 — val_acc: {val_acc:.1f}%"
                self.after(0, lambda p=pct, m=msg: (
                    self.progress_bar.config(value=p),
                    self.progress_lbl.config(text=m),
                ))

        def train():
            try:
                m, mn, st = train_and_save(progress_cb)
                self.after(0, lambda: self._on_training_done(m, mn, st))
            except Exception as e:
                self.after(0, lambda: self._on_model_error(str(e)))

        threading.Thread(target=train, daemon=True).start()

    def _on_training_done(self, model, mean, std):
        self.model = model
        self.mean  = mean
        self.std   = std
        self.progress_bar["value"] = 100
        self.status_lbl.config(text="● Дайын (Готов)", fg=ACCENT_GRN)
        self.progress_lbl.config(text="✅ Оқыту аяқталды! (Обучение завершено)")
        self.train_btn.config(state="normal")
        if self.img_path:
            self.classify_btn.config(state="normal")


# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()
