"""
build_exe.py — Сборка .exe через PyInstaller

Использование:
    python build_exe.py
"""
import subprocess, sys, os

def build():
    # Устанавливаем PyInstaller если нет
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    cmd = [
        "pyinstaller",
        "--onefile",                   # Один .exe файл
        "--windowed",                  # Без консоли
        "--name", "CIFAR10_Classifier",
        "--add-data", "cifar10_cnn.h5;.",   # Включить модель (Windows)
        "--add-data", "cifar10_norm.npy;.", # Включить нормализацию
        "app.py",
    ]

    print("=" * 55)
    print("  Сборка .exe файла...")
    print("  Бұл бірнеше минут алуы мүмкін (Займёт несколько минут)")
    print("=" * 55)

    subprocess.check_call(cmd)

    print("\n✅ Готово! .exe находится в папке dist/")
    print("   dist/CIFAR10_Classifier.exe")

if __name__ == "__main__":
    # Сначала нужно обучить модель
    if not os.path.exists("cifar10_cnn.h5"):
        print("⚠️  Сначала запустите app.py чтобы обучить модель!")
        print("   После обучения запустите этот скрипт снова.")
    else:
        build()
