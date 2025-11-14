#!/usr/bin/env python3
"""
Проверка зависимостей и системы для Kaleido ID
"""

import sys
import subprocess
import importlib
import pkg_resources

def check_python_version():
    """Проверка версии Python"""
    print("🔍 Проверка версии Python...")
    if sys.version_info < (3, 7):
        print(f"❌ Требуется Python 3.7 или выше, установлена {sys.version}")
        return False
    else:
        print(f"✅ Python {sys.version}")
        return True

def check_package(package_name, import_name=None):
    """Проверка наличия пакета"""
    if import_name is None:
        import_name = package_name
        
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name}")
        return True
    except ImportError:
        print(f"❌ {package_name} не установлен")
        return False

def install_package(package_name):
    """Установка пакета"""
    print(f"📦 Установка {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ {package_name} успешно установлен")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Ошибка установки {package_name}")
        return False

def check_system_requirements():
    """Проверка системных требований"""
    print("🔍 Проверка системных требований...")
    
    requirements = [
        ("opencv-python", "cv2"),
        ("mediapipe", "mediapipe"),
        ("Pillow", "PIL"),
        ("numpy", "numpy"),
        ("tkinter", "tkinter")  # Обычно входит в стандартную поставку Python
    ]
    
    missing_packages = []
    
    for package, import_name in requirements:
        if not check_package(package, import_name):
            missing_packages.append(package)
    
    return missing_packages

def main():
    """Основная функция проверки"""
    print("🎭 Kaleido ID - Проверка системы")
    print("=" * 50)
    
    # Проверка версии Python
    if not check_python_version():
        print("\n❌ Обновите Python до версии 3.7 или выше")
        return False
    
    print("\n🔍 Проверка необходимых пакетов...")
    missing_packages = check_system_requirements()
    
    if missing_packages:
        print(f"\n⚠️  Отсутствуют пакеты: {', '.join(missing_packages)}")
        response = input("Хотите установить отсутствующие пакеты? (y/n): ")
        if response.lower() == 'y':
            for package in missing_packages:
                if not install_package(package):
                    print(f"\n❌ Не удалось установить {package}")
                    return False
            print("\n✅ Все пакеты успешно установлены!")
        else:
            print("\n❌ Установите отсутствующие пакеты вручную:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    else:
        print("\n✅ Все необходимые пакеты установлены!")
    
    # Проверка доступности камеры
    print("\n🔍 Проверка доступности камеры...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Камера доступна")
            cap.release()
        else:
            print("⚠️  Камера не доступна (но это не критично)")
    except Exception as e:
        print(f"⚠️  Ошибка проверки камеры: {e}")
    
    print("\n🎉 Проверка системы завершена успешно!")
    print("\n🚀 Запуск приложения...")
    return True

if __name__ == "__main__":
    if main():
        # Запуск приложения
        try:
            from main_build.main import main as app_main
            app_main()
        except Exception as e:
            print(f"❌ Ошибка запуска приложения: {e}")
            print("\n💡 Попробуйте переустановить зависимости:")
            print("pip install opencv-python mediapipe Pillow numpy")
    else:
        print("\n❌ Проверка системы не пройдена")
        sys.exit(1)