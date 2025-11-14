#!/usr/bin/env python3
"""
Kaleido ID - Система распознавания лиц
Главный файл для запуска приложения
"""

import os
import sys

# Добавляем путь к модулям
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'main_build'))

from main_build.main import main

if __name__ == "__main__":
    main()