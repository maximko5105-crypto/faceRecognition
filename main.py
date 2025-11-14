import tkinter as tk
import logging
import os
import sys
from .database import FaceDatabase
from .face_recognizer import MediaPipeFaceRecognizer
from .gui import KaleidoIDGUI

def setup_logging():
    """Настройка логирования"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'kaleido_id.log'), encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Основная функция приложения"""
    try:
        # Настройка логирования
        setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("Starting Kaleido ID Application")
        
        # Инициализация базы данных
        database = FaceDatabase()
        
        # Инициализация распознавателя лиц
        face_recognizer = MediaPipeFaceRecognizer()
        
        # Загрузка эмбеддингов из базы данных
        loaded_count = face_recognizer.load_embeddings_from_database(database)
        logger.info(f"Loaded {loaded_count} embeddings from database")
        
        # Создание графического интерфейса
        root = tk.Tk()
        app = KaleidoIDGUI(root, database, face_recognizer)
        
        # Обработка закрытия приложения
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        logger.info("GUI initialized successfully")
        
        # Запуск основного цикла
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()