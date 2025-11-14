import sqlite3
import os
import json
import shutil
from datetime import datetime
from contextlib import contextmanager
import logging
import cv2

logger = logging.getLogger(__name__)

class FaceDatabase:
    def __init__(self, db_path="data/database.db"):
        self.db_path = db_path
        self.images_dir = "data/face_images"
        
        # Создаем необходимые директории
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        self.init_database()

    @contextmanager
    def get_connection(self):
        """Контекстный менеджер для работы с базой данных"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise e
        finally:
            conn.close()

    def init_database(self):
        """Инициализация структуры базы данных"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Таблица людей с дополнительными полями
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS people (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    last_name TEXT NOT NULL,
                    first_name TEXT NOT NULL,
                    middle_name TEXT,
                    age INTEGER,
                    position TEXT,
                    department TEXT,
                    phone TEXT,
                    email TEXT,
                    address TEXT,
                    notes TEXT,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')
            
            # Таблица фотографий пользователей (один пользователь - много фото)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS person_photos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    image_path TEXT NOT NULL,
                    face_embedding BLOB,
                    is_primary BOOLEAN DEFAULT 0,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (person_id) REFERENCES people (id)
                )
            ''')
            
            # Таблица сессий распознавания
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recognition_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    recognition_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL,
                    FOREIGN KEY (person_id) REFERENCES people (id)
                )
            ''')
            
            # Добавляем новые столбцы если их нет
            for column in ['phone', 'email', 'address']:
                try:
                    cursor.execute(f"ALTER TABLE people ADD COLUMN {column} TEXT")
                except sqlite3.OperationalError:
                    pass  # Столбец уже существует
            
            logger.info("Database initialized successfully")

    def safe_get(self, data, key, default=None):
        """Безопасное получение значения из словаря"""
        value = data.get(key, default)
        return value if value is not None else default

    def add_person(self, person_data):
        """Добавление нового человека в базу данных"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO people 
                    (last_name, first_name, middle_name, age, position, 
                     department, phone, email, address, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    self.safe_get(person_data, 'last_name', '').strip(),
                    self.safe_get(person_data, 'first_name', '').strip(),
                    self.safe_get(person_data, 'middle_name', '').strip(),
                    self.safe_get(person_data, 'age'),
                    self.safe_get(person_data, 'position', '').strip(),
                    self.safe_get(person_data, 'department', '').strip(),
                    self.safe_get(person_data, 'phone', '').strip(),
                    self.safe_get(person_data, 'email', '').strip(),
                    self.safe_get(person_data, 'address', '').strip(),
                    self.safe_get(person_data, 'notes', '').strip()
                ))
                
                person_id = cursor.lastrowid
                logger.info(f"Added person: {self.safe_get(person_data, 'last_name')} {self.safe_get(person_data, 'first_name')} (ID: {person_id})")
                return person_id
                
        except Exception as e:
            logger.error(f"Error adding person: {e}")
            return None

    def update_person(self, person_id, person_data):
        """Обновление данных человека"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE people 
                    SET last_name=?, first_name=?, middle_name=?, age=?, 
                        position=?, department=?, phone=?, email=?, address=?, notes=?,
                        last_updated=CURRENT_TIMESTAMP
                    WHERE id=?
                ''', (
                    self.safe_get(person_data, 'last_name', ''),
                    self.safe_get(person_data, 'first_name', ''),
                    self.safe_get(person_data, 'middle_name', ''),
                    self.safe_get(person_data, 'age'),
                    self.safe_get(person_data, 'position', ''),
                    self.safe_get(person_data, 'department', ''),
                    self.safe_get(person_data, 'phone', ''),
                    self.safe_get(person_data, 'email', ''),
                    self.safe_get(person_data, 'address', ''),
                    self.safe_get(person_data, 'notes', ''),
                    person_id
                ))
                
                success = cursor.rowcount > 0
                if success:
                    logger.info(f"Updated person ID: {person_id}")
                return success
                
        except Exception as e:
            logger.error(f"Error updating person {person_id}: {e}")
            return False

    def add_person_photo(self, person_id, image_path, embedding=None, is_primary=False):
        """Добавление фотографии пользователя"""
        try:
            # Сохраняем изображение
            saved_image_path = self._save_image(image_path, person_id, f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            if not saved_image_path:
                return None
                
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Если это основное фото, снимаем флаг с других фото
                if is_primary:
                    cursor.execute('''
                        UPDATE person_photos SET is_primary=0 WHERE person_id=?
                    ''', (person_id,))
                
                cursor.execute('''
                    INSERT INTO person_photos 
                    (person_id, image_path, face_embedding, is_primary)
                    VALUES (?, ?, ?, ?)
                ''', (person_id, saved_image_path, embedding, is_primary))
                
                photo_id = cursor.lastrowid
                logger.info(f"Added photo for person {person_id}, photo ID: {photo_id}")
                return photo_id
                
        except Exception as e:
            logger.error(f"Error adding person photo: {e}")
            return None

    def add_person_photo_from_memory(self, person_id, image_array, filename_prefix=None):
        """Добавление фотографии пользователя из массива изображения"""
        try:
            # Проверяем входные данные
            if image_array is None or image_array.size == 0:
                logger.error("Empty image array provided")
                return None, None
            
            if person_id is None:
                logger.error("No person ID provided")
                return None, None
            
            # Создаем папку если не существует
            os.makedirs(self.images_dir, exist_ok=True)
            
            if filename_prefix is None:
                filename_prefix = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Генерируем уникальное имя файла
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # с миллисекундами
            filename = f"{filename_prefix}_{person_id}_{timestamp}.jpg"
            saved_path = os.path.join(self.images_dir, filename)
            
            # Проверяем, не существует ли уже файл
            counter = 1
            while os.path.exists(saved_path):
                filename = f"{filename_prefix}_{person_id}_{timestamp}_{counter}.jpg"
                saved_path = os.path.join(self.images_dir, filename)
                counter += 1
                if counter > 100:  # Защита от бесконечного цикла
                    logger.error("Too many duplicate filenames")
                    return None, None
            
            # Сохраняем массив как изображение
            success = cv2.imwrite(saved_path, image_array)
            if not success:
                logger.error(f"Failed to save image to {saved_path}")
                return None, None
            
            # Проверяем, что файл действительно создался
            if not os.path.exists(saved_path):
                logger.error(f"File was not created: {saved_path}")
                return None, None
            
            file_size = os.path.getsize(saved_path)
            if file_size == 0:
                logger.error(f"Empty file created: {saved_path}")
                os.remove(saved_path)  # Удаляем пустой файл
                return None, None
            
            logger.info(f"Image saved successfully: {saved_path} ({file_size} bytes)")
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Проверяем, есть ли уже основное фото
                cursor.execute('SELECT COUNT(*) as count FROM person_photos WHERE person_id=? AND is_primary=1', (person_id,))
                has_primary = cursor.fetchone()['count'] > 0
                
                is_primary = not has_primary  # Если нет основного фото, делаем это основным
                
                cursor.execute('''
                    INSERT INTO person_photos 
                    (person_id, image_path, is_primary)
                    VALUES (?, ?, ?)
                ''', (person_id, saved_path, is_primary))
                
                photo_id = cursor.lastrowid
                logger.info(f"Added photo from memory for person {person_id}, photo ID: {photo_id}, primary: {is_primary}")
                return photo_id, saved_path
                
        except Exception as e:
            logger.error(f"Error adding person photo from memory: {e}")
            # Пытаемся удалить файл если он был создан, но произошла ошибка в БД
            if 'saved_path' in locals() and os.path.exists(saved_path):
                try:
                    os.remove(saved_path)
                    logger.info(f"Removed orphaned image: {saved_path}")
                except Exception as cleanup_error:
                    logger.error(f"Error cleaning up orphaned image: {cleanup_error}")
            return None, None

    def get_person_photos(self, person_id):
        """Получение всех фотографий пользователя"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM person_photos 
                WHERE person_id=? 
                ORDER BY is_primary DESC, created_date DESC
            ''', (person_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_person_photos_by_id(self, photo_id):
        """Получение фотографии по ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT pp.*, p.last_name, p.first_name 
                FROM person_photos pp
                JOIN people p ON pp.person_id = p.id
                WHERE pp.id=?
            ''', (photo_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_primary_photo(self, person_id):
        """Получение основной фотографии пользователя"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM person_photos 
                WHERE person_id=? AND is_primary=1
                LIMIT 1
            ''', (person_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def set_primary_photo(self, photo_id):
        """Установка фотографии как основной"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Получаем person_id для этой фотографии
                cursor.execute('SELECT person_id FROM person_photos WHERE id=?', (photo_id,))
                result = cursor.fetchone()
                if not result:
                    return False
                
                person_id = result['person_id']
                
                # Снимаем флаг со всех фото пользователя
                cursor.execute('UPDATE person_photos SET is_primary=0 WHERE person_id=?', (person_id,))
                
                # Устанавливаем основное фото
                cursor.execute('UPDATE person_photos SET is_primary=1 WHERE id=?', (photo_id,))
                
                success = cursor.rowcount > 0
                if success:
                    logger.info(f"Set photo {photo_id} as primary for person {person_id}")
                return success
                
        except Exception as e:
            logger.error(f"Error setting primary photo: {e}")
            return False

    def delete_photo(self, photo_id):
        """Удаление фотографии"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Получаем путь к изображению для удаления
                cursor.execute('SELECT image_path FROM person_photos WHERE id=?', (photo_id,))
                result = cursor.fetchone()
                if result and result['image_path']:
                    self._delete_image(result['image_path'])
                
                cursor.execute('DELETE FROM person_photos WHERE id=?', (photo_id,))
                
                success = cursor.rowcount > 0
                if success:
                    logger.info(f"Deleted photo ID: {photo_id}")
                return success
                
        except Exception as e:
            logger.error(f"Error deleting photo {photo_id}: {e}")
            return False

    def update_photo_embedding(self, photo_id, embedding):
        """Обновление эмбеддинга для фотографии"""
        try:
            embedding_data = None
            if embedding is not None:
                if hasattr(embedding, 'tolist'):
                    embedding_data = json.dumps(embedding.tolist()).encode('utf-8')
                else:
                    embedding_data = json.dumps(embedding).encode('utf-8')
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE person_photos 
                    SET face_embedding=?
                    WHERE id=?
                ''', (embedding_data, photo_id))
                
                success = cursor.rowcount > 0
                if success:
                    logger.info(f"Updated embedding for photo {photo_id}")
                return success
                
        except Exception as e:
            logger.error(f"Error updating photo embedding: {e}")
            return False

    def delete_person(self, person_id):
        """Мягкое удаление человека"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('UPDATE people SET is_active=0 WHERE id=?', (person_id,))
                
                success = cursor.rowcount > 0
                if success:
                    logger.info(f"Deleted person ID: {person_id}")
                return success
                
        except Exception as e:
            logger.error(f"Error deleting person {person_id}: {e}")
            return False

    def get_person(self, person_id):
        """Получение данных конкретного человека"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM people WHERE id=? AND is_active=1', (person_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_people(self, include_inactive=False):
        """Получение всех записей"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if include_inactive:
                cursor.execute('SELECT * FROM people ORDER BY last_name, first_name')
            else:
                cursor.execute('SELECT * FROM people WHERE is_active=1 ORDER BY last_name, first_name')
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def search_people(self, search_term, include_inactive=False):
        """Поиск людей по различным полям"""
        if not search_term:
            return self.get_all_people(include_inactive)
            
        search_pattern = f'%{search_term}%'
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if include_inactive:
                cursor.execute('''
                    SELECT * FROM people 
                    WHERE (last_name LIKE ? OR first_name LIKE ? OR 
                           middle_name LIKE ? OR position LIKE ? OR 
                           department LIKE ? OR phone LIKE ? OR 
                           email LIKE ? OR notes LIKE ?)
                    ORDER BY last_name, first_name
                ''', (search_pattern, search_pattern, search_pattern,
                      search_pattern, search_pattern, search_pattern,
                      search_pattern, search_pattern))
            else:
                cursor.execute('''
                    SELECT * FROM people 
                    WHERE (last_name LIKE ? OR first_name LIKE ? OR 
                           middle_name LIKE ? OR position LIKE ? OR 
                           department LIKE ? OR phone LIKE ? OR 
                           email LIKE ? OR notes LIKE ?)
                    AND is_active=1
                    ORDER BY last_name, first_name
                ''', (search_pattern, search_pattern, search_pattern,
                      search_pattern, search_pattern, search_pattern,
                      search_pattern, search_pattern))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def add_recognition_session(self, person_id, confidence):
        """Добавление записи о распознавании"""
        try:
            conf_value = 0.0
            if confidence is not None:
                try:
                    conf_value = float(confidence)
                except (TypeError, ValueError):
                    conf_value = 0.0
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO recognition_sessions (person_id, confidence)
                    VALUES (?, ?)
                ''', (person_id, conf_value))
                logger.debug(f"Added recognition session for person {person_id}")
                return True
        except Exception as e:
            logger.error(f"Error adding recognition session: {e}")
            return False

    def get_recognition_stats(self, person_id=None):
        """Получение статистики распознавания"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if person_id:
                cursor.execute('''
                    SELECT COUNT(*) as count, 
                           COALESCE(AVG(confidence), 0) as avg_confidence,
                           MAX(recognition_time) as last_seen
                    FROM recognition_sessions 
                    WHERE person_id=?
                ''', (person_id,))
            else:
                cursor.execute('''
                    SELECT COUNT(*) as count, 
                           COALESCE(AVG(confidence), 0) as avg_confidence
                    FROM recognition_sessions
                ''')
            row = cursor.fetchone()
            if row:
                result = dict(row)
                result['count'] = result.get('count', 0) or 0
                result['avg_confidence'] = float(result.get('avg_confidence', 0.0) or 0.0)
                result['last_seen'] = result.get('last_seen', '')
                return result
            else:
                return {'count': 0, 'avg_confidence': 0.0, 'last_seen': ''}

    def get_database_stats(self):
        """Получение общей статистики базы данных"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) as total FROM people WHERE is_active=1')
            total_row = cursor.fetchone()
            total_people = total_row['total'] if total_row and total_row['total'] is not None else 0
            
            cursor.execute('''
                SELECT COUNT(DISTINCT person_id) as with_embeddings 
                FROM person_photos 
                WHERE face_embedding IS NOT NULL
            ''')
            embeddings_row = cursor.fetchone()
            with_embeddings = embeddings_row['with_embeddings'] if embeddings_row and embeddings_row['with_embeddings'] is not None else 0
            
            cursor.execute('SELECT COUNT(*) as total_photos FROM person_photos')
            photos_row = cursor.fetchone()
            total_photos = photos_row['total_photos'] if photos_row else 0
            
            cursor.execute('''
                SELECT COUNT(*) as total_sessions,
                       COALESCE(AVG(confidence), 0) as avg_confidence
                FROM recognition_sessions
            ''')
            sessions_row = cursor.fetchone()
            if sessions_row:
                total_sessions = sessions_row['total_sessions'] or 0
                avg_confidence = float(sessions_row['avg_confidence'] or 0.0)
            else:
                total_sessions = 0
                avg_confidence = 0.0
            
            return {
                'total_people': total_people,
                'with_embeddings': with_embeddings,
                'total_photos': total_photos,
                'total_sessions': total_sessions,
                'avg_confidence': avg_confidence
            }

    def get_person_with_photos(self, person_id):
        """Получение данных человека вместе с фотографиями"""
        person = self.get_person(person_id)
        if person:
            person['photos'] = self.get_person_photos(person_id)
        return person

    def get_photos_without_embeddings(self):
        """Получение фотографий без эмбеддингов"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT pp.*, p.last_name, p.first_name 
                FROM person_photos pp
                JOIN people p ON pp.person_id = p.id
                WHERE pp.face_embedding IS NULL 
                AND pp.image_path IS NOT NULL
                ORDER BY p.last_name, p.first_name
            ''')
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_all_photos(self):
        """Получение всех фотографий"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT pp.*, p.last_name, p.first_name 
                FROM person_photos pp
                JOIN people p ON pp.person_id = p.id
                WHERE pp.image_path IS NOT NULL
                ORDER BY p.last_name, p.first_name
            ''')
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def _save_image(self, image_path, person_id, filename_prefix):
        """Сохранение изображения в папку"""
        try:
            file_ext = os.path.splitext(image_path)[1].lower()
            if file_ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
                raise ValueError(f"Unsupported image format: {file_ext}")
                
            filename = f"{filename_prefix}_{person_id}{file_ext}"
            saved_path = os.path.join(self.images_dir, filename)
            shutil.copy2(image_path, saved_path)
            return saved_path
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return None

    def _delete_image(self, image_path):
        """Удаление изображения"""
        try:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
                return True
        except Exception as e:
            logger.error(f"Error deleting image {image_path}: {e}")
        return False

    def cleanup_orphaned_images(self):
        """Очистка изображений без записей в базе"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT image_path FROM person_photos WHERE image_path IS NOT NULL')
                valid_images = {row['image_path'] for row in cursor.fetchall()}
            
            deleted_count = 0
            for filename in os.listdir(self.images_dir):
                file_path = os.path.join(self.images_dir, filename)
                if file_path not in valid_images and os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"Error deleting orphaned image {file_path}: {e}")
            
            logger.info(f"Cleaned up {deleted_count} orphaned images")
            return deleted_count
        except Exception as e:
            logger.error(f"Error cleaning up images: {e}")
            return 0