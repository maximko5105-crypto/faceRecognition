import cv2
import mediapipe as mp
import numpy as np
import json
import logging
from typing import List, Tuple, Optional, Dict, Any
import os

logger = logging.getLogger(__name__)

class MediaPipeFaceRecognizer:
    def __init__(self, min_detection_confidence: float = 0.5):
        """
        Инициализация MediaPipe для распознавания лиц
        """
        self.min_detection_confidence = min_detection_confidence
        
        # Инициализация MediaPipe компонентов
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Детектор лиц
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=min_detection_confidence
        )
        
        # Face Mesh для извлечения признаков
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        
        # Хранилище известных эмбеддингов
        self.known_embeddings: List[np.ndarray] = []
        self.known_names: List[str] = []
        self.known_ids: List[int] = []
        self.known_photo_ids: List[int] = []  # ID фотографий в базе данных
        
        # Настройки распознавания
        self.recognition_threshold: float = 0.6
        self.embedding_size: int = 468 * 3  # 468 landmarks × 3 координаты (x, y, z)
        
        logger.info("Kaleido ID Face Recognizer initialized")

    def safe_float(self, value, default=0.0):
        """Безопасное преобразование в float"""
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def extract_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Извлечение эмбеддинга лица из изображения
        """
        try:
            # Конвертируем в RGB для MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Обрабатываем изображение для извлечения landmarks
            results = self.face_mesh.process(rgb_image)
            
            if results.multi_face_landmarks:
                # Берем первое найденное лицо (самое крупное/четкое)
                face_landmarks = results.multi_face_landmarks[0]
                
                # Извлекаем координаты landmarks
                landmarks = []
                for landmark in face_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                # Преобразуем в numpy array
                landmarks_array = np.array(landmarks, dtype=np.float32)
                
                # Нормализуем эмбеддинг
                if len(landmarks_array) == self.embedding_size:
                    embedding = self._normalize_embedding(landmarks_array)
                    logger.debug("Face embedding extracted successfully")
                    return embedding
                else:
                    logger.warning(f"Unexpected embedding size: {len(landmarks_array)}")
            
            logger.warning("No face landmarks detected in image")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None

    def extract_embedding_from_file(self, image_path: str) -> Optional[np.ndarray]:
        """
        Извлечение эмбеддинга из файла изображения
        """
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None
                
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
                
            return self.extract_embedding(image)
        except Exception as e:
            logger.error(f"Error extracting embedding from file {image_path}: {e}")
            return None

    def extract_embedding_from_face_roi(self, image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Извлечение эмбеддинга из области лица
        """
        try:
            x, y, w, h = face_bbox
            # Вырезаем область лица
            face_roi = image[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                logger.warning("Empty face region")
                return None
                
            return self.extract_embedding(face_roi)
        except Exception as e:
            logger.error(f"Error extracting embedding from face ROI: {e}")
            return None

    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Обнаружение всех лиц в изображении
        """
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_image)
            
            faces = []
            if results.detections:
                for detection in results.detections:
                    if not detection.location_data:
                        continue
                        
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = image.shape
                    
                    # Конвертируем относительные координаты в абсолютные
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Добавляем отступы для лучшего захвата лица
                    padding = 20
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    width = min(w - x, width + 2 * padding)
                    height = min(h - y, height + 2 * padding)
                    
                    # Получаем уверенность
                    confidence = self.safe_float(detection.score[0] if detection.score else 0.0)
                    
                    face_info = {
                        'bbox': (x, y, width, height),
                        'confidence': confidence,
                        'keypoints': self._extract_keypoints(detection, w, h),
                        'area': width * height  # Площадь лица для сортировки
                    }
                    faces.append(face_info)
            
            # Сортируем лица по площади (от большего к меньшему)
            faces.sort(key=lambda x: x['area'], reverse=True)
            
            logger.debug(f"Detected {len(faces)} faces in image")
            return faces
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []

    def recognize_face(self, embedding: np.ndarray) -> Tuple[Optional[int], float, Optional[int]]:
        """
        Распознавание лица по эмбеддингу
        
        Returns:
            Tuple[person_id, confidence, photo_id]
        """
        if embedding is None:
            return None, 0.0, None
            
        if len(self.known_embeddings) == 0:
            logger.warning("No embeddings loaded for recognition")
            return None, 0.0, None
        
        best_match_id = None
        best_photo_id = None
        best_similarity = 0.0
        
        for i, known_embedding in enumerate(self.known_embeddings):
            similarity = self._calculate_similarity(embedding, known_embedding)
            
            if similarity > best_similarity and similarity > self.recognition_threshold:
                best_similarity = similarity
                best_match_id = self.known_ids[i]
                best_photo_id = self.known_photo_ids[i]
        
        logger.debug(f"Recognition result: ID={best_match_id}, confidence={best_similarity:.3f}")
        return best_match_id, best_similarity, best_photo_id

    def recognize_face_from_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Полный цикл распознавания: обнаружение + распознавание всех лиц в изображении
        """
        results = []
        faces = self.detect_faces(image)
        
        for face in faces:
            x, y, w, h = face['bbox']
            face_roi = image[y:y+h, x:x+w]
            
            embedding = self.extract_embedding(face_roi)
            person_id, confidence, photo_id = self.recognize_face(embedding)
            
            result = {
                'face_info': face,
                'person_id': person_id,
                'confidence': confidence,
                'photo_id': photo_id,
                'embedding': embedding
            }
            results.append(result)
        
        return results

    def train_from_image(self, image: np.ndarray, person_data: Dict[str, Any], photo_id: int = None) -> bool:
        """
        Обучение модели на основе изображения
        """
        try:
            embedding = self.extract_embedding(image)
            if embedding is not None:
                return self._add_embedding_to_memory(embedding, person_data, photo_id)
            else:
                logger.warning("No face detected for training")
                return False
                
        except Exception as e:
            logger.error(f"Error training from image: {e}")
            return False

    def train_from_image_file(self, image_path: str, person_data: Dict[str, Any], photo_id: int = None) -> bool:
        """
        Обучение модели на основе файла изображения
        """
        try:
            embedding = self.extract_embedding_from_file(image_path)
            if embedding is not None:
                return self._add_embedding_to_memory(embedding, person_data, photo_id)
            else:
                logger.warning(f"No face detected in image: {image_path}")
                return False
        except Exception as e:
            logger.error(f"Error training from image file {image_path}: {e}")
            return False

    def add_existing_embedding(self, embedding: np.ndarray, person_data: Dict[str, Any], photo_id: int = None) -> bool:
        """
        Добавление существующего эмбеддинга в модель
        """
        try:
            if embedding is not None and len(embedding) == self.embedding_size:
                return self._add_embedding_to_memory(embedding, person_data, photo_id)
            else:
                logger.warning("Invalid embedding provided")
                return False
        except Exception as e:
            logger.error(f"Error adding existing embedding: {e}")
            return False

    def batch_train_person(self, person_id: int, person_name: str, photo_ids: List[int], database) -> int:
        """
        Пакетное обучение для одного человека по всем его фотографиям
        """
        try:
            trained_count = 0
            photos = database.get_person_photos(person_id)
            
            for photo in photos:
                if photo.get('face_embedding'):
                    # Используем существующий эмбеддинг из базы
                    try:
                        embedding_data = photo['face_embedding']
                        if isinstance(embedding_data, bytes):
                            embedding_json = embedding_data.decode('utf-8')
                        else:
                            embedding_json = embedding_data
                            
                        embedding = np.array(json.loads(embedding_json), dtype=np.float32)
                        
                        if self.add_existing_embedding(embedding, {'id': person_id, 'last_name': person_name}, photo['id']):
                            trained_count += 1
                    except Exception as e:
                        logger.warning(f"Error loading embedding for photo {photo['id']}: {e}")
                        continue
                elif photo.get('image_path') and os.path.exists(photo['image_path']):
                    # Извлекаем эмбеддинг из изображения
                    if self.train_from_image_file(photo['image_path'], {'id': person_id, 'last_name': person_name}, photo['id']):
                        trained_count += 1
            
            logger.info(f"Batch trained {trained_count} photos for person {person_name}")
            return trained_count
            
        except Exception as e:
            logger.error(f"Error in batch training for person {person_id}: {e}")
            return 0

    def load_embeddings_from_database(self, database) -> int:
        """
        Загрузка эмбеддингов из базы данных
        """
        try:
            self.known_embeddings.clear()
            self.known_names.clear()
            self.known_ids.clear()
            self.known_photo_ids.clear()
            
            people = database.get_all_people()
            loaded_count = 0
            
            for person in people:
                photos = database.get_person_photos(person['id'])
                for photo in photos:
                    if photo.get('face_embedding'):
                        try:
                            # Декодируем эмбеддинг из бинарных данных
                            embedding_data = photo['face_embedding']
                            if isinstance(embedding_data, bytes):
                                embedding_json = embedding_data.decode('utf-8')
                            else:
                                embedding_json = embedding_data
                                
                            embedding = np.array(json.loads(embedding_json), dtype=np.float32)
                            
                            if len(embedding) == self.embedding_size:
                                self.known_embeddings.append(embedding)
                                
                                # Создаем имя для отображения
                                last_name = person.get('last_name', 'Unknown')
                                first_name = person.get('first_name', '')
                                display_name = f"{last_name} {first_name}".strip()
                                self.known_names.append(display_name)
                                
                                self.known_ids.append(person['id'])
                                self.known_photo_ids.append(photo['id'])
                                loaded_count += 1
                                
                        except Exception as e:
                            logger.warning(f"Error loading embedding for photo {photo.get('id')}: {e}")
                            continue
            
            logger.info(f"Loaded {loaded_count} embeddings from database")
            return loaded_count
            
        except Exception as e:
            logger.error(f"Error loading embeddings from database: {e}")
            return 0

    def batch_extract_embeddings(self, database, progress_callback=None):
        """
        Пакетное извлечение эмбеддингов для всех фотографий без эмбеддингов
        """
        try:
            photos = database.get_photos_without_embeddings()
            total_photos = len(photos)
            processed_count = 0
            success_count = 0
            
            logger.info(f"Starting batch embedding extraction for {total_photos} photos")
            
            for photo in photos:
                try:
                    if progress_callback:
                        progress_callback(processed_count, total_photos, f"Обработка {photo.get('last_name', '')}...")
                    
                    if photo.get('image_path') and os.path.exists(photo['image_path']):
                        embedding = self.extract_embedding_from_file(photo['image_path'])
                        
                        if embedding is not None:
                            # Сохраняем эмбеддинг в базу
                            database.update_photo_embedding(photo['id'], embedding)
                            success_count += 1
                            
                            # Добавляем в память модели
                            person_data = {
                                'id': photo['person_id'],
                                'last_name': photo.get('last_name', ''),
                                'first_name': photo.get('first_name', '')
                            }
                            self._add_embedding_to_memory(embedding, person_data, photo['id'])
                        
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing photo {photo.get('id')}: {e}")
                    processed_count += 1
                    continue
            
            logger.info(f"Batch embedding extraction completed: {success_count}/{total_photos} successful")
            return success_count, total_photos
            
        except Exception as e:
            logger.error(f"Error in batch embedding extraction: {e}")
            return 0, 0

    def extract_embedding_for_photo(self, photo_id, database):
        """Извлечение эмбеддинга для конкретной фотографии"""
        try:
            photos = database.get_person_photos_by_id(photo_id)
            if not photos:
                return False
                
            photo = photos[0]
            if not photo.get('image_path') or not os.path.exists(photo['image_path']):
                return False
            
            embedding = self.extract_embedding_from_file(photo['image_path'])
            if embedding is not None:
                # Сохраняем в базу
                database.update_photo_embedding(photo_id, embedding)
                
                # Добавляем в память модели
                person = database.get_person(photo['person_id'])
                if person:
                    self._add_embedding_to_memory(embedding, person, photo_id)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error extracting embedding for photo {photo_id}: {e}")
            return False

    def remove_embedding_by_photo_id(self, photo_id: int) -> bool:
        """
        Удаление эмбеддинга по ID фотографии
        """
        try:
            if photo_id in self.known_photo_ids:
                index = self.known_photo_ids.index(photo_id)
                self.known_embeddings.pop(index)
                self.known_names.pop(index)
                self.known_ids.pop(index)
                self.known_photo_ids.pop(index)
                logger.info(f"Removed embedding for photo ID: {photo_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing embedding for photo {photo_id}: {e}")
            return False

    def remove_embeddings_by_person_id(self, person_id: int) -> int:
        """
        Удаление всех эмбеддингов по ID человека
        """
        try:
            removed_count = 0
            indices_to_remove = []
            
            for i, known_id in enumerate(self.known_ids):
                if known_id == person_id:
                    indices_to_remove.append(i)
            
            # Удаляем в обратном порядке чтобы не сломать индексы
            for i in sorted(indices_to_remove, reverse=True):
                self.known_embeddings.pop(i)
                self.known_names.pop(i)
                self.known_ids.pop(i)
                self.known_photo_ids.pop(i)
                removed_count += 1
            
            if removed_count > 0:
                logger.info(f"Removed {removed_count} embeddings for person ID: {person_id}")
            
            return removed_count
        except Exception as e:
            logger.error(f"Error removing embeddings for person {person_id}: {e}")
            return 0

    def set_recognition_threshold(self, threshold: float):
        """Установка порога распознавания"""
        self.recognition_threshold = max(0.1, min(1.0, self.safe_float(threshold, 0.6)))
        logger.info(f"Recognition threshold set to {self.recognition_threshold}")

    def set_detection_confidence(self, confidence: float):
        """Установка уверенности обнаружения"""
        new_confidence = max(0.1, min(1.0, self.safe_float(confidence, 0.5)))
        self.min_detection_confidence = new_confidence
        
        # Пересоздаем детектор с новыми настройками
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=new_confidence
        )
        
        logger.info(f"Detection confidence set to {new_confidence}")

    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о модели"""
        unique_people = len(set(self.known_ids)) if self.known_ids else 0
        
        return {
            'loaded_embeddings': len(self.known_embeddings),
            'unique_people': unique_people,
            'recognition_threshold': self.recognition_threshold,
            'min_detection_confidence': self.min_detection_confidence,
            'embedding_size': self.embedding_size,
            'status': 'ready' if len(self.known_embeddings) > 0 else 'needs_training'
        }

    def get_embedding_statistics(self) -> Dict[str, Any]:
        """Получение статистики по эмбеддингам"""
        if not self.known_embeddings:
            return {
                'total_embeddings': 0,
                'unique_people': 0,
                'embeddings_per_person': {},
                'status': 'empty'
            }
        
        # Считаем количество эмбеддингов на человека
        embeddings_per_person = {}
        for person_id in self.known_ids:
            embeddings_per_person[person_id] = embeddings_per_person.get(person_id, 0) + 1
        
        return {
            'total_embeddings': len(self.known_embeddings),
            'unique_people': len(set(self.known_ids)),
            'embeddings_per_person': embeddings_per_person,
            'status': 'loaded'
        }

    def clear_model(self):
        """Очистка модели (удаление всех эмбеддингов)"""
        self.known_embeddings.clear()
        self.known_names.clear()
        self.known_ids.clear()
        self.known_photo_ids.clear()
        logger.info("Model cleared - all embeddings removed")

    def _add_embedding_to_memory(self, embedding: np.ndarray, person_data: Dict[str, Any], photo_id: int = None) -> bool:
        """Добавление эмбеддинга в память модели"""
        try:
            self.known_embeddings.append(embedding)
            
            # Создаем имя для отображения
            last_name = person_data.get('last_name', 'Unknown')
            first_name = person_data.get('first_name', '')
            display_name = f"{last_name} {first_name}".strip()
            self.known_names.append(display_name)
            
            # Используем существующий ID или создаем новый
            person_id = person_data.get('id')
            if person_id is None:
                person_id = len(self.known_ids) + 1
            self.known_ids.append(person_id)
            
            # Сохраняем ID фотографии
            self.known_photo_ids.append(photo_id)
            
            logger.info(f"Added embedding for {display_name}, photo ID: {photo_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding embedding to memory: {e}")
            return False

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Нормализация эмбеддинга"""
        mean = np.mean(embedding)
        std = np.std(embedding)
        
        if std > 0:
            normalized = (embedding - mean) / std
        else:
            normalized = embedding - mean
            
        return normalized

    def _calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Вычисление схожести между эмбеддингами"""
        try:
            # Косинусное сходство
            emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-10)
            emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-10)
            similarity = np.dot(emb1_norm, emb2_norm)
            
            # Приводим к диапазону [0, 1]
            normalized_similarity = float((similarity + 1) / 2)
            
            return normalized_similarity
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def _extract_keypoints(self, detection, image_width: int, image_height: int) -> List[Tuple[int, int]]:
        """Извлечение ключевых точек лица"""
        keypoints = []
        try:
            if hasattr(detection.location_data, 'relative_keypoints'):
                for keypoint in detection.location_data.relative_keypoints:
                    x = int(keypoint.x * image_width)
                    y = int(keypoint.y * image_height)
                    keypoints.append((x, y))
        except Exception as e:
            logger.warning(f"Error extracting keypoints: {e}")
            
        return keypoints

    def draw_detection(self, image: np.ndarray, face_info: Dict[str, Any], 
                      person_name: str = None, confidence: float = None, 
                      is_selected: bool = False) -> np.ndarray:
        """
        Отрисовка обнаруженного лица на изображении
        """
        try:
            x, y, w, h = face_info['bbox']
            
            # Выбираем цвет в зависимости от результата распознавания и выбора
            if is_selected:
                color = (255, 255, 0)  # Желтый для выбранного лица
                thickness = 3
            elif person_name and confidence is not None:
                color = (0, 255, 0)  # Зеленый для распознанных
                thickness = 2
            else:
                color = (0, 0, 255)  # Красный для неизвестных
                thickness = 2
            
            # Создаем текст для отображения
            if person_name and confidence is not None:
                conf_str = f"{self.safe_float(confidence):.2f}"
                label = f"{person_name} ({conf_str})"
            else:
                face_confidence = self.safe_float(face_info.get('confidence', 0.0))
                conf_str = f"{face_confidence:.2f}"
                label = f"Unknown ({conf_str})"
            
            # Рисуем прямоугольник вокруг лица
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
            
            # Рисуем подпись с фоном
            font_scale = 0.6
            label_thickness = 2
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_thickness)[0]
            
            # Вычисляем позицию для подписи (над прямоугольником)
            label_y = max(y - 10, label_size[1] + 10)
            
            # Рисуем фон для текста
            cv2.rectangle(image, 
                         (x, label_y - label_size[1] - 10), 
                         (x + label_size[0], label_y), 
                         color, -1)
            
            # Рисуем текст
            cv2.putText(image, label, (x, label_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), label_thickness)
            
            # Рисуем ключевые точки если есть
            for kp_x, kp_y in face_info.get('keypoints', []):
                cv2.circle(image, (kp_x, kp_y), 3, color, -1)
                
            return image
            
        except Exception as e:
            logger.error(f"Error drawing detection: {e}")
            return image

    def draw_multiple_detections(self, image: np.ndarray, recognition_results: List[Dict[str, Any]], 
                                selected_face_index: int = 0) -> np.ndarray:
        """
        Отрисовка нескольких обнаруженных лиц с распознаванием
        """
        try:
            for i, result in enumerate(recognition_results):
                face_info = result['face_info']
                person_name = None
                confidence = None
                
                if result['person_id'] is not None:
                    person_name = f"ID:{result['person_id']}"
                    confidence = result['confidence']
                
                # Помечаем выбранное лицо
                is_selected = (i == selected_face_index)
                
                image = self.draw_detection(image, face_info, person_name, confidence, is_selected)
            
            return image
            
        except Exception as e:
            logger.error(f"Error drawing multiple detections: {e}")
            return image

    def validate_embedding(self, embedding: np.ndarray) -> bool:
        """Проверка валидности эмбеддинга"""
        if embedding is None:
            return False
        
        if not isinstance(embedding, np.ndarray):
            return False
        
        if len(embedding) != self.embedding_size:
            return False
        
        # Проверяем что эмбеддинг не состоит из нулей
        if np.all(embedding == 0):
            return False
        
        return True

    def get_similarity_matrix(self) -> np.ndarray:
        """Получение матрицы схожести между всеми эмбеддингами"""
        if not self.known_embeddings:
            return np.array([])
        
        n = len(self.known_embeddings)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity_matrix[i, j] = self._calculate_similarity(
                        self.known_embeddings[i], 
                        self.known_embeddings[j]
                    )
        
        return similarity_matrix

    def find_similar_embeddings(self, embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """Поиск наиболее похожих эмбеддингов"""
        if not self.known_embeddings:
            return []
        
        similarities = []
        for i, known_embedding in enumerate(self.known_embeddings):
            similarity = self._calculate_similarity(embedding, known_embedding)
            similarities.append((i, similarity))
        
        # Сортируем по убыванию схожести
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]

    def cleanup(self):
        """Очистка ресурсов"""
        try:
            if hasattr(self, 'face_detection'):
                self.face_detection.close()
            if hasattr(self, 'face_mesh'):
                self.face_mesh.close()
            logger.info("MediaPipe resources cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")

    def __del__(self):
        """Деструктор для автоматической очистки ресурсов"""
        self.cleanup()