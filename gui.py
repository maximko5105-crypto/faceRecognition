import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import logging
import os
from datetime import datetime
from .database import FaceDatabase
from .face_recognizer import MediaPipeFaceRecognizer

logger = logging.getLogger(__name__)

class KaleidoIDGUI:
    def __init__(self, root, database, face_recognizer):
        self.root = root
        self.database = database
        self.face_recognizer = face_recognizer
        
        self.current_image = None
        self.current_photo_id = None
        self.current_person_id = None
        self.camera_active = False
        self.cap = None
        self.recognition_results = []
        self.selected_face_index = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        self.root.title("Kaleido ID - Система распознавания лиц")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Создаем стиль для виджетов
        self.setup_styles()
        
        # Главный фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Настройка весов строк и столбцов для растягивания
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Заголовок
        title_label = ttk.Label(main_frame, text="Kaleido ID", font=('Helvetica', 24, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Левая панель - управление
        left_frame = ttk.Frame(main_frame, padding="10", relief='raised')
        left_frame.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.W), padx=(0, 10))
        
        # Правая панель - отображение
        right_frame = ttk.Frame(main_frame, padding="10", relief='raised')
        right_frame.grid(row=1, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # Настройка весов для правой панели
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        
        # ===== Левая панель =====
        control_label = ttk.Label(left_frame, text="Управление", font=('Helvetica', 14, 'bold'))
        control_label.grid(row=0, column=0, pady=(0, 15))
        
        # Кнопки управления камерой
        camera_frame = ttk.LabelFrame(left_frame, text="Камера", padding="10")
        camera_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.camera_btn = ttk.Button(camera_frame, text="Включить камеру", command=self.toggle_camera)
        self.camera_btn.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.capture_btn = ttk.Button(camera_frame, text="Сделать снимок", command=self.capture_image, state=tk.DISABLED)
        self.capture_btn.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Кнопки загрузки изображения
        image_frame = ttk.LabelFrame(left_frame, text="Изображение", padding="10")
        image_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(image_frame, text="Загрузить изображение", command=self.load_image).grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(image_frame, text="Распознать лица", command=self.recognize_faces).grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Управление распознаванием
        recognize_frame = ttk.LabelFrame(left_frame, text="Распознавание", padding="10")
        recognize_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(recognize_frame, text="Выбрать следующее лицо", command=self.select_next_face).grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(recognize_frame, text="Добавить в базу", command=self.add_to_database).grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Информация о выбранном лице
        self.face_info_label = ttk.Label(recognize_frame, text="Лицо не выбрано", wraplength=200)
        self.face_info_label.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Управление базой данных
        db_frame = ttk.LabelFrame(left_frame, text="База данных", padding="10")
        db_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(db_frame, text="Просмотр базы", command=self.show_database).grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(db_frame, text="Настройки распознавания", command=self.show_settings).grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(db_frame, text="Статистика", command=self.show_statistics).grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Информация о системе
        info_frame = ttk.LabelFrame(left_frame, text="Информация о системе", padding="10")
        info_frame.grid(row=5, column=0, sticky=(tk.W, tk.E))
        
        self.system_info_label = ttk.Label(info_frame, text="Загрузка...", justify=tk.LEFT)
        self.system_info_label.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # ===== Правая панель =====
        # Заголовок изображения
        self.image_title = ttk.Label(right_frame, text="Изображение не загружено", font=('Helvetica', 12))
        self.image_title.grid(row=0, column=0, pady=(0, 10))
        
        # Холст для отображения изображения
        self.canvas = tk.Canvas(right_frame, bg='#34495e', highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # Полоса прокрутки для холста
        scroll_y = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        scroll_y.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.canvas.configure(yscrollcommand=scroll_y.set)
        
        # Фрейм для информации под изображением
        info_bottom_frame = ttk.Frame(right_frame)
        info_bottom_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.image_info_label = ttk.Label(info_bottom_frame, text="")
        self.image_info_label.grid(row=0, column=0, sticky=tk.W)
        
        # Обновление информации о системе
        self.update_system_info()
        
    def setup_styles(self):
        """Настройка стилей для виджетов"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Настраиваем цвета
        style.configure('TFrame', background='#2c3e50')
        style.configure('TLabel', background='#2c3e50', foreground='white')
        style.configure('TLabelframe', background='#2c3e50', foreground='white')
        style.configure('TLabelframe.Label', background='#2c3e50', foreground='white')
        style.configure('TButton', background='#3498db', foreground='black')
        style.configure('TScrollbar', background='#34495e')
        
    def update_system_info(self):
        """Обновление информации о системе"""
        try:
            model_info = self.face_recognizer.get_model_info()
            db_stats = self.database.get_database_stats()
            
            info_text = (f"Людей в базе: {db_stats['total_people']}\n"
                        f"Фотографий: {db_stats['total_photos']}\n"
                        f"Эмбеддингов: {model_info['loaded_embeddings']}\n"
                        f"Уникальных лиц: {model_info['unique_people']}\n"
                        f"Порог распознавания: {model_info['recognition_threshold']:.2f}")
            
            self.system_info_label.config(text=info_text)
            
        except Exception as e:
            logger.error(f"Error updating system info: {e}")
            self.system_info_label.config(text="Ошибка загрузки информации")
            
    def toggle_camera(self):
        """Включение/выключение камеры"""
        if not self.camera_active:
            self.start_camera()
        else:
            self.stop_camera()
            
    def start_camera(self):
        """Запуск камеры"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Ошибка", "Не удалось подключиться к камере")
                return
                
            self.camera_active = True
            self.camera_btn.config(text="Выключить камеру")
            self.capture_btn.config(state=tk.NORMAL)
            
            # Запускаем обновление кадров
            self.update_camera_frame()
            
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            messagebox.showerror("Ошибка", f"Ошибка запуска камеры: {e}")
            
    def stop_camera(self):
        """Остановка камеры"""
        self.camera_active = False
        if self.cap:
            self.cap.release()
            self.cap = None
            
        self.camera_btn.config(text="Включить камеру")
        self.capture_btn.config(state=tk.DISABLED)
        
        # Очищаем холст
        self.canvas.delete("all")
        self.image_title.config(text="Камера выключена")
        
    def update_camera_frame(self):
        """Обновление кадра с камеры"""
        if self.camera_active and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Обнаруживаем лица в реальном времени
                self.recognition_results = self.face_recognizer.recognize_face_from_image(frame)
                
                # Отрисовываем обнаруженные лица
                if self.recognition_results:
                    frame = self.face_recognizer.draw_multiple_detections(
                        frame, self.recognition_results, self.selected_face_index
                    )
                
                # Отображаем кадр
                self.display_image(frame, "Режим камеры - Обнаружено лиц: {}".format(len(self.recognition_results)))
            
            # Планируем следующее обновление
            if self.camera_active:
                self.root.after(10, self.update_camera_frame)
                
    def capture_image(self):
        """Снимок с камеры"""
        if self.camera_active and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.current_image = frame.copy()
                self.recognition_results = self.face_recognizer.recognize_face_from_image(frame)
                self.selected_face_index = 0
                self.display_image(frame, "Снимок с камеры - Обнаружено лиц: {}".format(len(self.recognition_results)))
                self.update_face_info()
                
    def load_image(self):
        """Загрузка изображения из файла"""
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                image = cv2.imread(file_path)
                if image is not None:
                    self.current_image = image
                    self.recognition_results = []
                    self.selected_face_index = 0
                    self.display_image(image, f"Загружено: {os.path.basename(file_path)}")
                    self.recognize_faces()
                else:
                    messagebox.showerror("Ошибка", "Не удалось загрузить изображение")
                    
            except Exception as e:
                logger.error(f"Error loading image: {e}")
                messagebox.showerror("Ошибка", f"Ошибка загрузки изображения: {e}")
                
    def recognize_faces(self):
        """Распознавание лиц на текущем изображении"""
        if self.current_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение или сделайте снимок")
            return
            
        try:
            self.recognition_results = self.face_recognizer.recognize_face_from_image(self.current_image)
            self.selected_face_index = 0
            
            # Обновляем отображение
            if self.recognition_results:
                displayed_image = self.current_image.copy()
                displayed_image = self.face_recognizer.draw_multiple_detections(
                    displayed_image, self.recognition_results, self.selected_face_index
                )
                self.display_image(displayed_image, f"Распознано лиц: {len(self.recognition_results)}")
                self.update_face_info()
            else:
                self.display_image(self.current_image, "Лица не обнаружены")
                self.face_info_label.config(text="Лица не обнаружены")
                
        except Exception as e:
            logger.error(f"Error recognizing faces: {e}")
            messagebox.showerror("Ошибка", f"Ошибка распознавания: {e}")
            
    def select_next_face(self):
        """Выбор следующего лица на изображении"""
        if self.recognition_results:
            self.selected_face_index = (self.selected_face_index + 1) % len(self.recognition_results)
            
            # Обновляем отображение
            displayed_image = self.current_image.copy()
            displayed_image = self.face_recognizer.draw_multiple_detections(
                displayed_image, self.recognition_results, self.selected_face_index
            )
            self.display_image(displayed_image)
            self.update_face_info()
            
    def update_face_info(self):
        """Обновление информации о выбранном лице"""
        if self.recognition_results and 0 <= self.selected_face_index < len(self.recognition_results):
            result = self.recognition_results[self.selected_face_index]
            face_info = result['face_info']
            
            info_text = (f"Лицо {self.selected_face_index + 1}/{len(self.recognition_results)}\n"
                        f"Размер: {face_info['bbox'][2]}x{face_info['bbox'][3]}\n"
                        f"Уверенность: {face_info['confidence']:.2f}")
            
            if result['person_id'] is not None:
                person = self.database.get_person(result['person_id'])
                if person:
                    name = f"{person.get('last_name', '')} {person.get('first_name', '')}".strip()
                    info_text += f"\nРаспознан как: {name}\nID: {result['person_id']}\nСхожесть: {result['confidence']:.2f}"
                else:
                    info_text += f"\nРаспознан как: ID {result['person_id']}\nСхожесть: {result['confidence']:.2f}"
            else:
                info_text += "\nНеизвестное лицо"
                
            self.face_info_label.config(text=info_text)
        else:
            self.face_info_label.config(text="Лицо не выбрано")
            
    def add_to_database(self):
        """Добавление выбранного лица в базу данных"""
        if not self.recognition_results or self.selected_face_index >= len(self.recognition_results):
            messagebox.showwarning("Предупреждение", "Сначала выберите лицо для добавления")
            return
            
        result = self.recognition_results[self.selected_face_index]
        
        # Если лицо уже распознано, спрашиваем хотим ли мы добавить еще одну фотографию
        if result['person_id'] is not None:
            person = self.database.get_person(result['person_id'])
            if person:
                name = f"{person.get('last_name', '')} {person.get('first_name', '')}".strip()
                response = messagebox.askyesno(
                    "Подтверждение", 
                    f"Это лицо уже распознано как {name} (ID: {result['person_id']}).\nДобавить еще одну фотографию этому человеку?"
                )
                if response:
                    self.add_photo_to_person(result['person_id'], result)
                return
                
        # Если лицо не распознано, открываем диалог добавления нового человека
        self.add_new_person(result)
        
    def add_new_person(self, recognition_result):
        """Добавление нового человека в базу данных"""
        dialog = AddPersonDialog(self.root, self.database)
        if dialog.result:
            person_data = dialog.result
            person_id = self.database.add_person(person_data)
            
            if person_id:
                # Добавляем фотографию
                self.add_photo_to_person(person_id, recognition_result, is_primary=True)
                
                # Обучаем модель на новом лице
                self.retrain_model_for_person(person_id, person_data)
                
                messagebox.showinfo("Успех", f"Человек успешно добавлен в базу с ID: {person_id}")
                self.update_system_info()
            else:
                messagebox.showerror("Ошибка", "Не удалось добавить человека в базу данных")
                
    def add_photo_to_person(self, person_id, recognition_result, is_primary=False):
        """Добавление фотографии к существующему человеку"""
        try:
            face_info = recognition_result['face_info']
            x, y, w, h = face_info['bbox']
            face_roi = self.current_image[y:y+h, x:x+w]
            
            # Сохраняем фотографию в базу
            photo_id, saved_path = self.database.add_person_photo_from_memory(
                person_id, face_roi, f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            if photo_id:
                # Обучаем модель на новой фотографии
                person = self.database.get_person(person_id)
                if person:
                    self.face_recognizer.train_from_image(face_roi, person, photo_id)
                    
                logger.info(f"Added photo {photo_id} to person {person_id}")
                return True
            else:
                logger.error(f"Failed to add photo to person {person_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding photo to person: {e}")
            messagebox.showerror("Ошибка", f"Ошибка добавления фотографии: {e}")
            return False
            
    def retrain_model_for_person(self, person_id, person_data):
        """Переобучение модели для конкретного человека"""
        try:
            # Загружаем все фотографии человека
            photos = self.database.get_person_photos(person_id)
            trained_count = 0
            
            for photo in photos:
                if photo.get('image_path') and os.path.exists(photo['image_path']):
                    image = cv2.imread(photo['image_path'])
                    if image is not None:
                        if self.face_recognizer.train_from_image(image, person_data, photo['id']):
                            trained_count += 1
                            
            logger.info(f"Retrained model for person {person_id} with {trained_count} photos")
            
        except Exception as e:
            logger.error(f"Error retraining model for person {person_id}: {e}")
            
    def display_image(self, image, title=None):
        """Отображение изображения на холсте"""
        try:
            if title:
                self.image_title.config(text=title)
                
            # Масштабируем изображение для отображения
            h, w = image.shape[:2]
            max_width = 800
            max_height = 600
            
            # Вычисляем новые размеры с сохранением пропорций
            if w > max_width or h > max_height:
                scale = min(max_width / w, max_height / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                new_w, new_h = w, h
                
            # Конвертируем для tkinter
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            self.tk_image = ImageTk.PhotoImage(pil_image)
            
            # Обновляем холст
            self.canvas.delete("all")
            self.canvas.config(scrollregion=(0, 0, new_w, new_h))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
            
            # Обновляем информацию
            self.image_info_label.config(text=f"Размер: {w}x{h} | Отображение: {new_w}x{new_h}")
            
        except Exception as e:
            logger.error(f"Error displaying image: {e}")
            
    def show_database(self):
        """Показ окна базы данных"""
        dialog = DatabaseDialog(self.root, self.database, self.face_recognizer)
        
    def show_settings(self):
        """Показ окна настроек"""
        dialog = SettingsDialog(self.root, self.face_recognizer)
        
    def show_statistics(self):
        """Показ статистики"""
        dialog = StatisticsDialog(self.root, self.database, self.face_recognizer)
        
    def on_closing(self):
        """Действия при закрытии приложения"""
        self.stop_camera()
        self.root.destroy()

class AddPersonDialog(tk.Toplevel):
    """Диалог добавления нового человека"""
    def __init__(self, parent, database):
        super().__init__(parent)
        self.database = database
        self.result = None
        
        self.title("Добавить нового человека")
        self.geometry("400x500")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        
        self.create_widgets()
        self.center_on_parent(parent)
        
    def center_on_parent(self, parent):
        """Центрирование диалога относительно родительского окна"""
        parent_x = parent.winfo_rootx()
        parent_y = parent.winfo_rooty()
        parent_width = parent.winfo_width()
        parent_height = parent.winfo_height()
        
        dialog_width = 400
        dialog_height = 500
        
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        self.geometry(f"+{x}+{y}")
        
    def create_widgets(self):
        """Создание виджетов диалога"""
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Поля ввода
        fields = [
            ('last_name', 'Фамилия*:', True),
            ('first_name', 'Имя*:', True),
            ('middle_name', 'Отчество:', False),
            ('age', 'Возраст:', False),
            ('position', 'Должность:', False),
            ('department', 'Отдел:', False),
            ('phone', 'Телефон:', False),
            ('email', 'Email:', False),
            ('address', 'Адрес:', False),
            ('notes', 'Примечания:', False)
        ]
        
        self.entries = {}
        row = 0
        
        for field, label, required in fields:
            ttk.Label(main_frame, text=label).grid(row=row, column=0, sticky=tk.W, pady=5)
            entry = ttk.Entry(main_frame, width=30)
            entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
            self.entries[field] = entry
            row += 1
            
        # Кнопки
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="Добавить", command=self.on_add).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Отмена", command=self.on_cancel).pack(side=tk.LEFT, padx=5)
        
        # Настройка весов для растягивания
        main_frame.columnconfigure(1, weight=1)
        
    def on_add(self):
        """Обработка нажатия кнопки Добавить"""
        # Проверяем обязательные поля
        last_name = self.entries['last_name'].get().strip()
        first_name = self.entries['first_name'].get().strip()
        
        if not last_name or not first_name:
            messagebox.showwarning("Предупреждение", "Поля 'Фамилия' и 'Имя' обязательны для заполнения")
            return
            
        # Собираем данные
        self.result = {
            'last_name': last_name,
            'first_name': first_name,
            'middle_name': self.entries['middle_name'].get().strip(),
            'age': self.parse_int(self.entries['age'].get()),
            'position': self.entries['position'].get().strip(),
            'department': self.entries['department'].get().strip(),
            'phone': self.entries['phone'].get().strip(),
            'email': self.entries['email'].get().strip(),
            'address': self.entries['address'].get().strip(),
            'notes': self.entries['notes'].get().strip()
        }
        
        self.destroy()
        
    def on_cancel(self):
        """Обработка нажатия кнопки Отмена"""
        self.result = None
        self.destroy()
        
    def parse_int(self, value):
        """Парсинг целого числа"""
        try:
            return int(value) if value.strip() else None
        except ValueError:
            return None

class DatabaseDialog(tk.Toplevel):
    """Диалог просмотра базы данных"""
    def __init__(self, parent, database, face_recognizer):
        super().__init__(parent)
        self.database = database
        self.face_recognizer = face_recognizer
        
        self.title("База данных - Kaleido ID")
        self.geometry("800x600")
        self.transient(parent)
        self.grab_set()
        
        self.create_widgets()
        self.load_data()
        
    def create_widgets(self):
        """Создание виджетов диалога"""
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Панель поиска
        search_frame = ttk.Frame(main_frame)
        search_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(search_frame, text="Поиск:").pack(side=tk.LEFT, padx=(0, 5))
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=30)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.search_entry.bind('<KeyRelease>', self.on_search)
        
        ttk.Button(search_frame, text="Обновить", command=self.load_data).pack(side=tk.LEFT)
        
        # Таблица
        columns = ('ID', 'Фамилия', 'Имя', 'Должность', 'Фотографии', 'Последнее распознавание')
        self.tree = ttk.Treeview(main_frame, columns=columns, show='headings', height=20)
        
        # Настраиваем заголовки
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)
            
        # Полоса прокрутки
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Кнопки управления
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Просмотреть", command=self.view_person).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Удалить", command=self.delete_person).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Обновить эмбеддинги", command=self.update_embeddings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Закрыть", command=self.destroy).pack(side=tk.RIGHT, padx=5)
        
    def load_data(self):
        """Загрузка данных в таблицу"""
        # Очищаем таблицу
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        search_term = self.search_var.get().strip()
        if search_term:
            people = self.database.search_people(search_term)
        else:
            people = self.database.get_all_people()
            
        for person in people:
            # Получаем статистику
            photos = self.database.get_person_photos(person['id'])
            stats = self.database.get_recognition_stats(person['id'])
            
            self.tree.insert('', tk.END, values=(
                person['id'],
                person.get('last_name', ''),
                person.get('first_name', ''),
                person.get('position', ''),
                len(photos),
                stats.get('last_seen', '')[:19] if stats.get('last_seen') else 'Никогда'
            ))
            
    def on_search(self, event):
        """Обработка поиска"""
        self.load_data()
        
    def view_person(self):
        """Просмотр выбранного человека"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Предупреждение", "Выберите человека для просмотра")
            return
            
        item = self.tree.item(selection[0])
        person_id = item['values'][0]
        
        # Здесь можно открыть детальный просмотр человека
        messagebox.showinfo("Информация", f"Просмотр человека ID: {person_id}")
        
    def delete_person(self):
        """Удаление выбранного человека"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Предупреждение", "Выберите человека для удаления")
            return
            
        item = self.tree.item(selection[0])
        person_id = item['values'][0]
        name = f"{item['values'][1]} {item['values'][2]}"
        
        if messagebox.askyesno("Подтверждение", f"Вы уверены, что хотите удалить {name} (ID: {person_id})?"):
            if self.database.delete_person(person_id):
                # Удаляем эмбеддинги из модели
                self.face_recognizer.remove_embeddings_by_person_id(person_id)
                messagebox.showinfo("Успех", "Человек удален из базы данных")
                self.load_data()
            else:
                messagebox.showerror("Ошибка", "Не удалось удалить человека")
                
    def update_embeddings(self):
        """Обновление эмбеддингов для всех фотографий"""
        if messagebox.askyesno("Подтверждение", 
                              "Это действие извлечет эмбеддинги для всех фотографий без эмбеддингов.\nЭто может занять некоторое время. Продолжить?"):
            
            success_count, total_count = self.face_recognizer.batch_extract_embeddings(self.database)
            messagebox.showinfo("Готово", f"Эмбеддинги обновлены: {success_count}/{total_count} фотографий обработано")

class SettingsDialog(tk.Toplevel):
    """Диалог настроек"""
    def __init__(self, parent, face_recognizer):
        super().__init__(parent)
        self.face_recognizer = face_recognizer
        
        self.title("Настройки распознавания")
        self.geometry("400x300")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        
        self.create_widgets()
        self.load_current_settings()
        
    def create_widgets(self):
        """Создание виджетов диалога"""
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="Параметры распознавания", font=('Helvetica', 14, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Порог распознавания
        ttk.Label(main_frame, text="Порог распознавания (0.1-1.0):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.recognition_threshold = tk.DoubleVar()
        threshold_scale = ttk.Scale(main_frame, from_=0.1, to=1.0, variable=self.recognition_threshold, orient=tk.HORIZONTAL)
        threshold_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        self.threshold_label = ttk.Label(main_frame, text="0.60")
        self.threshold_label.grid(row=1, column=2, padx=(5, 0))
        
        # Уверенность обнаружения
        ttk.Label(main_frame, text="Уверенность обнаружения (0.1-1.0):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.detection_confidence = tk.DoubleVar()
        detection_scale = ttk.Scale(main_frame, from_=0.1, to=1.0, variable=self.detection_confidence, orient=tk.HORIZONTAL)
        detection_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        self.detection_label = ttk.Label(main_frame, text="0.50")
        self.detection_label.grid(row=2, column=2, padx=(5, 0))
        
        # Привязка событий для обновления меток
        threshold_scale.configure(command=self.on_threshold_change)
        detection_scale.configure(command=self.on_detection_change)
        
        # Кнопки
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=20)
        
        ttk.Button(button_frame, text="Сохранить", command=self.save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Сброс", command=self.load_current_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Закрыть", command=self.destroy).pack(side=tk.LEFT, padx=5)
        
        # Настройка весов
        main_frame.columnconfigure(1, weight=1)
        
    def load_current_settings(self):
        """Загрузка текущих настроек"""
        model_info = self.face_recognizer.get_model_info()
        
        self.recognition_threshold.set(model_info['recognition_threshold'])
        self.detection_confidence.set(model_info['min_detection_confidence'])
        
        self.threshold_label.config(text=f"{model_info['recognition_threshold']:.2f}")
        self.detection_label.config(text=f"{model_info['min_detection_confidence']:.2f}")
        
    def on_threshold_change(self, value):
        """Обработка изменения порога распознавания"""
        self.threshold_label.config(text=f"{float(value):.2f}")
        
    def on_detection_change(self, value):
        """Обработка изменения уверенности обнаружения"""
        self.detection_label.config(text=f"{float(value):.2f}")
        
    def save_settings(self):
        """Сохранение настроек"""
        try:
            self.face_recognizer.set_recognition_threshold(self.recognition_threshold.get())
            self.face_recognizer.set_detection_confidence(self.detection_confidence.get())
            
            messagebox.showinfo("Успех", "Настройки сохранены")
            self.destroy()
            
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            messagebox.showerror("Ошибка", f"Ошибка сохранения настроек: {e}")

class StatisticsDialog(tk.Toplevel):
    """Диалог статистики"""
    def __init__(self, parent, database, face_recognizer):
        super().__init__(parent)
        self.database = database
        self.face_recognizer = face_recognizer
        
        self.title("Статистика системы")
        self.geometry("500x400")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        
        self.create_widgets()
        self.load_statistics()
        
    def create_widgets(self):
        """Создание виджетов диалога"""
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="Статистика системы", font=('Helvetica', 16, 'bold')).pack(pady=(0, 20))
        
        # Фрейм для статистики
        self.stats_frame = ttk.Frame(main_frame)
        self.stats_frame.pack(fill=tk.BOTH, expand=True)
        
        # Кнопки
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        ttk.Button(button_frame, text="Обновить", command=self.load_statistics).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Экспорт...", command=self.export_statistics).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Закрыть", command=self.destroy).pack(side=tk.RIGHT, padx=5)
        
    def load_statistics(self):
        """Загрузка статистики"""
        # Очищаем фрейм
        for widget in self.stats_frame.winfo_children():
            widget.destroy()
            
        # Получаем статистику
        db_stats = self.database.get_database_stats()
        model_info = self.face_recognizer.get_model_info()
        embedding_stats = self.face_recognizer.get_embedding_statistics()
        
        # Отображаем статистику
        stats_data = [
            ("База данных:", ""),
            ("Всего людей:", f"{db_stats['total_people']}"),
            ("Всего фотографий:", f"{db_stats['total_photos']}"),
            ("Сессий распознавания:", f"{db_stats['total_sessions']}"),
            ("Средняя уверенность:", f"{db_stats['avg_confidence']:.2f}"),
            ("", ""),
            ("Модель:", ""),
            ("Загружено эмбеддингов:", f"{model_info['loaded_embeddings']}"),
            ("Уникальных лиц:", f"{model_info['unique_people']}"),
            ("Порог распознавания:", f"{model_info['recognition_threshold']:.2f}"),
            ("Уверенность обнаружения:", f"{model_info['min_detection_confidence']:.2f}"),
            ("", ""),
            ("Эмбеддинги:", ""),
            ("Статус:", f"{embedding_stats['status']}"),
        ]
        
        row = 0
        for label, value in stats_data:
            if label == "" and value == "":
                # Пустая строка
                ttk.Label(self.stats_frame, text="").grid(row=row, column=0, columnspan=2, pady=5)
            else:
                ttk.Label(self.stats_frame, text=label, font=('Helvetica', 10, 'bold')).grid(row=row, column=0, sticky=tk.W, pady=2)
                ttk.Label(self.stats_frame, text=value).grid(row=row, column=1, sticky=tk.W, pady=2, padx=(10, 0))
            row += 1
            
    def export_statistics(self):
        """Экспорт статистики в файл"""
        # Здесь можно реализовать экспорт в CSV или JSON
        messagebox.showinfo("Экспорт", "Функция экспорта будет реализована в будущей версии")