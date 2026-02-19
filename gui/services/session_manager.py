"""
Менеджер сессий для GUI приложения GOP
"""

import uuid
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pathlib import Path


class SessionManager:
    """Менеджер для управления сессиями пользователей"""
    
    def __init__(self, storage_path: str = 'data/sessions'):
        """
        Инициализация менеджера сессий
        
        Args:
            storage_path: Путь для хранения сессий
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.sessions = {}  # In-memory кэш сессий
        self.session_timeout = 24  # часы
    
    def create_session(self, user_id: Optional[str] = None, expires_hours: int = 24) -> str:
        """
        Создание новой сессии
        
        Args:
            user_id: ID пользователя
            expires_hours: Время жизни сессии в часах
            
        Returns:
            ID созданной сессии
        """
        session_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(hours=expires_hours)
        
        session_data = {
            'session_id': session_id,
            'user_id': user_id,
            'created_at': datetime.utcnow().isoformat(),
            'expires_at': expires_at.isoformat(),
            'last_activity': datetime.utcnow().isoformat(),
            'projects': [],
            'current_project': None,
            'preferences': {
                'theme': 'light',
                'language': 'ru',
                'default_indices': ['NDVI', 'EVI']
            },
            'uploaded_files': [],
            'processing_history': []
        }
        
        # Сохранение в память
        self.sessions[session_id] = session_data
        
        # Сохранение на диск
        self._save_session_to_disk(session_id, session_data)
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение данных сессии
        
        Args:
            session_id: ID сессии
            
        Returns:
            Данные сессии или None если не найдена/просрочена
        """
        # Проверка в памяти
        if session_id in self.sessions:
            session_data = self.sessions[session_id]
        else:
            # Попытка загрузить с диска
            session_data = self._load_session_from_disk(session_id)
            if session_data:
                self.sessions[session_id] = session_data
            else:
                return None
        
        # Проверка срока действия
        expires_at = datetime.fromisoformat(session_data['expires_at'])
        if datetime.utcnow() > expires_at:
            self.delete_session(session_id)
            return None
        
        # Обновление времени последней активности
        session_data['last_activity'] = datetime.utcnow().isoformat()
        self._save_session_to_disk(session_id, session_data)
        
        return session_data
    
    def update_session(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """
        Обновление данных сессии
        
        Args:
            session_id: ID сессии
            session_data: Новые данные сессии
            
        Returns:
            True если успешно, False если сессия не найдена
        """
        if session_id not in self.sessions:
            # Попытка загрузить с диска
            existing_data = self._load_session_from_disk(session_id)
            if not existing_data:
                return False
            self.sessions[session_id] = existing_data
        
        # Обновление данных
        session_data['last_activity'] = datetime.utcnow().isoformat()
        self.sessions[session_id] = session_data
        
        # Сохранение на диск
        self._save_session_to_disk(session_id, session_data)
        
        return True
    
    def delete_session(self, session_id: str) -> bool:
        """
        Удаление сессии
        
        Args:
            session_id: ID сессии
            
        Returns:
            True если успешно, False если сессия не найдена
        """
        # Удаление из памяти
        if session_id in self.sessions:
            del self.sessions[session_id]
        
        # Удаление с диска
        session_file = self.storage_path / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()
            return True
        
        return False
    
    def add_project_to_session(self, session_id: str, project_data: Dict[str, Any]) -> bool:
        """
        Добавление проекта в сессию
        
        Args:
            session_id: ID сессии
            project_data: Данные проекта
            
        Returns:
            True если успешно
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        project_data['created_at'] = datetime.utcnow().isoformat()
        session['projects'].append(project_data)
        
        return self.update_session(session_id, session)
    
    def get_current_project(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение текущего проекта сессии
        
        Args:
            session_id: ID сессии
            
        Returns:
            Данные текущего проекта или None
        """
        session = self.get_session(session_id)
        if not session or not session['current_project']:
            return None
        
        project_id = session['current_project']
        for project in session['projects']:
            if project['id'] == project_id:
                return project
        
        return None
    
    def set_current_project(self, session_id: str, project_id: str) -> bool:
        """
        Установка текущего проекта
        
        Args:
            session_id: ID сессии
            project_id: ID проекта
            
        Returns:
            True если успешно
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        # Проверка существования проекта
        project_exists = any(p['id'] == project_id for p in session['projects'])
        if not project_exists:
            return False
        
        session['current_project'] = project_id
        return self.update_session(session_id, session)
    
    def add_uploaded_file(self, session_id: str, file_data: Dict[str, Any]) -> bool:
        """
        Добавление информации о загруженном файле
        
        Args:
            session_id: ID сессии
            file_data: Данные файла
            
        Returns:
            True если успешно
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        file_data['uploaded_at'] = datetime.utcnow().isoformat()
        session['uploaded_files'].append(file_data)
        
        return self.update_session(session_id, session)
    
    def add_processing_record(self, session_id: str, processing_data: Dict[str, Any]) -> bool:
        """
        Добавление записи об обработке
        
        Args:
            session_id: ID сессии
            processing_data: Данные обработки
            
        Returns:
            True если успешно
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        processing_data['timestamp'] = datetime.utcnow().isoformat()
        session['processing_history'].append(processing_data)
        
        return self.update_session(session_id, session)
    
    def cleanup_expired_sessions(self) -> int:
        """
        Очистка просроченных сессий
        
        Returns:
            Количество удаленных сессий
        """
        deleted_count = 0
        current_time = datetime.utcnow()
        
        # Проверка сессий в памяти
        expired_sessions = []
        for session_id, session_data in self.sessions.items():
            expires_at = datetime.fromisoformat(session_data['expires_at'])
            if current_time > expires_at:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.delete_session(session_id)
            deleted_count += 1
        
        # Проверка сессий на диске
        for session_file in self.storage_path.glob("*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                expires_at = datetime.fromisoformat(session_data['expires_at'])
                if current_time > expires_at:
                    session_file.unlink()
                    deleted_count += 1
            except (json.JSONDecodeError, KeyError, ValueError):
                # Удаление поврежденных файлов
                session_file.unlink()
                deleted_count += 1
        
        return deleted_count
    
    def _save_session_to_disk(self, session_id: str, session_data: Dict[str, Any]) -> None:
        """Сохранение сессии на диск"""
        session_file = self.storage_path / f"{session_id}.json"
        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения сессии {session_id}: {e}")
    
    def _load_session_from_disk(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Загрузка сессии с диска"""
        session_file = self.storage_path / f"{session_id}.json"
        if not session_file.exists():
            return None
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Ошибка загрузки сессии {session_id}: {e}")
            # Удаление поврежденного файла
            try:
                session_file.unlink()
            except:
                pass
            return None
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Получение статистики сессий
        
        Returns:
            Статистика сессий
        """
        total_sessions = len(self.sessions)
        active_sessions = 0
        
        current_time = datetime.utcnow()
        for session_data in self.sessions.values():
            expires_at = datetime.fromisoformat(session_data['expires_at'])
            if current_time <= expires_at:
                active_sessions += 1
        
        return {
            'total_sessions': total_sessions,
            'active_sessions': active_sessions,
            'storage_path': str(self.storage_path),
            'session_timeout_hours': self.session_timeout
        }