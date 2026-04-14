"""
Session manager for GOP GUI application
"""

import uuid
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class SessionManager:
    """Manager for user session management"""
    
    def __init__(self, storage_path: str = 'data/sessions') -> None:
        """
        Initialize session manager
        
        Args:
            storage_path: Path for session storage
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.sessions: Dict[str, Dict[str, Any]] = {}  # In-memory session cache
        self.session_timeout = 24  # hours
    
    def create_session(self, user_id: Optional[str] = None, expires_hours: int = 24) -> str:
        """
        Create a new session
        
        Args:
            user_id: User ID
            expires_hours: Session lifetime in hours
            
        Returns:
            Created session ID
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
        
        # Save to memory
        self.sessions[session_id] = session_data
        
        # Save to disk
        self._save_session_to_disk(session_id, session_data)
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session data
        
        Args:
            session_id: Session ID
            
        Returns:
            Session data or None if not found/expired
        """
        # Check in memory
        if session_id in self.sessions:
            session_data = self.sessions[session_id]
        else:
            # Try to load from disk
            session_data = self._load_session_from_disk(session_id)
            if session_data:
                self.sessions[session_id] = session_data
            else:
                return None
        
        # Check expiration
        expires_at = datetime.fromisoformat(session_data['expires_at'])
        if datetime.utcnow() > expires_at:
            self.delete_session(session_id)
            return None
        
        # Update last activity time
        session_data['last_activity'] = datetime.utcnow().isoformat()
        self._save_session_to_disk(session_id, session_data)
        
        return session_data
    
    def update_session(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """
        Update session data
        
        Args:
            session_id: Session ID
            session_data: New session data
            
        Returns:
            True if successful, False if session not found
        """
        if session_id not in self.sessions:
            # Try to load from disk
            existing_data = self._load_session_from_disk(session_id)
            if not existing_data:
                return False
            self.sessions[session_id] = existing_data
        
        # Update data
        session_data['last_activity'] = datetime.utcnow().isoformat()
        self.sessions[session_id] = session_data
        
        # Save to disk
        self._save_session_to_disk(session_id, session_data)
        
        return True
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete session
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful, False if session not found
        """
        # Delete from memory
        if session_id in self.sessions:
            del self.sessions[session_id]
        
        # Delete from disk
        session_file = self.storage_path / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()
            return True
        
        return False
    
    def add_project_to_session(self, session_id: str, project_data: Dict[str, Any]) -> bool:
        """
        Add project to session
        
        Args:
            session_id: Session ID
            project_data: Project data
            
        Returns:
            True if successful
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        project_data['created_at'] = datetime.utcnow().isoformat()
        session['projects'].append(project_data)
        
        return self.update_session(session_id, session)
    
    def get_current_project(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current project for session
        
        Args:
            session_id: Session ID
            
        Returns:
            Current project data or None
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
        Set current project for session
        
        Args:
            session_id: Session ID
            project_id: Project ID
            
        Returns:
            True if successful
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
        Add uploaded file information to session
        
        Args:
            session_id: Session ID
            file_data: File data
            
        Returns:
            True if successful
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        file_data['uploaded_at'] = datetime.utcnow().isoformat()
        session['uploaded_files'].append(file_data)
        
        return self.update_session(session_id, session)
    
    def add_processing_record(self, session_id: str, processing_data: Dict[str, Any]) -> bool:
        """
        Add processing record to session
        
        Args:
            session_id: Session ID
            processing_data: Processing data
            
        Returns:
            True if successful
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        processing_data['timestamp'] = datetime.utcnow().isoformat()
        session['processing_history'].append(processing_data)
        
        return self.update_session(session_id, session)
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions
        
        Returns:
            Number of deleted sessions
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
        """Save session to disk"""
        session_file = self.storage_path / f"{session_id}.json"
        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving session {session_id}: {e}")
    
    def _load_session_from_disk(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session from disk"""
        session_file = self.storage_path / f"{session_id}.json"
        if not session_file.exists():
            return None
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error loading session {session_id}: {e}")
            # Delete corrupted file
            try:
                session_file.unlink()
            except:
                pass
            return None
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get session statistics
        
        Returns:
            Session statistics
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