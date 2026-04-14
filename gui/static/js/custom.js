/**
 * Кастомный JavaScript для GUI приложения GOP
 */

// Глобальные переменные
let gopApp = {
    session: null,
    currentProject: null,
    processingTasks: new Map(),
    notifications: []
};

// Инициализация приложения при загрузке DOM
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

/**
 * Инициализация приложения
 */
function initializeApp() {
    console.log('Инициализация GOP GUI приложения...');
    
    // Инициализация компонентов
    initializeFileUpload();
    initializeModals();
    initializeTooltips();
    initializeNotifications();
    
    // Установка обработчиков событий
    setupEventListeners();
    
    // Проверка сессии
    checkSession();
    
    console.log('GOP GUI приложение инициализировано');
}

/**
 * Инициализация компонента загрузки файлов
 */
function initializeFileUpload() {
    // Обработка старых upload-area компонентов (для совместимости)
    const uploadAreas = document.querySelectorAll('.upload-area');
    
    uploadAreas.forEach(area => {
        // Предотвращение стандартного поведения drag & drop
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            area.addEventListener(eventName, preventDefaults, false);
        });
        
        // Подсветка при перетаскивании
        ['dragenter', 'dragover'].forEach(eventName => {
            area.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            area.addEventListener(eventName, unhighlight, false);
        });
        
        // Обработка сброшенных файлов
        area.addEventListener('drop', handleDrop, false);
        
        // Обработка клика для выбора файлов
        area.addEventListener('click', function() {
            const fileInput = area.querySelector('input[type="file"]');
            if (fileInput) {
                fileInput.click();
            }
        });
    });
    
    // Обработка новых компонентов загрузки файлов
    const dropzones = document.querySelectorAll('#file-upload-dropzone');
    const fileInputs = document.querySelectorAll('#file-upload-input');
    
    dropzones.forEach(dropzone => {
        // Предотвращение стандартного поведения drag & drop
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, preventDefaults, false);
        });
        
        // Подсветка при перетаскивании
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, unhighlight, false);
        });
        
        // Обработка сброшенных файлов
        dropzone.addEventListener('drop', function(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            const fileInput = document.querySelector('#file-upload-input');
            if (fileInput) {
                fileInput.files = files;
                handleFileSelect({ currentTarget: fileInput });
            }
        }, false);
        
        // Обработка клика для выбора файлов
        dropzone.addEventListener('click', function() {
            const fileInput = document.querySelector('#file-upload-input');
            if (fileInput) {
                fileInput.click();
            }
        });
    });
    
    fileInputs.forEach(input => {
        input.addEventListener('change', handleFileSelect);
    });
}

/**
 * Инициализация модальных окон
 */
function initializeModals() {
    const modals = document.querySelectorAll('.modal');
    
    modals.forEach(modal => {
        modal.addEventListener('show.bs.modal', function(event) {
            const button = event.relatedTarget;
            const modalId = modal.id;
            
            // Дополнительная логика при открытии модального окна
            console.log(`Открытие модального окна: ${modalId}`);
        });
        
        modal.addEventListener('hidden.bs.modal', function(event) {
            const modalId = modal.id;
            
            // Очистка формы при закрытии
            const form = modal.querySelector('form');
            if (form) {
                form.reset();
            }
            
            console.log(`Закрытие модального окна: ${modalId}`);
        });
    });
}

/**
 * Инициализация всплывающих подсказок
 */
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Инициализация системы уведомлений
 */
function initializeNotifications() {
    // Создание контейнера для уведомлений если его нет
    if (!document.getElementById('notification-container')) {
        const container = document.createElement('div');
        container.id = 'notification-container';
        container.className = 'notification-container';
        container.style.cssText = `
            position: fixed;
            top: 80px;
            right: 20px;
            z-index: 1050;
            max-width: 400px;
        `;
        document.body.appendChild(container);
    }
}

/**
 * Установка обработчиков событий
 */
function setupEventListeners() {
    // Обработка навигации
    const navLinks = document.querySelectorAll('[data-nav-target]');
    navLinks.forEach(link => {
        link.addEventListener('click', handleNavigation);
    });
    
    // Обработка форм
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', handleFormSubmit);
    });
    
    // Обработка кнопок (только для кнопок с data-action, исключая кнопки с id)
    const actionButtons = document.querySelectorAll('[data-action]:not([id])');
    actionButtons.forEach(button => {
        button.addEventListener('click', handleActionButton);
    });
    
    // Обработка изменения файлов
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', handleFileSelect);
    });
}

/**
 * Проверка сессии пользователя
 */
async function checkSession() {
    try {
        const response = await fetch('/api/session/check');
        const data = await response.json();
        
        if (data.valid) {
            gopApp.session = data.session;
            updateUIForAuthenticatedUser();
        } else {
            updateUIForAnonymousUser();
        }
    } catch (error) {
        console.error('Ошибка проверки сессии:', error);
        updateUIForAnonymousUser();
    }
}

/**
 * Обработка навигации
 */
function handleNavigation(event) {
    event.preventDefault();
    
    const target = event.currentTarget.dataset.navTarget;
    const mainContent = document.getElementById('main-content');
    
    // Показ индикатора загрузки
    showLoadingIndicator(mainContent);
    
    // Загрузка содержимого страницы
    loadPageContent(target);
}

/**
 * Обработка отправки форм
 */
async function handleFormSubmit(event) {
    event.preventDefault();
    
    const form = event.currentTarget;
    const formData = new FormData(form);
    const action = form.dataset.action;
    
    try {
        showLoadingIndicator(form);
        
        const response = await fetch(form.action, {
            method: form.method,
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            showNotification('success', data.message || 'Операция выполнена успешно');
            
            // Закрытие модального окна если форма в нем
            const modal = form.closest('.modal');
            if (modal) {
                bootstrap.Modal.getInstance(modal).hide();
            }
            
            // Обновление содержимого если нужно
            if (data.reload) {
                location.reload();
            }
        } else {
            showNotification('danger', data.error || 'Произошла ошибка');
        }
    } catch (error) {
        console.error('Ошибка отправки формы:', error);
        showNotification('danger', 'Произошла ошибка при отправке формы');
    } finally {
        hideLoadingIndicator(form);
    }
}

/**
 * Обработка кнопок действий
 */
async function handleActionButton(event) {
    const button = event.currentTarget;
    const action = button.dataset.action;
    const target = button.dataset.target;
    
    try {
        button.disabled = true;
        showLoadingIndicator(button);
        
        switch (action) {
            case 'create-project':
                await createProject();
                break;
            case 'upload-files':
                await uploadFiles();
                break;
            case 'start-processing':
                await startProcessing();
                break;
            case 'delete-project':
                await deleteProject(target);
                break;
            default:
                console.warn(`Неизвестное действие: ${action}`);
        }
    } catch (error) {
        console.error('Ошибка выполнения действия:', error);
        showNotification('danger', 'Произошла ошибка при выполнении действия');
    } finally {
        button.disabled = false;
        hideLoadingIndicator(button);
    }
}

/**
 * Обработка выбора файлов
 */
function handleFileSelect(event) {
    const input = event.currentTarget;
    const files = input.files;
    const fileList = document.getElementById('upload-file-list');
    const fileInfo = document.getElementById('selected-files-info');
    
    if (fileList && files.length > 0) {
        fileList.innerHTML = '';
        
        Array.from(files).forEach(file => {
            const fileItem = createFileItem(file);
            fileList.appendChild(fileItem);
        });
        
        // Показ информации о выбранных файлах
        if (fileInfo) {
            const totalSize = Array.from(files).reduce((sum, file) => sum + file.size, 0);
            fileInfo.innerHTML = `
                <div class="alert alert-info">
                    <strong>Выбрано файлов:</strong> ${files.length}<br>
                    <strong>Общий размер:</strong> ${formatFileSize(totalSize)}
                </div>
            `;
        }
        
        // Активация кнопки загрузки
        const uploadButton = document.getElementById('upload-files-btn');
        if (uploadButton) {
            uploadButton.disabled = false;
        }
    } else if (fileInfo) {
        fileInfo.innerHTML = '';
    }
}

/**
 * Создание элемента списка файлов
 */
function createFileItem(file) {
    const div = document.createElement('div');
    div.className = 'file-item d-flex justify-content-between align-items-center p-2 border-bottom';
    
    const fileInfo = document.createElement('div');
    fileInfo.innerHTML = `
        <div class="d-flex align-items-center">
            <i class="fas fa-file me-2 text-primary"></i>
            <div>
                <div class="fw-semibold">${file.name}</div>
                <small class="text-muted">${formatFileSize(file.size)}</small>
            </div>
        </div>
    `;
    
    const removeButton = document.createElement('button');
    removeButton.className = 'btn btn-sm btn-outline-danger';
    removeButton.innerHTML = '<i class="fas fa-times"></i>';
    removeButton.onclick = function() {
        div.remove();
        updateUploadButtonState();
    };
    
    div.appendChild(fileInfo);
    div.appendChild(removeButton);
    
    return div;
}

/**
 * Загрузка содержимого страницы
 */
async function loadPageContent(page) {
    try {
        const response = await fetch(`/pages/${page}.html`);
        const html = await response.text();
        
        const mainContent = document.getElementById('main-content');
        mainContent.innerHTML = html;
        
        // Переинициализация компонентов для нового содержимого
        initializeFileUpload();
        initializeTooltips();
        
    } catch (error) {
        console.error('Ошибка загрузки страницы:', error);
        showNotification('danger', 'Не удалось загрузить страницу');
    }
}

/**
 * Создание проекта
 */
async function createProject() {
    const name = document.getElementById('project-name-input').value;
    const description = document.getElementById('project-description-input').value;
    
    if (!name) {
        showNotification('warning', 'Введите название проекта');
        return;
    }
    
    try {
        const response = await fetch('/api/projects', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ name, description })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showNotification('success', 'Проект создан успешно');
            gopApp.currentProject = data;
            updateProjectList();
        } else {
            showNotification('danger', data.error || 'Ошибка создания проекта');
        }
    } catch (error) {
        console.error('Ошибка создания проекта:', error);
        showNotification('danger', 'Произошла ошибка при создании проекта');
    }
}

/**
 * Загрузка файлов с использованием потоковой передачи
 */
async function uploadFiles() {
    const fileInput = document.querySelector('#file-upload-input');
    const files = fileInput.files;
    
    if (files.length === 0) {
        showNotification('warning', 'Выберите файлы для загрузки');
        return;
    }
    
    // Проверка размера файлов
    const totalSize = Array.from(files).reduce((sum, file) => sum + file.size, 0);
    const maxSize = 10 * 1024 * 1024 * 1024; // 10GB
    
    if (totalSize > maxSize) {
        showNotification('danger', 'Общий размер файлов превышает допустимый лимит (10GB)');
        return;
    }
    
    const formData = new FormData();
    Array.from(files).forEach(file => {
        formData.append('files', file);
    });
    
    // Получение ID текущего проекта из URL или хранилища
    let projectId = null;
    if (gopApp.currentProject && gopApp.currentProject.id) {
        projectId = gopApp.currentProject.id;
    } else {
        // Попытка получить ID проекта из URL
        const urlPath = window.location.pathname;
        const projectMatch = urlPath.match(/\/project\/([^\/]+)/);
        if (projectMatch) {
            projectId = projectMatch[1];
        }
    }
    
    if (!projectId) {
        showNotification('danger', 'Не удалось определить проект для загрузки файлов');
        return;
    }
    
    const uploadUrl = `/api/projects/${projectId}/files/streaming`;
    
    try {
        // Показ индикатора загрузки
        const uploadButton = document.getElementById('upload-files-modal-btn');
        if (uploadButton) {
            uploadButton.disabled = true;
            uploadButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status"></span>Загрузка...';
        }
        
        const response = await fetch(uploadUrl, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showNotification('success', `Файлы загружены успешно (${data.uploaded_files.length})`);
            updateFileList(data.uploaded_files);
            
            // Закрытие модального окна
            const modal = document.getElementById('upload-files-modal');
            if (modal) {
                const modalInstance = bootstrap.Modal.getInstance(modal);
                if (modalInstance) {
                    modalInstance.hide();
                }
            }
            
            // Сброс формы
            const form = document.getElementById('file-upload-form');
            if (form) {
                form.reset();
            }
            
            // Очистка списка файлов
            const fileList = document.getElementById('upload-file-list');
            if (fileList) {
                fileList.innerHTML = '';
            }
            
            const fileInfo = document.getElementById('selected-files-info');
            if (fileInfo) {
                fileInfo.innerHTML = '';
            }
        } else {
            showNotification('danger', data.error || 'Ошибка загрузки файлов');
        }
    } catch (error) {
        console.error('Ошибка загрузки файлов:', error);
        showNotification('danger', 'Произошла ошибка при загрузке файлов: ' + error.message);
    } finally {
        // Восстановление кнопки
        const uploadButton = document.getElementById('upload-files-modal-btn');
        if (uploadButton) {
            uploadButton.disabled = false;
            uploadButton.innerHTML = 'Upload';
        }
    }
}

/**
 * Показ уведомления
 */
function showNotification(type, message) {
    const container = document.getElementById('notification-container');
    const notificationId = 'notification-' + Date.now();
    
    const notification = document.createElement('div');
    notification.id = notificationId;
    notification.className = `alert alert-${type} alert-dismissible fade show`;
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    container.appendChild(notification);
    
    // Автоматическое скрытие через 5 секунд
    setTimeout(() => {
        const alert = bootstrap.Alert.getOrCreateInstance(notification);
        alert.close();
    }, 5000);
}

/**
 * Показ индикатора загрузки
 */
function showLoadingIndicator(element) {
    const indicator = document.createElement('div');
    indicator.className = 'loading-overlay';
    indicator.innerHTML = `
        <div class="spinner-border text-primary" role="status">
            <span class="visually-hidden">Загрузка...</span>
        </div>
    `;
    
    element.style.position = 'relative';
    element.appendChild(indicator);
}

/**
 * Скрытие индикатора загрузки
 */
function hideLoadingIndicator(element) {
    const indicator = element.querySelector('.loading-overlay');
    if (indicator) {
        indicator.remove();
    }
}

/**
 * Вспомогательные функции
 */
function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight(e) {
    e.currentTarget.classList.add('drag-over');
}

function unhighlight(e) {
    e.currentTarget.classList.remove('drag-over');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    const fileInput = e.currentTarget.querySelector('input[type="file"]');
    if (fileInput) {
        fileInput.files = files;
        handleFileSelect({ currentTarget: fileInput });
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function updateUIForAuthenticatedUser() {
    // Обновление интерфейса для аутентифицированного пользователя
    console.log('Пользователь аутентифицирован');
}

function updateUIForAnonymousUser() {
    // Обновление интерфейса для анонимного пользователя
    console.log('Анонимный пользователь');
}

function updateProjectList() {
    // Обновление списка проектов
    console.log('Обновление списка проектов');
}

function updateFileList(files) {
    // Обновление списка файлов
    console.log('Обновление списка файлов:', files);
}

function updateUploadButtonState() {
    const fileInput = document.querySelector('#file-upload input[type="file"]');
    const uploadButton = document.getElementById('upload-files-btn');
    
    if (uploadButton && fileInput) {
        uploadButton.disabled = fileInput.files.length === 0;
    }
}

// Экспорт функций для использования в других скриптах
window.gopApp = gopApp;
window.showNotification = showNotification;
window.showLoadingIndicator = showLoadingIndicator;
window.hideLoadingIndicator = hideLoadingIndicator;