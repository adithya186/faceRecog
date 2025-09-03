import os
from typing import Optional


class Config:
    """Base configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-me')
    FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Database Configuration
    DATABASE_URL = os.environ.get('ATTENDANCE_DB_URL', 'sqlite:///attendance.db')
    
    # Face Recognition Configuration
    RECOG_UNKNOWN_DISTANCE = float(os.environ.get('RECOG_UNKNOWN_DISTANCE', '65'))
    
    # Server Configuration
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', '5000'))
    
    # Admin Configuration
    ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'admin123')
    
    # Security Configuration
    HTTPS_ONLY = os.environ.get('HTTPS_ONLY', 'False').lower() == 'true'
    SECURE_COOKIES = os.environ.get('SECURE_COOKIES', 'False').lower() == 'true'
    
    # File Upload Configuration
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'data', 'uploads')
    
    # Face Recognition Model Configuration
    MODEL_CONFIG = {
        'face_cascade_path': 'haarcascade_frontalface_default.xml',
        'face_size': (200, 200),
        'lbph_radius': 1,
        'lbph_neighbors': 8,
        'lbph_grid_x': 8,
        'lbph_grid_y': 8,
        'lbph_threshold': RECOG_UNKNOWN_DISTANCE
    }
    
    # Data Augmentation Configuration
    AUGMENTATION_CONFIG = {
        'num_variations': 5,
        'gaussian_blur_kernel': (3, 3),
        'bilateral_filter_d': 5,
        'bilateral_filter_sigma_color': 75,
        'bilateral_filter_sigma_space': 75
    }


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    FLASK_ENV = 'development'


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    FLASK_ENV = 'production'
    HTTPS_ONLY = True
    SECURE_COOKIES = True


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DATABASE_URL = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False


def get_config() -> Config:
    """Get configuration based on environment"""
    env = os.environ.get('FLASK_ENV', 'development')
    
    if env == 'production':
        return ProductionConfig()
    elif env == 'testing':
        return TestingConfig()
    else:
        return DevelopmentConfig()
