"""Desktop configuration for Face Verification System"""

import os
import yaml
from datetime import datetime

class DesktopConfig:
    """Desktop interface configuration"""
    
    def __init__(self, config_file=None):
        self.config_file = config_file or os.path.join(os.path.dirname(__file__), 'desktop.yaml')
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file"""
        default_config = {
            'interface': {
                'theme': 'default',
                'language': 'en',
                'window': {
                    'width': 1200,
                    'height': 800,
                    'maximized': False,
                    'remember_geometry': True
                },
                'display': {
                    'show_detections': True,
                    'show_landmarks': False,
                    'show_confidence': True
                }
            },
            'system': {
                'camera': {
                    'device_index': 0,
                    'resolution': '1280x720',
                    'fps': 30,
                    'buffer_size': 10
                },
                'processing': {
                    'max_fps': 30,
                    'multi_threading': True,
                    'batch_processing': True,
                    'cache_size': 100
                },
                'plugins': {
                    'default_detector': 'Basic Detector',
                    'default_recognizer': 'VGG Recognition',
                    'default_liveness': 'Basic Liveness'
                }
            },
            'performance': {
                'enable_gpu': False,
                'gpu_device': 0,
                'thread_pool_size': 4,
                'memory_limit_mb': 2048
            },
            'security': {
                'data_encryption': True,
                'log_all_operations': True,
                'require_admin_for_changes': False,
                'session_timeout_minutes': 60
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/desktop.log',
                'max_size_mb': 10,
                'backup_count': 5,
                'format': '[%(asctime)s] %(levelname)s - %(message)s'
            }
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    user_config = yaml.safe_load(f)
                    # Merge with defaults
                    self._merge_config(default_config, user_config)
                    return default_config
        except Exception as e:
            print(f"Warning: Could not load desktop config: {e}")
        
        return default_config
    
    def _merge_config(self, default, user):
        """Merge user configuration with defaults"""
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value
        return default
    
    def get(self, key, default=None):
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key, value):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        
        # Navigate to parent of key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        
        # Save to file
        self.save_config()
    
    def save_config(self):
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get_camera_config(self):
        """Get camera configuration"""
        return self.get('system.camera', {})
    
    def get_processing_config(self):
        """Get processing configuration"""
        return self.get('system.processing', {})
    
    def get_performance_config(self):
        """Get performance configuration"""
        return self.get('performance', {})
    
    def get_security_config(self):
        """Get security configuration"""
        return self.get('security', {})
    
    def get_interface_config(self):
        """Get interface configuration"""
        return self.get('interface', {})
    
    def update_from_command_line(self, args):
        """Update configuration from command line arguments"""
        # Process command line arguments
        if hasattr(args, 'config') and args.config:
            self.config_file = args.config
            self.config = self.load_config()
        
        if hasattr(args, 'gpu'):
            self.set('performance.enable_gpu', args.gpu)
        
        if hasattr(args, 'fps'):
            self.set('system.processing.max_fps', args.fps)
        
        if hasattr(args, 'resolution'):
            self.set('system.camera.resolution', args.resolution)

class DesktopIcons:
    """Icon management for desktop interface"""
    
    def __init__(self, config):
        self.config = config
        self.icon_cache = {}
    
    def get_icon_path(self, icon_name):
        """Get path to icon resource"""
        base_path = os.path.join(os.path.dirname(__file__), 'resources', 'icons')
        icon_path = os.path.join(base_path, f"{icon_name}.png")
        
        if os.path.exists(icon_path):
            return icon_path
        
        # Return default icon if not found
        return os.path.join(base_path, "default.png")
    
    def load_icon(self, icon_name, size=(32, 32)):
        """Load icon from file"""
        if icon_name in self.icon_cache:
            return self.icon_cache[icon_name]
        
        icon_path = self.get_icon_path(icon_name)
        if os.path.exists(icon_path):
            icon = QPixmap(icon_path).scaled(size[0], size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.icon_cache[icon_name] = icon
            return icon
        
        # Return default icon
        default_icon = QPixmap(size[0], size[1])
        default_icon.fill(Qt.lightGray)
        return default_icon

class DesktopTheme:
    """Theme management for desktop interface"""
    
    def __init__(self, config):
        self.config = config
        self.themes = self._load_themes()
    
    def _load_themes(self):
        """Load available themes"""
        themes = {
            'default': {
                'name': 'Default',
                'background': '#ffffff',
                'foreground': '#000000',
                'accent': '#007bff',
                'success': '#28a745',
                'warning': '#ffc107',
                'error': '#dc3545',
                'border': '#cccccc',
                'shadow': 'rgba(0, 0, 0, 0.1)'
            },
            'dark': {
                'name': 'Dark',
                'background': '#2b2b2b',
                'foreground': '#ffffff',
                'accent': '#4dabf7',
                'success': '#51cf66',
                'warning': '#ffd43b',
                'error': '#ff6b6b',
                'border': '#495057',
                'shadow': 'rgba(0, 0, 0, 0.3)'
            },
            'light': {
                'name': 'Light',
                'background': '#f8f9fa',
                'foreground': '#212529',
                'accent': '#0066cc',
                'success': '#28a745',
                'warning': '#ffc107',
                'error': '#dc3545',
                'border': '#dee2e6',
                'shadow': 'rgba(0, 0, 0, 0.05)'
            }
        }
        
        return themes
    
    def get_current_theme(self):
        """Get current theme configuration"""
        theme_name = self.config.get('interface.theme', 'default')
        return self.themes.get(theme_name, self.themes['default'])
    
    def get_stylesheet(self):
        """Generate stylesheet from current theme"""
        theme = self.get_current_theme()
        
        stylesheet = f"""
        QMainWindow {{
            background-color: {theme['background']};
            color: {theme['foreground']};
        }}
        
        QPushButton {{
            background-color: {theme['accent']};
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }}
        
        QPushButton:hover {{
            background-color: {self._adjust_color(theme['accent'], 1.1)};
        }}
        
        QPushButton:pressed {{
            background-color: {self._adjust_color(theme['accent'], 0.9)};
        }}
        
        QGroupBox {{
            border: 1px solid {theme['border']};
            border-radius: 4px;
            margin-top: 12px;
            padding-top: 8px;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 5px;
            background-color: {theme['background']};
            color: {theme['foreground']};
        }}
        
        QListWidget {{
            border: 1px solid {theme['border']};
            border-radius: 4px;
            background-color: {theme['background']};
            color: {theme['foreground']};
        }}
        
        QListWidget::item {{
            padding: 4px;
            border-bottom: 1px solid {theme['border']};
        }}
        
        QListWidget::item:hover {{
            background-color: {self._adjust_color(theme['background'], 0.95)};
        }}
        
        QTextEdit {{
            border: 1px solid {theme['border']};
            border-radius: 4px;
            background-color: {theme['background']};
            color: {theme['foreground']};
        }}
        
        QTabWidget::pane {{
            border: 1px solid {theme['border']};
            border-radius: 4px;
            background-color: {theme['background']};
        }}
        
        QTabBar::tab {{
            background-color: {theme['background']};
            color: {theme['foreground']};
            border: 1px solid {theme['border']};
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            padding: 6px 12px;
        }}
        
        QTabBar::tab:selected {{
            background-color: {theme['accent']};
            color: white;
        }}
        
        QStatusBar {{
            background-color: {theme['background']};
            color: {theme['foreground']};
            border-top: 1px solid {theme['border']};
        }}
        
        QProgressBar {{
            border: 1px solid {theme['border']};
            border-radius: 4px;
            text-align: center;
            background-color: {theme['background']};
        }}
        
        QProgressBar::chunk {{
            background-color: {theme['accent']};
        }}
        """
        
        return stylesheet
    
    def _adjust_color(self, color, factor):
        """Adjust color brightness"""
        # Simple color adjustment - in a real implementation, this would be more sophisticated
        if color.startswith('#'):
            # Convert hex to RGB
            hex_color = color[1:]
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            
            # Adjust
            r = int(min(255, max(0, r * factor)))
            g = int(min(255, max(0, g * factor)))
            b = int(min(255, max(0, b * factor)))
            
            # Convert back to hex
            return f"#{r:02x}{g:02x}{b:02x}"
        
        return color

def create_desktop_directories():
    """Create necessary directories for desktop application"""
    base_dir = os.path.dirname(__file__)
    
    directories = [
        'resources',
        'resources/icons',
        'resources/themes',
        'logs',
        'cache',
        'temp',
        'config'
    ]
    
    for directory in directories:
        path = os.path.join(base_dir, directory)
        os.makedirs(path, exist_ok=True)

def setup_desktop_environment():
    """Setup desktop application environment"""
    create_desktop_directories()
    
    # Create default configuration if it doesn't exist
    config_file = os.path.join(os.path.dirname(__file__), 'desktop.yaml')
    if not os.path.exists(config_file):
        config = DesktopConfig()
        config.save_config()