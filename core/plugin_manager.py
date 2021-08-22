"""Manages plugin discovery, loading and lifecycle."""

import os
import sys
import importlib
import importlib.util
from typing import Dict, List, Optional, Type, Any, Set
from pathlib import Path
import yaml
import json

from .base import (
    IPlugin, IDetector, IRecognizer, ILivenessDetector, 
    INotifier, IDevice, IProcessor, IStorage, DeviceType, PluginMetadata
)
from utils import get_logger

logger = get_logger('plugin_manager')





class PluginManager:
    """Plugin discovery and lifecycle management"""
    
    def __init__(self, plugin_dirs: Optional[List[str]] = None):
        self.plugin_dirs = plugin_dirs or []
        self.loaded_plugins: Dict[str, IPlugin] = {}
        self.plugin_types: Dict[str, List[IPlugin]] = {
            'detector': [],
            'recognizer': [],
            'liveness': [],
            'notifier': [],
            'device': [],
            'processor': [],
            'storage': []
        }
        self.plugin_registry: Dict[str, Dict[str, Any]] = {}
        self.device_plugins: Dict[DeviceType, IDevice] = {}
        
        # Add default plugin directories
        default_dirs = [
            os.path.join(os.path.dirname(__file__), '..', 'plugins'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plugins')
        ]
        
        # Add enhanced plugin directories
        enhanced_dirs = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'enhanced_detections'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'enhanced_recognition'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'enhanced_liveness'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'enhanced_notifications'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'enhanced_devices')
        ]
        
        all_dirs = default_dirs + enhanced_dirs
        
        for plugin_dir in all_dirs:
            if os.path.exists(plugin_dir) and plugin_dir not in self.plugin_dirs:
                self.plugin_dirs.append(plugin_dir)
    
    def add_plugin_directory(self, directory: str) -> bool:
        """Add a new plugin directory"""
        if os.path.exists(directory) and directory not in self.plugin_dirs:
            self.plugin_dirs.append(directory)
            logger.info(f"Added plugin directory: {directory}")
            return True
        return False
    
    def discover_plugins(self) -> List[Dict[str, Any]]:
        """Discover all available plugins in configured directories"""
        discovered_plugins = []
        
        for plugin_dir in self.plugin_dirs:
            if not os.path.exists(plugin_dir):
                continue
                
            logger.debug(f"Scanning for plugins in: {plugin_dir}")
            
            for root, dirs, files in os.walk(plugin_dir):
                # Look for plugin manifests or Python files
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Check for plugin manifest files
                    if file in ['plugin.yaml', 'plugin.yml', 'manifest.json']:
                        try:
                            manifest = self._load_plugin_manifest(file_path)
                            if manifest:
                                # Use the file field from manifest if present, otherwise use manifest path
                                if 'file' in manifest:
                                    python_file = os.path.join(root, manifest['file'])
                                    manifest['file_path'] = python_file
                                else:
                                    manifest['file_path'] = file_path
                                manifest['directory'] = root
                                discovered_plugins.append(manifest)
                        except Exception as e:
                            logger.error(f"Error loading plugin manifest {file_path}: {e}")
                    
                    # Check for Python plugin files
                    elif file.endswith('.py') and not file.startswith('__'):
                        try:
                            plugin_info = self._infer_plugin_info(file_path, root)
                            if plugin_info:
                                discovered_plugins.append(plugin_info)
                        except Exception as e:
                            logger.error(f"Error inferring plugin info {file_path}: {e}")
        
        logger.info(f"Discovered {len(discovered_plugins)} plugins")
        return discovered_plugins
    
    def _load_plugin_manifest(self, manifest_path: str) -> Optional[Dict[str, Any]]:
        """Load plugin manifest from YAML/JSON file"""
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                if manifest_path.endswith('.yaml') or manifest_path.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load plugin manifest {manifest_path}: {e}")
            return None
    
    def _infer_plugin_info(self, file_path: str, plugin_dir: str) -> Optional[Dict[str, Any]]:
        """Infer plugin information from Python file"""
        filename = os.path.basename(file_path)
        plugin_name = filename[:-3]  # Remove .py extension
        
        # Try to determine plugin type from directory structure
        relative_path = os.path.relpath(plugin_dir, os.path.dirname(file_path))
        plugin_type = os.path.basename(relative_path)
        
        # Validate plugin type
        valid_types = ['detection', 'recognition', 'liveness', 'notifications', 
                      'devices', 'processing', 'storage']
        if plugin_type not in valid_types:
            logger.debug(f"Skipping plugin with invalid type {plugin_type}: {file_path}")
            return None
        
        return {
            'name': plugin_name,
            'type': plugin_type,
            'file_path': file_path,
            'directory': plugin_dir,
            'version': '1.0.0',  # Default version
            'description': f'Inferred plugin from {filename}',
            'auto_load': True
        }
    
    def load_plugin(self, plugin_info: Dict[str, Any]) -> bool:
        """Load a single plugin"""
        try:
            plugin_name = plugin_info['name']
            file_path = plugin_info['file_path']
            
            # Check if already loaded
            if plugin_name in self.loaded_plugins:
                logger.warning(f"Plugin {plugin_name} already loaded, skipping")
                return True
            
            logger.info(f"Loading plugin: {plugin_name}")
            
            # Load the Python module
            spec = importlib.util.spec_from_file_location(plugin_name, file_path)
            if not spec or not spec.loader:
                logger.error(f"Could not load plugin spec for {plugin_name}")
                return False
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin class
            plugin_class = self._find_plugin_class(module, plugin_info['type'])
            if not plugin_class:
                logger.error(f"Could not find plugin class in {plugin_name}")
                return False
            
            # Instantiate plugin
            plugin_instance = plugin_class()
            
            # Get plugin metadata
            metadata = plugin_instance.get_metadata()
            logger.info(f"Plugin {metadata.name} v{metadata.version} loaded successfully")
            
            # Store plugin
            self.loaded_plugins[plugin_name] = plugin_instance
            
            # Add to type-specific list
            type_mapping = {
                'detection': 'detector',
                'recognition': 'recognizer', 
                'liveness': 'liveness',
                'notifications': 'notifier',
                'devices': 'device',
                'processing': 'processor',
                'storage': 'storage'
            }
            
            plugin_type = type_mapping.get(plugin_info['type'])
            if plugin_type:
                self.plugin_types[plugin_type].append(plugin_instance)
            
            # If device plugin, add to device plugins dict
            if isinstance(plugin_instance, IDevice):
                self.device_plugins[plugin_instance.get_device_type()] = plugin_instance
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_info.get('name', 'unknown')}: {e}")
            return False
    
    def _find_plugin_class(self, module, plugin_type: str) -> Optional[Type[IPlugin]]:
        """Find the plugin class in a module"""
        logger.debug(f"Searching for plugin class of type {plugin_type} in module {module.__name__}")
        
        # Mapping from plugin types to base classes
        type_to_base = {
            'detection': IDetector,
            'recognition': IRecognizer,
            'liveness': ILivenessDetector,
            'notifications': INotifier,
            'devices': IDevice,
            'processing': IProcessor,
            'storage': IStorage
        }
        
        base_class = type_to_base.get(plugin_type)
        if not base_class:
            logger.error(f"Unknown plugin type: {plugin_type}")
            return None
        
        logger.debug(f"Looking for classes inheriting from {base_class}")
        
        # Look for classes that inherit from the base class
        classes_found = []
        for name, obj in module.__dict__.items():
            if isinstance(obj, type):
                classes_found.append(f"{name}: {obj}")
                if (issubclass(obj, base_class) and 
                    obj != base_class):
                    logger.debug(f"Found plugin class: {name}")
                    return obj
        
        logger.debug(f"Classes found in module: {classes_found}")
        logger.debug(f"No plugin class found in module {module.__name__}")
        return None
    
    def load_plugins(self, plugin_list: Optional[List[Dict[str, Any]]] = None) -> int:
        """Load multiple plugins"""
        if plugin_list is None:
            plugin_list = self.discover_plugins()
        
        success_count = 0
        for plugin_info in plugin_list:
            if self.load_plugin(plugin_info):
                success_count += 1
        
        logger.info(f"Loaded {success_count}/{len(plugin_list)} plugins successfully")
        return success_count
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin"""
        if plugin_name not in self.loaded_plugins:
            logger.warning(f"Plugin {plugin_name} not loaded")
            return False
        
        try:
            plugin = self.loaded_plugins[plugin_name]
            plugin.cleanup()
            del self.loaded_plugins[plugin_name]
            
            # Remove from type-specific lists
            for plugin_list in self.plugin_types.values():
                if plugin in plugin_list:
                    plugin_list.remove(plugin)
            
            # Remove from device plugins if applicable
            for device_type, device_plugin in list(self.device_plugins.items()):
                if device_plugin == plugin:
                    del self.device_plugins[device_type]
            
            logger.info(f"Unloaded plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload plugin {plugin_name}: {e}")
            return False
    
    def get_plugins_by_type(self, plugin_type: str) -> List[IPlugin]:
        """Get all plugins of a specific type"""
        return self.plugin_types.get(plugin_type, [])
    
    def get_plugin_by_name(self, plugin_name: str) -> Optional[IPlugin]:
        """Get a specific plugin by name"""
        return self.loaded_plugins.get(plugin_name)
    
    def get_detector_plugins(self) -> List[IDetector]:
        """Get all face detection plugins"""
        return self.get_plugins_by_type('detector')
    
    def get_recognizer_plugins(self) -> List[IRecognizer]:
        """Get all face recognition plugins"""
        return self.get_plugins_by_type('recognizer')
    
    def get_liveness_detector_plugins(self) -> List[ILivenessDetector]:
        """Get all liveness detection plugins"""
        return self.get_plugins_by_type('liveness')
    
    def get_notifier_plugins(self) -> List[INotifier]:
        """Get all notification plugins"""
        return self.get_plugins_by_type('notifier')
    
    def get_device_plugins(self) -> Dict[DeviceType, IDevice]:
        """Get all device plugins"""
        return self.device_plugins
    
    def get_processor_plugins(self) -> List[IProcessor]:
        """Get all processing plugins"""
        return self.get_plugins_by_type('processor')
    
    def get_storage_plugins(self) -> List[IStorage]:
        """Get all storage plugins"""
        return self.get_plugins_by_type('storage')
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """Get information about all loaded plugins"""
        info = {
            'total_loaded': len(self.loaded_plugins),
            'by_type': {},
            'devices': list(self.device_plugins.keys()),
            'plugins': []
        }
        
        for plugin_type, plugins in self.plugin_types.items():
            info['by_type'][plugin_type] = len(plugins)
            
            for plugin in plugins:
                metadata = plugin.get_metadata()
                info['plugins'].append({
                    'name': metadata.name,
                    'version': metadata.version,
                    'type': plugin_type,
                    'author': metadata.author,
                    'description': metadata.description,
                    'device_compatibility': [dt.value for dt in metadata.device_compatibility] if metadata.device_compatibility else []
                })
        
        return info
    
    def initialize_plugins(self, config: Dict[str, Any]) -> bool:
        """Initialize all loaded plugins with configuration"""
        success_count = 0
        
        for plugin_name, plugin in self.loaded_plugins.items():
            try:
                # Get plugin-specific config
                plugin_config = config.get(plugin_name, {})
                # If the config has a 'config' key, use that (for nested configuration)
                if isinstance(plugin_config, dict) and 'config' in plugin_config:
                    plugin_config = plugin_config['config']
                logger.warning(f"Initializing plugin {plugin_name} with config: {plugin_config}")
                
                if plugin.initialize(plugin_config):
                    success_count += 1
                    logger.debug(f"Initialized plugin: {plugin_name}")
                else:
                    logger.warning(f"Failed to initialize plugin: {plugin_name}")
                    
            except Exception as e:
                logger.error(f"Error initializing plugin {plugin_name}: {e}")
        
        logger.info(f"Initialized {success_count}/{len(self.loaded_plugins)} plugins")
        return success_count == len(self.loaded_plugins)
    
    def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a plugin"""
        # First unload
        if plugin_name in self.loaded_plugins:
            if not self.unload_plugin(plugin_name):
                return False
        
        # Find plugin info and reload
        for plugin_dir in self.plugin_dirs:
            plugin_info = self._find_plugin_info_by_name(plugin_name, plugin_dir)
            if plugin_info:
                return self.load_plugin(plugin_info)
        
        logger.error(f"Could not find plugin info for {plugin_name}")
        return False
    
    def _find_plugin_info_by_name(self, plugin_name: str, plugin_dir: str) -> Optional[Dict[str, Any]]:
        """Find plugin info by name in a directory"""
        if not os.path.exists(plugin_dir):
            return None
        
        for root, dirs, files in os.walk(plugin_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith('.py') and not file.startswith('__'):
                    inferred_info = self._infer_plugin_info(file_path, root)
                    if inferred_info and inferred_info['name'] == plugin_name:
                        return inferred_info
        
        return None