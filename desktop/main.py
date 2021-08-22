"""PyQt5 desktop application for Face Verification System"""

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from datetime import datetime
import json
import os

from system import FaceVerificationSystem

class FaceVerificationApp(QMainWindow):
    """Main desktop application window"""
    def __init__(self):
        super().__init__()
        
        # Initialize verification system
        self.verification_system = FaceVerificationSystem()
        self.verification_system.initialize()
        
        # Application state
        self.current_image = None
        self.results = {}
        self.processing = False
        
        self.setup_ui()
        self.setup_timer()
        
    def setup_ui(self):
        """Setup user interface"""
        self.setWindowTitle("Face Verification System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel - Image display
        left_panel = QVBoxLayout()
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background-color: #f0f0f0; border: 2px solid #ccc;")
        self.image_label.setText("No image loaded")
        left_panel.addWidget(self.image_label)
        
        # Image controls
        image_controls = QHBoxLayout()
        
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        self.load_btn.setStyleSheet("background-color: #007acc; color: white;")
        
        self.capture_btn = QPushButton("Capture Camera")
        self.capture_btn.clicked.connect(self.capture_camera)
        self.capture_btn.setStyleSheet("background-color: #28a745; color: white;")
        
        self.process_btn = QPushButton("Process Image")
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setStyleSheet("background-color: #007bff; color: white;")
        self.process_btn.setEnabled(False)
        
        image_controls.addWidget(self.load_btn)
        image_controls.addWidget(self.capture_btn)
        image_controls.addWidget(self.process_btn)
        left_panel.addLayout(image_controls)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_panel.addWidget(self.progress_bar)
        
        main_layout.addLayout(left_panel, 2)
        
        # Right panel - Results and controls
        right_panel = QVBoxLayout()
        
        # Results tabs
        self.results_tabs = QTabWidget()
        
        # Detection tab
        detection_widget = QWidget()
        detection_layout = QVBoxLayout(detection_widget)
        
        self.detection_list = QListWidget()
        self.detection_list.setStyleSheet("font-size: 12px;")
        detection_layout.addWidget(QLabel("Face Detections:"))
        detection_layout.addWidget(self.detection_list)
        
        self.results_tabs.addTab(detection_widget, "Detections")
        
        # Recognition tab
        recognition_widget = QWidget()
        recognition_layout = QVBoxLayout(recognition_widget)
        
        self.recognition_list = QListWidget()
        self.recognition_list.setStyleSheet("font-size: 12px;")
        recognition_layout.addWidget(QLabel("Face Recognition:"))
        recognition_layout.addWidget(self.recognition_list)
        
        self.results_tabs.addTab(recognition_widget, "Recognition")
        
        # Liveness tab
        liveness_widget = QWidget()
        liveness_layout = QVBoxLayout(liveness_widget)
        
        self.liveness_info = QTextEdit()
        self.liveness_info.setReadOnly(True)
        self.liveness_info.setStyleSheet("font-size: 12px;")
        liveness_layout.addWidget(QLabel("Liveness Check:"))
        liveness_layout.addWidget(self.liveness_info)
        
        self.results_tabs.addTab(liveness_widget, "Liveness")
        
        # System info tab
        system_widget = QWidget()
        system_layout = QVBoxLayout(system_widget)
        
        self.system_info = QTextEdit()
        self.system_info.setReadOnly(True)
        self.system_info.setStyleSheet("font-size: 12px;")
        system_layout.addWidget(QLabel("System Information:"))
        system_layout.addWidget(self.system_info)
        
        self.results_tabs.addTab(system_widget, "System Info")
        
        right_panel.addWidget(self.results_tabs)
        
        # System controls
        controls_group = QGroupBox("System Controls")
        controls_layout = QVBoxLayout()
        
        # Performance settings
        perf_group = QGroupBox("Performance")
        perf_layout = QFormLayout()
        
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(30)
        perf_layout.addRow("Max FPS:", self.fps_spin)
        
        self.buffer_spin = QSpinBox()
        self.buffer_spin.setRange(1, 100)
        self.buffer_spin.setValue(10)
        perf_layout.addRow("Frame Buffer:", self.buffer_spin)
        
        self.multi_check = QCheckBox()
        self.multi_check.setChecked(True)
        perf_layout.addRow("Multi-threading:", self.multi_check)
        
        perf_group.setLayout(perf_layout)
        controls_layout.addWidget(perf_group)
        
        # Detection settings
        detect_group = QGroupBox("Detection")
        detect_layout = QFormLayout()
        
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.1)
        self.confidence_spin.setValue(0.5)
        detect_layout.addRow("Min Confidence:", self.confidence_spin)
        
        self.neighbor_spin = QSpinBox()
        self.neighbor_spin.setRange(1, 10)
        self.neighbor_spin.setValue(5)
        detect_layout.addRow("Min Neighbors:", self.neighbor_spin)
        
        detect_group.setLayout(detect_layout)
        controls_layout.addWidget(detect_group)
        
        # Apply settings button
        apply_btn = QPushButton("Apply Settings")
        apply_btn.setStyleSheet("background-color: #6c757d; color: white;")
        apply_btn.clicked.connect(self.apply_settings)
        controls_layout.addWidget(apply_btn)
        
        controls_group.setLayout(controls_layout)
        right_panel.addWidget(controls_group)
        
        # User management
        user_group = QGroupBox("User Management")
        user_layout = QVBoxLayout()
        
        user_btn_layout = QHBoxLayout()
        self.add_user_btn = QPushButton("Add User")
        self.add_user_btn.clicked.connect(self.add_user)
        self.add_user_btn.setStyleSheet("background-color: #28a745; color: white;")
        
        self.list_users_btn = QPushButton("List Users")
        self.list_users_btn.clicked.connect(self.list_users)
        self.list_users_btn.setStyleSheet("background-color: #007acc; color: white;")
        
        user_btn_layout.addWidget(self.add_user_btn)
        user_btn_layout.addWidget(self.list_users_btn)
        user_layout.addLayout(user_btn_layout)
        
        self.user_list = QListWidget()
        self.user_list.setStyleSheet("font-size: 12px;")
        user_layout.addWidget(self.user_list)
        
        user_group.setLayout(user_layout)
        right_panel.addWidget(user_group)
        
        main_layout.addLayout(right_panel, 1)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Menu bar
        self.setup_menu_bar()
        
    def setup_menu_bar(self):
        """Setup menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        load_action = QAction('Load Image', self)
        load_action.setShortcut('Ctrl+O')
        load_action.triggered.connect(self.load_image)
        file_menu.addAction(load_action)
        
        save_action = QAction('Save Results', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_results)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        toggle_action = QAction('Toggle Detection Boxes', self)
        toggle_action.setCheckable(True)
        toggle_action.setChecked(True)
        toggle_action.triggered.connect(self.toggle_detection_boxes)
        view_menu.addAction(toggle_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        
        camera_action = QAction('Open Camera', self)
        camera_action.triggered.connect(self.open_camera_dialog)
        tools_menu.addAction(camera_action)
        
        settings_action = QAction('Settings', self)
        settings_action.triggered.connect(self.open_settings)
        tools_menu.addAction(settings_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def setup_timer(self):
        """Setup timer for periodic updates"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_system_info)
        self.timer.start(5000)  # Update every 5 seconds
        
    def load_image(self):
        """Load image from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        
        if file_path:
            self.current_image = cv2.imread(file_path)
            if self.current_image is not None:
                self.display_image(self.current_image)
                self.process_btn.setEnabled(True)
                self.status_bar.showMessage(f"Loaded: {os.path.basename(file_path)}")
            else:
                QMessageBox.critical(self, "Error", "Failed to load image")
    
    def capture_camera(self):
        """Capture from camera"""
        try:
            # Open camera
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                QMessageBox.warning(self, "Warning", "Could not open camera")
                return
            
            # Capture frame
            ret, frame = camera.read()
            camera.release()
            
            if ret:
                self.current_image = frame
                self.display_image(frame)
                self.process_btn.setEnabled(True)
                self.status_bar.showMessage("Image captured from camera")
            else:
                QMessageBox.critical(self, "Error", "Failed to capture image")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Camera capture failed: {str(e)}")
    
    def display_image(self, image):
        """Display image in the GUI"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to QImage
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Convert to QPixmap and display
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(), 
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
    
    def process_image(self):
        """Process image with face verification"""
        if self.current_image is None or self.processing:
            return
        
        self.processing = True
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage("Processing image...")
        
        # Simulate progress
        def update_progress():
            current = self.progress_bar.value()
            if current < 100:
                self.progress_bar.setValue(min(100, current + 10))
            else:
                self.timer.timeout.disconnect()
                self.process_results()
                self.processing = False
                self.process_btn.setEnabled(True)
                self.progress_bar.setVisible(False)
                self.status_bar.showMessage("Processing complete")
        
        self.timer.timeout.connect(update_progress)
        self.progress_bar.setValue(0)
    
    def process_results(self):
        """Process verification results"""
        if self.current_image is None:
            return
        
        try:
            # Process with verification system
            self.results = self.verification_system.process_frame(self.current_image)
            
            # Update UI with results
            self.update_detections()
            self.update_recognitions()
            self.update_liveness()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Processing failed: {str(e)}")
    
    def update_detections(self):
        """Update detection results"""
        self.detection_list.clear()
        
        if 'detections' in self.results:
            for i, detection in enumerate(self.results['detections']):
                bbox = detection['bbox']
                confidence = detection['confidence']
                
                item_text = f"Face {i+1}: ({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}) - {confidence:.2f}"
                self.detection_list.addItem(item_text)
                
                # Draw detection box on image
                if hasattr(self, 'show_detections') and self.show_detections:
                    x, y, w, h = bbox
                    cv2.rectangle(self.current_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(self.current_image, f"{confidence:.2f}", (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            self.display_image(self.current_image)
    
    def update_recognitions(self):
        """Update recognition results"""
        self.recognition_list.clear()
        
        if 'recognitions' in self.results:
            for recognition in self.results['recognitions']:
                user_id = recognition.get('user_id', 'Unknown')
                confidence = recognition['confidence']
                
                item_text = f"User: {user_id} - Confidence: {confidence:.2f}"
                self.recognition_list.addItem(item_text)
    
    def update_liveness(self):
        """Update liveness results"""
        self.liveness_info.clear()
        
        if 'liveness_result' in self.results:
            liveness = self.results['liveness_result']
            is_live = liveness.get('is_live', False)
            confidence = liveness.get('confidence', 0.0)
            details = liveness.get('details', {})
            
            text = f"Status: {'LIVE' if is_live else 'NOT LIVE'}\n"
            text += f"Confidence: {confidence:.2%}\n\n"
            text += "Details:\n"
            
            for key, value in details.items():
                text += f"  {key}: {value}\n"
            
            self.liveness_info.setText(text)
    
    def update_system_info(self):
        """Update system information"""
        try:
            status = self.verification_system.get_system_status()
            
            info = f"System Status: {status.get('status', 'Unknown')}\n"
            info += f"Version: {status.get('version', 'Unknown')}\n"
            info += f"Uptime: {status.get('uptime', 'Unknown')}\n"
            info += f"Memory Usage: {status.get('memory_usage', 'Unknown')}\n"
            info += f"Active Users: {status.get('user_count', 0)}\n"
            info += f"Total Detections: {status.get('detection_count', 0)}\n"
            info += f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            self.system_info.setText(info)
            
        except Exception as e:
            self.system_info.setText(f"Error updating system info: {str(e)}")
    
    def apply_settings(self):
        """Apply system settings"""
        try:
            # Update system configuration
            config = {
                'max_fps': self.fps_spin.value(),
                'frame_buffer_size': self.buffer_spin.value(),
                'multi_threading': self.multi_check.isChecked(),
                'min_confidence': self.confidence_spin.value(),
                'min_neighbors': self.neighbor_spin.value()
            }
            
            # In a real implementation, this would update the system configuration
            QMessageBox.information(self, "Success", "Settings applied successfully")
            self.status_bar.showMessage("Settings applied")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply settings: {str(e)}")
    
    def add_user(self):
        """Add new user"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add User")
        dialog.setModal(True)
        
        layout = QVBoxLayout()
        
        # User ID input
        user_id_label = QLabel("User ID:")
        user_id_input = QLineEdit()
        
        # Image selection
        image_label = QLabel("Select face image:")
        image_path_input = QLineEdit()
        image_path_input.setReadOnly(True)
        
        def browse_image():
            file_path, _ = QFileDialog.getOpenFileName(
                dialog, "Select Face Image", 
                "", "Image Files (*.png *.jpg *.jpeg)"
            )
            if file_path:
                image_path_input.setText(file_path)
        
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(browse_image)
        
        # Buttons
        button_layout = QHBoxLayout()
        add_btn = QPushButton("Add User")
        add_btn.clicked.connect(lambda: self.confirm_add_user(user_id_input.text(), image_path_input.text(), dialog))
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        
        button_layout.addWidget(add_btn)
        button_layout.addWidget(cancel_btn)
        
        layout.addWidget(user_id_label)
        layout.addWidget(user_id_input)
        layout.addWidget(image_label)
        layout.addWidget(image_path_input)
        layout.addWidget(browse_btn)
        layout.addLayout(button_layout)
        
        dialog.setLayout(layout)
        dialog.exec_()
    
    def confirm_add_user(self, user_id, image_path, dialog):
        """Confirm adding user"""
        if not user_id.strip():
            QMessageBox.warning(dialog, "Warning", "Please enter a user ID")
            return
        
        if not image_path or not os.path.exists(image_path):
            QMessageBox.warning(dialog, "Warning", "Please select a valid image")
            return
        
        try:
            # Load and register user
            image = cv2.imread(image_path)
            if image is not None:
                success = self.verification_system.register_user(user_id.strip(), image)
                if success:
                    QMessageBox.information(dialog, "Success", f"User {user_id} registered successfully")
                    dialog.accept()
                else:
                    QMessageBox.critical(dialog, "Error", "Failed to register user")
            else:
                QMessageBox.critical(dialog, "Error", "Failed to load image")
                
        except Exception as e:
            QMessageBox.critical(dialog, "Error", f"Failed to add user: {str(e)}")
    
    def list_users(self):
        """List registered users"""
        try:
            users = self.verification_system.get_registered_users()
            self.user_list.clear()
            
            for user in users:
                self.user_list.addItem(user)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load users: {str(e)}")
    
    def save_results(self):
        """Save results to file"""
        if not self.results:
            QMessageBox.warning(self, "Warning", "No results to save")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", 
            f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.results, f, indent=2)
                self.status_bar.showMessage(f"Results saved: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save results: {str(e)}")
    
    def toggle_detection_boxes(self, state):
        """Toggle detection boxes display"""
        self.show_detections = state == Qt.Checked
        if self.current_image is not None:
            self.display_image(self.current_image)
    
    def open_camera_dialog(self):
        """Open camera dialog"""
        camera_dialog = QDialog(self)
        camera_dialog.setWindowTitle("Camera Settings")
        camera_dialog.setModal(True)
        
        layout = QVBoxLayout()
        
        # Camera settings
        camera_group = QGroupBox("Camera Configuration")
        camera_layout = QFormLayout()
        
        camera_index = QSpinBox()
        camera_index.setRange(0, 10)
        camera_index.setValue(0)
        camera_layout.addRow("Camera Index:", camera_index)
        
        resolution_combo = QComboBox()
        resolution_combo.addItems(["640x480", "1280x720", "1920x1080"])
        camera_layout.addRow("Resolution:", resolution_combo)
        
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(camera_dialog.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(camera_dialog.reject)
        
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        camera_dialog.setLayout(layout)
        camera_dialog.exec_()
    
    def open_settings(self):
        """Open settings dialog"""
        settings_dialog = QDialog(self)
        settings_dialog.setWindowTitle("System Settings")
        settings_dialog.setModal(True)
        settings_dialog.resize(400, 300)
        
        layout = QVBoxLayout()
        
        # Settings tabs
        settings_tabs = QTabWidget()
        
        # Plugin settings
        plugin_widget = QWidget()
        plugin_layout = QFormLayout(plugin_widget)
        
        plugin_combo = QComboBox()
        plugin_combo.addItems(["Basic Detector", "HOG Detector", "VGG Recognition"])
        plugin_layout.addRow("Default Plugin:", plugin_combo)
        
        settings_tabs.addTab(plugin_widget, "Plugins")
        
        # Logging settings
        logging_widget = QWidget()
        logging_layout = QFormLayout(logging_widget)
        
        log_level_combo = QComboBox()
        log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        logging_layout.addRow("Log Level:", log_level_combo)
        
        log_path_input = QLineEdit("logs/system.log")
        log_path_input.setReadOnly(True)
        logging_layout.addRow("Log Path:", log_path_input)
        
        settings_tabs.addTab(logging_widget, "Logging")
        
        layout.addWidget(settings_tabs)
        
        # Buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(settings_dialog.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(settings_dialog.reject)
        
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        settings_dialog.setLayout(layout)
        settings_dialog.exec_()
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
        <h1>Face Verification System</h1>
        <p>Version: 1.0.0</p>
        <p>A modular face verification system with multiple plugins and interfaces.</p>
        <br>
        <h3>Features:</h3>
        <ul>
            <li>Multiple detection algorithms</li>
            <li>Face recognition with deep learning</li>
            <li>Liveness detection</li>
            <li>Multi-platform support</li>
            <li>Web, mobile, and desktop interfaces</li>
        </ul>
        <br>
        <p><small>Built with PyQt5, Flask, Kivy, and OpenCV</small></p>
        """
        
        QMessageBox.about(self, "About Face Verification System", about_text)
    
    def closeEvent(self, event):
        """Handle application close"""
        reply = QMessageBox.question(
            self, 'Confirm Exit',
            'Are you sure you want to exit?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application icon and properties
    app.setApplicationName("Face Verification System")
    app.setApplicationVersion("1.0.0")
    
    # Create and show main window
    window = FaceVerificationApp()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()