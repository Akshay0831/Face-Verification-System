"""Kivy mobile application for Face Verification System"""

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.camera import Camera
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.utils import platform
import cv2
import numpy as np
import base64
import io
from datetime import datetime
import json
import os

from system import FaceVerificationSystem

class MainScreen(Screen):
    """Main screen with camera preview"""
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        
        # Initialize verification system
        self.verification_system = FaceVerificationSystem()
        self.verification_system.initialize()
        
        self.setup_ui()
        self.is_processing = False
        
    def setup_ui(self):
        """Setup user interface"""
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Title
        title = Label(
            text='Face Verification',
            font_size='24sp',
            size_hint_y=None,
            height=50,
            halign='center',
            color=(0, 0, 0, 1)
        )
        title.bind(size=title.setter, texture=title.setter)
        main_layout.add_widget(title)
        
        # Camera widget
        if platform == 'android':
            self.camera = Camera(resolution=(640, 480), play=True)
        else:
            # Fallback for desktop
            self.camera = Image(source='', size_hint_y=0.7)
            # Create a placeholder for the camera feed
            self.camera.bind(on_texture=self.update_camera_placeholder)
        
        main_layout.add_widget(self.camera)
        
        # Control buttons
        button_layout = BoxLayout(size_hint_y=None, height=60, spacing=10)
        
        capture_btn = Button(
            text='Capture',
            background_color=(0, 0.8, 0, 1),
            font_size='18sp'
        )
        capture_btn.bind(on_press=self.capture_image)
        
        switch_btn = Button(
            text='Switch Camera',
            background_color=(0.5, 0.5, 1, 1),
            font_size='18sp'
        )
        switch_btn.bind(on_press=self.switch_camera)
        
        button_layout.add_widget(capture_btn)
        button_layout.add_widget(switch_btn)
        main_layout.add_widget(button_layout)
        
        # Status label
        self.status_label = Label(
            text='Ready to capture',
            font_size='14sp',
            size_hint_y=None,
            height=40,
            halign='center',
            color=(0, 0, 0, 1)
        )
        main_layout.add_widget(self.status_label)
        
        # Progress bar
        self.progress = ProgressBar(
            max=100,
            value=0,
            size_hint_y=None,
            height=20
        )
        self.progress.opacity = 0
        main_layout.add_widget(self.progress)
        
        self.add_widget(main_layout)
        
    def update_camera_placeholder(self, instance, value):
        """Update camera placeholder (desktop fallback)"""
        # In a real implementation, this would connect to a camera feed
        pass
    
    def capture_image(self, instance):
        """Capture image from camera"""
        if self.is_processing:
            return
            
        if platform == 'android':
            # Get camera image
            buffer = self.camera.texture.pixels
            img = np.frombuffer(buffer, dtype=np.uint8)
            img = img.reshape(self.camera.texture.size[1], self.camera.texture.size[0], 4)
            # Convert RGBA to BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            # For desktop, use a test image
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "Test Image", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        self.process_image(img)
        
    def process_image(self, image):
        """Process captured image for face verification"""
        self.is_processing = True
        self.status_label.text = 'Processing...'
        self.progress.opacity = 1
        self.progress.value = 0
        
        def update_progress(dt):
            current = self.progress.value
            if current < 100:
                self.progress.value = min(100, current + 10)
            else:
                Clock.unschedule(update_progress)
        
        # Start progress animation
        Clock.schedule_interval(update_progress, 0.1)
        
        # Process the image
        try:
            result = self.verification_system.process_frame(image)
            
            # Convert image to displayable format
            display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = display_img.shape[:2]
            texture = Texture.create(size=(width, height))
            texture.blit_buffer(display_img.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
            
            # Update UI with results
            self.show_results(result, texture)
            
        except Exception as e:
            self.status_label.text = f'Error: {str(e)}'
        finally:
            self.is_processing = False
            self.progress.opacity = 0
            Clock.unschedule(update_progress)
    
    def show_results(self, result, image_texture):
        """Show verification results"""
        result_popup = Popup(
            title='Verification Results',
            size_hint=(0.9, 0.8),
            auto_dismiss=True
        )
        
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Display captured image
        img_display = Image(texture=image_texture, size_hint_y=0.4)
        layout.add_widget(img_display)
        
        # Results text
        results_text = Label(
            text=self.format_results(result),
            font_size='14sp',
            size_hint_y=None,
            height=100,
            halign='left',
            valign='top'
        )
        results_text.bind(width=lambda instance, value: instance.setter('text_size')(instance, (value, None)))
        layout.add_widget(results_text)
        
        # Action buttons
        button_layout = BoxLayout(size_hint_y=None, height=50, spacing=10)
        
        if 'recognitions' in result and result['recognitions']:
            # User recognized - show options
            register_btn = Button(text='Register Face', background_color=(0, 0.8, 0, 1))
            register_btn.bind(on_press=lambda x: self.register_face(result))
            button_layout.add_widget(register_btn)
        
        close_btn = Button(text='Close', background_color=(0.8, 0, 0, 1))
        close_btn.bind(on_press=lambda x: result_popup.dismiss())
        button_layout.add_widget(close_btn)
        
        layout.add_widget(button_layout)
        result_popup.content = layout
        result_popup.open()
    
    def format_results(self, result):
        """Format results as readable text"""
        text = f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        if 'detections' in result:
            text += f"Faces Detected: {len(result['detections'])}\n"
            for i, detection in enumerate(result['detections']):
                text += f"  Face {i+1}: Confidence {(detection['confidence'] * 100):.1f}%\n"
            text += "\n"
        
        if 'recognitions' in result:
            text += f"Users Recognized: {len(result['recognitions'])}\n"
            for i, recognition in enumerate(result['recognitions']):
                text += f"  {recognition.get('user_id', 'Unknown')}: {(recognition['confidence'] * 100):.1f}%\n"
            text += "\n"
        
        if 'liveness_result' in result:
            liveness = result['liveness_result']
            text += f"Liveness Check: {'Live' if liveness.get('is_live', False) else 'Not Live'}\n"
            text += f"Confidence: {(liveness.get('confidence', 0) * 100):.1f}%\n"
        
        return text
    
    def register_face(self, result):
        """Register face for recognized user"""
        if 'recognitions' not in result or not result['recognitions']:
            return
        
        # Get first recognized user
        user_id = result['recognitions'][0].get('user_id', 'unknown')
        
        # Get image from result
        if 'image' in result:
            img_data = result['image']
            success = self.verification_system.register_user(user_id, img_data)
            
            if success:
                self.status_label.text = f'Face registered for {user_id}'
            else:
                self.status_label.text = f'Failed to register face for {user_id}'
    
    def switch_camera(self, instance):
        """Switch between front/back camera (Android only)"""
        if platform == 'android' and hasattr(self, 'camera'):
            # This would implement camera switching in a real app
            self.status_label.text = 'Camera switching not implemented yet'
    
    def on_leave(self):
        """Clean up when leaving screen"""
        if platform == 'android' and hasattr(self, 'camera'):
            self.camera.play = False

class UsersScreen(Screen):
    """User management screen"""
    def __init__(self, **kwargs):
        super(UsersScreen, self).__init__(**kwargs)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup user interface"""
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Title
        title = Label(
            text='User Management',
            font_size='24sp',
            size_hint_y=None,
            height=50,
            halign='center',
            color=(0, 0, 0, 1)
        )
        main_layout.add_widget(title)
        
        # Users list (placeholder)
        self.users_label = Label(
            text='Loading users...',
            font_size='16sp',
            size_hint_y=None,
            height=200,
            halign='left',
            valign='top'
        )
        self.users_label.bind(width=lambda instance, value: instance.setter('text_size')(instance, (value, None)))
        main_layout.add_widget(self.users_label)
        
        # Add user button
        add_user_btn = Button(
            text='Add User',
            background_color=(0, 0.8, 0, 1),
            font_size='18sp',
            size_hint_y=None,
            height=50
        )
        add_user_btn.bind(on_press=self.add_user)
        main_layout.add_widget(add_user_btn)
        
        # Back button
        back_btn = Button(
            text='Back',
            background_color=(0.8, 0, 0, 1),
            font_size='18sp',
            size_hint_y=None,
            height=50
        )
        back_btn.bind(on_press=self.go_back)
        main_layout.add_widget(back_btn)
        
        self.add_widget(main_layout)
        
        # Load users
        Clock.schedule_once(self.load_users, 0.5)
        
    def load_users(self, dt):
        """Load users from system"""
        try:
            users = self.verification_system.get_registered_users()
            text = "Registered Users:\n\n"
            for user in users:
                text += f"• {user}\n"
            
            if not users:
                text = "No registered users found"
            
            self.users_label.text = text
        except Exception as e:
            self.users_label.text = f"Error loading users: {str(e)}"
    
    def add_user(self, instance):
        """Show add user dialog"""
        add_popup = Popup(
            title='Add User',
            size_hint=(0.9, 0.6),
            auto_dismiss=True
        )
        
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # User ID input
        user_id_input = TextInput(
            hint_text='Enter User ID',
            font_size='16sp',
            size_hint_y=None,
            height=50
        )
        layout.add_widget(Label(text='User ID:'))
        layout.add_widget(user_id_input)
        
        # Buttons
        button_layout = BoxLayout(size_hint_y=None, height=50, spacing=10)
        
        add_btn = Button(text='Add', background_color=(0, 0.8, 0, 1))
        add_btn.bind(on_press=lambda x: self.add_user_confirm(user_id_input.text, add_popup))
        
        cancel_btn = Button(text='Cancel', background_color=(0.8, 0, 0, 1))
        cancel_btn.bind(on_press=lambda x: add_popup.dismiss())
        
        button_layout.add_widget(add_btn)
        button_layout.add_widget(cancel_btn)
        layout.add_widget(button_layout)
        
        add_popup.content = layout
        add_popup.open()
    
    def add_user_confirm(self, user_id, popup):
        """Confirm adding user"""
        if user_id.strip():
            # In a real implementation, this would capture a face image
            try:
                # Create a dummy image for registration
                dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
                success = self.verification_system.register_user(user_id.strip(), dummy_image)
                
                if success:
                    self.load_users(1)  # Reload users list
                    popup.dismiss()
                    show_simple_popup("Success", f"User {user_id} registered successfully")
                else:
                    show_simple_popup("Error", "Failed to register user")
            except Exception as e:
                show_simple_popup("Error", str(e))
        else:
            show_simple_popup("Error", "Please enter a user ID")
    
    def go_back(self, instance):
        """Go back to main screen"""
        self.manager.current = 'main'

class FaceVerificationApp(App):
    """Main Kivy application"""
    def build(self):
        """Build the application"""
        # Create screen manager
        sm = ScreenManager()
        
        # Add screens
        main_screen = MainScreen(name='main')
        users_screen = UsersScreen(name='users')
        
        sm.add_widget(main_screen)
        sm.add_widget(users_screen)
        
        return sm

def show_simple_popup(title, message):
    """Show a simple popup with message"""
    popup = Popup(
        title=title,
        size_hint=(0.8, 0.4),
        auto_dismiss=True
    )
    
    layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
    layout.add_widget(Label(text=message, font_size='16sp'))
    close_btn = Button(text='OK', size_hint_y=None, height=50)
    close_btn.bind(on_press=lambda x: popup.dismiss())
    layout.add_widget(close_btn)
    
    popup.content = layout
    popup.open()

if __name__ == '__main__':
    FaceVerificationApp().run()