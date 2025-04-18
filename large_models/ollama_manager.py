import os
import subprocess
import requests
import json
import time
from pathlib import Path
import base64
from PIL import Image
import io
import numpy as np

class OllamaMacManager:
    def __init__(self, model, port, service_name):
        self.home = str(Path.home())
        self.launch_agents_dir = os.path.join(self.home, "Library", "LaunchAgents")
        self.ensure_directory_exists(self.launch_agents_dir)
        self.model, self.port, self.service_name = model, port, service_name
        
    def ensure_directory_exists(self, path):
        os.makedirs(path, exist_ok=True)
    
    def install_ollama(self):
        """Install Ollama if not already installed"""
        if not os.path.exists("/usr/local/bin/ollama"):
            print("Installing Ollama...")
            subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
            print("Ollama installed successfully.")
        else:
            print("Ollama is already installed.")
    
    def pull_model(self, ):
        model_name = self.model
        """Pull a specific model"""
        print(f"Pulling model: {model_name}...")
        subprocess.run(["ollama", "pull", model_name], check=True)
        print(f"Model {model_name} pulled successfully.")
    
    def create_service(self,):
        service_name = self.service_name
        port = self.port
        model = self.model
        """Create a launchd service for Ollama"""
        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{service_name}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/ollama</string>
        <string>serve</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>OLLAMA_HOST</key>
        <string>127.0.0.1:{port}</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/{service_name}.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/{service_name}.err</string>
</dict>
</plist>"""
        plist_path = os.path.join(self.launch_agents_dir, f"{service_name}.plist")
        with open(plist_path, 'w') as f:
            f.write(plist_content)
        
        print(f"Service {service_name} created at {plist_path}")
        # Load the service
        subprocess.run(["launchctl", "enable", f"gui/501/{service_name}"], check=True)
        subprocess.run(["launchctl", "kickstart", f"gui/501/{service_name}"], check=True)
        print(f"Service {service_name} loaded successfully on port {port}")
        
        if model:
            time.sleep(2)  # Give the service time to start
            self.set_default_model()

    def set_default_model(self, ):
        port = self.port
        model_name = self.model
        """Set the default model for a service"""
        url = f"http://localhost:{port}/api/pull"
        headers = {"Content-Type": "application/json"}
        data = {"name": model_name}
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            print(f"Default model set to {model_name} for service on port {port}")
        except Exception as e:
            print(f"Error setting default model: {e}")
    
    def start_service(self, ):
        service_name = self.service_name
        """Start a specific service"""
        plist_path = os.path.join(self.launch_agents_dir, f"{service_name}.plist")
        subprocess.run(["launchctl", "start", service_name], check=True)
        print(f"Service {service_name} started")
    
    def stop_service(self, ):
        """Stop a specific service"""
        service_name = self.service_name
        subprocess.run(["launchctl", "stop", service_name], check=True)
        print(f"Service {service_name} stopped")
    
    def restart_service(self, ):
        service_name = self.service_name
        """Restart a specific service"""
        self.stop_service(service_name)
        time.sleep(1)
        self.start_service(service_name)
    
    def remove_service(self, ):
        """Remove a service completely"""
        service_name = self.service_name
        self.stop_service(service_name)
        plist_path = os.path.join(self.launch_agents_dir, f"{service_name}.plist")
        if os.path.exists(plist_path):
            os.remove(plist_path)
            print(f"Service {service_name} removed")
        else:
            print(f"Service file not found at {plist_path}")
    
    def encode_image(image_path):
        with Image.open(image_path) as img:
            # Convert to RGB if not already (handles PNG transparency)
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Resize if too large (optional)
            max_size = 1024
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size))
            
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=90)
            return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    
    def query_service(self, messages,timeout=30):
        """Query a running Ollama service"""
        port = self.port
        url = f"http://localhost:{port}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
            
        data = {
            "messages": messages,
            "stream": False,
            "model": self.model
        }
        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=data)
            print(f"time spent: {time.time() - start_time} seconds")
            response.raise_for_status()
            response_text = response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error querying service on port {port}: {e}")
            return None
        import ipdb;ipdb.set_trace()
        return response_text

    
    def list_services(self):
        """List all Ollama services"""
        services = []
        for file in os.listdir(self.launch_agents_dir):
            if file.startswith("ollama") and file.endswith(".plist"):
                services.append(file.replace(".plist", ""))
        return services
    
    def check_service_status(self, ):
        """Check if a service is running on a specific port"""
        port = self.port
        try:
            result = subprocess.run(["lsof", "-i", f":{port}"], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE,
                                  text=True)
            return "ollama" in result.stdout
        except subprocess.CalledProcessError:
            return False


class OllamaUbuntuManager:
    def __init__(self, model, port, service_name):
        self.model_name = model
        self.port = port
        self.service_name = service_name
        self.systemd_dir = "/etc/systemd/system"
        self.ollama_bin = "/usr/bin/ollama"
        self.ensure_directory_exists(self.systemd_dir)
    
    def ensure_directory_exists(self, path):
        os.makedirs(path, exist_ok=True)
    
    def run_command(self, cmd, sudo=False):
        """Run a shell command with optional sudo"""
        if sudo:
            cmd = ["sudo"] + cmd
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Command failed: {' '.join(cmd)}")
            print(f"Error: {result.stderr}")
            return False
        return True
    
    def install_ollama(self):
        """Install Ollama on Ubuntu"""
        if not os.path.exists(self.ollama_bin):
            print("Installing Ollama...")
            commands = [
                ["curl", "-fsSL", "https://ollama.com/install.sh"],
                ["sudo", "bash"]
            ]
            for cmd in commands:
                if not self.run_command(cmd):
                    return False
            print("Ollama installed successfully.")
        else:
            print("Ollama is already installed.")
        return True
    
    def pull_model(self, ):
        """Pull a specific model"""
        model_name = self.model_name

        print(f"Pulling model: {model_name}...")
        return self.run_command(["ollama", "pull", model_name])
    
    def create_service(self,):
        """Create a systemd service for Ollama"""
        service_name = self.service_name
        model = self.model
        port = self.port
        service_content = f"""[Unit]
Description=Ollama {service_name} Service
After=network-online.target

[Service]
ExecStart={self.ollama_bin} serve
Environment=OLLAMA_HOST=127.0.0.1:{port}
User={os.getenv('USER')}
Group={os.getenv('USER')}
Restart=always
RestartSec=3
StandardOutput=file:/var/log/{service_name}.log
StandardError=file:/var/log/{service_name}.err

[Install]
WantedBy=multi-user.target
"""
        
        service_path = os.path.join(self.systemd_dir, f"{service_name}.service")

        try:
            with open(service_path, 'w') as f:
                f.write(service_content)
            
            # Reload systemd and enable service
            commands = [
                ["systemctl", "daemon-reload"],
                ["systemctl", "enable", f"{service_name}.service"],
                ["systemctl", "start", f"{service_name}.service"]
            ]
            
            for cmd in commands:
                if not self.run_command(cmd, sudo=True):
                    return False
            
            print(f"Service {service_name} created and started on port {port}")
            
            if model:
                time.sleep(2)  # Give the service time to start
                self.set_default_model(port, model)
            
            return True
        except Exception as e:
            print(f"Error creating service: {e}")
            return False
    
    def set_default_model(self,):
        port = self.port
        model_name = self.model_name
        """Set the default model for a service"""
        url = f"http://localhost:{port}/api/pull"
        headers = {"Content-Type": "application/json"}
        data = {"name": model_name}
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            print(f"Default model set to {model_name} for service on port {port}")
            return True
        except Exception as e:
            print(f"Error setting default model: {e}")
            return False
    
    def service_action(self, service_name, action):
        """Start/stop/restart a service"""
        actions = ["start", "stop", "restart", "status"]
        if action not in actions:
            print(f"Invalid action. Must be one of: {', '.join(actions)}")
            return False
        
        return self.run_command(["systemctl", action, f"{service_name}.service"], sudo=True)
    
    def remove_service(self, ):
        """Remove a service completely"""
        service_name = self.service_name
        commands = [
            ["systemctl", "stop", f"{service_name}.service"],
            ["systemctl", "disable", f"{service_name}.service"],
            ["rm", "-f", f"/etc/systemd/system/{service_name}.service"],
            ["systemctl", "daemon-reload"]
        ]
        
        success = True
        for cmd in commands:
            if not self.run_command(cmd, sudo=True):
                success = False
        
        if success:
            print(f"Service {service_name} removed successfully")
        return success
    
    def query_service(self, messages,  timeout=60):
        port = self.port
        """Query a running Ollama service"""
        url = f"http://localhost:{port}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "messages": messages,
            "stream": False,
            "model":self.model_name
        }
        try:
            response = requests.post(url, headers=headers, json=data, )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error querying service on port {port}: {e}")
            return None
    
    def list_services(self):
        """List all Ollama services"""
        try:
            result = subprocess.run(["systemctl", "list-units", "--all", "--type=service", "--no-pager", "|", "grep", "ollama"], 
                                  shell=True,
                                  capture_output=True,
                                  text=True)
            services = [line.split()[0] for line in result.stdout.splitlines() if "ollama" in line]
            return services
        except Exception as e:
            print(f"Error listing services: {e}")
            return []
    
    def check_service_status(self, port):
        """Check if a service is running on a specific port"""
        port = self.port

        try:
            result = subprocess.run(["ss", "-tulnp", "|", "grep", f":{port}"], 
                                  shell=True,
                                  capture_output=True,
                                  text=True)
            return "ollama" in result.stdout
        except Exception as e:
            print(f"Error checking port status: {e}")
            return False

import platform

def get_os_info():
    system = platform.system().lower()
    if system == "darwin":
        return "mac"
    elif system == "linux":
        # Check for specific Linux distribution
        try:
            with open('/etc/os-release') as f:
                os_release = f.read()
            if 'ubuntu' in os_release.lower():
                return "ubuntu"
            return "linux"  # Generic Linux if not Ubuntu
        except FileNotFoundError:
            return "linux"
    else:
        return "unknown"

current_os = get_os_info()

if current_os == "mac":
    OllamaManager = OllamaMacManager
elif current_os == "ubuntu":
    OllamaManager = OllamaUbuntuManager
else:
    raise NotImplementedError(f"Unsupported OS: {current_os}")

def main():
    manager = OllamaManager()
    
    # Example setup for two services
    SERVICE1_PORT = 11434
    SERVICE2_PORT = 11435
    MODEL1 = "llama2"
    MODEL2 = "mistral"
    
    # Install Ollama if needed
    manager.install_ollama()
    
    # Pull models
    manager.pull_model(MODEL1)
    manager.pull_model(MODEL2)
    
    # Create and start services
    manager.create_service("ollama1", SERVICE1_PORT, MODEL1)
    manager.create_service("ollama2", SERVICE2_PORT, MODEL2)
    
    # Wait for services to initialize
    time.sleep(5)
    
    # Example queries
    prompt = "Explain the concept of artificial intelligence in simple terms"
    
    print("\nQuerying first service (Llama2):")
    response1 = manager.query_service(prompt, SERVICE1_PORT)
    print(response1)
    
    print("\nQuerying second service (Mistral):")
    response2 = manager.query_service(prompt, SERVICE2_PORT, MODEL2)
    print(response2)
    
    # List all services
    print("\nCurrent services:")
    print(manager.list_services())

if __name__ == "__main__":
    main()