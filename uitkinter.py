import tkinter as tk
from tkinter import simpledialog, messagebox
import requests
import pyttsx3
import threading
import time
from datetime import datetime

class AmbulanceConsole:
    def __init__(self, root):
        self.root = root
        self.root.withdraw()
        
        # --- 1. SETUP CONNECTION ---
        server_ip = simpledialog.askstring("System Boot", "ENTER SERVER IP (Laptop B):", initialvalue="192.168.1.X")
        if not server_ip:
            root.destroy()
            return

        self.server_url = f"http://{server_ip}:5000/api/assignments"
        
        # --- 2. CONFIGURE WINDOW ---
        self.root.deiconify()
        self.root.title("MANIPUR EMERGENCY COMMAND - UNIT 1")
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg="#050505") # Deep Black

        # Fonts
        self.FONT_HEADER = ("Segoe UI", 24, "bold")
        self.FONT_STATUS = ("Impact", 72)
        self.FONT_DEST = ("Segoe UI", 48, "bold")
        self.FONT_METRIC_LABEL = ("Segoe UI", 10, "bold")
        self.FONT_METRIC_VAL = ("Consolas", 28, "bold")
        self.FONT_LOG = ("Consolas", 10)

        # Variables
        self.engine = pyttsx3.init()
        self.last_req_id = None
        self.is_connected = False

        # --- 3. BUILD UI LAYOUT ---
        self.setup_ui()
        
        # --- 4. START SYSTEMS ---
        self.log_system("SYSTEM BOOT INITIATED...")
        self.log_system(f"TARGET SERVER: {server_ip}")
        self.root.bind('<q>', lambda e: root.destroy()) # Q to Quit
        
        self.running = True
        threading.Thread(target=self.poll_server, daemon=True).start()
        self.update_clock()

    def setup_ui(self):
        # === TOP BAR (Header + Clock) ===
        top_frame = tk.Frame(self.root, bg="#1a1a1a", height=80)
        top_frame.pack(fill="x", side="top")
        top_frame.pack_propagate(False)

        tk.Label(top_frame, text=" MANIPUR EMS | UNIT 01", font=("Segoe UI", 16, "bold"), 
                 bg="#1a1a1a", fg="#aaaaaa").pack(side="left", padx=30)
        
        self.clock_label = tk.Label(top_frame, text="00:00:00", font=("Consolas", 16), 
                                  bg="#1a1a1a", fg="#00ff00")
        self.clock_label.pack(side="right", padx=30)
        
        # Connection Status Dot
        self.conn_dot = tk.Label(top_frame, text="●", font=("Arial", 20), bg="#1a1a1a", fg="#333")
        self.conn_dot.pack(side="right")

        # === MAIN LAYOUT (Split Left/Right) ===
        container = tk.Frame(self.root, bg="#050505")
        container.pack(fill="both", expand=True, padx=40, pady=40)

        # LEFT SIDE: INFO DASHBOARD
        self.dash_frame = tk.Frame(container, bg="#050505")
        self.dash_frame.pack(side="left", fill="both", expand=True)

        # Status Banner
        self.status_label = tk.Label(self.dash_frame, text="SYSTEM STANDBY", 
                                   font=self.FONT_STATUS, bg="#050505", fg="#333333")
        self.status_label.pack(pady=(20, 0))

        # Destination Box
        self.dest_frame = tk.Frame(self.dash_frame, bg="#111", highlightbackground="#333", highlightthickness=2)
        self.dest_frame.pack(fill="x", pady=40, ipady=20)
        
        tk.Label(self.dest_frame, text="TARGET DESTINATION", font=("Arial", 10), bg="#111", fg="#777").pack(pady=5)
        self.dest_label = tk.Label(self.dest_frame, text="--", font=self.FONT_DEST, bg="#111", fg="#aaa")
        self.dest_label.pack()

        # Metrics Grid
        metrics_frame = tk.Frame(self.dash_frame, bg="#050505")
        metrics_frame.pack(fill="x")
        
        self.create_metric(metrics_frame, "SEVERITY LEVEL", "severity_val", 0)
        self.create_metric(metrics_frame, "TRAFFIC CONDITION", "traffic_val", 1)
        self.create_metric(metrics_frame, "EST. ARRIVAL", "eta_val", 2)

        # RIGHT SIDE: SYSTEM LOGS
        log_frame = tk.Frame(container, bg="#0a0a0a", width=300, highlightbackground="#333", highlightthickness=1)
        log_frame.pack(side="right", fill="y", padx=(40, 0))
        log_frame.pack_propagate(False)

        tk.Label(log_frame, text="SYSTEM LOGS", font=("Segoe UI", 10, "bold"), bg="#222", fg="#aaa").pack(fill="x", ipady=5)
        
        self.log_list = tk.Listbox(log_frame, bg="#0a0a0a", fg="#00ff00", font=self.FONT_LOG, 
                                 borderwidth=0, highlightthickness=0)
        self.log_list.pack(fill="both", expand=True, padx=10, pady=10)

    def create_metric(self, parent, title, var_name, col):
        frame = tk.Frame(parent, bg="#111", highlightbackground="#333", highlightthickness=1)
        frame.pack(side="left", fill="x", expand=True, padx=10)
        
        tk.Label(frame, text=title, font=self.FONT_METRIC_LABEL, bg="#111", fg="#555").pack(pady=(15, 5))
        val = tk.Label(frame, text="-", font=self.FONT_METRIC_VAL, bg="#111", fg="#fff")
        val.pack(pady=(0, 15))
        setattr(self, var_name, val)

    def log_system(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_list.insert(0, f"[{timestamp}] {message}") # Insert at top
        if self.log_list.size() > 50: self.log_list.delete(50) # Keep list clean

    def update_clock(self):
        now = datetime.now().strftime("%H:%M:%S")
        self.clock_label.config(text=now)
        self.root.after(1000, self.update_clock)

    def speak(self, text):
        def _speak():
            self.engine.say(text)
            self.engine.runAndWait()
        threading.Thread(target=_speak, daemon=True).start()

    def poll_server(self):
        while self.running:
            try:
                response = requests.get(self.server_url, timeout=2)
                
                # Connection Status Indicator
                if response.status_code == 200:
                    if not self.is_connected:
                        self.is_connected = True
                        self.conn_dot.config(fg="#00ff00") # Green Dot
                        self.log_system("CONNECTION ESTABLISHED")
                    
                    data = response.json()
                    if data:
                        self.update_mission(data)
                    else:
                        self.reset_ui("AWAITING DISPATCH")
                        
                else:
                    raise Exception("Status Code Error")

            except Exception:
                if self.is_connected:
                    self.is_connected = False
                    self.conn_dot.config(fg="#ff0000") # Red Dot
                    self.log_system("CONNECTION LOST - RETRYING...")
                self.reset_ui("SEARCHING SIGNAL...")
            
            time.sleep(1)

    def update_mission(self, data):
        current_id = data.get('patient', 'Unknown')
        
        # Determine Alert Level
        mission_type = str(data.get('type', ''))
        hospital = str(data.get('hospital', ''))
        is_critical = "ALS" in mission_type or "Trauma" in hospital or "CRITICAL" in str(data.get('priority', ''))

        # Check for New Mission (Audio Trigger)
        if current_id != self.last_req_id:
            self.last_req_id = current_id
            self.log_system(f"NEW MISSION PACKET: {current_id}")
            self.log_system(f"ROUTING TO: {hospital}")
            
            if is_critical:
                self.speak("Critical Dispatch. Proceed to Trauma Center Immediately.")
            else:
                self.speak("Mission Assigned. Proceed to Destination.")

        # COLOR THEME
        # Critical = Red, Standard = Blue (More professional than Green)
        bg_color = "#3d0000" if is_critical else "#001a33" 
        fg_color = "#ff4d4d" if is_critical else "#4da6ff"
        
        # Update Main UI
        self.root.configure(bg=bg_color)
        self.dash_frame.configure(bg=bg_color)
        self.status_label.configure(text=data.get('type', 'ASSIGNED').upper(), bg=bg_color, fg=fg_color)
        
        self.dest_frame.configure(bg="#000", highlightbackground=fg_color)
        self.dest_label.configure(text=hospital.upper(), bg="#000", fg=fg_color)

        # Update Metrics
        self.severity_val.config(text=f"{data.get('pain_level')}/10")
        self.traffic_val.config(text=str(data.get('location', 'Unknown'))[:15]) # Limit char length
        self.eta_val.config(text=f"{data.get('eta')} MIN")

    def reset_ui(self, status):
        self.root.configure(bg="#050505")
        self.dash_frame.configure(bg="#050505")
        self.status_label.configure(text=status, bg="#050505", fg="#333")
        self.dest_frame.configure(bg="#111", highlightbackground="#333")
        self.dest_label.configure(text="--", bg="#111", fg="#555")
        
        # Reset Metrics
        self.severity_val.config(text="-")
        self.traffic_val.config(text="-")
        self.eta_val.config(text="-")

if __name__ == "__main__":
    root = tk.Tk()
    app = AmbulanceConsole(root)
    root.mainloop()