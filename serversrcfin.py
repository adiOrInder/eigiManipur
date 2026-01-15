import pandas as pd
import numpy as np
import sqlite3
import os
from flask import Flask, request, jsonify
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

app = Flask(__name__)

CSV_FILE = 'ambulance_ml_dataset_300_rows_with_drivers.csv'  # ← Updated filename
DB_FILE = 'ambulance_data.db'

class IntelligentDispatch:
    def __init__(self, csv_path, db_path):
        self.db_path = db_path
        self.knn = None
        self.rf_classifier = None
        self.eta_regressor = None
        self.encoders = {}

        if os.path.exists(csv_path):
            print(f"[SYSTEM] Loading data from {csv_path}...")
            self.df = pd.read_csv(csv_path)
            
            # Normalize ambulance_type: ICU/ALS → ALS
            self.df['ambulance_type'] = self.df['ambulance_type'].replace({'ICU': 'ALS'})
            
            conn = sqlite3.connect(self.db_path)
            self.df.to_sql('ambulances', conn, if_exists='replace', index=False)
            conn.close()
            
            self.train_models()
        else:
            print(f"[ERROR] {csv_path} not found.")

    def train_models(self):
        print("[AI] Training Models...")

        # --- Random Forest (Predict Ambulance Type) ---
        X_cls = self.df[["age", "oxygen_required", "conscious", "trauma"]]
        y_cls = self.df["ambulance_type"]
        self.encoders['type'] = LabelEncoder()
        y_cls_encoded = self.encoders['type'].fit_transform(y_cls)
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_classifier.fit(X_cls, y_cls_encoded)

        # --- Linear Regression (ETA) ---
        df_eta = self.df.copy()
        for col in ["time_of_day", "traffic_level", "priority"]:
            le = LabelEncoder()
            df_eta[col] = le.fit_transform(df_eta[col])
            self.encoders[col] = le

        X_eta = df_eta[["distance_km", "time_of_day", "traffic_level", "priority"]]
        y_eta = df_eta["eta_minutes"]
        self.eta_regressor = LinearRegression()
        self.eta_regressor.fit(X_eta, y_eta)

        # --- KNN (for initial fit, but we use dynamic per-query) ---
        available_units = self.df[self.df["available"] == 1]
        coords = np.radians(available_units[["ambulance_lat", "ambulance_lon"]])
        self.knn = NearestNeighbors(n_neighbors=1, metric="haversine")
        self.knn.fit(coords)
        print("[AI] All models trained.")

    def predict_need(self, age, oxy, con, tra):
        pred_code = self.rf_classifier.predict([[age, oxy, con, tra]])[0]
        return self.encoders['type'].inverse_transform([pred_code])[0]

    def predict_eta(self, distance, time_str, traffic_str, priority_str):
        try:
            t_code = self.encoders['time_of_day'].transform([time_str])[0]
            tr_code = self.encoders['traffic_level'].transform([traffic_str])[0]
            p_code = self.encoders['priority'].transform([priority_str])[0]
            eta = self.eta_regressor.predict([[distance, t_code, tr_code, p_code]])[0]
            return max(1, int(eta))
        except:
            return int(distance * 2)

    def find_nearest_smart(self, user_lat, user_lon, needed_type):
        conn = sqlite3.connect(self.db_path)
        # Only select available AND unassigned units of the needed type
        query = """
            SELECT * FROM ambulances 
            WHERE available = 1 AND assigned = 0 AND ambulance_type = ?
        """
        candidates = pd.read_sql(query, conn, params=(needed_type,))
        conn.close()

        if candidates.empty:
            print(f"[WARNING] No {needed_type} available. Using any type.")
            conn = sqlite3.connect(self.db_path)
            candidates = pd.read_sql(
                "SELECT * FROM ambulances WHERE available = 1 AND assigned = 0", conn
            )
            conn.close()

        if candidates.empty:
            raise Exception("No available ambulances!")

        candidate_coords = np.radians(candidates[["ambulance_lat", "ambulance_lon"]])
        temp_knn = NearestNeighbors(n_neighbors=1, metric="haversine")
        temp_knn.fit(candidate_coords)
        user_loc = np.radians([[user_lat, user_lon]])
        _, index = temp_knn.kneighbors(user_loc)
        return candidates.iloc[index[0][0]].to_dict()

# --- Initialize System ---
dispatch_system = IntelligentDispatch(CSV_FILE, DB_FILE)
current_mission = {}

@app.route('/location', methods=['POST'])
def receive_request():
    global current_mission
    data = request.json
    print(f"\n[INCOMING] Request: {data}")

    try:
        lat = float(data.get('lat', 24.8170))
        lon = float(data.get('lon', 93.9368))
        age = int(data.get('age', 30))
        oxy = 1 if data.get('oxygen') == "Yes" else 0
        con = 1 if data.get('conscious') == "Yes" else 0
        tra = 1 if data.get('trauma') == "Yes" else 0

        # Predict needed type (ALS/BLS/DeadBody)
        needed_type = dispatch_system.predict_need(age, oxy, con, tra)
        print(f"[AI] Predicted need: {needed_type}")

        # Find nearest unassigned ambulance
        match = dispatch_system.find_nearest_smart(lat, lon, needed_type)
        amb_id = match['ambulance_id']

        # Update DB: mark as assigned
        conn = sqlite3.connect(DB_FILE)
        conn.execute("UPDATE ambulances SET assigned = 1 WHERE ambulance_id = ?", (amb_id,))
        conn.commit()
        conn.close()

        # Predict ETA using real time context
        current_hour = datetime.now().hour
        time_of_day = (
            "Morning" if 6 <= current_hour < 12 else
            "Afternoon" if 12 <= current_hour < 18 else
            "Evening" if 18 <= current_hour < 22 else
            "Night"
        )
        traffic = "High" if 17 <= current_hour <= 19 else "Medium"
        priority = "Critical" if needed_type != "BLS" else "Medium"

        predicted_eta = dispatch_system.predict_eta(
            match['distance_km'], time_of_day, traffic, priority
        )

        # Build response
        current_mission = {
            "patient": f"REQ-{np.random.randint(1000,9999)}",
            "assigned_unit": amb_id,
            "driver": match.get('driver_name', 'Unknown'),
            "type": needed_type,
            "hospital": "RIMS Trauma" if tra else "JNIMS General",
            "pain_level": "CRITICAL" if not con else "STABLE",
            "location": f"{lat:.4f}, {lon:.4f}",
            "eta": predicted_eta,
            "priority": "RED" if priority == "Critical" else "YELLOW",
            "distance": f"{match['distance_km']:.2f} km"
        }

        print(f"[DISPATCH] Assigned {amb_id} ({needed_type}) - ETA: {predicted_eta}m")
        return jsonify({"status": "success", "mission": current_mission})

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"status": "error"}), 500

@app.route('/api/assignments', methods=['GET'])
def send_to_dashboard():
    return jsonify(current_mission)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)