import pandas as pd
import numpy as np
df=pd.read_csv(r"e:\ambulance_ml_dataset_300_rows.csv")
df.head()
df.drop(axis=1,columns='available')

def cat(s):
    if s=="ICU":
        return "ALS"
    elif s=="BLS":
        return "BLS"
    elif s=="DeadBody":
        return "DeadBody"
    else:
        return "ALS"
    
       
df['ambulance_type']=df['ambulance_type'].apply(cat)
ambulances = df[
    ["ambulance_id", "ambulance_lat", "ambulance_lon"]
].drop_duplicates()

from sklearn.neighbors import NearestNeighbors

knn = NearestNeighbors(n_neighbors=1, metric="haversine")

coords = np.radians(ambulances[["ambulance_lat", "ambulance_lon"]])
knn.fit(coords)

user_location = np.radians([[24.8200, 93.9400]])

distances, indices = knn.kneighbors(user_location)

nearest_ambulances = ambulances.iloc[indices[0]]
print("Nearest ambulances:")
print(nearest_ambulances)




X_cls = df[["age", "oxygen_required", "conscious", "trauma"]]
y_cls = df["ambulance_type"]

from sklearn.preprocessing import LabelEncoder

le_cls = LabelEncoder()
y_cls_encoded = le_cls.fit_transform(y_cls)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_cls, y_cls_encoded, test_size=0.2, random_state=42
)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = rf.predict(X_test)
print("Classification Accuracy:", accuracy_score(y_test, y_pred))

sample_patient = [[65, 1, 0, 0]] 
pred_class = rf.predict(sample_patient)
print("Predicted Ambulance Type:", le_cls.inverse_transform(pred_class)[0])





df_eta = df.copy()

from sklearn.preprocessing import LabelEncoder

for col in ["time_of_day", "traffic_level", "priority"]:
    df_eta[col] = LabelEncoder().fit_transform(df_eta[col])

X_eta = df_eta[["distance_km", "time_of_day", "traffic_level", "priority"]]
y_eta = df_eta["eta_minutes"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_eta, y_eta, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

eta_model = LinearRegression()
eta_model.fit(X_train, y_train)

print("ETA Model R² Score:", eta_model.score(X_test, y_test))

sample_eta = [[3.5, 1, 2, 3]]  
print("Predicted ETA (min):", int(eta_model.predict(sample_eta)[0]))





def calculate_fare(distance_km, ambulance_type, priority):
    base = {"BLS": 200, "ALS": 400, "ICU": 700, "DeadBody": 500}
    priority_extra = {"Low": 0, "Medium": 100, "High": 200, "Critical": 300}

    fare = base[ambulance_type] + distance_km * 200 + priority_extra[priority]
    return int(fare)

print("Estimated Fare:",
      calculate_fare(4.2, "ALS", "High")) 

coords = df[["user_lat", "user_lon"]]





from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(coords)

df["zone"] = kmeans.labels_

print("High demand zones (cluster centers):")
print(kmeans.cluster_centers_)
