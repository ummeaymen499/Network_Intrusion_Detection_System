import joblib
import numpy as np
import pandas as pd
from scapy.all import IP, TCP, UDP, rdpcap

# Load pre-trained components
model = joblib.load("network_intrusion_model.pkl")
scaler = joblib.load("network_intrusion_scaler.pkl")
label_encoders = joblib.load("network_intrusion_label_encoders.pkl")
feature_order = joblib.load("network_intrusion_feature_order.pkl")

# Mapping model predictions to attack names
attack_mapping = {
    0: "Normal",
    1: "Attack",
    2: "Probe",
    3: "R2L",
    4: "U2R"
}

def safe_label_transform(encoder, val):
    """Safely transform a value using a LabelEncoder."""
    if val in encoder.classes_:
        return encoder.transform([val])[0]
    else:
        # Add 'other' to encoder classes if not present
        if 'other' not in encoder.classes_:
            encoder.classes_ = np.append(encoder.classes_, 'other')
        return encoder.transform(['other'])[0]

def prepare_features(features_dict):
    """Prepare features for prediction by aligning with the feature order."""
    return pd.DataFrame([[features_dict.get(f, 0) for f in feature_order]], columns=feature_order)

def extract_features(pkt, packet_index, packets):
    """Extract features from a packet."""
    features = {
        "network_packet_size": len(pkt),  # Packet size
        "failed_logins": 0,  # Replace with actual data if available
        "ip_reputation_score": 0.0,  # Replace with actual data if available
        "login_attempts": 0,  # Replace with actual data if available
        "session_duration": 0  # Calculated based on packet timestamps
    }

    # Calculate session duration (time difference between packets)
    if packet_index > 0:
        features["session_duration"] = packets[packet_index].time - packets[packet_index - 1].time

    # Example: Replace placeholders with actual data
    # Use an IP reputation database or API to assign a reputation score
    if pkt.haslayer(IP):
        ip_address = pkt[IP].src
        features["ip_reputation_score"] = get_ip_reputation_score(ip_address)  # Implement this function

    # Replace placeholders for failed logins and login attempts with actual data if available
    features["failed_logins"] = get_failed_logins(pkt)  # Implement this function
    features["login_attempts"] = get_login_attempts(pkt)  # Implement this function

    return features

def predict(features_dict):
    """Predict the attack type based on extracted features."""
    # Encode categorical features
    for feature in ["protocol_type", "encryption_used", "browser_type"]:
        encoder = label_encoders.get(feature)
        val = features_dict.get(feature, "other")
        if encoder:
            features_dict[feature] = safe_label_transform(encoder, val)

    # Prepare features and scale
    feature_values_df = prepare_features(features_dict)
    scaled = scaler.transform(feature_values_df)
    prediction_encoded = model.predict(scaled)
    predicted_class = int(prediction_encoded[0])
    
    return attack_mapping.get(predicted_class, "Unknown")

def predict_with_probabilities(features_dict):
    """Predict the attack type and return probabilities."""
    # Encode categorical features
    for feature in ["protocol_type", "encryption_used", "browser_type"]:
        encoder = label_encoders.get(feature)
        val = features_dict.get(feature, "other")
        if encoder:
            features_dict[feature] = safe_label_transform(encoder, val)

    # Prepare features and scale
    feature_values_df = prepare_features(features_dict)
    scaled = scaler.transform(feature_values_df)
    prediction_encoded = model.predict(scaled)
    prediction_probabilities = model.predict_proba(scaled)
    predicted_class = int(prediction_encoded[0])
    
    return attack_mapping.get(predicted_class, "Unknown"), prediction_probabilities

def predict_from_pcap(pcap_path, verbose=False):
    """Predict attack types from a pcap file."""
    try:
        packets = rdpcap(pcap_path)
        results = []

        for i, pkt in enumerate(packets):
            try:
                if not pkt.haslayer(IP):
                    continue

                features = extract_features(pkt, i, packets)
                prediction_label, prediction_probabilities = predict_with_probabilities(features)
                ip_address = pkt[IP].src if pkt.haslayer(IP) else "N/A"

                if verbose:
                    print(f"Packet {i}: {features}")
                    print(f"Predicted: {prediction_label}, Probabilities: {prediction_probabilities}")

                results.append({
                    'Packet Index': i,
                    'IP Address': ip_address,
                    'Attack Type': prediction_label,
                    'Probabilities': prediction_probabilities.tolist()
                })
            except Exception as e:
                results.append({
                    'Packet Index': i,
                    'IP Address': "N/A",
                    'Attack Type': f"Error: {str(e)}"
                })

        return results
    except Exception as e:
        return [{"Packet Index": "N/A", "IP Address": "N/A", "Attack Type": f"Error: {str(e)}"}]

def analyze_feature_importance():
    """Analyze and display feature importance."""
    import matplotlib.pyplot as plt

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        plt.figure(figsize=(10, 6))
        plt.barh(feature_order, importances)
        plt.xlabel("Importance")
        plt.title("Feature Importances")
        plt.tight_layout()
        plt.show()
    else:
        print("The model does not support feature importance analysis.")

def get_ip_reputation_score(ip_address):
    """Fetch the reputation score for the given IP address."""
    # Placeholder implementation
    # Replace this with an actual API call to fetch the reputation score
    try:
        # Example: Assign a dummy reputation score based on IP address
        if ip_address.startswith("192.168"):
            return 0.8  # High reputation for private IPs
        elif ip_address.startswith("10."):
            return 0.7  # Medium reputation for private IPs
        else:
            return 0.5  # Default reputation for public IPs
    except Exception as e:
        print(f"Error fetching IP reputation for {ip_address}: {e}")
        return 0.0

def get_failed_logins(pkt):
    """Extract the number of failed login attempts from the packet."""
    try:
        # Example: Check for specific patterns in packet payload (e.g., "login failed")
        if pkt.haslayer(TCP) and hasattr(pkt[TCP], 'payload'):
            payload = str(pkt[TCP].payload)
            if "login failed" in payload.lower():
                return 1  # Increment failed login count
        return 0
    except Exception as e:
        print(f"Error extracting failed logins: {e}")
        return 0

def get_login_attempts(pkt):
    """Extract the number of login attempts from the packet."""
    try:
        # Example: Check for specific patterns in packet payload (e.g., "login attempt")
        if pkt.haslayer(TCP) and hasattr(pkt[TCP], 'payload'):
            payload = str(pkt[TCP].payload)
            if "login" in payload.lower():
                return 1  # Increment login attempt count
        return 0
    except Exception as e:
        print(f"Error extracting login attempts: {e}")
        return 0

