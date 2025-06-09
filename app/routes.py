from flask import Flask, render_template, request, redirect, url_for, jsonify
from scapy.all import sniff, wrpcap, get_if_list, conf, IP
from app.utils import extract_features, predict, predict_from_pcap
from collections import Counter
import numpy as np
import os
import json

# Optional: import label_encoder if available
try:
    from app.utils import label_encoder
except ImportError:
    label_encoder = None

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def describe_interface(iface):
    if "eth" in iface:
        return "Ethernet"
    elif "wlan" in iface:
        return "Wi-Fi"
    elif "lo" in iface:
        return "Loopback"
    return "Other"

def is_interface_active(iface):
    try:
        pkts = sniff(iface=iface, count=1, timeout=2)
        return len(pkts) > 0
    except Exception:
        return False

@app.route('/')
def index():
    raw_interfaces = get_if_list()
    default_iface = conf.route.route("0.0.0.0")[0]
    interfaces = [{
        'name': iface,
        'desc': describe_interface(iface),
        'is_default': iface == default_iface,
        'is_active': is_interface_active(iface)
    } for iface in raw_interfaces]
    return render_template('index.html', interfaces=interfaces)

@app.route('/results', methods=['POST'])
def results():
    try:
        # Check if a file is uploaded
        if 'file' not in request.files:
            return render_template('error.html', error="No file part in the request.")

        file = request.files['file']
        if file.filename == '':
            return render_template('error.html', error="No file selected for upload.")

        # Save the uploaded file
        pcap_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(pcap_path)

        # Process the PCAP file
        output = predict_from_pcap(pcap_path)

        # Build attack type count dictionary for graphs
        attack_data = {}
        for row in output:
            attack_type = row.get('Attack Type', 'Unknown')
            attack_data[attack_type] = attack_data.get(attack_type, 0) + 1

        return render_template('results.html', data=output, attack_data=attack_data)

    except Exception as e:
        print(f"Error in results route: {str(e)}")
        return render_template('error.html', error=str(e))


@app.route('/live_capture', methods=['POST'])
def live_capture():
    try:
        iface = request.form.get('iface')  # Get the selected interface
        packets = sniff(iface=iface, count=20)
        path = os.path.join(app.config['UPLOAD_FOLDER'], "live_capture.pcap")
        wrpcap(path, packets)

        return jsonify({'success': True, 'message': 'Live traffic successfully captured and saved to live_capture.pcap.'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error capturing live traffic: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
