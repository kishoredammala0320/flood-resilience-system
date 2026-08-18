import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
from datetime import datetime
from twilio.rest import Client

# --- SECURE CREDENTIAL LOADING ---
# Try importing from local config.py (works on laptop).
# If missing, fall back to Streamlit Cloud Secrets.
try:
    import config
    OPENWEATHER_API_KEY = config.OPENWEATHER_API_KEY
    TWILIO_SID = config.TWILIO_SID
    TWILIO_AUTH_TOKEN = config.TWILIO_AUTH_TOKEN
    TWILIO_NUMBER = config.TWILIO_NUMBER
    TARGET_PHONE = config.TARGET_PHONE
except ModuleNotFoundError:
    OPENWEATHER_API_KEY = st.secrets["OPENWEATHER_API_KEY"]
    TWILIO_SID = st.secrets["TWILIO_SID"]
    TWILIO_AUTH_TOKEN = st.secrets["TWILIO_AUTH_TOKEN"]
    TWILIO_NUMBER = st.secrets["TWILIO_NUMBER"]
    TARGET_PHONE = st.secrets["TARGET_PHONE"]

# --- 1. SMS FUNCTION (Single Target) ---
def send_sms_alert(city, risk_level):
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        
        # Send to only the single target phone defined in config/secrets
        client.messages.create(
            body=f"🚨 AP FLOOD INTEL: High Risk detected in {city}. Risk Index: {risk_level:.2f}. Please stay alert.",
            from_=TWILIO_NUMBER,
            to=TARGET_PHONE
        )
        return True
    except Exception as e:
        st.error(f"SMS Failed: {e}")
        return False

# --- 2. PREMIUM UI DESIGN ---
st.set_page_config(page_title="AP Flood Intelligence", page_icon="🛰️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #f0f2f6; }
    div[data-testid="stMetric"] { background: white; border-radius: 15px; padding: 15px 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    section[data-testid="stSidebar"] { background-color: #e2e8f0; }
    section[data-testid="stSidebar"] * { color: #1e293b !important; }
    .status-banner { padding: 30px; border-radius: 20px; text-align: center; font-size: 32px; font-weight: 800; color: white; margin: 25px 0; text-transform: uppercase; letter-spacing: 2px; }
    .subtitle { color: #64748b; font-size: 18px; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CONFIG & ORDERED CITY PROFILES ---
API_KEY = OPENWEATHER_API_KEY
CITY_PROFILES = {
    "Kakinada": {"Urban": 10, "Drain": 5, "Coast": 14, "Pop": 7, "Topo": 3, "River": 8},
    "Rajahmundry": {"Urban": 10, "Drain": 5, "Coast": 0, "Pop": 9, "Topo": 4, "River": 15},
    "Guntur": {"Urban": 11, "Drain": 7, "Coast": 0, "Pop": 10, "Topo": 6, "River": 4},
    "Srikakulam": {"Urban": 7, "Drain": 4, "Coast": 12, "Pop": 7, "Topo": 10, "River": 14},
    "Nellore": {"Urban": 9, "Drain": 5, "Coast": 13, "Pop": 8, "Topo": 3, "River": 10},
    "Vijayawada": {"Urban": 13, "Drain": 6, "Coast": 0, "Pop": 12, "Topo": 5, "River": 15},
    "Tirupati": {"Urban": 9, "Drain": 9, "Coast": 0, "Pop": 11, "Topo": 13, "River": 3},
    "Kurnool": {"Urban": 8, "Drain": 6, "Coast": 0, "Pop": 9, "Topo": 7, "River": 12},
    "Anantapur": {"Urban": 7, "Drain": 10, "Coast": 0, "Pop": 8, "Topo": 9, "River": 2},
    "Visakhapatnam": {"Urban": 15, "Drain": 8, "Coast": 15, "Pop": 14, "Topo": 11, "River": 4}
}

@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("flood_model.json")
    return model

def fetch_weather(city_name):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={API_KEY}&units=metric"
    try:
        r = requests.get(url).json()
        return r if r.get("cod") == 200 else None
    except: return None

# --- 4. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4005/4005831.png", width=100)
    st.title("Command Center")
    selected_city = st.selectbox("📍 Select Monitoring Site", list(CITY_PROFILES.keys()))
    st.markdown("---")
    
    st.subheader("🛠️ Simulation Suite")
    demo_mode = st.toggle("Enable Manual Override")
    if demo_mode:
        manual_rain = st.slider("Simulate Rainfall (mm/h)", 0, 200, 0)
    else:
        manual_rain = 0
        
    st.write("🛰️ **Data Streams:** Active")
    st.write("🤖 **ML Model:** XGBoost v2.1")
    if st.button("🔄 Force Re-Sync"):
        st.rerun()

# --- 5. MAIN DASHBOARD ---
st.markdown(f"<h1>🛰️ Flood Resilience Intelligence: <span style='color:#3b82f6;'>{selected_city}</span></h1>", unsafe_allow_html=True)
st.markdown(f"<p class='subtitle'>Real-time Satellite Feed | {datetime.now().strftime('%d %B %Y, %H:%M:%S')}</p>", unsafe_allow_html=True)

model = load_model()
data = fetch_weather(selected_city)

if data:
    api_rain = data.get('rain', {}).get('1h', 0)
    current_rain = float(manual_rain) if demo_mode else float(api_rain)
    prof = CITY_PROFILES[selected_city]

    # Sensitivity Logic
    inputs = {
        'MonsoonIntensity': np.clip(current_rain * 0.05, 0, 15),
        'TopographyDrainage': float(prof["Topo"]),
        'RiverManagement': float(prof["River"]),
        'Deforestation': 4.0, 'Urbanization': float(prof["Urban"]),
        'ClimateChange': 5.0, 'DamsQuality': 6.0, 'Siltation': 3.0,
        'AgriculturalPractices': 4.0, 'Encroachments': 4.0, 'IneffectiveDisasterPreparedness': 3.0,
        'DrainageSystems': float(np.clip(prof["Drain"] - (current_rain * 0.02), 0, 15)),
        'InadequatePlanning': 5.0, 'Watersheds': 5.0, 'DeterioratingInfrastructure': 4.0,
        'PopulationScore': float(prof["Pop"]), 'WetlandLoss': 3.0, 'PoliticalFactors': 3.0,
        'CoastalVulnerability': float(prof["Coast"]), 'Landslides': 2.0 if prof["Topo"] < 10 else 8.0
    }

    df = pd.DataFrame([inputs])
    df['fsum'] = df.sum(axis=1)
    df = df[['MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality', 'Siltation', 'AgriculturalPractices', 'Encroachments', 'IneffectiveDisasterPreparedness', 'DrainageSystems', 'InadequatePlanning', 'Watersheds', 'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss', 'PoliticalFactors', 'CoastalVulnerability', 'Landslides', 'fsum']]

    try:
        raw_prob = model.predict(df)[0]
        # 30mm Safety Buffer Logic
        final_prob = np.clip(raw_prob - 0.25, 0.25, 0.41) if current_rain < 30 else raw_prob

        # UI Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("🌡️ Temperature", f"{data['main']['temp']}°C")
        col2.metric("💧 Humidity", f"{data['main']['humidity']}%")
        col3.metric("🌧️ Precip (Used)", f"{current_rain}mm", delta="Simulation" if demo_mode else "Live")

        # Alert Logic
        if final_prob < 0.42: bg, msg, icon = "#10b981", "✅ LOW INDICATION", "🛡️"
        elif 0.42 <= final_prob < 0.58: bg, msg, icon = "#f59e0b", "⚠️ MODERATE RISK", "⚡"
        else: bg, msg, icon = "#ef4444", "🚨 HIGH ALERT", "🌊"

        st.markdown(f'<div class="status-banner" style="background-color: {bg};">{icon} {msg}</div>', unsafe_allow_html=True)
        
        # PROBABILITY METER
        c1, c2 = st.columns([1, 4])
        c1.write("### AI Prediction:")
        c1.write(f"## {final_prob:.4f}")
        c2.write("### Risk Probability Bar")
        c2.progress(float(np.clip(final_prob, 0.0, 1.0)))

        # EMERGENCY ALERT BUTTON
        if final_prob >= 0.58:
            st.divider()
            st.subheader("📢 Emergency Broadcast")
            if st.button("🚀 Send SMS Alert to Residents"):
                with st.spinner("Broadcasting..."):
                    if send_sms_alert(selected_city, final_prob):
                        st.success(f"SMS Alert successfully sent for {selected_city}!")
        
        st.info(f"**Insight:** This prediction combines live intensity ({current_rain}mm/h) with local urban vulnerability.")

    except Exception as e: st.error(f"Error: {e}")
else:
    st.error("Station Offline. Ensure API key is valid.")
