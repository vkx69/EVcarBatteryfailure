
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="VoltGuard AI | Battery Intelligence",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "battery_model.pkl"
SCALER_PATH = Path(__file__).parent / "scaler.pkl"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_artifacts()
FEATURES = list(model.feature_names_in_)

# The first 60 columns are continuous model inputs; the rest are one-hot encoded categories.
NUMERIC_FEATURES = FEATURES[:60]
CATEGORICAL_FEATURES = FEATURES[60:]

BRANDS = [x.replace("vehicle_brand_", "") for x in FEATURES if x.startswith("vehicle_brand_")]
MODELS = [x.replace("vehicle_model_", "") for x in FEATURES if x.startswith("vehicle_model_")]
TYPES = [x.replace("vehicle_type_", "") for x in FEATURES if x.startswith("vehicle_type_")]
BATTERY_MAKERS = [x.replace("battery_manufacturer_", "") for x in FEATURES if x.startswith("battery_manufacturer_")]
CHEMISTRIES = [x.replace("battery_chemistry_", "") for x in FEATURES if x.startswith("battery_chemistry_")]
DRIVES = [x.replace("drive_type_", "") for x in FEATURES if x.startswith("drive_type_")]
FLEET = [x.replace("fleet_or_private_", "") for x in FEATURES if x.startswith("fleet_or_private_")]
TERRAINS = [x.replace("terrain_type_", "") for x in FEATURES if x.startswith("terrain_type_")]

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1450px;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.15);}
.hero {
    padding: 28px 30px; border-radius: 24px; margin-bottom: 22px;
    background: linear-gradient(135deg, #101827 0%, #182d3f 55%, #0f766e 100%);
    color: white; box-shadow: 0 16px 45px rgba(0,0,0,.18);
}
.hero h1 {font-size: 2.35rem; margin: 0 0 8px 0;}
.hero p {font-size: 1.02rem; opacity: .82; margin: 0;}
.metric-card {
    padding: 18px; border: 1px solid rgba(128,128,128,.20);
    border-radius: 18px; background: rgba(128,128,128,.055);
}
.section-title {font-size: 1.15rem; font-weight: 700; margin: 8px 0 12px;}
.small-muted {opacity:.68; font-size:.86rem;}
.risk-good {color:#16a34a; font-weight:800;}
.risk-bad {color:#dc2626; font-weight:800;}
</style>
""", unsafe_allow_html=True)

# Defaults from the training scaler make the app usable even when exact training ranges are unknown.
defaults = dict(zip(NUMERIC_FEATURES, scaler.mean_))

def numeric_input(name, key=None):
    mean = float(defaults.get(name, 0))
    scale = float(scaler.scale_[FEATURES.index(name)]) if name in FEATURES else 1.0
    # Broad but stable bounds around the training distribution.
    lo, hi = mean - 4 * scale, mean + 4 * scale
    if scale == 0:
        lo, hi = mean - 1, mean + 1
    if name in {"manufacturing_year"}:
        lo, hi, step = 1990, 2030, 1
        mean = min(max(mean, lo), hi)
    elif name in {"vehicle_age_years", "battery_health_percent", "state_of_charge",
                  "depth_of_discharge", "state_of_health", "charge_efficiency",
                  "discharge_efficiency", "capacity_loss_percent",
                  "fast_charge_ratio", "slow_charge_ratio", "overnight_charging_ratio",
                  "home_charging_ratio", "cooling_system_health", "maintenance_score",
                  "thermal_runaway_risk", "voltage_imbalance", "temperature_variance",
                  "battery_stress_index", "aging_score", "thermal_health_score",
                  "charging_quality_score", "driving_stress_score"}:
        lo, hi, step = 0.0, 100.0, 0.1
    else:
        step = max(abs(scale) / 10, 0.01)
        # Keep floating ranges practical.
        if abs(hi - lo) > 1e7:
            lo, hi = max(0, mean - 2e6), mean + 2e6
    return st.number_input(
        name.replace("_", " ").title(),
        min_value=float(lo), max_value=float(hi),
        value=float(np.clip(mean, lo, hi)),
        step=float(step), key=key or name,
        help=f"Model feature: {name}"
    )

def make_input():
    x = {f: 0.0 for f in FEATURES}

    # Main numeric panels
    panels = {
        "Vehicle": ["manufacturing_year","battery_capacity_kwh","odometer_km","vehicle_age_years","average_speed","average_trip_distance","daily_distance"],
        "Battery": ["cycle_count","battery_health_percent","state_of_charge","depth_of_discharge","state_of_health",
                    "cell_voltage_avg","cell_voltage_std","pack_voltage","cell_temperature_avg","cell_temperature_max",
                    "internal_resistance","charge_efficiency","discharge_efficiency","remaining_capacity","capacity_loss_percent"],
        "Charging": ["charging_cycles_last_month","fast_charge_ratio","slow_charge_ratio","average_charge_power_kw",
                     "average_charging_time","overnight_charging_ratio","home_charging_ratio","charging_interruptions","overcharge_events"],
        "Driving": ["aggressive_acceleration_score","hard_braking_score","regenerative_braking_usage",
                    "highway_driving_ratio","city_driving_ratio"],
        "Environment": ["average_ambient_temperature","maximum_temperature","minimum_temperature","humidity","altitude","dust_exposure"],
        "Maintenance & BMS": ["last_service_days","cooling_system_health","firmware_updates","previous_faults","maintenance_score",
                              "thermal_runaway_risk","voltage_imbalance","temperature_variance","sensor_fault_count",
                              "BMS_warning_count","abnormal_voltage_events","battery_stress_index","aging_score",
                              "thermal_health_score","charging_quality_score","driving_stress_score","predicted_remaining_life_cycles"],
    }

    for title, fields in panels.items():
        with st.expander(title, expanded=(title in ["Vehicle", "Battery"])):
            cols = st.columns(3)
            for i, f in enumerate(fields):
                with cols[i % 3]:
                    x[f] = numeric_input(f)

    st.markdown("### Vehicle configuration")
    c1,c2,c3,c4 = st.columns(4)
    with c1: brand = st.selectbox("Brand", BRANDS, index=BRANDS.index("Tesla") if "Tesla" in BRANDS else 0)
    with c2: vehicle_model = st.selectbox("Model", MODELS, index=MODELS.index("Model 3") if "Model 3" in MODELS else 0)
    with c3: vehicle_type = st.selectbox("Vehicle type", TYPES, index=TYPES.index("Sedan") if "Sedan" in TYPES else 0)
    with c4: battery_maker = st.selectbox("Battery manufacturer", BATTERY_MAKERS, index=0)

    c1,c2,c3,c4 = st.columns(4)
    with c1: chemistry = st.selectbox("Battery chemistry", CHEMISTRIES, index=CHEMISTRIES.index("NMC") if "NMC" in CHEMISTRIES else 0)
    with c2: drive = st.selectbox("Drive type", DRIVES, index=DRIVES.index("RWD") if "RWD" in DRIVES else 0)
    with c3: ownership = st.selectbox("Usage", FLEET, index=FLEET.index("Private") if "Private" in FLEET else 0)
    with c4: terrain = st.selectbox("Terrain", TERRAINS, index=TERRAINS.index("Flat") if "Flat" in TERRAINS else 0)

    for prefix, value in [
        ("vehicle_brand_", brand), ("vehicle_model_", vehicle_model),
        ("vehicle_type_", vehicle_type), ("battery_manufacturer_", battery_maker),
        ("battery_chemistry_", chemistry), ("drive_type_", drive),
        ("fleet_or_private_", ownership), ("terrain_type_", terrain)
    ]:
        col = prefix + value
        if col in x: x[col] = 1.0

    return pd.DataFrame([[x[f] for f in FEATURES]], columns=FEATURES)

st.markdown("""
<div class="hero">
  <div style="opacity:.7; letter-spacing:.12em; font-size:.78rem;">PREMIUM BATTERY INTELLIGENCE</div>
  <h1>VoltGuard AI</h1>
  <p>Professional battery-risk screening powered by your Random Forest model + StandardScaler.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🔋 VoltGuard AI")
    st.caption("Battery Risk Prediction Suite")
    st.divider()
    st.markdown("### Model status")
    st.success("Model loaded")
    st.caption(f"Random Forest • {len(model.estimators_)} trees • {len(FEATURES)} features")
    st.divider()
    st.markdown("### Workflow")
    st.write("1. Enter vehicle telemetry")
    st.write("2. Select configuration")
    st.write("3. Run AI assessment")
    st.write("4. Review probability & signals")

tab1, tab2 = st.tabs(["⚡ Live Assessment", "📊 Model Details"])

with tab1:
    data = make_input()

    st.divider()
    left, right = st.columns([1, 1])
    with left:
        st.markdown("### Ready for assessment")
        st.caption("The model receives the same 156-column feature layout used during training.")
    with right:
        run = st.button("🚀 Run Premium AI Assessment", type="primary", use_container_width=True)

    if run:
        scaled = scaler.transform(data)
        pred = int(model.predict(scaled)[0])
        proba = model.predict_proba(scaled)[0]
        risk_prob = float(proba[1]) if len(proba) > 1 else float(pred)

        # Treat class 1 as the positive/risk class. Confirm label meaning against the training dataset if available.
        if pred == 1:
            title = "Elevated Battery Risk"
            desc = "The model classified this vehicle into the positive risk class."
            status = "ATTENTION"
        else:
            title = "Lower Battery Risk"
            desc = "The model classified this vehicle into the negative risk class."
            status = "HEALTHY"

        st.markdown("## Assessment result")
        a,b,c = st.columns(3)
        with a:
            st.metric("AI classification", status)
        with b:
            st.metric("Risk probability", f"{risk_prob*100:.1f}%")
        with c:
            st.metric("Model confidence", f"{max(proba)*100:.1f}%")

        st.success(f"### {title}\n\n{desc}")

        st.markdown("### Probability profile")
        chart = pd.DataFrame({"Probability": proba}, index=[str(c) for c in model.classes_])
        st.bar_chart(chart)

        st.markdown("### Input snapshot")
        shown = data[NUMERIC_FEATURES].T.rename(columns={0:"Value"})
        st.dataframe(shown, use_container_width=True, height=430)

        st.warning("This is a machine-learning screening result, not a safety certification or diagnosis. The meaning of class 0/1 should be verified against the dataset labels used to train the model.")

with tab2:
    st.markdown("### Model architecture")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Estimator", "Random Forest")
    c2.metric("Trees", len(model.estimators_))
    c3.metric("Features", len(FEATURES))
    c4.metric("Classes", len(model.classes_))

    st.markdown("### Feature groups")
    groups = {
        "Continuous telemetry": len(NUMERIC_FEATURES),
        "Vehicle / battery one-hot fields": len(CATEGORICAL_FEATURES),
    }
    st.bar_chart(pd.Series(groups))

    st.markdown("### Deployment notes")
    st.info("Keep battery_model.pkl and scaler.pkl in the same directory as app.py. The app automatically preserves the trained feature order and one-hot columns.")
