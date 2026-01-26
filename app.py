import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from joblib import load

# ---------------------- Page Setup ---------------------- #
st.set_page_config(
    page_title="Flight Ticket Price Prediction",
    layout="centered"
)

st.markdown(
    "<h1 style='text-align:center; white-space: nowrap;'>✈️ Flight Ticket Price Prediction 🎟️</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>ML-based real-world airfare estimation</p>",
    unsafe_allow_html=True
)

# ---------------------- Load Artifacts ---------------------- #
@st.cache_resource
def load_artifacts():
    model = load("flight_price_model.pkl")
    categorical_columns = load("categorical_features.pkl")
    numerical_columns = load("numeric_features.pkl")
    return model, categorical_columns, numerical_columns

@st.cache_data
def load_route_summary():
    df = pd.read_csv("route_summary.csv")
    df.columns = df.columns.str.upper()  # Ensure uppercase columns
    return df

model, categorical_columns, numerical_columns = load_artifacts()
route_summary = load_route_summary()

input_data = {}

# ---------------------- Helpers ---------------------- #
def hours_to_hrs_mins(hours_float):
    hrs = int(hours_float)
    mins = int(round((hours_float - hrs) * 60))
    if mins == 60:
        hrs += 1
        mins = 0
    return hrs, mins

def get_time_bucket(hour):
    if 5 <= hour < 9:
        return "Early Morning"
    elif 9 <= hour < 13:
        return "Morning"
    elif 13 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    elif 21 <= hour < 24:
        return "Night"
    else:
        return "Late Night"

# ---------------------- Route Selection ---------------------- #
st.markdown("### 🔍 Route")

c1, c2 = st.columns(2)

with c1:
    input_data['SOURCE CITY'] = st.selectbox(
        "From",
        sorted(route_summary['SOURCE CITY'].unique())
    )

with c2:
    input_data['DESTINATION CITY'] = st.selectbox(
        "To",
        sorted(route_summary['DESTINATION CITY'].unique())
    )

# Get route info
route = route_summary[
    (route_summary['SOURCE CITY'] == input_data['SOURCE CITY']) &
    (route_summary['DESTINATION CITY'] == input_data['DESTINATION CITY'])
]

if not route.empty:
    duration = float(route['MIN_DURATION'].iloc[0])
else:
    st.warning("⚠️ Route not found. Using default duration of 1 hr.")
    duration = 1.0

input_data['DURATION'] = duration
hrs, mins = hours_to_hrs_mins(duration)

# Display route with duration below it
st.markdown(
    f"<h3 style='text-align:center;'>"
    f"{input_data['SOURCE CITY']} ✈️ {input_data['DESTINATION CITY']}"
    f"</h3>",
    unsafe_allow_html=True
)
st.markdown(
    f"<p style='text-align:center; font-size:16px;'>Duration: {hrs} hrs {mins} mins</p>",
    unsafe_allow_html=True
)

# ---------------------- Dates ---------------------- #
st.markdown("### 📅 Travel Dates")

d1, d2 = st.columns(2)

with d1:
    booking_date = st.date_input("Booking Date", value=datetime.today())

with d2:
    flight_date = st.date_input(
        "Flight Date",
        value=booking_date + timedelta(days=1),
        min_value=booking_date + timedelta(days=1)
    )

days_left = (flight_date - booking_date).days

input_data['BOOKING YEAR'] = booking_date.year
input_data['BOOKING MONTH'] = booking_date.month
input_data['BOOKING DAY'] = booking_date.day
input_data['FLIGHT YEAR'] = flight_date.year
input_data['FLIGHT MONTH'] = flight_date.month
input_data['FLIGHT DAY'] = flight_date.day
input_data['DAYS LEFT'] = days_left

st.info(f"🕒 Days until departure: **{days_left} days**")

# ---------------------- Flight Details ---------------------- #
st.markdown("### 🧳 Flight Details")

f1, f2, f3 = st.columns(3)

with f1:
    input_data['AIRLINE'] = st.selectbox(
        "Airline",
        ['SpiceJet', 'AirAsia', 'Vistara', 'GO FIRST', 'Indigo', 'Air India']
    )

with f2:
    input_data['STOPS'] = st.selectbox(
        "Stops",
        ['Zero', 'One', 'Two or More']
    )

with f3:
    input_data['PRICE CLASS'] = st.selectbox(
        "Class",
        ['Economy', 'Premium Economy', 'Business', 'First Class', 'Luxury']
    )

# ---------------------- Time Logic ---------------------- #
st.markdown("### ⏰ Time")

departure_time = st.selectbox(
    "Departure Time",
    ['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late Night']
)
input_data['DEPARTURE TIME'] = departure_time

departure_hour_map = {
    'Early Morning': 6,
    'Morning': 10,
    'Afternoon': 14,
    'Evening': 18,
    'Night': 22,
    'Late Night': 23
}

dep_hour = departure_hour_map[departure_time]

arrival_dt = datetime(2000, 1, 1, dep_hour, 0) + timedelta(hours=duration)
arrival_bucket = get_time_bucket(arrival_dt.hour)

input_data['ARRIVAL TIME'] = arrival_bucket

st.text_input(
    "Arrival Time (Auto)",
    value=f"{arrival_bucket} ({arrival_dt.strftime('%H:%M')})",
    disabled=True
)

# ---------------------- Passengers ---------------------- #
st.markdown("### 👥 Passengers")

passengers = st.number_input("Number of Passengers", 1, 9, 1)

# ---------------------- Prediction ---------------------- #
st.markdown("---")

if st.button("🔮 Predict Ticket Price", use_container_width=True):

    X_columns = [
        'AIRLINE', 'SOURCE CITY', 'STOPS', 'DESTINATION CITY', 'PRICE CLASS',
        'BOOKING YEAR', 'BOOKING MONTH', 'BOOKING DAY', 'DAYS LEFT',
        'FLIGHT YEAR', 'FLIGHT MONTH', 'FLIGHT DAY',
        'DEPARTURE TIME', 'DURATION', 'ARRIVAL TIME'
    ]

    input_df = pd.DataFrame([[input_data[col] for col in X_columns]], columns=X_columns)

    try:
        base_price = model.predict(input_df)[0]

        cgst = base_price * 0.025
        sgst = base_price * 0.025
        ticket_price = base_price + cgst + sgst
        total_price = ticket_price * passengers

        st.success(f"🎫 Price per Ticket: ₹ {ticket_price:,.2f}")
        st.info(f"🧾 CGST (2.5%): ₹ {cgst:,.2f} | SGST (2.5%): ₹ {sgst:,.2f}")
        st.success(f"💰 Total Price ({passengers} passengers): ₹ {total_price:,.2f}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")