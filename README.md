# ✈️ Flight Ticket Price Prediction 🎟️

A **Machine Learning–powered Streamlit web application** that predicts **domestic flight ticket prices in India** based on airline, route, travel dates, duration, class, stops, and time of travel.

🔗 **Live App:**  
https://ticket-prediction-mrbtyekmakupnorweqlkc8.streamlit.app/

---

## 📌 Project Overview

Flight ticket prices vary dynamically due to multiple factors such as demand, airline policies, booking time, travel class, route distance, and departure time.  
This project simulates a **real-world airfare pricing system** using a trained ML regression model and an interactive Streamlit interface.

Users can:
- Select source & destination cities
- Choose airline, class, and stops
- Pick booking and flight dates
- View auto-calculated duration & arrival time
- Get ticket price with GST
- Calculate total cost for multiple passengers

---

## 🚀 Key Features

- 🔍 Route-based flight duration detection  
- 📅 Smart booking & flight date handling  
- ⏰ Automatic arrival time calculation  
- 🧠 ML-based airfare prediction  
- 💸 CGST (2.5%) + SGST (2.5%) calculation  
- 👥 Multi-passenger pricing support  
- ⚡ Optimized performance using caching  
- 🌐 Clean, responsive Streamlit UI  

---

## 🧠 Machine Learning Model

- **Problem Type:** Regression  
- **Framework:** scikit-learn  
- **Model Serialization:** joblib  
- **Target Variable:** Base airfare (₹)

The model was trained on historical flight pricing data using both categorical and numerical features.

---

## 📊 Feature Details

### 🔤 Categorical Features
- AIRLINE: SpiceJet, AirAsia, Vistara, GO FIRST, Indigo, Air India  
- SOURCE CITY: Delhi, Mumbai, Bangalore, Kolkata, Hyderabad, Chennai  
- DESTINATION CITY: Mumbai, Bangalore, Kolkata, Hyderabad, Chennai, Delhi  
- STOPS: Zero, One, Two or More  
- PRICE CLASS: Economy, Premium Economy, Business, First Class, Luxury  
- DEPARTURE TIME: Early Morning, Morning, Afternoon, Evening, Night, Late Night  
- ARRIVAL TIME: Auto-derived  

### 🔢 Numerical Features
- BOOKING YEAR: 2024 – 2025  
- BOOKING MONTH: 1 – 12  
- BOOKING DAY: 1 – 31  
- DAYS LEFT: 1 – 49  
- FLIGHT YEAR: 2024 – 2026  
- FLIGHT MONTH: 1 – 12  
- FLIGHT DAY: 1 – 31  
- DURATION (hours): 0.83 – 30.17  

---

## 🧾 Price Calculation Logic

1. Base price predicted by ML model  
2. CGST (2.5%) added  
3. SGST (2.5%) added  
4. Final ticket price calculated  
5. Total fare = Ticket price × Number of passengers  

---

## 🛠️ Tech Stack

- Python  
- pandas  
- scikit-learn  
- joblib  
- Streamlit  
- datetime  

---

## Future Enhancements

- Airline-wise surge pricing
- Distance-based duration estimation
- Seat availability modeling
- Seasonal & festival demand effects
- Real-time flight API integration

## 👤 Author

**Shubham S. Ghanwat**  
MBA – Data Science & Business Analytics  
Machine Learning | Streamlit | Predictive Analytics  

## 📜 License


This project is licensed under the **Apache License, Version 2.0**.
You may obtain a copy of the License at:


http://www.apache.org/licenses/LICENSE-2.0


Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an **"AS IS" BASIS**,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
