# 🏎️ DriveMatch India

### 📌 Project Overview
DriveMatch India is an advanced, end-to-end Machine Learning web application designed to simplify and revolutionize the car-buying decision for Indian consumers. Buying a vehicle in India involves navigating complex factors such as regional state RTO taxes, safety crash-test ratings, fluctuating fuel prices, and varying insurance costs. DriveMatch India addresses these challenges by combining vector-similarity algorithms and ensemble regression models into a unified platform. Users can effortlessly discover their ideal car based on personal budget, body style, and safety preferences while obtaining real-time, city-accurate on-road price estimates and complete financial breakdowns.

---

### ✨ Key Features & Capabilities
The application packs a comprehensive suite of intelligent tools tailored specifically for the Indian automotive market. At its core, the **AI Precision Matchmaker** leverages K-Nearest Neighbors vector similarity to evaluate multi-dimensional vehicle attributes and surface top matching models across more than 30 brands. Safety is prioritized through dedicated **BNCAP & GNCAP Safety Filtering**, allowing users to filter cars strictly by crash-test star ratings. The **Regional On-Road Price Engine** estimates total purchase costs by computing state-level RTO taxes, insurance costs, and registration fees across major Indian cities. Furthermore, the **Operating Cost Estimator** predicts monthly and annual fuel expenses across Petrol, Diesel, CNG, Hybrid, and EV powertrains based on custom monthly driving distances, while the **Interactive Loan EMI Calculator** assists buyers in financial planning with customizable down payment, interest rate, and tenure options.

---

### 📂 Repository Architecture & Files
The repository is structured logically to keep deployment clean, lightweight, and efficient. The core user interface and logic reside in **`app.py`**, which hosts the complete Streamlit dashboard. The underlying dataset containing vehicle specifications, pricing attributes, and state tax metrics is stored in **`indian_cars_ml_dataset_10k_2.csv`**. Due to file size constraints on GitHub, the complete pre-trained machine learning artifact containing the vector scalers, Nearest Neighbors model, and Random Forest pricing regressor is stored in a single compressed file named **`car_model.zip`**. Additional repository management files include **`requirements.txt`** for environment dependencies and **`README.md`** for project documentation.

---

### 🚀 Setup & Installation Guide
Setting up and running DriveMatch India locally requires only a few straightforward steps. First, clone the repository using `git clone https://github.com/your-username/drivematch-india.git` and navigate into the project root directory using `cd drivematch-india`. Next, establish an isolated virtual environment by executing `python -m venv venv` and activate it via `source venv/bin/activate` on macOS/Linux or `venv\\Scripts\\activate` on Windows. Once activated, install all required dependencies by running `pip install -r requirements.txt`.

---

### ⚠️ Critical Step: Extracting the Machine Learning Model
Before launching the application, you must extract the pre-trained machine learning model artifact. Because GitHub enforces strict file size limits on individual uploads, the single unified model binary **`car_model.pkl`** has been compressed into **`car_model.zip`**. You must extract **`car_model.zip`** directly into the root directory of the project alongside `app.py`. On Linux or macOS, run `unzip car_model.zip` in your terminal; on Windows PowerShell, run `Expand-Archive -Path car_model.zip -DestinationPath .`. Alternatively, you can right-click the zip file in your system file explorer and extract it manually. Ensure `car_model.pkl` is located in the root folder before proceeding.

---

### 🎛️ Running the Application
After verifying that `car_model.pkl` is extracted in the root directory, launch the interactive web dashboard by executing `streamlit run app.py` in your terminal. Streamlit will start the local server and automatically open the application in your default web browser at `http://localhost:8501`. From there, you can adjust budget sliders, set city locations, filter by safety ratings, and explore personalized vehicle recommendations and financial analytics in real time.

---

### 🛠️ Tech Stack & Dependencies
DriveMatch India is built entirely in Python, utilizing industry-standard libraries for machine learning, data processing, and web display. **Streamlit** powers the interactive glassmorphic web dashboard, while **Scikit-Learn** handles the machine learning modeling, including StandardScaler, NearestNeighbors, and RandomForestRegressor. Data manipulation and numerical operations are handled by **Pandas** and **NumPy**, while interactive analytics and chart rendering are driven by **Plotly Express**. Model serialization and loading are managed via **Joblib**.

---

### 📜 Dataset Note & Attribution
Note on Data Source: This machine learning model and demonstration dashboard were trained and evaluated utilizing a synthesized benchmarking dataset generated to simulate current Indian automotive market trends, RTO tax slabs, and regional pricing metrics.

---

### 🎥 Working Video at My Linkedin Profile 👇🏻
https://www.linkedin.com/posts/krishna-gautam-562198326_machinelearning-python-streamlit-ugcPost-7496526673962496000-ykdM/?utm_source=social_share_send&utm_medium=android_app&rcm=ACoAAFJAvhoBPNDTRDLup_mVzpv0FBujo7kCynY&utm_campaign=copy_link

---

### ©️ Copyright 2026, Maintained and Created By "KRISHNA"
