# WattWise


WattWise is an intelligent, AI-powered energy management platform designed to help households and businesses monitor energy usage, receive consumption forecasts, and get actionable recommendations to reduce costs and environmental impact. This Flask-based web application integrates real-time data visualization with a robust machine learning backend for predictive analytics.

## Key Features

*   **Real-time Dashboard**: A dynamic dashboard to monitor live energy consumption, costs, and carbon emissions.
*   **AI-Powered Forecasting**: Utilizes LSTM and XGBoost models to provide 24-hour energy consumption forecasts, identify peak usage periods, and run what-if scenarios.
*   **Smart Recommendations & Alerts**: Generates personalized, rule-based recommendations for saving energy and provides alerts for abnormal usage patterns.
*   **Data Simulation**: Built-in data simulator to demonstrate the platform's capabilities by streaming realistic energy readings.
*   **Device Management**: Users can register, manage, and monitor individual smart devices and appliances.
*   **Cost & Carbon Tracking**: Automatically calculates estimated electricity costs and CO₂ emissions based on usage.
*   **Gamification**: Engages users with daily goals and achievement badges to encourage sustainable habits.
*   **User Authentication**: Secure user registration and login system to manage personal profiles and devices.

## Technology Stack

*   **Backend**: Python, Flask, Flask-SQLAlchemy, Flask-Login, Flask-Migrate
*   **Frontend**: HTML, CSS, JavaScript, Bootstrap, Chart.js
*   **Database**: MySQL (Production), SQLite (Testing)
*   **Machine Learning**:
    *   **Frameworks**: TensorFlow, Keras, Scikit-learn, XGBoost
    *   **Time Series**: Prophet, Statsmodels
    *   **Data Handling**: Pandas, NumPy
*   **Tooling**: Gunicorn, python-dotenv, PyMySQL

## Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

*   Python 3.8+
*   Pip
*   Virtualenv
*   MySQL Server

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/malottey1/wattwise.git
    cd wattwise
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Unix/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up the database:**
    *   Ensure your MySQL server is running.
    *   Create a new database named `wattwise`.
    *   Update the `SQLALCHEMY_DATABASE_URI` in `config.py` with your MySQL credentials, for example:
        ```python
        SQLALCHEMY_DATABASE_URI = "mysql+pymysql://user:password@host/wattwise"
        ```

5.  **Initialize the database:**
    *The project uses Flask-Migrate, but migration files are not included. You may need to initialize it manually or just create the tables directly from the models.*
    ```bash
    # From the root directory, run the Flask shell
    flask shell

    # Inside the shell, create the database tables
    >>> from app import db
    >>> db.create_all()
    >>> exit()
    ```

6.  **Populate the database with historical data:**
    Run the provided script to populate the `energy_readings` table with data from the UCI dataset. This is essential for the ML model to generate forecasts.
    ```bash
    python scripts/populate_energy_data.py
    ```

7.  **Run the application:**
    ```bash
    flask run
    ```
    The application will be available at `http://127.0.0.1:5000`.

## Machine Learning Pipeline

The project's core is its predictive modeling capability, developed using the "Individual household electric power consumption" dataset from the UCI Machine Learning Repository.

1.  **Data Exploration & Preprocessing**: The `eda_wakanda_tech.ipynb` notebook covers the initial exploratory data analysis, data cleaning, and resampling of the raw minute-level data into an hourly format.

2.  **Model Development & Comparison**: The `watt_wise_lstm_model.ipynb` notebook documents the end-to-end process of training and evaluating multiple time-series forecasting models, including SARIMA, Prophet, XGBoost, and an LSTM-based sequence-to-sequence model. The XGBoost model was identified as the best-performing model based on RMSE.

3.  **Model Integration**:
    *   The trained XGBoost model (`best_model.pkl`) and the corresponding data scaler (`scaler.pkl`) are saved in the `ml_models/` directory.
    *   The Flask application loads these artifacts using helper functions in `app/utils/ml_utils.py`.
    *   The `/api/forecast` endpoint fetches recent data from the database, preprocesses it, and uses the loaded model to generate and return a 24-hour forecast.

## Project Structure

```
.
├── app/                  # Main Flask application directory
│   ├── static/           # Static files (CSS, JS, Images)
│   ├── templates/        # HTML templates
│   ├── utils/            # Utility modules for forecasting, costs, etc.
│   ├── __init__.py       # Application factory
│   ├── forms.py          # WTForms definitions
│   ├── models.py         # SQLAlchemy ORM models
│   └── routes.py         # Application routes and API endpoints
├── ml_models/            # Trained machine learning models and scalers
│   ├── best_model.pkl    # The deployed XGBoost model
│   └── scaler.pkl        # The scikit-learn scaler
├── scripts/              # Helper scripts for DB seeding and testing
├── *.ipynb               # Jupyter notebooks for EDA and model training
├── requirements.txt      # Python dependencies
└── run.py                # Main entry point to run the application
