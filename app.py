# Import necessary libraries
from flask import Flask, render_template, request, jsonify  # Flask web framework components
import joblib  # For loading the trained machine learning model
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation (if needed)
import psycopg2
import psycopg
from psycopg2.extras import RealDictCursor
import os
import random
from dotenv import load_dotenv
# Initialize Flask application
app = Flask(__name__)

# ==============================================
# DATABASE CONFIGURATION
# ==============================================
load_dotenv()
# Configure PostgreSQL connection settings

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        cursor_factory=RealDictCursor
    )
import os
import psycopg2

from dotenv import load_dotenv
load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS customers (
    Customer_ID SERIAL PRIMARY KEY,
    Age INT,
    Gender INT,
    District INT,
    Region INT,
    Location_Type INT,
    Customer_Type INT,
    Employment_Status INT,
    Income_Level FLOAT,
    Education_Level INT,
    Tenure INT,
    Balance FLOAT,
    Credit_Score INT,
    Outstanding_Loans FLOAT,
    Num_Of_Products INT,
    Mobile_Banking_Usage INT,
    Number_of_Transactions_per_Month INT,
    Num_Of_Complaints INT,
    Proximity_to_NearestBranch_or_ATM_km FLOAT,
    Mobile_Network_Quality INT,
    Owns_Mobile_Phone INT,
    prediction INT,
    Churn_Probability FLOAT
);
""")


conn.commit()
cursor.close()
conn.close()




# ==============================================
# MODEL LOADING
# ==============================================

# Load the pre-trained LightGBM model from file
model = joblib.load('lgb_model.pkl')
# ==============================================
# ROUTE DEFINITIONS
# ==============================================

# Home page route - redirects to login
@app.route('/')
def home():
    """Render the login page as the home page"""
    return render_template('index.html')

# About page route
@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')

# Contacts page route
@app.route('/contacts')
def contacts():
    """Render the contacts page"""
    return render_template('contacts.html')

# Login page route
@app.route('/index')
def login():
    """Render the login page"""
    return render_template('index.html')

# Main dashboard/churn analysis page
@app.route('/churn')
def dashboard():
    """
    Render the churn analysis dashboard with customer data
    and total customer count
    """
    try:
        # Create a database cursor
        conn = get_db_connection()
        cursor = conn.cursor()

        
        # Execute query to get customer data
        cursor.execute("SELECT * FROM customers LIMIT 10")  # Sample customers
        customers = cursor.fetchall()
        
        # Execute query to get total customer count
        cursor.execute("SELECT COUNT(*) as total_customers FROM customers")
        count_result = cursor.fetchone()
        total_customers = count_result['total_customers']
        
        # Close the cursor
        cursor.close()
        
        # Render template with both customer data and total count
        return render_template('churn.html', 
                            customers=customers,
                            total_customers=total_customers)
    
    except Exception as e:
        print("Database error:", str(e))
        return render_template('churn.html', 
                            customers=[], 
                            total_customers=0,
                            error=str(e))
# Prediction API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests from the frontend
    
    Receives customer data as JSON, makes churn prediction using the ML model,
    stores the prediction in the database, and returns results to the frontend
    """
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        # ==============================================
        # DATA PREPARATION FOR MODEL
        # ==============================================
        
        # Convert input data to numpy array in the correct feature order
        # Note: The order must match exactly how the model was trained
        features = np.array([
            int(data['Age']),    # Customer age
            int(data['Gender']),  # Gender (encoded as number)
            int(data['District']),  # District code
            int(data['Region']),  # Region code
            int(data['Location_Type']),  # Urban/rural etc.
            int(data['Customer_Type']),  # Type of customer
            int(data['Employment_Status']),  # Employment status
            float(data['Income_Level']),  # Income bracket
            int(data['Education_Level']),  # Education level
            int(data['Tenure']),  # How long customer has been with bank
            float(data['Balance']),  # Account balance
            int(data['Credit_Score']),  # Credit score
            float(data['Outstanding_Loans']),  # Number of active loans
            int(data['Num_Of_Products']),  # Number of bank products used
            int(data['Mobile_Banking_Usage']),  # Mobile banking frequency
            int(data['Number_of_Transactions_per_Month']),  # Transaction count
            int(data['Num_Of_Complaints']),  # Number of complaints
            float(data['Proximity_to_NearestBranch_or_ATM_km']),  # Distance to nearest branch
            int(data['Mobile_Network_Quality']),  # Network quality rating
            int(data['Owns_Mobile_Phone'])  # Whether customer owns mobile phone
        ]).reshape(1, -1)  # Reshape for single prediction

        # ==============================================
        # MAKE PREDICTION
        # ==============================================
        
        # Get prediction probabilities from model
        probabilities = model.predict_proba(features)
        
        # Extract probability of churn (class 1)
        prob_churn = float(probabilities[0][1])  
        
        # Convert probability to binary prediction (using 0.5 threshold)
        prediction = 1 if prob_churn >= 0.5 else 0
        
        # Convert to percentage with 2 decimal places
        probability = round(prob_churn * 100, 2)
        
        # Create human-readable result
        result = "Customer will leave" if prediction == 1 else "Customer will stay"
        
        return jsonify({
            'prediction': result, 
            'probability': probability,
            'success': True
        })

    except Exception as e:
        # Log the full error for debugging
        print("Prediction error:", str(e))
        
        # Return error message to frontend
        return jsonify({
            'error': f"Prediction failed: {str(e)}",
            'success': False
        })
        
# Add this new endpoint to handle batch predictions
@app.route('/api/customers/predict_all', methods=['POST'])
def predict_all_customers():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()


        # Fetch customers needing predictions
        cursor.execute("""
            SELECT * FROM customers 
            WHERE prediction IS NULL
            LIMIT 10000
        """)
        customers = cursor.fetchall()

        if not customers:
            cursor.close()
            return jsonify({
                'success': True,
                'processed': 0,
                'message': "No customers left to process."
            })

        # List of features in the order expected by the model
        model_features = getattr(model, 'feature_name_', [
            'Age', 'Gender', 'District', 'Region', 'Location_Type', 
            'Customer_Type', 'Employment_Status', 'Income_Level',
            'Education_Level', 'Tenure', 'Balance', 'Credit_Score',
            'Outstanding_Loans', 'Num_Of_Products', 'Mobile_Banking_Usage',
            'Number_of_Transactions_per_Month', 'Num_Of_Complaints',
            'Proximity_to_NearestBranch_or_ATM_km', 'Mobile_Network_Quality',
            'Owns_Mobile_Phone'
        ])

        # Mappings from your data
        region_map = { "Southern": 0, "Northern": 1, "Central": 2 }
        gender_map = { "Male": 1, "Female": 0 }
        district_map = {
            "Dedza": 6, "Dowa": 26, "Kasungu": 10, "Lilongwe": 22, "Mchinji": 8,
            "Nkhotakota": 12, "Ntcheu": 27, "Ntchisi": 9, "Salima": 21, "Chitipa": 11,
            "Karonga": 5, "Likoma": 20, "Mzimba": 16, "Nkhata Bay": 3, "Rumphi": 24,
            "Balaka": 25, "Blantyre": 17, "Chikwawa": 2, "Chiradzulu": 7,
            "Machinga": 4, "Mangochi": 14, "Mulanje": 15, "Mwanza": 23,
            "Nsanje": 18, "Thyolo": 1, "Phalombe": 19, "Zomba": 0, "Neno": 13
        }
        customertype_map = { "Retail": 0, "SME": 1, "Corporate": 2 }
        employmentstatus_map = { "Self Employed": 0, "Not Employed": 1, "Employed": 2 }
        educationlevel_map = { "Primary": 0, "Secondary": 1, "Tertiary": 2 }
        netquality_map = { "Fair": 0, "Poor": 1, "Good": 2 }
        phone_map = { "Yes": 0, "No": 1 }
        mobilebank_map = { "No": 0, "Yes": 1 }
        locationtype_map = { "Rural": 0, "Urban": 1, "Semi Urban": 2 }

        processed_count = 0

        def safe_int(val):
            return int(val) if val not in (None, '', 'NULL') else 0

        def safe_float(val):
            return float(val) if val not in (None, '', 'NULL') else 0.0

        for customer in customers:
            try:
                feature_values = {
                    'Age': safe_int(customer.get('Age')),
                    'Gender': gender_map.get(customer.get('Gender'), 0),
                    'District': district_map.get(customer.get('District'), 0),
                    'Region': region_map.get(customer.get('Region'), 0),
                    'Location_Type': locationtype_map.get(customer.get('Location_Type'), 0),
                    'Customer_Type': customertype_map.get(customer.get('Customer_Type'), 0),
                    'Employment_Status': employmentstatus_map.get(customer.get('Employment_Status'), 0),
                    'Income_Level': safe_float(customer.get('Income_Level')),
                    'Education_Level': educationlevel_map.get(customer.get('Education_Level'), 0),
                    'Tenure': safe_int(customer.get('Tenure')),
                    'Balance': safe_float(customer.get('Balance')),
                    'Credit__Score': safe_int(customer.get('Credit_Score')),
                    'Outstanding_Loans': safe_float(customer.get('Outstanding_Loans')),
                    'Num_Of_Products': safe_int(customer.get('Num_Of_Products')),
                    'Mobile_Banking_Usage': mobilebank_map.get(customer.get('Mobile_Banking_Usage'), 0),
                    'Number_of__Transactions_per/Month': safe_int(customer.get('Number_of_Transactions_per_Month')),
                    'Num_Of_Complaints': safe_int(customer.get('Num_Of_Complaints')),
                    'Proximity_to_NearestBranch_or_ATM_(km)': safe_float(customer.get('Proximity_to_NearestBranch_or_ATM_km')),
                    'Mobile_Network_Quality': netquality_map.get(customer.get('Mobile_Network_Quality'), 0),
                    'Owns_Mobile_Phone': phone_map.get(customer.get('Owns_Mobile_Phone'), 0)
                }

                features = np.array([feature_values[col] for col in model_features]).reshape(1, -1)

                prob_churn = model.predict_proba(features)[0][1]
                
                prediction = int(prob_churn >= 0.5)
                
                customer_id = safe_int(customer.get('Customer_ID'))
                
                cursor.execute("""
                    UPDATE customers 
                    SET prediction = %s,
                        Churn_Probability = %s
                    WHERE Customer_ID = %s
                """, (prediction, float(prob_churn), customer_id))

                if cursor.rowcount == 0:
                    print(f"No rows updated for Customer_ID {customer_id}. Check if ID exists.")
                else:
                    print(f"Updated Customer_ID {customer_id}")
                    processed_count += 1

            except Exception as e:
                print(f"Error processing customer {customer.get('Customer_ID')}: {str(e)}")
                continue


        conn.commit()
        cursor.close()
        conn.close()


        return jsonify({
            'success': True,
            'processed': processed_count,
            'message': f"Successfully updated {processed_count} customers"
        })

    except Exception as e:
        print(f"Global error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/customers')
def customers_page():
    return render_template("customers.html")
    
@app.route('/api/customers/churn_count', methods=['GET'])
def churn_count():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        
        cursor.execute("""
            SELECT COUNT(*) as churn_count
            FROM customers
            WHERE prediction = 1
        """)
        
        result = cursor.fetchone()
        churn_count = result['churn_count'] if result else 0
        print('churn_count')
        cursor.close()
        
        return jsonify({'churn_count': churn_count})
    
    except Exception as e:
        print("Error fetching churn count:", str(e))
        return jsonify({'error': str(e)}), 500
@app.route('/api/customers/churn_summary', methods=['GET'])
def churn_summary():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        attributes = [
            'mobile_banking_usage',
            'mobile_network_quality',
            'num_of_complaints',
            'proximity_to_nearestbranch_or_atm_km',
            'education_level',
            'district',
            'location_type',
            'region',
            'employment_status',
            'num_of_products'
        ]

        summary = {}

        # total churners to calculate percentages
        cursor.execute("""
            SELECT COUNT(*) FROM customers
            WHERE prediction = 1
        """)
        total_churners = cursor.fetchone()[0]
        if total_churners == 0:
            total_churners = 1  # avoid division by zero

        for attr in attributes:
            if attr == 'proximity_to_nearestbranch_or_atm_km':
                # Group distance into bands
                query = f"""
                    SELECT
                        CASE
                            WHEN proximity_to_nearestbranch_or_atm_km >= 0.5 AND proximity_to_nearestbranch_or_atm_km <= 16 THEN '0.5 - 16 km'
                            WHEN proximity_to_nearestbranch_or_atm_km >= 17 AND proximity_to_nearestbranch_or_atm_km <= 32 THEN '17 - 32 km'
                            WHEN proximity_to_nearestbranch_or_atm_km >= 33 AND proximity_to_nearestbranch_or_atm_km <= 50 THEN '33 - 50 km'
                            ELSE 'Unknown'
                        END AS value,
                        COUNT(*) * 100.0 / %s AS percentage
                    FROM customers
                    WHERE prediction = 1
                    GROUP BY value
                    ORDER BY percentage DESC
                """
                cursor.execute(query, (total_churners,))
            else:
                # Regular attributes
                query = f"""
                    SELECT
                        {attr} AS value,
                        COUNT(*) * 100.0 / %s AS percentage
                    FROM customers
                    WHERE prediction = 1
                    GROUP BY {attr}
                    ORDER BY percentage DESC
                """
                cursor.execute(query, (total_churners,))

            rows = cursor.fetchall()
            # Convert rows to list of dicts
            summary[attr] = [{"value": r[0], "percentage": float(r[1])} for r in rows]

        cursor.close()
        conn.close()
        return jsonify(summary)

    except Exception as e:
        print("Error generating churn summary:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route("/api/customers/predicted")
def predicted_customers():
    threshold = request.args.get("threshold", default=50, type=int)
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
        SELECT * FROM customers
        WHERE churn_probability >= %s
        ORDER BY churn_probability DESC
    """
    cursor.execute(query, (threshold / 100,))
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    return jsonify(rows)


if __name__ == '__main__':
    app.run(debug=os.getenv('FLASK_DEBUG', 'false').lower() == 'true')
    
