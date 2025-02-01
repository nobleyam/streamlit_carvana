import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.linear_model import LogisticRegression



def load_and_check_model(model_path):
    """Load model and handle errors."""
    try:
        model = joblib.load(model_path)
        st.write(f"✅ Model loaded: {model_path}")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model {model_path}: {e}")
        return None

# Load models
encoder = load_and_check_model('models/self made/one_hot_encoder.pkl')  # Load the encoder here
scaler = load_and_check_model('models/self made/scaler3.pkl')
decision_tree_model = load_and_check_model('models/catboost_model.pkl')

def run_ml_app():
    st.title("ML Prediction App")

    # Numerical features
    numerical_input = pd.DataFrame([[st.slider("Vehicle Age", 0, 30, 10),
                                      st.slider("Vehicle Odometer", 0, 120000, 50000, step=500),
                                      st.slider("MMRAcquisition Auction Average Price", 800, 46000, 10000),
                                      st.slider("MMRAcquisition Auction Clean Price", 1000, 46000, 10000),
                                      st.slider("MMRAcquisition Retail Average Price", 1000, 46000, 10000),
                                      st.slider("MMRAcquisition Retail Clean Price", 1000, 46000, 10000),
                                      st.slider("MMRCurrent Auction Average Price", 300, 46000, 10000),
                                      st.slider("MMRCurrent Auction Clean Price", 400, 46000, 10000),
                                      st.slider("MMRCurrent Retail Average Price", 800, 46000, 10000),
                                      st.slider("MMRCurrent Retail Clean Price", 1000, 46000, 10000),
                                      st.number_input("Vehicle Base Cost", 1000, 46000, 15000, step=500),
                                      st.number_input("Warranty Cost", 400, 8000, 2000, step=100)]],
                                    columns=['VehicleAge', 'VehOdo', 'MMRAcquisitionAuctionAveragePrice',
                                             'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice',
                                             'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice',
                                             'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice',
                                             'MMRCurrentRetailCleanPrice', 'VehBCost', 'WarrantyCost'])

    # Categorical features
    categorical_input = pd.DataFrame([[st.selectbox("Auction", ['ADESA', 'MANHEIM', 'OTHER']),
                                       st.selectbox("Color", ['BLACK', 'OTHER', 'BLUE', 'GREY', 'SILVER', 'WHITE',
                                                              'GOLD', 'RED', 'MAROON', 'BEIGE', 'GREEN']),
                                       st.selectbox("Transmission", ['AUTO', 'MANUAL']),
                                       st.selectbox("Wheel Type", ['Alloy', 'Covers', 'Special']),
                                       st.selectbox("Nationality", ['OTHER ASIAN', 'AMERICAN', 'TOP LINE ASIAN']),
                                       st.selectbox("Size", ['SMALL SUV', 'MEDIUM SUV', 'MEDIUM', 'VAN', 'LARGE TRUCK',
                                                             'LARGE', 'COMPACT', 'LARGE SUV', 'SPECIALTY',
                                                             'CROSSOVER', 'SMALL TRUCK', 'SPORTS']),
                                       st.selectbox("Top Three American Name", ['OTHER', 'CHRYSLER', 'FORD']),
                                       st.selectbox("PRIMEUNIT", ['NO', 'YES', 'Unknown']),
                                       st.selectbox("AUCGUART", ['GREEN', 'RED', 'Unknown']),
                                       st.selectbox("Make", ['MAZDA', 'CHEVROLET', 'CHRYSLER', 'SUZUKI', 'PONTIAC',
                                                             'DODGE', 'TOYOTA', 'FORD', 'JEEP', 'MERCURY', 'OTHER',
                                                             'KIA', 'SATURN', 'HYUNDAI', 'MITSUBISHI', 'NISSAN']),
                                       st.selectbox("Is Online Sale", ['NO', 'YES'])]],
                                     columns=['Auction', 'Color', 'Transmission', 'WheelType', 'Nationality',
                                              'Size', 'TopThreeAmericanName', 'PRIMEUNIT', 'AUCGUART', 'Make', 'IsOnlineSale'])

    # Combine input data
    final_input = pd.concat([numerical_input, categorical_input], axis=1)


    # Scale numerical features
    final_input_scaled = scaler.transform(final_input.select_dtypes(include=[np.number]))

    # Encode categorical features
    final_input_encoded = encoder.transform(final_input.select_dtypes(include=[object]))

    # Combine processed features
    final_input_processed = np.concatenate([final_input_scaled, final_input_encoded.toarray()], axis=1)

    # Prediction button
    if st.button('Show Prediction'):
        with st.expander('Prediction Result:'):
            prediction = decision_tree_model.predict(final_input_processed)
            pred_prob = decision_tree_model.predict_proba(final_input_processed)

            st.write(f"Prediction: {prediction[0]}")
            st.write(f"Prediction Probability: {pred_prob[0]}")

            if prediction[0] == 1:
                st.warning("⚠️ You did a bad buy!")
            else:
                st.success("✅ You did not do a bad buy!")

    # Model Summary
    st.write(f"Model Summary: {decision_tree_model.get_params()}")
