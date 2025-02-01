import streamlit as st
import streamlit.components.v1 as stc

from eda_app import run_eda_app
from ml_app import run_ml_app

html_temp = """
		<div style="background-color:#3872fb;padding:5px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Predict if the car purchased at the Auction is a good / bad buy?! </h1>
		<h4 style="color:white;text-align:center;">Don't Get Kicked! </h4>
		</div>
		"""


def main():
    # st.title("main application")
    stc.html(html_temp)
    Menu = ['home', 'EDA', 'ML', 'About']
    choice = st.sidebar.selectbox('Menu', Menu)

    if choice == 'home':
        # st.subheader('home')
        st.write("""
            ### One of the biggest challenges for an auto dealership purchasing a used car at an auto auction  
            is the risk that the vehicle might have serious issues preventing it from being sold to customers.  
            The auto community calls these unfortunate purchases **"kicks"**.  

            Kicked cars often result from tampered odometers, mechanical issues that the dealer cannot address,  
            issues with obtaining the vehicle title from the seller, or other unforeseen problems.  
            Kicked cars can be very costly to dealers due to transportation costs, wasted repair work,  
            and market losses when reselling the vehicle.  

            Modelers who can identify cars with a higher risk of being kicked can provide real value  
            to dealerships aiming to offer the best inventory selection to their customers.  
            The challenge of this competition is to predict whether a car purchased at the auction is a Kick (bad buy).  

            #### Data Source  
            - [Kaggle Competition: Don't Get Kicked](https://www.kaggle.com/competitions/DontGetKicked)  

            #### App Content  
            - **EDA Section**: Exploratory Data Analysis  
            - **ML Section**: Machine Learning Prediction App  
        """)


    elif choice == 'ML':
        run_ml_app()
    elif choice == 'EDA':
        run_eda_app()
    elif choice == 'About':
        st.subheader('About')

if __name__ == '__main__':
             main()
