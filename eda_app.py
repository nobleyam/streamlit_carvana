import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
import plotly.express as px
import seaborn as sn
@st.cache_data
def load_data(data):
    try:
        df = pd.read_csv(data)
        return df
    except FileNotFoundError:
        st.error("File not found.")
        return None


def run_eda_app():
    st.title("Exploratory Data Analysis (EDA) App")
    #df = pd.read_csv('data/training.csv')
    df = load_data('data/training.csv')


    submenu = st.sidebar.selectbox('SubMenu', ['descriptive', 'plots'])
    if submenu == 'descriptive':
        # st.header('Descriptive Statistics')
        # st.write(df.describe())
        # st.header('Nulls Summary')
        # st.write(df.isnull().sum())
        st.header('Check Data frame')
        st.dataframe(df)
        with st.expander('Data Type'):
            st.dataframe(df.dtypes)
        with st.expander('descriptive summary'):
            st.dataframe(df.describe())
        with st.expander('Nulls Summary'):
            st.dataframe(df.isnull().sum())
        with st.expander('Is Bad Buy Distribution'):
            st.dataframe(df['IsBadBuy'].value_counts())
        with st.expander("Unique Values Per Column"):
            st.dataframe(df.nunique().to_frame(name="Unique Values"))
        with st.expander("Skewness & Kurtosis"):
            numeric_cols = df.select_dtypes(include=["number"]).columns
            skew_kurt = df[numeric_cols].agg(["skew", "kurtosis"]).T
            st.dataframe(skew_kurt)






    elif submenu == 'plots':

        # Ensure df exists and is not empty

        if df is None or df.empty:
            st.error("DataFrame is not loaded or is empty.")

            return

        # Numerical column check

        numerical_columns = ['VehYear', 'VehicleAge', 'VehOdo', 'MMRAcquisitionAuctionAveragePrice',

                             'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice',

                             'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice',

                             'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice',

                             'MMRCurrentRetailCleanPrice', 'VehBCost', 'WarrantyCost']

        existing_numerical_columns = [col for col in numerical_columns if col in df.columns]

        if existing_numerical_columns:

            with st.expander("📊 Heatmap of Numerical Columns"):

                correlation = df[existing_numerical_columns].corr()

                fig, ax = plt.subplots(figsize=(12, 10))

                sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={'shrink': .8},
                            ax=ax)

                ax.set_title('Heatmap of Numerical Columns')

                st.pyplot(fig)

        else:

            st.warning("No valid numerical columns found for the heatmap.")

        # Categorical column check (Independent)

        categorical_columns = ['Auction', 'Color', 'Transmission', 'WheelType',

                               'Nationality', 'Size', 'TopThreeAmericanName',

                               'PRIMEUNIT', 'AUCGUART', 'IsOnlineSale', 'Make']

        existing_categorical_columns = [col for col in categorical_columns if col in df.columns]

        if existing_categorical_columns:

            with st.expander("📊 Categorical Feature Distributions"):

                for col in existing_categorical_columns:

                    if df[col].nunique() < 50:  # Skip high-cardinality features

                        fig, ax = plt.subplots(figsize=(12, 6))

                        sns.countplot(x=df[col], order=df[col].value_counts().index, palette="viridis", ax=ax)

                        ax.set_title(f"Distribution of {col}")

                        plt.xticks(rotation=45)

                        st.pyplot(fig)

                    else:

                        st.write(f"Skipping `{col}` - too many unique values ({df[col].nunique()}).")

        else:

            st.warning("No valid categorical columns found for visualization.")

        # Power Transformation Visualization

        selected_features = ["VehBCost", "WarrantyCost"]

        existing_features = [col for col in selected_features if col in df.columns]

        if existing_features:

            with st.expander("⚡ Power Transformation Visualization"):

                transformed_df = df.copy()  # Prevent modifying df

                for feature in existing_features:

                    has_negative_values = (df[feature] <= 0).any()

                    transformer = PowerTransformer(method='yeo-johnson' if has_negative_values else 'box-cox',
                                                   standardize=False)

                    transformed_df[f"{feature}_transformed"] = transformer.fit_transform(df[[feature]])

                    lambda_value = transformer.lambdas_[0]

                    st.write(f"**Lambda for {feature}:** {lambda_value:.4f}")

                    # Plot histograms

                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

                    axes[0].hist(df[feature], bins=30, color='blue', alpha=0.7)

                    axes[0].set_title(f'Original {feature} Histogram')

                    if f"{feature}_transformed" in transformed_df.columns:
                        axes[1].hist(transformed_df[f"{feature}_transformed"], bins=30, color='green', alpha=0.7)

                        axes[1].set_title(f'Transformed {feature} Histogram')

                    plt.tight_layout()

                    st.pyplot(fig)

        else:

            st.warning("No valid numerical columns found for Power Transformation.")




