import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Petrol Consumption Predictor",
    page_icon="⛽",
    layout="wide"
)

# Title
st.title("⛽ Petrol Consumption Prediction App")
st.markdown("Predict petrol consumption based on economic and infrastructure factors")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    return df

# Load and display data
df = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page", ["Data Overview", "Model Training", "Make Prediction", "Model Insights"])

if page == "Data Overview":
    st.header("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        st.metric("Avg Petrol Consumption", f"{df['Petrol_Consumption'].mean():.2f}")
    with col4:
        st.metric("Avg Petrol Tax", f"{df['Petrol_tax'].mean():.2f}")
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Dataset Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.subheader("Data Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x='Petrol_Consumption', nbins=20, 
                          title='Distribution of Petrol Consumption')
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.scatter(df, x='Petrol_tax', y='Petrol_Consumption',
                        title='Petrol Tax vs Consumption')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(df, x='Average_income', y='Petrol_Consumption',
                        title='Average Income vs Consumption')
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.scatter(df, x='Population_Driver_licence(%)', y='Petrol_Consumption',
                        title='Driver Licence % vs Consumption')
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation matrix
    st.subheader("Feature Correlation Matrix")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                   title="Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Model Training":
    st.header("🤖 Model Training")
    
    st.subheader("Model Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider("Number of Trees (n_estimators)", 50, 500, 200, 50)
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    
    with col2:
        random_state = st.number_input("Random State", min_value=0, max_value=100, value=42)
        max_depth = st.slider("Max Depth (0 = unlimited)", 0, 20, 0)
    
    if st.button("Train Model", type="primary"):
        with st.spinner("Training model..."):
            # Prepare data
            X = df[['Petrol_tax', 'Average_income', 'Paved_Highways', 'Population_Driver_licence(%)']]
            y = df['Petrol_Consumption']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Create and train model
            model_params = {
                'n_estimators': n_estimators,
                'random_state': random_state
            }
            if max_depth > 0:
                model_params['max_depth'] = max_depth
            
            model = RandomForestRegressor(**model_params)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            # Store model in session state
            st.session_state['model'] = model
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.session_state['y_train_pred'] = y_train_pred
            st.session_state['y_test_pred'] = y_test_pred
            
            st.success("Model trained successfully!")
            
            # Display metrics
            st.subheader("Model Performance Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Training Set")
                st.metric("R² Score", f"{train_r2:.4f}")
                st.metric("Mean Absolute Error", f"{train_mae:.2f}")
                st.metric("Root Mean Squared Error", f"{train_rmse:.2f}")
            
            with col2:
                st.markdown("### Test Set")
                st.metric("R² Score", f"{test_r2:.4f}")
                st.metric("Mean Absolute Error", f"{test_mae:.2f}")
                st.metric("Root Mean Squared Error", f"{test_rmse:.2f}")
            
            # Feature importance
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(feature_importance, x='Importance', y='Feature', 
                        orientation='h', title='Feature Importance')
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction vs Actual plots
            st.subheader("Prediction vs Actual")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_train, y=y_train_pred, mode='markers', 
                                        name='Training', marker=dict(color='blue')))
                fig.add_trace(go.Scatter(x=[y_train.min(), y_train.max()], 
                                        y=[y_train.min(), y_train.max()], 
                                        mode='lines', name='Perfect Prediction', 
                                        line=dict(color='red', dash='dash')))
                fig.update_layout(title='Training Set: Actual vs Predicted',
                                xaxis_title='Actual Consumption',
                                yaxis_title='Predicted Consumption')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_test, y=y_test_pred, mode='markers', 
                                        name='Test', marker=dict(color='green')))
                fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                        y=[y_test.min(), y_test.max()], 
                                        mode='lines', name='Perfect Prediction', 
                                        line=dict(color='red', dash='dash')))
                fig.update_layout(title='Test Set: Actual vs Predicted',
                                xaxis_title='Actual Consumption',
                                yaxis_title='Predicted Consumption')
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Click 'Train Model' to start training the Random Forest model.")

elif page == "Make Prediction":
    st.header("🔮 Make Prediction")
    
    if 'model' not in st.session_state:
        st.warning("⚠️ Please train the model first in the 'Model Training' page.")
    else:
        st.subheader("Enter Feature Values")
        
        col1, col2 = st.columns(2)
        
        with col1:
            petrol_tax = st.number_input("Petrol Tax", min_value=0.0, max_value=20.0, 
                                        value=8.0, step=0.1, 
                                        help="Tax rate on petrol")
            average_income = st.number_input("Average Income", min_value=0, 
                                           max_value=10000, value=4000, step=100,
                                           help="Average income of the population")
        
        with col2:
            paved_highways = st.number_input("Paved Highways", min_value=0, 
                                            max_value=20000, value=5000, step=100,
                                            help="Number of paved highways")
            driver_licence_pct = st.slider("Population Driver Licence (%)", 
                                          min_value=0.0, max_value=1.0, 
                                          value=0.55, step=0.01,
                                          help="Percentage of population with driver's licence")
        
        if st.button("Predict Petrol Consumption", type="primary"):
            # Prepare input
            input_data = [[petrol_tax, average_income, paved_highways, driver_licence_pct]]
            
            # Make prediction
            prediction = st.session_state['model'].predict(input_data)[0]
            
            # Display result
            st.success(f"### Predicted Petrol Consumption: **{prediction:.2f}**")
            
            # Show feature values used
            st.subheader("Input Features")
            input_df = pd.DataFrame({
                'Feature': ['Petrol Tax', 'Average Income', 'Paved Highways', 
                           'Population Driver Licence (%)'],
                'Value': [petrol_tax, average_income, paved_highways, driver_licence_pct]
            })
            st.dataframe(input_df, use_container_width=True, hide_index=True)
            
            # Show comparison with dataset statistics
            st.subheader("Comparison with Dataset")
            avg_consumption = df['Petrol_Consumption'].mean()
            diff = prediction - avg_consumption
            diff_pct = (diff / avg_consumption) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Consumption", f"{prediction:.2f}")
            with col2:
                st.metric("Average Consumption", f"{avg_consumption:.2f}")
            with col3:
                st.metric("Difference", f"{diff:+.2f}", f"{diff_pct:+.2f}%")

elif page == "Model Insights":
    st.header("💡 Model Insights")
    
    if 'model' not in st.session_state:
        st.warning("⚠️ Please train the model first in the 'Model Training' page.")
    else:
        model = st.session_state['model']
        
        st.subheader("Feature Importance Analysis")
        feature_importance = pd.DataFrame({
            'Feature': ['Petrol_tax', 'Average_income', 'Paved_Highways', 
                       'Population_Driver_licence(%)'],
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        st.dataframe(feature_importance, use_container_width=True, hide_index=True)
        
        # Feature importance visualization
        fig = px.pie(feature_importance, values='Importance', names='Feature',
                    title='Feature Importance Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Model Performance Summary")
        y_test = st.session_state['y_test']
        y_test_pred = st.session_state['y_test_pred']
        
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{test_r2:.4f}", 
                     help="Higher is better. Measures how well the model explains the variance.")
        with col2:
            st.metric("MAE", f"{test_mae:.2f}",
                     help="Mean Absolute Error - average prediction error in the same units as target.")
        with col3:
            st.metric("RMSE", f"{test_rmse:.2f}",
                     help="Root Mean Squared Error - penalizes larger errors more.")
        
        st.subheader("Residual Analysis")
        residuals = y_test - y_test_pred
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(x=y_test_pred, y=residuals, 
                           title='Residuals vs Predicted Values',
                           labels={'x': 'Predicted Values', 'y': 'Residuals'})
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(x=residuals, nbins=20, 
                             title='Distribution of Residuals',
                             labels={'x': 'Residuals', 'y': 'Frequency'})
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This app uses Random Forest Regression to predict petrol consumption "
    "based on economic and infrastructure factors."
)

