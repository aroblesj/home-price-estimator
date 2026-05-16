import pandas as pd
import streamlit as st
import joblib
import plotly.express as px

#Cache the files in RAM
@st.cache_resource

#Create function for loading information
def load_data():
    model = joblib.load('estimate_model.joblib')
    city_map = joblib.load('city_map.joblib')
    state_map = joblib.load('state_map.joblib')
    metadata = joblib.load('metadata.joblib')

    return model, city_map, state_map, metadata

#Load model and location information
model, city_map, state_map, metadata = load_data()

#Load location references
df_locations = pd.read_csv('location_reference.csv')

#Create header
st.title("Home Price Estimator")
st.write("Enter your desired features for a data-driven market evaluation.")

#Create layout for the inputs
col1, col2 = st.columns(2)

#Configure first column
with col1:
    st.subheader("Features")
    beds = st.slider("Number of bedrooms:", min_value=1, max_value=10)
    baths = st.slider("Number of bathrooms:", min_value=1, max_value=10)
    size = st.slider("Square Footage:", min_value=200, max_value=15000)

#Configure second column
with col2:
    st.subheader('Location')

    #State selection
    states = sorted(state_map.keys())
    selected_state = st.selectbox("Select State:", states)

    #Filter cities based on state
    filtered_cities = df_locations[df_locations['state'] == selected_state]['city'].unique()
    filtered_cities = sorted([c for c in filtered_cities if c in city_map])
    selected_city = st.selectbox("Select City:", filtered_cities)

#Prediction processing
if st.button("Estimate Price"):
    state_val = state_map.get(selected_state, metadata['state_val'])
    city_val = city_map.get(selected_city, metadata['city_val'])

    input_df = pd.DataFrame({'bed': [beds],
                             'bath': [baths],
                             'city_mapping': [city_val],
                             'state_mapping': [state_val],
                             'house_size': [size]
                             })
    
    prediction = model.predict(input_df)[0]

    st.success(f"Estimated market Value: ${prediction:,.2f}")
    st.info (f"Location: {selected_city}, {selected_state} | Size: {size} sqft.")

    #Visual aids
    st.divider()
    st.header("Market Insight and Data Analysis")

    #Visual 1
    st.subheader("Biggest Price Factors")
    price_factors = pd.DataFrame({'Features': ['Bedrooms', 'Bathrooms', 'City Location', 'State Location', 'Size'],
                                'Price Factor': model.feature_importances_.round(4)
                                })

    st.bar_chart(data=price_factors, x='Features', y='Price Factor')
    st.caption('The higher the bar, the bigger the impact on price.')

    #Visual 2
    st.subheader('Price by Square Footage')
    size_options = [size * 0.5, size * 0.75, size, size * 1.25, size * 1.5]
    trend_results = [model.predict(pd.DataFrame({
        'bed': [beds], 'bath': [baths], 
        'city_mapping': [city_val], 'state_mapping': [state_val], 
        'house_size': [s]
    }))[0] for s in size_options]

    trend_data = pd.DataFrame({
        'Square Footage': size_options,
        'Estimated Price': trend_results
    })

    fig_line = px.line(trend_data, x='Square Footage', y='Estimated Price', markers=True)
    fig_line.update_layout(yaxis_tickformat='$,.0f')
    st.plotly_chart(fig_line, use_container_width=True)

    #Visual 3
    st.subheader("Market Comparison")

    compare_data = pd.DataFrame({
        'Location': ['Your Estimate', f'{selected_city} Avg', f'{selected_state} Avg'],
                    'Price': [prediction, city_val, state_val]
    })

    fig_bar = px.bar(compare_data, x='Location', y='Price', color='Location', text_auto='$,.0f')
    fig_bar.update_traces(textposition='outside')
    fig_bar.update_layout(yaxis_tickformat='$,.0f', showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)