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

#Site logo
st.logo("resources/logo.svg", size="large")

#Create header
st.sidebar.title("Home Price Estimator", text_alignment="center")
st.sidebar.markdown("Select your options:", text_alignment="center")


#Configure Feature Selection
st.sidebar.subheader("Features", text_alignment="center")
beds = st.sidebar.slider("Number of bedrooms:", min_value=1, max_value=10)
baths = st.sidebar.slider("Number of bathrooms:", min_value=1, max_value=10)
size = st.sidebar.slider("Square Footage:", min_value=200, max_value=15000)

#Configure location selection
st.sidebar.subheader('Location', text_alignment="center")
states = sorted(state_map.keys())
selected_state = st.sidebar.selectbox("Select State:", states)

#Filter cities based on state
filtered_cities = df_locations[df_locations['state'] == selected_state]['city'].unique()
filtered_cities = sorted([c for c in filtered_cities if c in city_map])
selected_city = st.sidebar.selectbox("Select City:", filtered_cities)

#Prediction processing
if st.sidebar.button("Estimate Price"):
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
    st.header("Market Insight and Data Analysis", text_alignment="center")
    st.text("Explore the factors influencing your home's value and how it compares to the market.", text_alignment="center")

    tab1, tab2, tab3 = st.tabs(["Price Factors", "Price Trend", "Market Comparison"])

#Bar chart of feature importance
    with tab1:
        st.subheader("Biggest Price Factors")
        price_factors = pd.DataFrame({'Features': 
                                      ['Bedrooms', 
                                       'Bathrooms', 
                                       'City Location', 
                                       'State Location', 
                                       'Size'], 
                                       'Price Factor': 
                                       model.feature_importances_.round(4)})

        fig_factors = px.bar(price_factors, x='Features', y='Price Factor')
        fig_factors.update_layout(xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True))

        st.plotly_chart(fig_factors, config={'scrollZoom': False, 'displayModeBar': False})

        st.text("The chart shows the relative importance of each feature in determining home value. Size and location are the most influential, followed by bedrooms and bathrooms.", text_alignment="center")

    #Plot of price trend by square footage
    with tab2:
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
        st.plotly_chart(fig_line)

        st.text("The plot shows how estimated prices change with different square footage values, while keeping other features constant. It helps you understand the impact of size on home value.", text_alignment="center")

    #Bar chart comparing user's estimate to city and state averages
    with tab3:
        st.subheader("Market Comparison")

        compare_data = pd.DataFrame({
            'Location': ['Your Estimate', f'{selected_city} Avg', f'{selected_state} Avg'],
                        'Price': [prediction, city_val, state_val]
        })

        fig_bar = px.bar(compare_data, x='Location', y='Price', color='Location', text_auto='$,.0f')
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(yaxis_tickformat='$,.0f', showlegend=False)
        st.plotly_chart(fig_bar)

        st.text("The comparison shows your home’s estimated value against the average prices in your city and state, providing context for its local market position.", text_alignment="center")

else:
    st.info("ℹ️ Configure sidebar options and click ‘Estimate Price’ to see your home’s estimated market value.")
    st.image("resources/landing_image.svg")