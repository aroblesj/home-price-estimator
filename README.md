Home Price Estimator
Predictive Real Estate Analysis Tool

This application is a data-driven tool designed to estimate residential property values across the United States. It utilizes a Random Forest Regressor trained on over 1 million real estate records to provide instant valuations based on specific physical and geographic features.

Key Features
Instant Valuation: Adjust bedrooms, bathrooms, and square footage to receive a predicted market price.

Market Comparison: A descriptive benchmarking tool that compares your specific estimate against city and state averages.

Predictive Trends: A Plotly-powered visualization showing how price scales with square footage in your chosen area.

Optimized Performance: The underlying model is serialized into a lightweight .joblib format, ensuring fast loading and a small storage footprint.

Getting Started: Running the Web Application
Follow these steps to launch the interactive dashboard on your local machine:

Environment: Ensure Python 3.11 or higher is installed.

Installation: Open your terminal in the project directory and install the dependencies:
pip install streamlit pandas scikit-learn joblib plotly

Launch: Execute the following command to start the Streamlit server:
streamlit run app.py

Access: A browser window will open automatically. If it does not, copy the Local URL printed in your terminal and enter it into your browser.

Interact: Use the sliders and dropdown menus to configure a property, then click "Estimate Price" to generate the valuation and charts.

Advanced: Training the Model Locally
If you wish to verify the methodology or re-train the model using the raw source data:

Dataset Acquisition: Download the "USA Real Estate Dataset" from Kaggle via https://www.kaggle.com/datasets/yasirumanujith/usa-real-estate-dataset

File Placement: Place the raw USA Real Estate Dataset.csv file into the root project folder.

Execute Training: Open the USA_Estimator_Training.ipynb notebook in VS Code or Jupyter.

Process: Run the cells sequentially to perform the 80/20 train/test split, execute Target Mean Encoding for geographic features, and train the Random Forest Regressor.

Output: The notebook will save updated .joblib artifacts, which the web application will automatically use upon the next launch.

File Structure
app.py: The main Streamlit dashboard source code.

USA_Estimator_Training.ipynb: The Jupyter Notebook containing data exploration, cleaning, and training logic.

estimate_model.joblib: The serialized Random Forest model weights.

location_reference.csv: A lightweight helper file used to populate location selectors in the UI.

city_map.joblib / state_map.joblib / metadata.joblib: Supporting artifacts for geographic encoding and market benchmarking.

Technical Notes
Data Privacy: To ensure ethical data usage and prevent algorithmic bias, the dataset is stripped of protected-class indicators, focusing solely on objective physical and geographic traits.

Storage Optimization: The raw dataset is excluded from the main repository due to size; however, the application remains fully functional using the provided pre-trained model artifacts.
