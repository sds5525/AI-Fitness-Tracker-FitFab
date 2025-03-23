import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import time

import warnings
warnings.filterwarnings('ignore')

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True
)
wtype = ''

st.write("## AI Fitness Tracker ~ FitFab")
#st.image("", use_column_width=True)
st.write("In this WebApp you will be able to calculate your BMI, track your sleep, observe the predicted calories burned in your body after a workout. Pass your parameters into this WebApp and see the predicted values.")

st.header("Your Details ")

def user_input_features():
    age = st.slider("Age: ", 10, 100, 30)
    weight = st.slider("Weight(kg): ", 15, 300, 70)
    height = st.slider("Height(cm): ", 50, 500, 170)
    gender_button = st.radio("Gender: ", ("Male", "Female"))
    st.header("Workout: ")
    work_type = ["Cardio","Strength Training","Yoga/Pilates","Sports","Dance-Based","Other"]
    wtype = st.selectbox("Workout type: ",work_type)
    duration = st.slider("Duration (min): ", 0, 60, 15)
    heart_rate = st.slider("Heart Rate: ", 60, 130, 80)
    body_temp = st.slider("Body Temperature (C): ", 36, 42, 38)
    
    bmi = weight/(height*height/100)

    gender = 1 if gender_button == "Male" else 0

    # Use column names to match the training data
    data_model = {
        "Age": age,
        "BMI": bmi,
        "Type": wtype,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender  # Gender is encoded as 1 for male, 0 for female
    }

    features = pd.DataFrame(data_model, index=[0])
    return features

df = user_input_features()


# Create or append user parameters to a CSV file
def save_user_input(data):
    file_path = "user_para.csv"
    try:
        existing_data = pd.read_csv(file_path)
        updated_data = pd.concat([existing_data, data], ignore_index=True)
    except FileNotFoundError:
        updated_data = data
    
    updated_data.to_csv(file_path, index=False)

save_user_input(df)


st.write("---")
st.header("Your Parameters: ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)
st.write(df)


calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

# Add BMI column to both training and test sets
for data in [exercise_train_data, exercise_test_data]:
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
    data["BMI"] = round(data["BMI"], 2)

# Prepare the training and testing sets
exercise_train_data = exercise_train_data[["Gender", "Age", "BMI","Type", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_test_data = exercise_test_data[["Gender", "Age", "BMI","Type", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

# Separate features and labels
X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]

X_test = exercise_test_data.drop("Calories", axis=1)
y_test = exercise_test_data["Calories"]

# Train the model
random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
random_reg.fit(X_train, y_train)

# Align prediction data columns with training data
df = df.reindex(columns=X_train.columns, fill_value=0)

# Make prediction
prediction = random_reg.predict(df)

st.write("---")
st.header("Prediction: ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

st.write(f"{round(prediction[0], 2)} **kilocalories**")


st.write("---")
st.header("Similar User Results: ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

# Find recent results based on predicted calories
calorie_range = [prediction[0] - 10, prediction[0] + 10]

# Filter recent data and select only required columns
recent_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & 
                          (exercise_df["Calories"] <= calorie_range[1])]

# Sort by most recent values
recent_data = recent_data[["Type", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]\
                .sort_index(ascending=False)

# Display the top 5 recent results
st.write(recent_data.head(5))



st.write("---")
st.header("General Information: ")
boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
if "Type" in df.columns:
    type_column = f"Type_{df['Type'].values[0]}"
    if type_column in exercise_df.columns:
        boolean_type = (exercise_df[type_column] == 1).tolist()
    else:
        boolean_type = [False] * len(exercise_df)  # If not present, return False
else:
    boolean_type = [False] * len(exercise_df)


boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()


st.write("You are older than", round(sum(boolean_age) / len(boolean_age), 3) * 100, "% of other people.")
st.write("Your exercise duration is higher than", round(sum(boolean_duration) / len(boolean_duration), 2) * 100, "% of other people.")
st.write("You have a higher heart rate than", round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100, "% of other people during exercise.")
st.write("You have a higher body temperature than", round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100, "% of other people during exercise.")
