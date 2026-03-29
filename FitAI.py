import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="AI Fitness Planner", layout="wide")

# ---------------------------
# DARK THEME (YOUR STYLE)
# ---------------------------
st.markdown("""
<style>
.stApp { background-color: #0E1117; color: white; }
h1, h2, h3 { color: white; font-family: "Segoe UI"; }
label { color: white !important; }
.stButton>button {
    background-color: #2563EB;
    color: white;
    font-weight: 600;
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# TITLE
# ---------------------------
st.title("AI-Based Personalized Workout Planner")

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv("fitness_exercises_large.csv")
df.dropna(inplace=True)

# ---------------------------
# FEATURE ENGINEERING
# ---------------------------
le_muscle = LabelEncoder()
le_diff = LabelEncoder()

df["Muscle_enc"] = le_muscle.fit_transform(df["MuscleGroup"])
df["Difficulty_enc"] = le_diff.fit_transform(df["Difficulty"])

df["Gym_Machine"] = df["Gym_Machine"].map({"Yes":1,"No":0})
df["Home_Compatible"] = df["Home_Compatible"].map({"Yes":1,"No":0})

# ---------------------------
# LABEL GENERATION
# ---------------------------
def generate_label(row):
    score = 0
    if row["Difficulty"] == "Beginner":
        score += 2
    if row["Home_Compatible"] == 1:
        score += 1
    if row["Gym_Machine"] == 1:
        score += 1
    return score

df["suitability"] = df.apply(generate_label, axis=1)

features = ["Muscle_enc","Difficulty_enc","Gym_Machine","Home_Compatible"]
X = df[features]
y = df["suitability"]

# ---------------------------
# TRAIN MULTIPLE MODELS
# ---------------------------
gb_model = GradientBoostingRegressor()
rf_model = RandomForestRegressor()
lr_model = LinearRegression()

gb_model.fit(X, y)
rf_model.fit(X, y)
lr_model.fit(X, y)

gb_score = mean_squared_error(y, gb_model.predict(X))
rf_score = mean_squared_error(y, rf_model.predict(X))
lr_score = mean_squared_error(y, lr_model.predict(X))

scores = {
    "GradientBoosting": gb_score,
    "RandomForest": rf_score,
    "LinearRegression": lr_score
}

best_model_name = min(scores, key=scores.get)

if best_model_name == "GradientBoosting":
    model = gb_model
elif best_model_name == "RandomForest":
    model = rf_model
else:
    model = lr_model

# ---------------------------
# BMI + FITNESS SCORE
# ---------------------------
def calculate_bmi(weight, height):
    if height == 0:
        return 0
    return weight / ((height/100)**2)

def fitness_score(bmi, level):
    score = 0

    if 18.5 <= bmi <= 24.9:
        score += 40
    elif 25 <= bmi <= 29.9:
        score += 25
    else:
        score += 10

    if level == "Beginner":
        score += 20
    elif level == "Intermediate":
        score += 30
    else:
        score += 40

    return score

# ---------------------------
# PROGRESSION
# ---------------------------
def generate_progression(level):
    if level == "Beginner":
        return [
            "Week 1: 2 sets x 10 reps",
            "Week 2: 3 sets x 10 reps",
            "Week 3: 3 sets x 12 reps",
            "Week 4: 3 sets x 12–15 reps"
        ]
    elif level == "Intermediate":
        return [
            "Week 1: 3 sets x 10 reps",
            "Week 2: 4 sets x 10 reps",
            "Week 3: 4 sets x 12 reps",
            "Week 4: 4 sets x 12–15 reps"
        ]
    else:
        return [
            "Week 1: 4 sets x 8 reps",
            "Week 2: 4 sets x 10 reps",
            "Week 3: 5 sets x 10 reps",
            "Week 4: 5 sets x 12 reps"
        ]

# ---------------------------
# WEEKLY SPLIT
# ---------------------------
def weekly_split(days):
    return {
        "Day 1":["Chest","Triceps"],
        "Day 2":["Back","Biceps"],
        "Day 3":["Legs"],
        "Day 4":["Shoulders","Core"],
        "Day 5":["Biceps","Triceps"],
        "Day 6":["Core"]
    }

# ---------------------------
# AGE FILTER
# ---------------------------
def age_filter(data, age):
    if age >= 50:
        data = data[data["Difficulty"] != "Advanced"]
        return data.head(3)
    return data

# ---------------------------
# RECOMMENDER
# ---------------------------
def recommend_exercises(muscles, location, injuries, level, age):

    subset = df[df["MuscleGroup"].isin(muscles)]

    if location == "Home":
        subset = subset[subset["Home_Compatible"] == 1]
    else:
        subset = subset[subset["Gym_Machine"] == 1]

    if injuries:
        subset = subset[~subset["Avoid_If_Injury"].isin(injuries)]

    subset = age_filter(subset, age)

    subset["Difficulty_match"] = (subset["Difficulty"] == level).astype(int)

    subset["ML_score"] = model.predict(subset[features])

    subset["Final_score"] = subset["ML_score"] + subset["Difficulty_match"] * 2

    subset = subset.sort_values("Final_score", ascending=False)

    top = subset.head(5)

    # Explainability
    explanations = []
    for _, row in top.iterrows():
        reasons = []
        if row["Difficulty"] == level:
            reasons.append("Matches your level")
        if location == "Home" and row["Home_Compatible"] == 1:
            reasons.append("Home compatible")
        if location == "Gym" and row["Gym_Machine"] == 1:
            reasons.append("Gym suitable")
        if not injuries:
            reasons.append("Safe for your condition")
        explanations.append(", ".join(reasons))

    top["Why Recommended"] = explanations

    return top[["Exercise","MuscleGroup","Difficulty","Why Recommended"]]

def bmi_category(bmi):

    if bmi < 18.5:
        return "Underweight", "Focus on strength and nutrition"
    
    elif bmi < 25:
        return "Fit", "Maintain consistency and balanced training"
    
    elif bmi < 30:
        return "Overweight", "Include cardio and moderate intensity workouts"
    
    else:
        return "Obese", "Start with low-impact and gradual progression"
    
def fitness_score_category(score):

    if score >= 70:
        return "Excellent", "You have a strong fitness base. You can handle higher intensity and volume."

    elif score >= 50:
        return "Moderate", "You have an average fitness level. Focus on consistency and gradual progression."

    else:
        return "Beginner", "Start with low intensity and build your fitness gradually."

# ---------------------------
# USER INPUT
# ---------------------------
st.subheader("User Profile")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 15, 70)

with col2:
    weight = st.number_input("Weight (kg)")

with col3:
    height = st.number_input("Height (cm)")

st.subheader("Workout Preferences")

col4, col5, col6 = st.columns(3)

with col4:
    location = st.selectbox("Location", ["Home","Gym"])

with col5:
    level = st.selectbox("Experience", ["Beginner","Intermediate","Advanced"])

with col6:
    pass

injuries = st.multiselect("Injuries", ["knee","lower back","shoulder","elbow"])

# ---------------------------
# DAY SELECTION
# ---------------------------
st.subheader("Workout Frequency")

days = st.radio(
    "Select number of workout days per week",
    [1, 2, 3, 4, 5, 6],
    horizontal=True
)

# ---------------------------
# HEALTH PANEL
# ---------------------------
# ---------------------------
# HEALTH ANALYSIS
# ---------------------------

bmi = calculate_bmi(weight, height)
score = fitness_score(bmi, level)

bmi_cat, bmi_suggestion = bmi_category(bmi)
fit_cat, fit_suggestion = fitness_score_category(score)

st.subheader("Health Analysis")

colA, colB = st.columns(2)

with colA:
    st.metric("BMI", round(bmi, 2))
    st.write(f"Category: {bmi_cat}")
    st.write(f"Guidance: {bmi_suggestion}")

with colB:
    st.metric("Fitness Score", score)
    st.write(f"Level: {fit_cat}")
    st.write(f"Meaning: {fit_suggestion}")

# ---------------------------
# GENERATE PLAN
# ---------------------------

if st.button("Generate Plan"):

    st.write(f"Selected Model: {best_model_name}")

    progression = generate_progression(level)

    st.subheader("4-Week Progression Plan")
    for week in progression:
        st.write(week)

    split = weekly_split(days)

    # show only required number of days
    split = {k: split[k] for k in list(split.keys())[:days]}

    for day, muscles in split.items():
        st.subheader(day)
        result = recommend_exercises(muscles, location, injuries, level, age)
        st.dataframe(result, use_container_width=True)