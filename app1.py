import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="AI Fitness Planner", layout="wide", page_icon="⚡")

# ---------------------------
# PROFESSIONAL DARK THEME
# ---------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>
/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

.block-container {
    padding-top: 0rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 1200px !important;
}

#MainMenu, header, footer { visibility: hidden; }

.stApp {
    background-color: #FAF8F5;
    background-image:
        radial-gradient(ellipse at 10% 0%, rgba(99,102,241,0.07) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 100%, rgba(16,185,129,0.06) 0%, transparent 50%);
    color: #2B2B2B;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #2B2B2B 0%, #292524 60%, #2B2B2B 100%);
    border-radius: 24px;
    padding: 3rem 3.2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(28,25,23,0.18), 0 4px 16px rgba(28,25,23,0.12);
}

.hero-banner::before {
    content: "";
    position: absolute;
    top: -80px; right: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(99,102,241,0.2) 0%, transparent 65%);
    border-radius: 50%;
}

.hero-banner::after {
    content: "";
    position: absolute;
    bottom: -50px; left: 35%;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(16,185,129,0.15) 0%, transparent 65%);
    border-radius: 50%;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 800;
    color:#2B2B2B !important;
    letter-spacing: -0.5px;
    margin: 0 0 0.4rem 0;
    line-height: 1.2;
}

.hero-title span {
    background: linear-gradient(90deg, #7BAE8B, #C58C65);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1rem;
    color:#6B7280 !important;
    font-weight: 300;
    margin: 0;
    letter-spacing: 0.3px;
}

.hero-badge {
    display: inline-block;
    background: rgba(129,140,248,0.15);
    border: 1px solid rgba(129,140,248,0.35);
    color: #E2E8F0 !important;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 0.9rem;
}

/* ── Section Label ── */
.section-label {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #5C8D76;
    margin-bottom: 1.2rem;
    margin-top: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.section-label::after {
    content: "";
    flex: 1;
    height: 1px;
    background: rgba(99,102,241,0.2);
}

/* ── Metric Cards ── */
.metric-card {
    background: #FFFFFF;
    border: 1px solid rgba(28,25,23,0.08);
    border-radius: 16px;
    padding: 1.5rem 1.7rem;
    position: relative;
    overflow: hidden;
    margin-bottom: 0.5rem;
    box-shadow: 0 2px 12px rgba(28,25,23,0.06), 0 1px 3px rgba(28,25,23,0.04);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}

.metric-card:hover {
    box-shadow: 0 8px 28px rgba(28,25,23,0.1), 0 2px 6px rgba(28,25,23,0.06);
    transform: translateY(-1px);
}

.metric-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #5C8D76, #7BAE8B);
}

.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 2.5rem;
    font-weight: 800;
    color: #2B2B2B;
    line-height: 1;
    margin-bottom: 0.2rem;
}

.metric-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #A8A29E;
    margin-bottom: 0.8rem;
}

.metric-tag {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    margin-bottom: 0.4rem;
}

.tag-fit    { background: rgba(16,185,129,0.1);  color: #059669; border: 1px solid rgba(16,185,129,0.3); }
.tag-warn   { background: rgba(245,158,11,0.1);  color: #D97706; border: 1px solid rgba(245,158,11,0.3); }
.tag-danger { background: rgba(239,68,68,0.1);   color: #DC2626; border: 1px solid rgba(239,68,68,0.3); }
.tag-blue   { background: rgba(99,102,241,0.1);  color: #4F46E5; border: 1px solid rgba(99,102,241,0.3); }

.metric-hint {
    font-size: 0.8rem;
    color: #78716C;
    margin-top: 0.3rem;
    line-height: 1.5;
}

/* ── Day Header ── */
.day-header {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: #2B2B2B;
    padding: 0.6rem 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.day-dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    background: linear-gradient(135deg, #5C8D76, #7BAE8B);
    display: inline-block;
    flex-shrink: 0;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15);
}

/* ── Global Widget Styling ── */
h1, h2, h3, h4 {
    font-family: 'Playfair Display', serif !important;
    color: #2B2B2B !important;
}

p, span, div, label { color: #2B2B2B !important; }

/* Inputs */
.stNumberInput input,
.stTextInput input {
    background-color:#6B7280 !important;
    border: 1.5px solid rgba(28,25,23,0.15) !important;
    border-radius: 10px !important;
    color: #2B2B2B !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    box-shadow: 0 1px 4px rgba(28,25,23,0.05) !important;
}

.stNumberInput input:focus,
.stTextInput input:focus {
    border-color: #5C8D76 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div,
[data-baseweb="select"] > div {
    background-color:#6B7280 !important;
    border: 1.5px solid rgba(28,25,23,0.15) !important;
    border-radius: 10px !important;
    color: #2B2B2B !important;
    box-shadow: 0 1px 4px rgba(28,25,23,0.05) !important;
}

[data-baseweb="popover"] ul {
    background-color:#6B7280 !important;
    border: 1px solid rgba(28,25,23,0.1) !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 30px rgba(28,25,23,0.12) !important;
}

[data-baseweb="popover"] li {
    background-color:#6B7280 !important;
    color: #2B2B2B !important;
}

[data-baseweb="popover"] li:hover {
    background-color: rgba(99,102,241,0.07) !important;
}

/* Multiselect */
[data-baseweb="tag"] {
    background-color: rgba(99,102,241,0.12) !important;
    color: #4F46E5 !important;
    border-radius: 6px !important;
}

/* Radio */
[data-testid="stRadio"] label { color: #2B2B2B !important; }
[data-testid="stRadio"] p { color: #2B2B2B !important; }

/* Widget labels */
[data-testid="stWidgetLabel"] p,
.stSelectbox label,
.stNumberInput label,
.stMultiSelect label {
    color: #78716C !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Metric overrides */
[data-testid="stMetricValue"] { color: #2B2B2B !important; }
[data-testid="stMetricLabel"] { color: #78716C !important; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #4F46E5, #5C8D76) !important;
    color: white !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 16px rgba(79,70,229,0.35) !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #4338CA, #4F46E5) !important;
    box-shadow: 0 8px 24px rgba(79,70,229,0.45) !important;
    transform: translateY(-2px) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(28,25,23,0.08) !important;
    box-shadow: 0 2px 10px rgba(28,25,23,0.05) !important;
}

hr { border-color: rgba(28,25,23,0.08) !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# HERO BANNER
# ---------------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">⚡ Personalized Fitness</div>
    <div class="hero-title">Personalized <span>Workout Planner</span></div>
    <div class="hero-subtitle">Elegant, personalized workout recommendations tailored to your body, fitness level and lifestyle.</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv("fitness_exercises_large.csv")
df.dropna(inplace=True)

# ---------------------------
# OUTLIER REMOVAL
# ---------------------------
# Remove extreme outliers using IQR on numeric columns only
for col in df.select_dtypes(include=[np.number]).columns:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR))]
df.reset_index(drop=True, inplace=True)

# ---------------------------
# FEATURE ENGINEERING
# ---------------------------
le_muscle = LabelEncoder()
le_diff   = LabelEncoder()

df["Muscle_enc"]     = le_muscle.fit_transform(df["MuscleGroup"])
df["Difficulty_enc"] = le_diff.fit_transform(df["Difficulty"])
df["Gym_Machine"]    = df["Gym_Machine"].map({"Yes": 1, "No": 0})
df["Home_Compatible"]= df["Home_Compatible"].map({"Yes": 1, "No": 0})

# Improved multi-factor suitability label
def generate_label(row):
    score = 0
    # Difficulty — equal 1 point spread
    if row["Difficulty"] == "Beginner":      score += 1
    elif row["Difficulty"] == "Intermediate": score += 2
    else:                                     score += 3
    # Location compatibility — same weight as difficulty
    if row["Home_Compatible"] == 1: score += 2
    if row["Gym_Machine"] == 1:     score += 2
    # Muscle group variety bonus (encoded value spread)
    score += round(row["Muscle_enc"] % 3)
    return score

df["suitability"] = df.apply(generate_label, axis=1)

features = ["Muscle_enc", "Difficulty_enc", "Gym_Machine", "Home_Compatible"]
X = df[features]
y = df["suitability"]

# ---------------------------
# TRAIN / TEST SPLIT
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# PIPELINES & MODEL TRAINING
# ---------------------------

MODEL_FILE = "best_fitness_model.pkl"

@st.cache_resource(show_spinner="🔧 Training models...")
def train_all_models():

    def make_pipeline(model):
        return Pipeline([("scaler", StandardScaler()), ("model", model)])

    # n_estimators bumped to 100 for better accuracy
    gb_pipe = make_pipeline(GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
    rf_pipe = make_pipeline(RandomForestRegressor(
        n_estimators=100, max_depth=8, n_jobs=1, random_state=42))
    lr_pipe = make_pipeline(LinearRegression())

    gb_pipe.fit(X_train, y_train)
    rf_pipe.fit(X_train, y_train)
    lr_pipe.fit(X_train, y_train)

    def calc_metrics(pipe, name):
        tr_pred = pipe.predict(X_train)
        te_pred = pipe.predict(X_test)
        # Real 5-fold cross-validation on full dataset
        cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="neg_mean_squared_error")
        real_cv_mse = round(-cv_scores.mean(), 4)
        return {
            "name":      name,
            "train_mse": round(mean_squared_error(y_train, tr_pred), 4),
            "test_mse":  round(mean_squared_error(y_test,  te_pred), 4),
            "train_acc": round(r2_score(y_train, tr_pred) * 100, 2),
            "test_acc":  round(r2_score(y_test,  te_pred) * 100, 2),
            "cv_mse":    real_cv_mse,   # ← real 5-fold CV now
        }

    all_metrics = [
        calc_metrics(gb_pipe, "GradientBoosting"),
        calc_metrics(rf_pipe, "RandomForest"),
        calc_metrics(lr_pipe, "LinearRegression"),
    ]

    best_m    = min(all_metrics, key=lambda d: d["test_mse"])
    best_name = best_m["name"]
    pipes     = {"GradientBoosting": gb_pipe,
                 "RandomForest":     rf_pipe,
                 "LinearRegression": lr_pipe}
    best_pipe = pipes[best_name]

    # Feature importance
    feat_imp = None
    inner    = best_pipe.named_steps["model"]
    if hasattr(inner, "feature_importances_"):
        feat_imp = dict(zip(features, inner.feature_importances_))

    # Collect actual hyperparameters of the best model
    best_params = {k.replace("model__", ""): v
                   for k, v in best_pipe.get_params().items()
                   if k.startswith("model__") and not callable(v)}

    joblib.dump(best_pipe, MODEL_FILE)
    return best_pipe, best_name, all_metrics, feat_imp, best_params, pipes

# Load from disk if already trained; retrain only on first run
if os.path.exists(MODEL_FILE):
    try:
        _loaded = joblib.load(MODEL_FILE)
        # Still run train_all_models for metrics/feat_imp (cached after first call)
        best_pipe, best_model_name, all_metrics, feat_imp, best_params, all_pipes = train_all_models()
        model = _loaded
    except Exception:
        best_pipe, best_model_name, all_metrics, feat_imp, best_params, all_pipes = train_all_models()
        model = best_pipe
else:
    best_pipe, best_model_name, all_metrics, feat_imp, best_params, all_pipes = train_all_models()
    model = best_pipe

# ---------------------------
# HELPERS
# ---------------------------
def calculate_bmi(weight, height):
    if height == 0: return 0
    return weight / ((height / 100) ** 2)

def fitness_score(bmi, level):
    score = 40 if 18.5 <= bmi <= 24.9 else (25 if bmi <= 29.9 else 10)
    score += {"Beginner": 20, "Intermediate": 30, "Advanced": 40}[level]
    return score

def bmi_category(bmi):
    if bmi < 18.5:  return "Underweight", "Focus on strength and nutrition", "tag-warn"
    elif bmi < 25:  return "Healthy",     "Maintain consistency and balanced training", "tag-fit"
    elif bmi < 30:  return "Overweight",  "Include cardio and moderate intensity workouts", "tag-warn"
    else:           return "Obese",       "Start with low-impact and gradual progression", "tag-danger"

def fitness_score_category(score):
    if score >= 70:   return "Excellent", "Strong fitness base. Ready for high intensity.", "tag-fit"
    elif score >= 50: return "Moderate",  "Average level. Focus on consistency.", "tag-blue"
    else:             return "Beginner",  "Start light and build gradually.", "tag-warn"

def generate_progression(level):
    plans = {
        "Beginner":     ["2 sets × 10 reps", "3 sets × 10 reps", "3 sets × 12 reps", "3 sets × 12–15 reps"],
        "Intermediate": ["3 sets × 10 reps", "4 sets × 10 reps", "4 sets × 12 reps", "4 sets × 12–15 reps"],
        "Advanced":     ["4 sets × 8 reps",  "4 sets × 10 reps", "5 sets × 10 reps", "5 sets × 12 reps"],
    }
    return plans[level]

def weekly_split(days):
    all_days = {
        "Day 1": ["Chest", "Triceps"],
        "Day 2": ["Back", "Biceps"],
        "Day 3": ["Legs"],
        "Day 4": ["Shoulders", "Core"],
        "Day 5": ["Biceps", "Triceps"],
        "Day 6": ["Core"],
    }
    return {k: all_days[k] for k in list(all_days.keys())[:days]}

def age_filter(data, age):
    if age >= 50:
        data = data[data["Difficulty"] != "Advanced"]
        return data.head(3)
    return data

def recommend_exercises(muscles, location, injuries, level, age):
    subset = df[df["MuscleGroup"].isin(muscles)].copy()
    if location == "Home":
        subset = subset[subset["Home_Compatible"] == 1]
    else:
        subset = subset[subset["Gym_Machine"] == 1]
    if injuries:
        subset = subset[~subset["Avoid_If_Injury"].isin(injuries)]
    subset = age_filter(subset, age)
    if subset.empty:
        return pd.DataFrame(columns=["Exercise", "MuscleGroup", "Difficulty", "Why Recommended"])
    subset["Difficulty_match"] = (subset["Difficulty"] == level).astype(int)
    subset["ML_score"]   = model.predict(subset[features])
    subset["Final_score"]= subset["ML_score"] + subset["Difficulty_match"] * 2
    subset = subset.sort_values("Final_score", ascending=False)
    top = subset.head(5)

    explanations = []
    for _, row in top.iterrows():
        reasons = []
        if row["Difficulty"] == level: reasons.append("Matches your level")
        if location == "Home" and row["Home_Compatible"] == 1: reasons.append("Home compatible")
        if location == "Gym" and row["Gym_Machine"] == 1: reasons.append("Gym suitable")
        if not injuries: reasons.append("Safe for your condition")
        explanations.append(", ".join(reasons))

    top = top.copy()
    top["Why Recommended"] = explanations
    return top[["Exercise", "MuscleGroup", "Difficulty", "Why Recommended"]]

# ---------------------------
# USER PROFILE
# ---------------------------
st.markdown('<div class="section-label">👤 &nbsp; User Profile</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1: age    = st.number_input("Age",         min_value=15, max_value=70, value=25)
with col2: weight = st.number_input("Weight (kg)", value=70.0)
with col3: height = st.number_input("Height (cm)", value=170.0)

st.markdown('<div class="section-label" style="margin-top:1.5rem;">🏋️ &nbsp; Workout Preferences</div>', unsafe_allow_html=True)

col4, col5 = st.columns(2)
with col4: location = st.selectbox("Workout Location", ["Home", "Gym"])
with col5: level    = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced"])

injuries = st.multiselect("Injuries / Areas to Avoid", ["knee", "lower back", "shoulder", "elbow"],
                          placeholder="Select if applicable...")

st.markdown('<div class="section-label" style="margin-top:1.5rem;">📅 &nbsp; Workout Frequency</div>', unsafe_allow_html=True)
days = st.radio("Days per week", [1, 2, 3, 4, 5, 6], horizontal=True, index=2)

# ---------------------------
# HEALTH ANALYSIS (live)
# ---------------------------
bmi   = calculate_bmi(weight, height)
score = fitness_score(bmi, level)
bmi_cat,  bmi_hint,  bmi_tag  = bmi_category(bmi)
fit_cat,  fit_hint,  fit_tag  = fitness_score_category(score)

st.markdown('<div class="section-label" style="margin-top:1.5rem;">📊 &nbsp; Health Analysis</div>', unsafe_allow_html=True)

colA, colB = st.columns(2)

with colA:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Body Mass Index</div>
        <div class="metric-value">{round(bmi, 1)}</div>
        <span class="metric-tag {bmi_tag}">{bmi_cat}</span>
        <div class="metric-hint">{bmi_hint}</div>
    </div>
    """, unsafe_allow_html=True)

with colB:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Fitness Score</div>
        <div class="metric-value">{score}<span style="font-size:1rem; color:#4B5563;">/80</span></div>
        <span class="metric-tag {fit_tag}">{fit_cat}</span>
        <div class="metric-hint">{fit_hint}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------
# MODEL ACCURACY (always visible)
# ---------------------------
st.markdown('<div class="section-label">🎯 &nbsp; Model Accuracy</div>', unsafe_allow_html=True)

acc_cols = st.columns(3)
model_icons = {"GradientBoosting": "🚀", "RandomForest": "🌲", "LinearRegression": "📐"}
for col, m in zip(acc_cols, all_metrics):
    is_best   = m["name"] == best_model_name
    gap       = round(abs(m["test_acc"] - m["train_acc"]), 1)
    overfit   = "⚠️ Overfit risk" if gap > 20 else "✅ Stable"
    icon      = model_icons.get(m["name"], "🤖")
    label     = f"{icon} {m['name']} {'★ BEST' if is_best else ''}"
    with col:
        st.metric(
            label=label,
            value=f"{m['test_acc']}%",
            delta=f"Train: {m['train_acc']}%  |  MSE: {m['test_mse']}"
        )
        st.caption(f"{overfit}  (gap {gap}%)  |  5-Fold CV MSE: {m['cv_mse']}")

# Feature Importance
if feat_imp:
    st.markdown('<div class="section-label" style="margin-top:1.5rem;">📊 &nbsp; Feature Importance</div>', unsafe_allow_html=True)
    sorted_imp = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
    labels = {"Muscle_enc": "Muscle Group", "Difficulty_enc": "Difficulty",
              "Gym_Machine": "Gym Machine", "Home_Compatible": "Home Compat."}
    fi_cols = st.columns(len(sorted_imp))
    for col, (feat, val) in zip(fi_cols, sorted_imp):
        pct = round(val * 100, 1)
        with col:
            st.metric(label=labels.get(feat, feat), value=f"{pct}%")
            st.progress(int(pct))

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------
# GENERATE BUTTON
# ---------------------------
col_btn, _ = st.columns([1, 3])
with col_btn:
    generate = st.button("Generate Workout →", use_container_width=True)

# ---------------------------
# PLAN OUTPUT
# ---------------------------
if generate:

    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:0.6rem; margin: 1.5rem 0 0.5rem 0;">
        <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.65rem; font-weight:700;
                    letter-spacing:2px; text-transform:uppercase; color:#059669;">
            ✅ &nbsp; Plan Generated
        </div>
        <div style="flex:1; height:1px; background:rgba(5,150,105,0.2);"></div>
        <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.72rem; color:#78716C;">
            Model: {best_model_name} &nbsp;·&nbsp; Accuracy: {[m for m in all_metrics if m['name']==best_model_name][0]['test_acc']}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -- 4-Week Progression --
    st.markdown('<div class="section-label" style="margin-top:1rem;">📈 &nbsp; 4-Week Progression</div>', unsafe_allow_html=True)

    progression = generate_progression(level)
    prog_cols = st.columns(4)
    for i, (col, week_text) in enumerate(zip(prog_cols, progression)):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center; padding:1.2rem 1rem;">
                <div class="metric-label">Week {i+1}</div>
                <div style="font-family:'Playfair Display',serif; font-size:1rem;
                            font-weight:700; color:#2B2B2B; line-height:1.4;">
                    {week_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # -- Weekly Split --
    st.markdown('<div class="section-label" style="margin-top:1.5rem;">🗓️ &nbsp; Weekly Split</div>', unsafe_allow_html=True)

    split = weekly_split(days)

    for day, muscles in split.items():
        st.markdown(f"""
        <div class="day-header">
            <span class="day-dot"></span> {day}
            <span style="font-size:0.75rem; color:#A8A29E; font-weight:400; margin-left:0.2rem;">
                — {' · '.join(muscles)}
            </span>
        </div>
        """, unsafe_allow_html=True)

        result = recommend_exercises(muscles, location, injuries, level, age)
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div style='margin-top:50px;padding:20px;text-align:center;color:#8A817C;font-size:14px'>
<hr>
Built with ❤️ using Streamlit & Scikit-learn · <b>Palak Virk</b>
</div>
""", unsafe_allow_html=True)
