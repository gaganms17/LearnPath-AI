from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

NUMERIC_FEATURES = [
    "age", "weekly_study_hours", "engagement_score",
    "courses_completed", "quiz_average", "activity_days"
]

SEGMENT_NAMES = {
    0: "Consistent Learners",
    1: "Career-Focused Learners",
    2: "Exploratory Learners",
}

def load_data(students_path, courses_path):
    students = pd.read_csv(students_path)
    courses = pd.read_csv(courses_path)
    required_s = {"student_id", "age", "weekly_study_hours", "engagement_score",
                  "courses_completed", "quiz_average", "activity_days",
                  "interest", "skill_level", "career_goal"}
    required_c = {"course_id", "course_name", "category", "level", "career_goal",
                  "rating", "duration_hours", "description"}
    if not required_s.issubset(students.columns):
        raise ValueError(f"Student data missing columns: {sorted(required_s-set(students.columns))}")
    if not required_c.issubset(courses.columns):
        raise ValueError(f"Course data missing columns: {sorted(required_c-set(courses.columns))}")
    if students.empty or courses.empty:
        raise ValueError("Student and course datasets must not be empty.")
    if students[NUMERIC_FEATURES].isna().any().any():
        raise ValueError("Student numerical features contain missing values.")
    return students, courses

def train_segmenter(students, n_clusters=3, random_state=42):
    if len(students) < n_clusters:
        raise ValueError("Number of students must be at least the number of clusters.")
    scaler = StandardScaler()
    X = scaler.fit_transform(students[NUMERIC_FEATURES])
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    cluster_ids = model.fit_predict(X)
    score = silhouette_score(X, cluster_ids) if len(set(cluster_ids)) > 1 else 0.0
    # Stable, human-readable labels based on cluster profile, not arbitrary K-Means IDs.
    profile = students.assign(_cluster=cluster_ids).groupby("_cluster")[
        ["engagement_score", "courses_completed", "weekly_study_hours"]
    ].mean()
    order = profile["engagement_score"].sort_values().index.tolist()
    names = ["Exploratory Learners", "Consistent Learners", "Career-Focused Learners"]
    mapping = {cluster_id: names[i] for i, cluster_id in enumerate(order)}
    metadata = {"silhouette_score": float(score), "cluster_mapping": mapping}
    return model, scaler, NUMERIC_FEATURES, metadata

def segment_students(students, model, scaler, features):
    result = students.copy()
    raw = model.predict(scaler.transform(result[features]))
    # Label clusters according to profile ranking for consistency with training.
    profile = result.assign(_cluster=raw).groupby("_cluster")[
        ["engagement_score", "courses_completed", "weekly_study_hours"]
    ].mean()
    order = profile["engagement_score"].sort_values().index.tolist()
    names = ["Exploratory Learners", "Consistent Learners", "Career-Focused Learners"]
    mapping = {cluster_id: names[i] for i, cluster_id in enumerate(order)}
    result["segment"] = pd.Series(raw, index=result.index).map(mapping)
    return result

def segment_summary(segmented):
    cols = ["engagement_score", "weekly_study_hours", "courses_completed", "quiz_average"]
    return segmented.groupby("segment", as_index=False)[cols].mean().round(1)

def recommend_courses(student, courses, top_n=5):
    """Transparent hybrid ranking; scores are heuristic relevance, not probabilities."""
    ranked = courses.copy()
    interest = str(student["interest"]).strip().lower()
    goal = str(student["career_goal"]).strip().lower()
    skill = str(student["skill_level"]).strip().lower()

    ranked["interest_match"] = ranked["category"].str.lower().eq(interest).astype(float)
    ranked["goal_match"] = ranked["career_goal"].str.lower().eq(goal).astype(float)
    level_order = {"beginner": 0, "intermediate": 1, "advanced": 2}
    student_level = level_order.get(skill, 0)
    ranked["level_match"] = ranked["level"].str.lower().map(
        lambda x: 1.0 if level_order.get(x, 0) == student_level else
                  0.55 if abs(level_order.get(x, 0)-student_level) == 1 else 0.2
    )
    rating = ranked["rating"].clip(0, 5) / 5
    ranked["match_score"] = (
        45*ranked["interest_match"] + 30*ranked["goal_match"] +
        15*ranked["level_match"] + 10*rating
    ).clip(0, 100)
    ranked = ranked.sort_values(["match_score", "rating"], ascending=False)
    return ranked.head(top_n).reset_index(drop=True)