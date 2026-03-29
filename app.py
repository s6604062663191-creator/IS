import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# LOAD DATA
ratings = pd.read_csv("data/u.data", sep="\t",
                      names=["user", "movie", "rating", "timestamp"])

movies = pd.read_csv("data/u.item", sep="|", encoding="latin-1", header=None)
movies = movies[[0, 1] + list(range(5, 24))]
movies.columns = ["movie", "title"] + [f"genre_{i}" for i in range(19)]

df = pd.merge(ratings, movies, on="movie")

# LOAD MODEL
ml_model = joblib.load("ml_model.pkl")
nn_model = tf.keras.models.load_model("nn_model.h5")

# GENRE
genre_names = [
    "Action","Adventure","Animation","Children","Comedy",
    "Crime","Documentary","Drama","Fantasy","Film-Noir","Horror",
    "Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"
]

# USER ANALYSIS
def get_user_preference(user_id):
    user_data = df[df["user"] == user_id]
    scores = {}

    for i in range(19):
        g = user_data[user_data[f"genre_{i}"] == 1]
        if len(g) > 0:
            scores[i] = g["rating"].mean()

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# MENU
if "page" not in st.session_state:
    st.session_state.page = "ML Model"

st.sidebar.title("🎬 Movie Recommender")
st.sidebar.markdown("---")

if st.sidebar.button("ML Model"):
    st.session_state.page = "ML Model"

if st.sidebar.button("Neural Network"):
    st.session_state.page = "Neural Network"

if st.sidebar.button("Test ML"):
    st.session_state.page = "Test ML"

if st.sidebar.button("Test NN"):
    st.session_state.page = "Test NN"

st.sidebar.markdown("---")

page = st.session_state.page

# ML PAGE

if page == "ML Model":
    st.title("📊 Machine Learning Model (Ensemble)")

    st.header("1. Dataset")
    st.write("""
    Dataset ที่ใช้คือ MovieLens 100K จาก GroupLens Research
    เป็นข้อมูลจริงของการให้คะแนนภาพยนตร์จากผู้ใช้

    โครงสร้างข้อมูล:
    - user: รหัสผู้ใช้
    - movie: รหัสภาพยนตร์
    - rating: คะแนน (1–5)
    - timestamp: เวลา

    ข้อมูลภาพยนตร์:
    - title: ชื่อหนัง
    - genre: ประเภทหนัง (One-hot encoding 19 ค่า)

    ลักษณะสำคัญ:
    - เป็นข้อมูลแบบ Sparse (ไม่ได้ให้คะแนนทุกเรื่อง)
    - มีความไม่สมบูรณ์และ noise
    """)

    st.header("2. Data Preparation")
    st.write("""
    ขั้นตอนเตรียมข้อมูล:

    1. รวมข้อมูล ratings และ movies ด้วย movie id
    2. แปลง genre เป็น One-hot encoding
    3. เลือก feature:
       - user
       - movie
       - genre
    4. ตรวจสอบ missing values และแปลงชนิดข้อมูล

    ผลลัพธ์:
    ได้ dataset ที่พร้อมสำหรับ Machine Learning
    """)

    st.header("3. Model (Ensemble Learning)")
    st.write("""
    โมเดลที่ใช้เป็น Ensemble Learning ประกอบด้วย:

    - Random Forest
    - Gradient Boosting
    - Extra Trees

    ใช้ Voting Regressor เพื่อรวมผลลัพธ์

    แนวคิด:
    รวมหลายโมเดลเพื่อเพิ่มความแม่นยำและลด error
    """)

    st.header("4. หลักการทำงาน")
    st.write("""
    Input:
    - User ID
    - Movie ID
    - Genre

    กระบวนการ:
    1. เรียนรู้พฤติกรรมผู้ใช้จากข้อมูลอดีต
    2. วิเคราะห์ว่าผู้ใช้ชอบ genre ใด
    3. เปรียบเทียบกับ movie ที่เลือก
    4. ทำนายคะแนน (rating)

    Output:
    - คะแนนที่คาดว่าผู้ใช้จะให้
    """)

    st.header("5. ข้อดีและข้อจำกัด")
    st.write("""
    ข้อดี:
    - ลด overfitting
    - เพิ่มความแม่นยำ
    - ทำงานได้เร็ว

    ข้อจำกัด:
    - ไม่สามารถเรียนรู้ pattern ซับซ้อนมาก
    - ใช้ feature ที่กำหนดเท่านั้น
    """)
    st.header("6. 📈 พฤติกรรมผู้ใช้ (User Behavior Analysis)")

    user_sample = st.selectbox("เลือก User เพื่อวิเคราะห์", sorted(df["user"].unique()))

    user_data = df[df["user"] == user_sample]

    # จำนวนการให้คะแนน
    st.write("จำนวนหนังที่ user ให้คะแนน:", len(user_data))

    # ค่าเฉลี่ย rating
    st.write("ค่าเฉลี่ย rating:", round(user_data["rating"].mean(), 2))

    # กราฟ distribution rating
    st.subheader("Distribution ของ Rating")
    st.bar_chart(user_data["rating"].value_counts().sort_index())

    # กราฟ genre ที่ดูบ่อย
    genre_count = []
    for i in range(19):
        genre_count.append(user_data[f"genre_{i}"].sum())

    genre_df = pd.DataFrame({
        "genre": genre_names,
        "count": genre_count[:len(genre_names)]
    })

    st.subheader("Genre ที่ดูบ่อย")
    st.bar_chart(genre_df.set_index("genre"))


# NN PAGE
elif page == "Neural Network":
    st.title("🤖 Neural Network Model")

    st.header("1. แนวคิด")
    st.write("""
    Neural Network เป็นโมเดลที่เลียนแบบการทำงานของสมองมนุษย์
    สามารถเรียนรู้ pattern ที่ซับซ้อนได้

    เหมาะสำหรับ Recommendation System
    เพราะสามารถเรียนรู้รสนิยมของผู้ใช้ได้โดยอัตโนมัติ
    """)

    st.header("2. โครงสร้างโมเดล")
    st.write("""
    โมเดลใช้ข้อมูล:
    - User ID
    - Movie ID
    - Genre

    โครงสร้าง:

    1. User Embedding
       แปลง user เป็น vector

    2. Movie Embedding
       แปลง movie เป็น vector

    3. Genre Input
       ใช้ One-hot encoding

    4. Concatenate
       รวมข้อมูลทั้งหมด

    5. Dense Layers
       เรียนรู้ pattern

    6. Output Layer
       ทำนาย rating
    """)

    st.header("3. ขั้นตอนการทำงาน")
    st.write("""
    1. รับ input (user, movie, genre)
    2. แปลง ID เป็น embedding vector
    3. รวมข้อมูลเข้าด้วยกัน
    4. ผ่าน neural network
    5. ได้ผลลัพธ์เป็น rating

    โมเดลจะเรียนรู้โดยปรับ weight ให้ error ลดลง
    """)

    st.header("4. การ Train โมเดล")
    st.write("""
    - Loss Function: Mean Squared Error (MSE)
    - Optimizer: Adam
    - Epochs: 5

    โมเดลเรียนรู้จากความผิดพลาด
    และปรับตัวให้แม่นยำขึ้นในแต่ละรอบ
    """)

    st.header("5. ข้อดีและข้อจำกัด")
    st.write("""
    ข้อดี:
    - เรียนรู้ pattern ซับซ้อน
    - แม่นยำสูง
    - ค้นหา latent feature ได้

    ข้อจำกัด:
    - ใช้เวลา train มาก
    - ต้องปรับ parameter
    - interpret ยาก (black box)
    """)
    st.header("6. 🧠 เปรียบเทียบ ML vs Neural Network")

    compare_df = pd.DataFrame({
        "หัวข้อ": [
            "ความซับซ้อน (Complexity)",
            "ความเร็ว (Speed)",
            "ความแม่นยำ (Accuracy)",
            "การเรียนรู้ Pattern",
            "การตีความ (Interpretability)"
        ],
        "Machine Learning": [
            "ต่ำ-กลาง",
            "เร็ว",
            "ปานกลาง",
            "จำกัด",
            "เข้าใจง่าย"
        ],
        "Neural Network": [
            "สูง",
            "ช้า",
            "สูง",
            "ซับซ้อน",
            "ยาก (Black Box)"
        ]
    })

    st.table(compare_df)

    st.subheader("📊 Visualization")

    score_compare = pd.DataFrame({
        "Model": ["ML", "NN"],
        "Accuracy": [70, 85],
        "Complexity": [40, 90],
        "Speed": [85, 50]
    })

    st.bar_chart(score_compare.set_index("Model"))


# TEST ML
elif page == "Test ML":
    st.title("Test ML")

    user = st.number_input("User ID", min_value=1)
    movie_choice = st.selectbox("เลือกหนัง", movies["title"])

    if st.button("Predict ML"):
        row = movies[movies["title"] == movie_choice].iloc[0]

        movie_id = row["movie"]
        genres = row[[f"genre_{i}" for i in range(19)]].values.astype("float32")

        pred = ml_model.predict([[user, movie_id] + list(genres)])

        scores = get_user_preference(user)
        movie_genres = [genre_names[i] for i, g in enumerate(genres) if g == 1 and i != 0]

        st.write("🎬 Movie:", movie_choice)
        st.write("🎭 Genre:", ", ".join(movie_genres))
        st.success(f"⭐ ML Rating: {pred[0]:.2f}")

        st.subheader("🔥 User ชอบ:")
        for i, s in scores[:3]:
            if i != 0:
                st.write(f"{genre_names[i]} ⭐ {s:.2f}")

# TEST NN
elif page == "Test NN":
    st.title("Test NN")

    user = st.number_input("User ID", min_value=1)
    movie_choice = st.selectbox("เลือกหนัง", movies["title"])

    if st.button("Predict NN"):
        row = movies[movies["title"] == movie_choice].iloc[0]

        movie_id = row["movie"]
        genres = row[[f"genre_{i}" for i in range(19)]].values.astype("float32")

        pred = nn_model.predict([
            np.array([user-1], dtype="int32"),
            np.array([movie_id-1], dtype="int32"),
            np.array([genres], dtype="float32")
        ])

        scores = get_user_preference(user)

        st.success(f"⭐ NN Rating: {pred[0][0]:.2f}")

        st.subheader("🔥 User ชอบ:")
        for i, s in scores[:3]:
            if i != 0:
                st.write(f"{genre_names[i]} ⭐ {s:.2f}")