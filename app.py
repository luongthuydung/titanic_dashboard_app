import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 페이지 설정
st.set_page_config(page_title="🚢 타이타닉 생존자 분석 대시보드", layout="wide")

# 데이터 로딩
@st.cache_data
def load_data():
    return pd.read_csv("train.csv"), pd.read_csv("test.csv"), pd.read_csv("gender_submission.csv")

train_df, test_df, gender_df = load_data()

# 헤더
st.markdown("<h1 style='text-align: center; color: navy;'>🚢 타이타닉 생존자 분석 대시보드</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>탑승자들의 생존 여부를 예측하고 시각적으로 분석합니다.</p>", unsafe_allow_html=True)

# 탭 메뉴
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏠 홈", "📑 데이터", "🔍 고객 검색", "📊 기초 분석", "📌 고급 시각화", "🔬 예측 모델", "📈 예측 결과"
])

with tab1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", use_column_width=True)
    st.markdown("#### 데이터 출처: [Kaggle Titanic Challenge](https://www.kaggle.com/c/titanic)")

with tab2:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("학습 데이터")
        st.dataframe(train_df)
    with col2:
        st.subheader("테스트 데이터")
        st.dataframe(test_df)
    with col3:
        st.subheader("예측 제출 샘플")
        st.dataframe(gender_df)

with tab3:
    st.subheader("🔎 이름으로 승객 검색")
    name_input = st.text_input("승객 이름 입력 (예: Allen, Braund 등)", "").strip().lower()
    if name_input:
        result = train_df[train_df['Name'].str.lower().str.contains(name_input)]
        if not result.empty:
            st.success(f"{len(result)}명의 승객을 찾았습니다.")
            st.dataframe(result)
        else:
            st.warning("검색 결과가 없습니다.")

with tab4:
    st.subheader("📊 기본 시각화 분석")

    st.markdown("**1️⃣ 성별 분포**")
    fig1 = px.histogram(train_df, x='Sex', color='Sex', title='성별 분포',
                      color_discrete_sequence=px.colors.qualitative.Set2,
                      animation_frame=None)
    st.plotly_chart(fig1)

    st.markdown("**2️⃣ 성별 생존률**")
    gender_survival = train_df.groupby('Sex')['Survived'].mean().reset_index()
    fig2 = px.bar(gender_survival, x='Sex', y='Survived', color='Sex',
                  title='성별 생존률', labels={'Survived': '생존률'},
                  color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig2)

    st.markdown("**3️⃣ 선실 등급과 성별 생존률**")
    pclass_sex = train_df.groupby(['Pclass', 'Sex'])['Survived'].mean().reset_index()
    fig3 = px.bar(pclass_sex, x='Pclass', y='Survived', color='Sex',
                  barmode='group', title='선실 등급과 성별 생존률',
                  color_discrete_sequence=px.colors.sequential.Bluered)
    st.plotly_chart(fig3)

    st.markdown("**4️⃣ 생존자와 비생존자의 나이 분포**")
    fig4 = px.histogram(train_df, x='Age', color='Survived', nbins=30,
                    barmode='overlay', title='나이별 생존/사망 분포',
                    color_discrete_sequence=px.colors.sequential.Viridis)
    st.plotly_chart(fig4)

with tab5:
    st.subheader("📌 고급 시각화")

    st.markdown("**🎯 생존 여부에 따른 나이 BoxPlot**")
    fig5 = px.box(train_df, x='Survived', y='Age', color='Survived',
                 points="all", title='생존 여부에 따른 나이 분포',
                 color_discrete_sequence=px.colors.qualitative.Set1)
    st.plotly_chart(fig5)

    st.markdown("**🎯 성별 & Pclass에 따른 나이 분포 (ViolinPlot)**")
    fig6 = px.violin(train_df, x='Pclass', y='Age', color='Sex', box=True,
                  points='all', title='성별과 등급에 따른 나이 분포',
                  color_discrete_sequence=px.colors.qualitative.Bold)
    st.plotly_chart(fig6)

    st.markdown("**📊 상관관계 히트맵**")
    corr = train_df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr()
    fig7 = px.imshow(corr, text_auto=True, title='상관관계 히트맵')
    st.plotly_chart(fig7)

    st.markdown("**🎯 생존자 비율 파이차트**")
    pie_values = train_df['Survived'].value_counts()
    fig8 = px.pie(values=pie_values.values, names=['사망', '생존'],
                title='생존자 비율', color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig8)

with tab6:
    st.subheader("🤖 랜덤 포레스트 예측")
    df_model = train_df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]].dropna()
    df_model["Sex"] = df_model["Sex"].map({"male": 0, "female": 1})

    X = df_model.drop("Survived", axis=1)
    y = df_model["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.markdown("**📋 분류 리포트**")
    st.text(classification_report(y_test, y_pred))

    st.markdown("**📉 혼동 행렬**")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="예측값", y="실제값"), x=['사망', '생존'], y=['사망', '생존'], title="혼동 행렬")
    st.plotly_chart(fig_cm)

with tab7:
    st.subheader("📈 테스트 예측 결과")
    test_merged = pd.merge(test_df, gender_df, on="PassengerId")
    pred_group = test_merged.groupby("Sex")["Survived"].mean().reset_index()
    fig_result = px.bar(pred_group, x='Sex', y='Survived', title='성별에 따른 예측 생존률')
    st.plotly_chart(fig_result)

st.markdown("<hr><center>🚢 타이타닉 분석 대시보드 • AI 도우미와 함께 개발되었습니다</center>", unsafe_allow_html=True)