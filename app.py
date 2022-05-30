import pandas as pd
import numpy as np
import streamlit as st
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix


def main():
    st.title('Uber pickups in NYC')
    df = pd.read_csv('https://pastebin.com/raw/NaZLBe1N')
    df.drop(df.columns[0], axis=1, inplace=True)
    X = np.array(df.drop(['Classes  '], axis=1))
    y = np.array(df['Classes  '])
    y = y.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    page = st.sidebar.selectbox("Выберите страницу", ('Основная', 'Все, связанное с моделью'))
    if page == 'Основная':
        st.title('Исследуем датасет с пожарами в Алжире')
        st.write("Просто датафрейм")
        st.dataframe(df)
    elif page == 'Все, связанное с моделью':
        st.title('Исследуем датасет с пожарами в Алжире')
        st.title('Наконец заработало!!')
        st.markdown('### Анализ связей столбцов')
        st.text('Корреляция:')
        st.text('Влияние различных классов')
        st.title('Моделируем')
        st.write("Точность, просто датафрейм, предсказание")
        st.write("Точность")
        st.write(str(clf.score(X_test, y_test)))
        st.write("Просто датафрейм")
        st.dataframe(df)
        st.write("Предсказание")
        df2 = pd.DataFrame({'Реальность': y_test, 'Предсказание': clf.predict(X_test)})
        st.dataframe(df2)


if __name__ == '__main__':
    main()