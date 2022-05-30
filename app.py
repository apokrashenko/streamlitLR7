import streamlit as st
import altair as alt
import pandas as pd
import pickle

def main():
    df = load_data()
    page = st.sidebar.selectbox("Choose a page", ["Homepage", "Exploration","Model"])

    with open('sgd_ppl_clf.pkl', 'rb') as pkl_file:
        sgd_ppl_clf = pickle.load(pkl_file)

    if page == "Homepage":
        st.header("Данные для задачи классификации заголовков.")
        st.write("Данные для датасета были извлечены из изображений, снятых с подлинных и поддельных банкнотоподобных образцов. Для оцифровки использовалась промышленная камера, обычно используемая для проверки печати.Инструмент Wavelet Transform использовался для извлечения признаков из изображений.")
        st.write("1) variance of Wavelet Transformed image (дисперсия вейвлет-преобразованного изображения), тип вещественный.")
        st.write("2) skewness of Wavelet Transformed image (асимметрия вейвлет-преобразованного изображения), тип вещественный.")
        st.write("3) curtosis of Wavelet Transformed image (эксцесс преобразованного изображения), тип вещественный.")
        st.write("4) entropy of image (энтропия изображения), тип вещественный.")
        st.write(df)
    elif page == "Exploration":
        st.title("Data Exploration")
        visualize_data(df)
    else:
        st.title("Model ")


@st.cache
def load_data():
    df = pd.read_csv('train_data_preprocessed.tsv', sep='\t')
    return df

def visualize_data(df):
    c = alt.Chart(df).mark_circle().encode(x='variance', y='skewness',
                                       color='class')
    st.write(c)

if __name__ == "main":
    main()
