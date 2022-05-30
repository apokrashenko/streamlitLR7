import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import altair as alt
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


@st.cache
def load_data():
    df = pd.read_csv('train_data_preprocessed.tsv', sep='\t')
    return df


def visualize_data(df):
    c = alt.Chart(df).mark_circle().encode(x='variance', y='skewness',
                                       color='class')
    st.write(c)

    
def main():
    df = load_data()
    page = st.sidebar.selectbox("Choose a page", ["Homepage", "Exploration","Model"])

    with open('sgd_ppl_clf.pkl', 'rb') as pkl_file:
        sgd_ppl_clf = pickle.load(pkl_file)

    if page == "Homepage":
        st.header("Данные для задачи классификации заголовков.")
    elif page == "Exploration":
        st.title("Data Exploration")
        visualize_data(df)
    else:
        st.title("Model ")


if __name__ == '__main__':
    main()
