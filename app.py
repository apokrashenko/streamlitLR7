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
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

@st.cache
def load_data():
    df = pd.read_csv('train_data_preprocessed.tsv', sep='\t')
    return df


def visualize_data(df):
    c = alt.Chart(df).mark_circle().encode(x='variance', y='skewness',
                                       color='class')
    st.write(c)
    
def str_corpus(corpus): # Получаем из списка слов текстовую строку
    str_corpus = ''
    for i in corpus:
        str_corpus += ' ' + i
    str_corpus = str_corpus.strip()
    return str_corpus

def get_corpus(data): # Получаем список всех слов в corpus
    corpus = []
    for phrase in data:
        for word in phrase.split():
            corpus.append(word)
    return corpus

def get_wordCloud(corpus): # Получаем облако слов
    wordCloud = WordCloud(background_color='white',
                              stopwords=STOPWORDS,
                              width=3000,
                              height=2500,
                              max_words=200,
                              random_state=42
                         ).generate(str_corpus(corpus))
    return wordCloud

    
def main():
    df = load_data()
    corpus_clean = get_corpus(df['title'].values)
    procWordCloud = get_wordCloud(corpus_clean)
    num_words = len(set(corpus_clean))
    page = st.sidebar.selectbox("Choose a page", ["Homepage", "Exploration","Model"])

    with open('sgd_ppl_clf.pkl', 'rb') as pkl_file:
        sgd_ppl_clf = pickle.load(pkl_file)

    if page == "Homepage":
        st.header("Данные для задачи классификации заголовков.")
        st.write("Датасет представляет из себя заголвки с правдивыми и ложными новостями.")
        st.dataframe(df)
    elif page == "Exploration":
        st.header("Визуализация и анализирование датасета")
        st.title("После обработки")
        st.write("Облако слов обработанного набора данных содержит уникальных слов:")
        print(num_words)
        #visualize_data(df)
        fig = plt.figure(figsize=(20, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(procWordCloud)
        plt.axis('off')
        plt.subplot(1, 2, 1)
    else:
        st.title("Model ")
    
if __name__ == '__main__':
    main()
