from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup as bsp
import streamlit as st
import pandas as pd
import random as rd
import requests

class film_data():

    def __init__(self, url):
        self.url = url
        self.titles = []
        self.year = []
        self.rates = []
        self.descriptions = []
        self.genres = []
        self.scrapped = 0
        self.data = pd.DataFrame()
        self.similarity = None
        self.handler = None
 
    def film_scrap(self):
#         while self.url is not None:
#         try:
        response = requests.get(self.url)

        html = response.content

        soup = bsp(html, 'lxml')

        with open('IMDB_HTML_LXML', 'wb') as file:
            file.write(soup.prettify('utf-8'))

        item_head = soup.find_all('div', class_='lister-item-content')
        item_head[1].find('p', class_='').text.strip('\n      ')

        self.titles.extend([t.find('a').text for t in item_head])
        self.year.extend([y.find('span', class_='lister-item-year text-muted unbold').text.strip(')').strip('(') for y in item_head])
        self.rates.extend([r.find('span', class_='ipl-rating-star__rating').text for r in item_head])
        self.descriptions.extend([d.find('p', class_='').text.strip('\n      ') for d in item_head])
        self.genres.extend([g.find('span', class_='genre').text.strip()for g in item_head])

        item_footer = soup.find_all('div', class_='footer filmosearch')
        for x in item_footer:
            self.url = x.find_all('a')[1]['href']
        if self.url.startswith("/list"):
            self.url = "https://www.imdb.com" + self.url
        else:
            self.url = None
#         except:
#             print("There is a problem scrapping this URL :" + self.url)
        print("Data scrapped successfully !")
        print(self.titles)

      
    def sentence(self):
        # Encoding the descriptions and do the cosine similarity

        bert = SentenceTransformer('bert-base-nli-mean-tokens')
        sentence_embeddings = bert.encode(self.data['descriptions'].tolist())
        similarity = cosine_similarity(sentence_embeddings)
        self.similarity = similarity
        
    def film_table(self):
        if not 'titles' in self.data.columns:
            self.film_scrap()
            imdb_films = pd.DataFrame()
            imdb_films['titles'] = self.titles
            imdb_films['year'] = self.year
            imdb_films['rates'] = self.rates
            imdb_films['descriptions'] = self.descriptions
            imdb_films['genres'] = self.genres
            self.handler = movie_rec(imdb_films)
            self.data = self.handler.get_movies()
            self.sentence()
        return self.data

class movie_rec():

    def __init__(self, data):
        self.movies = data.dropna()
        self.movies['index'] = [i for i in range(0, len(self.movies))]

    def get_movies(self):
        return self.movies

    def get_title(self, index):
        return self.movies[self.movies.index == index]["titles"].values[0]

    def get_index(self, title):
        return self.movies[self.movies.titles == title]["index"].values[0]

    def get_rate(self, index):
        return self.movies[self.movies.index == index]["rates"].values[0]

    def get_genre(self, index):
        return self.movies[self.movies.index == index]["genres"].values[0]

    def get_description(self, index):
        return self.movies[self.movies.index == index]["descriptions"].values[0]

    def get_year(self, index):
        return self.movies[self.movies.index == index]["year"].values[0]

    def get_movies_genre(self, genre):
        return self.movies[self.movies.genres == genre]


# Template part
st.title("Movie recommendation :")

box = film_data("https://www.imdb.com/list/ls068082370/")
box.film_table()
choosed = st.selectbox("Please Choose the movie for which you want recommendations :",
                       tuple(box.data['titles']))



if st.button("Top 5 recommendation"):
    index_choosed = box.handler.get_index(choosed)
    recommendations = sorted(list(enumerate(
        box.similarity[index_choosed])), key=lambda x: x[1], reverse=True)
    recommendations = [r if r[0] != index_choosed else continue for r in recommendations]
    st.header("The top 5 recommendations for" + " " + choosed + " " + "are: ")
    st.success(box.handler.get_title(recommendations[0][0]))
    st.success(box.handler.get_title(recommendations[1][0]))
    st.success(box.handler.get_title(recommendations[2][0]))
    st.success(box.handler.get_title(recommendations[3][0]))
    st.success(box.handler.get_title(recommendations[4][0]))
    st.success(box.handler.get_title(recommendations[5][0]))
