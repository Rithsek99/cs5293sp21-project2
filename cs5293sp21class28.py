import pandas as pd
df = pd.read_csv("tmdb_5000_movies.csv")
df.info()
df.head()
df = df[['title', 'tagline', 'overview', 'genres', 'popularity']]
df.info()
df.tagline.fillna('', inplace=True)
df['description'] = df['tagline'].map(str) + ' ' + df['overview']
df['description'][0]
df['tagline'][0]
df['title'][0]
df.info()
df.dropna(inplace=True)
df.info()
df['popularity'][0]
import nltk
import re
import numpy as np
stop_words = nltk.corpus.stopwords.words('english')
stop_words = nltk.corpus.stopwords.words('english')
stop_words
stop_words[:20]
len(stop_words)
def normalize_document(txt):
    txt = re.sub(r'[^a-zA-Z0-9\s]', '', txt, re.I|re.A)
    txt = txt.lower()
    txt = txt.strip()
    tokens = nltk.word_tokenize(txt)
    clean_tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(clean_tokens)
normalize_document(df['description'][0])
normalize_corpus = np.vectorize(normalize_document)
# New function that can take a list of items
norm_corpus = normalize_corpus(list(df['description']))
len(norm_corpus)
norm_corpus[0]
norm_corpus[-1]
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
tfidf_matrix = tf.fit_transform(norm_corpus)
tfidf_matrix.shape
from sklearn.metrics.pairwise import cosine_similarity
import sklearn
#help(sklearn.metric.pairwise)
#doc(sklearn.metric.pairwise)
#help(sklearn.metric.pairwise)
#help(sklearn.metrics.pairwise)
doc_sim = cosine_similarity(tfidf_matrix)
doc_sim_df = pd.DataFrame(doc_sim)
doc_sim_df.info()
doc_sim_df.head()
movies_list = df['title'].values
movies_list[0:3]
movies_list, movies_list.shape
np.where(movies_list == 'Minions')[0][0]
movie_idx = np.where(movies_list == 'Minions')[0][0]
movie_similarity = doc_sim_df.iloc[movie_idx].values
movie_similarity
similar_movie_idxs = np.argsort(-movie_similarity)[1:6]
similar_movie_idxs
movies_list[similar_movie_idxs]
# K Means clustering
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(1,2), min_df=10, max_df=0.8, stop_words=stop_words)
cv_matrix = cv.fit_transform(norm_corpus)
cv_matrix.shape
from sklearn.cluster import KMeans
#help(sklearn.cluster)
#help(KMeans)
km = KMeans(n_clusters=6, max_iter=10000, n_init=50, random_state=42)
km.fit(cv_matrix)
#%time
#%timeit
km.labels_
from collections import Counter
Counter(km.labels_)
df['kmeans_cluster'] = km.labels_
movie_clusters = (df[['title', 'kmeans_cluster', 'popularity']]
                  .sort_values(by=['kmeans_cluster', 'popularity'], 
                               ascending=False)
                  .groupby('kmeans_cluster').head(3))
movie_clusters
cv.get_feature_names()
feature_name = cv.get_feature_names()
topn_features = 9
ordered_centroids = km.cluster_centers_.argsort()[:, ::-1]
ordered_centroids

feature_names = cv.get_feature_names()
for cluster_num in range(6):
    key_features = [feature_names[index] 
                        for index in ordered_centroids[cluster_num, :topn_features]]
    movies = movie_clusters[movie_clusters['kmeans_cluster'] == cluster_num]['title'].values.tolist()
    print('CLUSTER #'+str(cluster_num+1))
    print('Key Features:', key_features)
    print('Popular Movies:', movies)
    print('-'*80)
#%history -f recsys-clustering.py
