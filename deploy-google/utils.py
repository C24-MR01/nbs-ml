import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import os
import json

import tensorflow as tf
import pickle
from tensorflow import keras
from keras import layers
import urllib.request

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pickle
import zipfile

def get_model():
    """
    https://drive.usercontent.google.com/download?id=1tKJNWOa5Nqskdm-YAo-X1avsoDOx8EW7&export=download&authuser=0&confirm=t&uuid=677a441e-52cf-4042-89c0-adc4342a3a57&at=APZUnTXApeyO239jIVEinBEsWZqZ%3A1718211135110
    """
    model_url = 'https://drive.usercontent.google.com/download?id=1tKJNWOa5Nqskdm-YAo-X1avsoDOx8EW7&export=download&authuser=0&confirm=t&uuid=677a441e-52cf-4042-89c0-adc4342a3a57&at=APZUnTXApeyO239jIVEinBEsWZqZ%3A1718211135110'
    current_dir = os.getcwd()
    model_path = os.path.join(current_dir)
    print("Downloading model to {}".format(model_path))
    os.makedirs(model_path, exist_ok=True)
    # Define the full path for the downloaded file
    target_file_path = os.path.join(model_path, 'model.zip')
    if not os.path.exists(target_file_path):
        urllib.request.urlretrieve(model_url, target_file_path)
        print(f'File downloaded and saved to {target_file_path}')
        # Unzip the model.zip
        with zipfile.ZipFile(target_file_path, 'r') as zip_ref:
            zip_ref.extractall(model_path)
        print(f'File unzipped to {model_path}')
    else:
        print(f'File already exists at {target_file_path}, skipping download.')

get_model()
def get_dataset():
    # Get the current directory of the scrip

    data_url_1 = 'https://drive.usercontent.google.com/download?id=1rACBSh5FWqP5S_xMn3Ty382BSjGZC6U0&export=download&authuser=0&confirm=t&uuid=ee5921d6-dc36-4593-8662-f5e7490f590f&at=APZUnTXz447GE_ox2yw3NvJM1NLN%3A1717769617938'
    # Get the current working directory
    # Get the current working directory
    current_dir = os.getcwd()

    # Get the parent directory of the current working directory
    parent_dir = os.path.dirname(current_dir)

    data_path = os.path.join(parent_dir, 'data')
    print("Downloading movie dataset to {}".format(data_path))
    # Create the target directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)

    # Define the full path for the downloaded file
    target_file_path = os.path.join(data_path, '10000-movie.csv')

    if not os.path.exists(target_file_path):
        # Download the file to the target directory
        urllib.request.urlretrieve(data_url_1, target_file_path)
        print(f'File downloaded and saved to {target_file_path}')
    else:
        print(f'File already exists at {target_file_path}, skipping download.')

get_dataset()
movies = pd.read_csv("https://drive.usercontent.google.com/download?id=1rACBSh5FWqP5S_xMn3Ty382BSjGZC6U0&export=download&authuser=0&confirm=t&uuid=ee5921d6-dc36-4593-8662-f5e7490f590f&at=APZUnTXz447GE_ox2yw3NvJM1NLN%3A1717769617938")
movies = movies.iloc[:, 0:7]
movies = movies.dropna(subset=['cast', 'crew'])

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
movies['overview'] = movies['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies['overview'])
tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(movies.index, index=movies['id'])
# ------------------------------------------------------------ CONTENT BASED BY GENRE
from ast import literal_eval
features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    movies[feature] = movies[feature].apply(literal_eval)
def get_director(crew):
    # Mengurai string JSON menjadi list of dictionaries
    try:
        crew_list = json.loads(crew)
    except (TypeError, ValueError):
        return None

    for i in crew_list:
        if i['job'] == 'Director':
            return i['name']
    return None

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []

movies['director'] = movies['crew'].apply(get_director)
features = ['cast', 'keywords', 'genres']
for feature in features:
    movies[feature] = movies[feature].apply(get_list)

#clean data
movies = movies.drop(columns=['crew'])
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

#apply clean data
features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    movies[feature] = movies[feature].apply(clean_data)

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

movies['soup'] = movies.apply(create_soup, axis=1)

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(movies['soup'])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)


def get_recommendations(movie_id, cosine_sim=cosine_sim):
    try:
        id_to_find = movie_id
        title = movies.loc[movies['id'] == id_to_find, 'title'].values[0]

        # Get indices corresponding to the title
        idx = indices[movie_id]

        # Convert idx to a list if it's not already
        if not isinstance(idx, list):
            idx = [idx]

        sim_scores = []
        for index in idx:
            # Retrieve cosine similarities for the current index
            cosine_sims = cosine_sim[index]

            # Extend sim_scores with the enumerated cosine similarities
        sim_scores.extend(list(enumerate(cosine_sims)))

        # Sort the sim_scores list based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Retrieve top 5 similar movies
        sim_scores = sim_scores[1:6]
        # Extract movie indices from sim_scores
        movie_indices = [i[0] for i in sim_scores]

        return movies['id'].iloc[movie_indices]
    except IndexError as e:
    # Handle the error
        return("Movie ID is not found in dataset")

# ============Collaborative===============
# ================CLASS===================
class MovieLensTrainDataset(Dataset):
    """MovieLens PyTorch Dataset for Training"""

    def __init__(self, ratings, all_movieIds):
        self.users, self.items, self.labels = self.get_dataset(ratings, all_movieIds)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_dataset(self, ratings, all_movieIds):
        users, items, labels = [], [], []
        user_item_set = set(zip(ratings['user'], ratings['movieId']))
        num_negatives = 4
        for u, i in user_item_set:
            users.append(u)
            items.append(i)
            labels.append(1)
            for _ in range(num_negatives):
                negative_item = np.random.choice(all_movieIds)
                while (u, negative_item) in user_item_set:
                    negative_item = np.random.choice(all_movieIds)
                users.append(u)
                items.append(negative_item)
                labels.append(0)
        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)


class NCF(pl.LightningModule):
    """Neural Collaborative Filtering (NCF)"""

    def __init__(self, num_users, num_items, ratings, all_movieIds):
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=8)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=8)
        self.fc1 = nn.Linear(in_features=16, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)
        self.ratings = ratings
        self.all_movieIds = all_movieIds

    def forward(self, user_input, item_input):
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        vector = torch.cat([user_embedded, item_embedded], dim=-1)
        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))
        pred = nn.Sigmoid()(self.output(vector))
        return pred

    def training_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        predicted_labels = self(user_input, item_input)
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(MovieLensTrainDataset(self.ratings, self.all_movieIds),
                          batch_size=512, num_workers=0)


def load_model():
    folder = './model/'
    ratings = pickle.load(open(folder + 'ratings_fix_df.pkl', 'rb'))
    user_id_encoded = pickle.load(open(folder + 'user_id_encoded_fix.pkl', 'rb'))
    movie_id_encoded = pickle.load(open(folder + 'movie_id_encoded_fix.pkl', 'rb'))

    num_users = ratings['user'].max() + 1
    num_items = ratings['movieId'].max() + 1

    all_movieIds = ratings['movieId'].unique()
    ratings['rank_latest'] = ratings.groupby(['user'])['timestamp'].rank(method='first', ascending=False)
    train_ratings = ratings[ratings['rank_latest'] != 1]
    train_ratings = train_ratings[['user', 'movieId', 'rating']]
    train_ratings.loc[:, 'rating'] = 1

    model = NCF(num_users, num_items, train_ratings, all_movieIds)
    model.load_state_dict(torch.load('model/model_fix_state_dict.pth'))
    model.eval()

    return model, ratings, user_id_encoded, movie_id_encoded, all_movieIds


model, ratings, user_id_encoded, movie_id_encoded, all_movieIds = load_model()


def get_collab_recommendations(user_id):
    if user_id not in user_id_encoded.keys():
        # Get the top 10 most common movieIds
        result_list = ratings['movieId'].value_counts().head(100).index.tolist()
        result_list = np.random.choice(result_list, 10, replace=False)
        return [int(movie_id) for movie_id in result_list]

    user_encoder = user_id_encoded.get(user_id)
    user_interacted_items = ratings.groupby('user')['movieId'].apply(list).to_dict()
    interacted_items = user_interacted_items[user_encoder]
    not_interacted_items = set(all_movieIds) - set(interacted_items)
    selected_not_interacted = list(np.random.choice(list(not_interacted_items), 100))
    test_items = selected_not_interacted

    predicted_labels = np.squeeze(model(torch.tensor([user_encoder] * 100),
                                        torch.tensor(test_items)).detach().numpy())

    top10_items = [test_items[i] for i in np.argsort(predicted_labels)[::-1][0:10].tolist()]

    return [int(movie_id) for movie_id in top10_items]
