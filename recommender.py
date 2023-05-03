'''Implements functions for making predictions.'''
import random
from sklearn.decomposition import NMF
import pickle
import numpy as np
import pandas as pd

#ratings = pd.read_csv()

R = pd.read_csv("week_10_project.out.csv")
MOVIES = [
    'Avatar',
    'The Great Beauty',
    'Star Wars',
    'Interstelar',
]

model = NMF(n_components=20, max_iter=1000)

def random_recommender():
    random.shuffle(MOVIES)
    top_two = MOVIES[0:2]
    return top_two

def cos_sim():
    pass

def NMF_model(R, model):
    model.fit(R)
    with open("NMF_fitted.pkl", "wb") as model_out:
        pickle.dump(model, model_out)

def NMF(model, query, k):
    with open("NMF_fitted.pkl", "rb") as model_in:
        model_fitted = pickle.load(model_in)
    df = pd.DataFrame(query, columns=list(R.columns), index=["new_user"])
    Q = model_fitted.components_
    user_transform = model.transform(df)
    R_hat_new_user_matrix = np.dot(user_transform, Q)
    R_hat_new_user = pd.DataFrame(data=R_hat_new_user_matrix,
                            columns=R.columns,
                            index = ['new_user'])
    sorted_list = R_hat_new_user.transpose().sort_values(by="new_user", ascending=False).index.to_list()
    rated_movies = list(query.keys())
    recommended_movies = [movie for movie in sorted_list if movie not in rated_movies]
    recommended_movies = recommended_movies[:k] #shorten to Top 10
    return recommended_movies
    


    
    
    

    


if __name__=="__main__":
    #
    #model = NMF_model(R, model)
    
    print(type(model_fitted))
