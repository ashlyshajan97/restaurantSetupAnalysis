import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


import seaborn as sns
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# Import label encoder
from sklearn import preprocessing

from django.shortcuts import render
# Importing the datasets

def sample(request):
    location=request.session['location']
    datasets = pd.read_csv('search/ranking.csv')
    global loc 
    #loc = input("Enter location: ")
    ser_his=[]
    location = location.capitalize()
    ser_his.append(location)
    unique = set(ser_his)
    df = pd.read_csv('search/ekm_town_location_data.csv')
    locs = list(df['Town Name'])
    # label_encoder object knows how to understand word labels.
    label_encoder = preprocessing.LabelEncoder()
    # Encode labels 
    datasets['Town Name']= label_encoder.fit_transform(datasets['Town Name'])
    datasets['Town Name'].unique()
    ## Computing Mean Rating
    restaurants = list(datasets['Town Name'].unique())
    datasets['Mean Rating'] = 0
    for i in range(len(restaurants)):
        datasets['Mean Rating'][datasets['Town Name'] == restaurants[i]] = datasets['Overall Rating'][datasets['Town Name'] == restaurants[i]].mean()   
    #Scaling the mean rating values
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range = (1,5))
    datasets[['Mean Rating']] = scaler.fit_transform(datasets[['Mean Rating']]).round(2)
    indices = pd.Series(datasets.index)
    # Creating tf-idf matrix
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(datasets['review'])
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    def recommend(Name, cosine_similarities = cosine_similarities):

        recommend_restaurant = []
    
        if Name=='Aluva'or 'aluva'or 'ALUVA':
            Name=0
        elif Name=='Angamaly'or 'angamaly'or 'ANGAMALY':
            Name=1
        elif Name=='Ernakulam'or 'ernakulam'or 'ERNAKULAM':
            Name=2
        elif Name=='Kakkanad'or 'kakkanad'or 'KAKKANAD':
            Name=3
        elif Name=='Kalamassery'or 'kalamassery'or 'KALAMASSERY':
            Name=4
        elif Name=='Kochi'or 'kochi'or 'KOCHI':
            Name=5
        elif Name=='Kumbalam'or 'kumbalam'or 'KUMBALAM':
            Name=6
        elif Name=='Kumbalangy'or 'kumbalangy'or 'KUMBALANGY':
            Name=7
        elif Name=='Maradu'or 'maradu'or 'MARADU':
            Name=8
        elif Name=='Mulanthuruthy'or 'mulanthuruthy'or 'MULANTHURUTHY':
            Name=9
        elif Name=='Mulavukadu'or 'mulavukadu'or 'MULAVUKAD':
            Name=10
        elif Name=='Muvattupuzha'or 'muvattupuzha'or 'MUVATTUPUZHA':
            Name=11
        elif Name=='Nedubassery'or 'nedubassery'or 'NEDUBASSERY':
            Name=12
        elif Name=='Paravur'or 'paravur'or 'PARAVUR':
            Name=13
        elif Name=='Perumbavoor'or 'perumbavoor'or 'PERUMBAVOOR':
            Name=14
        elif Name=='Thrippunithura'or 'thrippunithura'or 'THRIPPUNITHURA':
            Name=15
        elif Name=='Vazhakkala'or 'vazhakkala'or 'VAZHAKKALA':
            Name=16
    
        # Find the index of the location
        idx = indices[indices==Name].index[0]

        # Find the restaurants with a similar cosine-sim value and order them from bigges number
        score_series = pd.Series(cosine_similarities[idx])
        idx = indices[indices==Name].index[0]

        # Find the restaurants with a similar cosine-sim value and order them from bigges number
        score_series = pd.Series(cosine_similarities[idx])

        # Extract restaurant indexes with a similar cosine-sim value
        top_indexes = list(score_series.iloc[0:].index)

        # Names of the  restaurants
        for each in top_indexes:
            recommend_restaurant.append(list(datasets.index)[each])

        # Creating the new data set to show similar restaurants
        df_new = pd.DataFrame(columns=['Location','Cuisines', 'Mean Rating','Best Selling','Hot Trending'])
    
        # Create the  similar restaurants with some of their columns
        for each in recommend_restaurant:
            df_new = df_new.append(pd.DataFrame(datasets[['Location','Cuisines','Mean Rating','Best Selling','Hot Trending']][datasets.index == each].sample()))
        k=df_new[df_new["Location"]==loc]
        print('TOP %s Cusines WITH SIMILAR REVIEWS: ' % (str(len(k))))
        recommend(loc)
        return render(request,'search.html',context=data)
