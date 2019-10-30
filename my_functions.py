import random
import pandas as pd
import re

import pandas as pd
import random
from sklearn.model_selection import train_test_split
import numpy as np
import re
from sklearn.metrics import accuracy_score
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from sklearn.utils import resample

nltk.download('wordnet')
nltk.download('punkt')

def strip(l):
  z= []
  for i in l:
    z.append(i.strip())
  return z

def my_func(x, languages, lang):
  if lang in strip(x.split(',')):
      return 1
  else:
    return 0

def create_random_input(names, langs, genres):
  #languages
  languages = []
  n_langs = random.randint(0, len(langs))
  for i in range(n_langs):
    languages.append(random.choice(langs))
  
  #names
  name = random.choice(names)
  
  #in-app-purchase
  app_purchase = random.randint(0, 1)
  
  #price
  price = random.randint(0, 1)
  
  #original release date
  original_date = random.randint(2008, 2019)
  
  #current release
  current_release = random.randint(2008, 2019)
  
  #age rating
  age_rating = random.choice(['4+', '9+', '12+', '17+'])
  
  #size
  size = random.randint(51328, 4005591040)
  
  #genres
  g = []
  n_genres = random.randint(0, len(genres))
  for i in range(n_genres):
    g.append(random.choice(genres))
    
  return pd.DataFrame({'Languages':[','.join(languages)], 'In-app Purchases':[app_purchase], 'Price':[price], 'Name':[name], 'Age Rating':[age_rating], 'Size':[size], 
              'Genres':[','.join(g)],	'Original Release Date':['11/07/'+str(original_date)],	'Current Version Release Date':['22/07/'+str(current_release)]})

def prcess_input(df, available_languages, available_genres, ages, vectorizer, svd, yeo, scaler, yeo2, scaler2, a_map, b_map, best_features):
  # LANGUAGES
  languages = available_languages
  
  for i in languages:
    df[i] = df['Languages'].apply(my_func, args=[languages, i])

  df.drop('Languages', axis=1, inplace=True)
  
  # NAME
  df['Name'] = df['Name'].apply(lambda x: re.sub('free', '', x.lower()))
  names = [name for name in df['Name'].values]
  result = []
  lemmatizer = WordNetLemmatizer()
  sno = PorterStemmer()
  for name in names:
    new_name = name.lower()
    new_name = re.sub(r'[^а-яА-Яa-zA-Z0-9]', ' ', new_name)
    new_name = re.sub(r'\s+', ' ', new_name)
    words=''
    for word in new_name.split():
      words+=sno.stem(lemmatizer.lemmatize(word)) + ' '
    result.append(words.strip())
  df['Name'] = result
  
  names_vector = vectorizer.transform(df['Name'])
  names_vector = normalize(svd.transform(names_vector))
  
  for i in range(names_vector.shape[1]):
    df[str(i)] = names_vector[:, i]
  
  df.drop('Name', axis=1, inplace=True)
  
  # User Rating Count и Size
  df[['Size']] = yeo.transform(df[['Size']].values.reshape(df.shape[0],-1))
  df[['Size']] = scaler.transform(df[['Size']])
  
  # Genres 
  genres = available_genres
  for i in genres:
    df[i] = df['Genres'].apply(my_func, args=[genres, i])

  df.drop('Genres', axis=1, inplace=True)
  
  # Age Rating
  for i in ages:
    df[i] = df['Age Rating'].apply(my_func, args=[ages, i])
    
  df.drop('Age Rating', axis=1, inplace=True)
  
  # Original Release Date и Current Version Release Date
  
  df['Original Release Date'] = df['Original Release Date'].apply(lambda x: x.split('/')[-1])
  df['Current Version Release Date'] = df['Current Version Release Date'].apply(lambda x: x.split('/')[-1])
  
  
  df['Original Release Date'] = df['Original Release Date'].map(a_map)
  df['Current Version Release Date'] = df['Current Version Release Date'].map(b_map)
  
  df[['Original Release Date', 'Current Version Release Date']] = yeo2.transform(df[['Original Release Date', 'Current Version Release Date']].values.reshape(df.shape[0],-1))
  df[['Original Release Date', 'Current Version Release Date']] = scaler2.transform(df[['Original Release Date', 'Current Version Release Date']])
  
  return df[best_features]
