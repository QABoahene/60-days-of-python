import pandas as pd 
import numpy as np 
import glob 
import os 
import string
import re 
from matplotlib import pyplot as plt
%matplotlib inline
import nltk
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.linear_model import RidgeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.pipeline import Pipeline

ps = PorterStemmer()
stop_words = nltk.corpus.stopwords.words('english')

bbc_folder = "./bbc/"
category_folders = ['business', 'entertainment', 'politics', 'tech', 'sport']
os.listdir(bbc_folder)

text_files = os.listdir(bbc_folder + 'business')
category_list = [f for f in os.listdir(bbc_folder) if not f.startswith('.')]

news_articles = []
news_category = []

for folder in category_folders:
    category_path = bbc_folder + folder + '/'
    category_text_files = os.listdir(category_path)
    for text_file in category_text_files:
        text_file_path = category_path + text_file
        with open(text_file_path, errors = 'replace') as f:
            news_data = f.readlines()
        news_data = ' '.join(news_data)
        news_articles.append(news_data)
        news_category.append(folder)

bbc_dict = {'article': news_articles, 'category': news_category}
data = pd.DataFrame(bbc_dict)
#data.to_csv(bbc_folder + '/bbc_news.csv')

data['category_id'] = data['category'].factorize()[0]
column_list = ['articles', 'category', 'category_id']
data.columns = column_list
#data.head()

#data.groupby('category').articles.count().plot.bar(ylim = 0)

#data.sample(10, random_state = 0)

contractions_dict = { "ain't": "are not","'s":" is", "aren't": "are not",
                     "can't": "cannot","can't've": "cannot have",
                     "'cause": "because","could've": "could have","couldn't": "could not",
                     "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                     "don't": "do not", "gonna": "going to", "Gonna": "going to", "hadn't": "had not","hadn't've": "had not have",
                     "hasn't": "has not","haven't": "have not","he'd": "he would", "he's": "he is",
                     "he'd've": "he would have","He'll": "he will", "he'll've": "he will have",
                     "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                     "I'd": "i would", "I'd've": "i would have","I'll": "i will",
                     "I'll've": "i will have","I'm": "i am","I've": "i have", "isn't": "is not",
                     "it'd": "it would","it'd've": "it would have","it'll": "it will",
                     "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                     "mayn't": "may not","might've": "might have","mightn't": "might not", 
                     "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                     "mustn't've": "must not have", "needn't": "need not",
                     "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                     "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
                     "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
                     "she'll": "she will", "She'll": "she will", "she'll've": "she will have","should've": "should have",
                     "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
                     "that'd": "that would","that'd've": "that would have", "there'd": "there would",
                     "there'd've": "there would have", "they'd": "they would",
                     "they'd've": "they would have","They'll": "they will", "they'll": "they will",
                     "they'll've": "they will have", "They're": "they are", "they're": "they are", "They've": "they have", "they've": "they have",
                     "to've": "to have","tryna": "trying to", "Tryna": "trying to", "wasn't": "was not","we'd": "we would",
                     "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                     "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                     "what'll've": "what will have","what're": "what are", "what've": "what have",
                     "when've": "when have","where'd": "where did", "where've": "where have",
                     "who'll": "who will","who'll've": "who will have","who've": "who have",
                     "why've": "why have","will've": "will have","won't": "will not",
                     "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                     "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                     "y'all'd've": "you all would have","y'all're": "you all are",
                     "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                     "you'll": "you will","you'll've": "you will have", "you're": "you are",
                     "you've": "you have"}

# Regular expression for finding contractions
contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

# Function for expanding contractions
def expand_contractions(text, contractions_dict = contractions_dict):
  def replace(match):
    return contractions_dict[match.group(0)]
  return contractions_re.sub(replace, text)

#Function to clean news articles
def initial_clean(article):
    article = article.lower()
    article = re.sub('[%s]' % re.escape(string.punctuation), '', article)
    article = re.sub('\[.*?\]', '', article)
    article = re.sub('\n', ' ', article)
    article = re.sub('[''""’—...]', '', article)
    return article

# Applying the contraction expansion function
data['processed_articles'] = data['articles'].apply(lambda x: expand_contractions(x))

# Applying final cleaning to the news articles
data['processed_articles'] = data['processed_articles'].apply(initial_clean)

#Final Cleaning
data['processed_articles'] = data['processed_articles'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
data['processed_articles'] = data['processed_articles'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))
frequency_one = pd.Series(' '.join(data['processed_articles']).split()).value_counts()
frequency_two = frequency_one[frequency_one <= 3]
frequency_three = list(frequency_two.index.values)
data['processed_articles'] = data['processed_articles'].apply(lambda x: ' '.join([word for word in x.split() if word not in (frequency_three)]))

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2))

features = tfidf.fit_transform(data['processed_articles']).toarray()
labels = data['category_id']
#features.shape

category_id_df = data[['category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'category']].values)

N = 5
for category, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    #print("# '{}':".format(category))
    #print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
    #print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))

# Sampling a subset of our dataset because t-SNE is computationally expensive
SAMPLE_SIZE = int(len(features) * 0.3)
np.random.seed(0)
indices = np.random.choice(range(len(features)), size = SAMPLE_SIZE, replace = False)
projected_features = TSNE(n_components=2, random_state = 0).fit_transform(features[indices])
colors = ['pink', 'green', 'midnightblue', 'orange', 'darkgrey']

# for category, category_id in sorted(category_to_id.items()):
#     points = projected_features[(labels[indices] == category_id).values]
#     plt.scatter(points[:, 0], points[:, 1], s = 30, c = colors[category_id], label = category)
# plt.title("tf-idf feature vector for each article, projected on 2 dimensions.",
#           fontdict=dict(fontsize=15))
# plt.legend()

models = [
    RandomForestClassifier(n_estimators = 200, max_depth=3, random_state=0),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

# sns.boxplot(x='model_name', y='accuracy', data=cv_df)
# sns.stripplot(x='model_name', y='accuracy', data=cv_df,size=8, jitter=True, edgecolor="gray", linewidth=2)

X_train, X_test, y_train, y_test = train_test_split(data.articles, data.category, test_size=0.2, random_state=42)

folds = KFold(n_splits = 5, shuffle = True, random_state = 1)

vect = TfidfVectorizer(stop_words='english', ngram_range = (1,2), max_features = 20000)
Ridge = RidgeClassifier()
XGB = XGBClassifier(objective = 'multi:softprob')

def fit_cv(clf , name):
    pipe = Pipeline([
    ('vectorize', vect),
    (name, clf)])
    result = cross_validate(pipe, X_train, y_train, cv = folds, return_train_score=True,scoring = ('accuracy', 
                                                                                       'f1_weighted', 
                                                                                       'precision_weighted', 
                                                                                       'recall_weighted'))
    return result

ridge = fit_cv(Ridge, 'Ridge')
xgb = fit_cv(XGB, 'XGB')

ridge
xgb