#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 00:22:16 2022

@author: leemingjun
"""

#Data Scraping
import tweepy # Manages Data encoding and decoding, HTTP Requests, Results Pagination, Authentication and Rate Limits 
import pandas as pd

#Data Cleaning
import string
import re
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import gensim
import gensim.corpora as corpora
import numpy as np

#Data Analysis
from nltk import FreqDist, bigrams
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim.models import CoherenceModel
import pyLDAvis.gensim_models
import networkx as nx
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


consumer_key = "Lua2Q0jGg6L6vEBouRlVI1jSE" # API_Key
consumer_secret = "D306gpTDlsesRcEyznmh5z9YBB9rToTyqBIZOwCW65yvy2UuQL" # API_Secret
access_key = "1519331418085937152-sGabcnJWNM20L3sZqsnXJTHfcTZjSN" # Access_Token
access_secret = "sk4zsPofphKQlwHKBWyqkFLTuXDwj4O0VtHzthts7CeqA" # Access_Secret

#Function to Authenticate Twitter API Keys
def twitter_setup():
    
    # Authentication and access using keys
    auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth,wait_on_rate_limit=True)
    
    try:
        api.verify_credentials()
        print("Authentication OK")
    except:
        print("Error during Authentication")
        
    return api

# We create an extractor object (holding the api data) by calling in our twitter_setup() function
extractor = twitter_setup()

def keyword_tweets(api, keyword):
    new_keyword = keyword + " -filter:retweets" #Filter out retweets
    
    tweets = []
    # Collect tweets using the Cursor object
    # .Cursor() returns an object that you can iterate or loop over to access the data collected.
    # Specify the language of the tweets
    # Scraping popular tweets for the last 7 days (twitter rule: can only scrape tweets within 7 days)
    for status in tweepy.Cursor(api.search_tweets, q=new_keyword, lang="en").items(1000): #To specify number of tweets
        tweets.append(status) # Scraped tweets metadata are appended
    
    return tweets

# Calling of function to authenticate API key, then to to scrape Tweets that contain uber
keyword_alltweets = keyword_tweets(extractor, 'uber')

# Show 5 recents tweet that contain the keyword uber
print("5 recent tweets:\n")
for tweet in keyword_alltweets[:5]:
    print(tweet.text)

# Tweets are stored in a pandas dataframe
data = pd.DataFrame()

data['Username'] = [tweet.user.screen_name for tweet in keyword_alltweets]
data['Date']  = [tweet.created_at for tweet in keyword_alltweets]
data['Retweets'] = [tweet.retweet_count for tweet in keyword_alltweets]
data['Likes']   = [tweet.favorite_count for tweet in keyword_alltweets]
data['Tweets']=[tweet.text for tweet in keyword_alltweets]



#################### Exportation of Data into CSV #########################



#data.to_csv('/Users/leemingjun/desktop/uber.csv', index=False)

data=pd.read_csv('/Users/leemingjun/Desktop/uber.csv')


####################### Data Cleaning ##########################


# Stemming words to its root form, Ex. driving to driv. Because stemming does not consider the context of te word.
# Lemmatization is the process of grouping different inflected forms of word as a single item.
# Lemmatisation is a form of stemming but it considers the context of the word.
# For example, driving to drive
def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer() # Assign variable to the function
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n' # tagging noun tokens with 'n' for lemmatizing process
            lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence
    #print(word, tag, pos)

def remove_noise(tweet_tokens, stop_words):
    cleaned_tokens = []
    for token in tweet_tokens:
        token = re.sub('http[s]?','',token) # Remove http or https
        token = re.sub('//t.co/[A-Za-z0-9]+','',token) # Remove //t.co/ and the words behind it
        token = re.sub('@[A-Za-z0-9_]+','',token) # Remove words start with @
        token = re.sub('[0-9]','',token) # Remove numbers
        token = re.sub(r'[^\w]', '', token) # Remove non-alphabet character like symbol and emoji
        if (len(token) >3) and (token not in string.punctuation) and (token.lower() not in stop_words):
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


# Tokenisation of Tweets
tweets_token=data['Tweets'].apply(word_tokenize).tolist()

stop_words = stopwords.words('english')
stop_words.extend(['uber', 'today', 'thank', 'thanks', 'really'])

cleaned_tokens = []
for tokens in tweets_token:
    lemma_tokens= lemmatize_sentence(tokens)
    rm_noise=remove_noise(lemma_tokens, stop_words)
    cleaned_tokens.append(rm_noise)
    


#################### Assigning id to each word ######################



# Using Gensim Dictionary to assign each word a unique id
id2word = corpora.Dictionary(cleaned_tokens)
print(id2word.token2id)

# Using filter to filter extreme words
id2word.filter_extremes(no_below=6, no_above=0.6)
print(id2word.token2id)


##################### Analysis ##########################


##################### Sentiment Analysis ##########################


text_blob=[]
for tweet in data['Tweets'].tolist():
    analysis = TextBlob (tweet)
    if analysis.sentiment.polarity == 0:
        sentiment = "Neutral"
    elif analysis.sentiment.polarity > 0:
        sentiment = "Positive"
    elif analysis.sentiment.polarity < 0:
        sentiment = "Negative"
    text_blob.append(sentiment)
    
data['Sentiment'] = text_blob

labelled_tweets= data[['Tweets','Sentiment']]
labelled_tweets.drop(labelled_tweets.loc[labelled_tweets['Sentiment']=='Neutral'].index, inplace=True)

sentiment_plot = pd.Series(labelled_tweets['Sentiment']).value_counts().plot(kind="bar")


##################### Machine Learning #########################


# Stemming words to its root form, Ex. driving to driv. Because stemming does not consider the context of te word.
# Lemmatization is the process of grouping different inflected forms of word as a single item.
# Lemmatisation is a form of stemming but it considers the context of the word.
# For example, driving to drive
def lemmatize_sentence_ML(tokens):
    lemmatizer = WordNetLemmatizer() # Assign variable to the function
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n' # tagging noun tokens with 'n' for lemmatizing process
            lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence
    #print(word, tag, pos)

def remove_noise_ML(tweet_tokens, stop_words):
    cleaned_tokens = []
    for token in tweet_tokens:
        token = re.sub('http[s]?','',token) # Remove http or https
        token = re.sub('//t.co/[A-Za-z0-9]+','',token) # Remove //t.co/ and the words behind it
        token = re.sub('@[A-Za-z0-9_]+','',token) # Remove words start with @
        token = re.sub('[0-9]','',token) # Remove numbers
        token = re.sub(r'[^\w]', '', token) # Remove non-alphabet character like symbol and emoji
        if (len(token) >3) and (token not in string.punctuation) and (token.lower() not in stop_words):
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

# Tokenisation of Tweets
ML_tokens=labelled_tweets['Tweets'].apply(word_tokenize).tolist()


cleaned_ML_tokens = []
for machine in ML_tokens:
    lemma_ML_tokens= lemmatize_sentence_ML(machine)
    rm_noise_ML=remove_noise_ML(lemma_ML_tokens, stop_words)
    cleaned_ML_tokens.append(rm_noise_ML)

machine_tweet=[]
for line in cleaned_ML_tokens:
    line=' '.join(line)
    print(type(line))
    machine_tweet.append (line)

print (type(machine_tweet))
print (machine_tweet)

tf = TfidfVectorizer(max_features=1000)
X =tf.fit_transform(machine_tweet).toarray()
columns = tf.get_feature_names_out()
df = pd.DataFrame(X, columns= columns)

y=labelled_tweets['Sentiment']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=0)

model=MultinomialNB()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

cf=classification_report(y_test,y_pred)
print(cf)


##################### WordCloud #########################


#Generate WordCloud Image
keyword_wordcloud = WordCloud(width = 800, height = 800,
                              background_color = 'white',
                              min_font_size = 10).generate(str(cleaned_tokens))

plt.imshow(keyword_wordcloud)
plt.axis("off")
plt.show()



##################### Coherence Score Model #########################



# Corpus only contains token id or word id and its corresponding frequency.
# Corpus is the word being search through different document.
corpus = [id2word.doc2bow(text) for text in cleaned_tokens]

# Calling of ldamodel from gensim library and fit corpus dictionary to its parameters
ldamodel = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           passes=50, # Number of training iteration through the entire corpus
                                           iterations=50, # Maximum iteration over each documents to reach the convergence
                                           num_topics=5, # Search for 5 topics only to be included
                                           random_state=1)

# Print the topics and its associating top keywords
ldamodel.print_topics(num_words=5) # Only print the top 5 keywords

# Calling of coherence statistical model to get the coherence score
coherence_model_lda = CoherenceModel(model=ldamodel, texts=cleaned_tokens, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

lda_display = pyLDAvis.gensim_models.prepare(ldamodel, corpus, id2word)
pyLDAvis.save_html(lda_display, 'uber.html')


##################### Topic Rename #########################


data['Tweets'][0]
tweet_topic=ldamodel[corpus][0]
tweets_topic=sorted(tweet_topic, key=lambda tweet_topic: tweet_topic[1], reverse=True)

tp_list=[]
for i in range(len(ldamodel[corpus])):
    tp=ldamodel[corpus][i]
    tp=sorted(tp, key=lambda tp: tp[1], reverse=True)
    tp_list.append(tp[0][0])

new_df=pd.DataFrame(data['Tweets'])
new_df['Topic'] = tp_list
new_df['Topic'] = new_df['Topic'].replace([0,1,2,3,4],
                                          ['Leaked files about Uber by Mark Macgann',
                                           "Leaked documents about Uber's secret deal with the French government",
                                           'Whistleblower about Uber',
                                           'Leaked government report about airbnb and doordash',
                                           'Complaints filed about Uber drivers on Uber_Support Twitter'])


##################### Network Graph #########################


#Function to yield(generator) tokens
def get_all_words(cleaned_tokens_list):
    tokens=[]
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

#Tokens Frequency
tokens_flat=get_all_words(cleaned_tokens)
freq_dist = FreqDist(tokens_flat)
print(freq_dist.most_common(50))


bigram_list=[list(bigrams(tweet)) for tweet in cleaned_tokens]

bigrams_flat=get_all_words(bigram_list)

freq_dist_bigrams = FreqDist(bigrams_flat)

print(freq_dist_bigrams.most_common(10))

# Create network graph
network_token_df = pd.DataFrame(freq_dist_bigrams.most_common(50), columns=[ 'token', 'count'])

bigrams_d=network_token_df.set_index('token').T.to_dict('records')

network_graph = nx.Graph()

# Create connections between nodes
for k, v in bigrams_d[0].items():
    network_graph.add_edge(k[0],k[1],weight=(v * 10))
    
fiq, ax = plt.subplots(figsize=(20,17))

pos = nx.spring_layout(network_graph, k=1)

# Plot networks
nx.draw_networkx(network_graph, pos,
                 font_size=20,
                 width=3,
                 node_size=50,
                 edge_color='grey',
                 node_color='blue',
                 with_labels = True,
                 ax=ax)



#################### Searching for bad keyword #########################



def key_word_intersection(tweets_token, bad_keyword):
  summaries = []
  results = []
  for x in tweets_token:
      summaries.append(np.array(x)[[i for i, keyword in enumerate(x) if (keyword.lower() in bad_keyword)]])
  for summary in summaries:
      temp = np.unique(summary)
      results.append(list(temp))
  return results

bad_keyword= ['idiot', 'stupid', 'useless', 'punch', 'kick', 'loser', 'liar', 'disgusting', 
              'shit', 'bitch', 'trash', 'bully', 'bullies', 'annoying', 'fuck']
review_keyword = key_word_intersection(tweets_token, bad_keyword)

print(review_keyword)
print(cleaned_tokens)


##################### Observing word frequency #########################


#Tokens Frequency
tokens_flat=get_all_words(review_keyword)
freq_dist = FreqDist(tokens_flat)
print(freq_dist.most_common(50))


##################### WordCloud: Bad_Keyword #########################


#Generate WordCloud Image
keyword_wordcloud = WordCloud(width = 800, height = 800,
                              background_color = 'white',
                              min_font_size = 10).generate(str(review_keyword))

plt.imshow(keyword_wordcloud)
plt.axis("off")
plt.show()

##################### Creating Bar Graph for Bad Keyword #########################


testing = list(filter(None, review_keyword))
print(testing)
        
bad_plot = pd.Series(testing).value_counts().plot(kind="bar")
