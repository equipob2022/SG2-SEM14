import streamlit as st
import sys
sys.path.append('../')
from conf import load_tweet
import re
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import pandas as pd
import nltk
from nltk.tokenize import WordPunctTokenizer
from textblob import TextBlob
import io
import matplotlib.pyplot as plt
##
import asyncio
##

# Set page name and favicon
st.set_page_config(page_title='Twitter scraper',page_icon=':iphone:')


def app():
    nltk.download('punkt')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Modelo twitter')
    st.title('Scraping Twitter')
    keyword = st.text_input('Input Keyword for Screep','Elon Musk')
    limit = st.number_input('Input Limit for Screep', 20, step=20)
    tweet_data = None 
    df = None

    if st.button("Start"):
        if keyword != "" and limit != "":
            st.text("Keyword : "+keyword)
            st.text("Limit : "+str(limit))
            st.subheader("Data cruda")  
            data_tweet = st.text('Loading data...')
            df = load_tweet(keyword,limit)
            data_tweet.text('Loading data... done!')
            st.table(df['tweet'].head(3))
            st.write(df)

        if keyword != "" and limit != "":
            st.subheader("Remover caracteres")
            removing_data = st.text('Removing not required characters...')
            df['tweet'] = df['tweet'].apply(lambda text: tweet_cleaner(text))
            removing_data.text('Removing... done!')
            st.table(df['tweet'].head(3))
            st.write(df)

        if keyword != "" and limit != "":
            st.subheader("Normalizar texto")
            normalize_data = st.text('Normalizing data...')
            df["tweet"] = df["tweet"].apply(lambda x: word_normalize(x))
            normalize_data.text('Normalizing... done!')
            st.table(df['tweet'].head(3))
            st.write(df)

        if keyword != "" and limit != "":
            st.subheader("Data labeada")
            labeling_data = st.text('Labeling data...')
            df["sentiment"] = df["tweet"].apply(lambda x: sentimenLabeling(x))
            labeling_data.text('Labeling... done!')
            st.table(df['tweet'].head(3))
            st.write(df)
        
        if keyword != "" and limit != "":
            st.subheader("Agregar un keyword a las columnas")
            df["keyword"] = keyword
            st.table(df['tweet'].head(3))
            st.write(df)
        
        labels = 'Positivo', 'Neutral', 'Negativo'
        pos = df[df.sentiment == 1].shape[0]
        net = df[df.sentiment == 0].shape[0]
        neg = df[df.sentiment == -1].shape[0]
        sizes = [pos, net, neg]
        colors = ['lightskyblue', 'gold', 'lightcoral']
        explode = (0.1, 0, 0)  # explode 1st slice

        # Plot
        plt.pie(sizes, explode=explode  , labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=140)

        plt.axis('equal')
        st.pyplot()
            
        # To remove a specific field name from all documents
        # client = mp.client()
        # client.update('tweets','date')
        # st.success('Update success')

        # To remove all documents
        # client = mp.client()
        # client.deleteall('tweets')
        # client.deleteall('keywords')

        # To remove documents with category
        # client = mp.client()
        # client.deleteByValue(keyword)
        # st.success('Delete data success')

        with io.open('stopword_list_TALA.txt', encoding="utf-8") as f:
            stoptext = f.read().lower()
            stopword = nltk.word_tokenize(stoptext)

        st.subheader("WordCloud Positivo Tweet")
        asyncio.set_event_loop(asyncio.new_event_loop())
        showWordCloud(df, 1, stopword)

        st.subheader("WordCloud Neutral Tweet")
        
        asyncio.set_event_loop(asyncio.new_event_loop())
        showWordCloud(df, 0, stopword)

        st.subheader("WordCloud Negative Tweet")
        asyncio.set_event_loop(asyncio.new_event_loop())
        showWordCloud(df, -1, stopword)

@st.cache
def tweet_cleaner(text):
    tok = WordPunctTokenizer()
    pat1 = r'@[A-Za-z0-9]+'
    pat2 = r'https?://[A-Za-z0-9./]+'
    pat3 = r'pic.twitter.com/[A-Za-z0-9./]+'
    combined_pat = r'|'.join((pat1, pat2, pat3))
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()

@st.cache
def word_normalize(text):
    norm = pd.read_csv('key_norm.csv')

    norm_dic = pd.Series(norm.hasil.values,index=norm.singkat).to_dict()  
    text_tokenized = nltk.word_tokenize(text.lower())
    text = " ".join(word if word not in norm_dic else norm_dic[word] for word in text_tokenized)
    return text

def sentimenLabeling(text):
    analyticts  = TextBlob(text)
    an = analyticts
        
    try:
        if an.detect_language() != 'en':
            an = analyticts.translate(from_lang='id', to='en')
            print(an)
    except:
        print("Skip")
        
    if an.sentiment.polarity > 0:
        return 1
    elif an.sentiment.polarity == 0:
        return 0
    else:
        return -1

def showWordCloud (df,sentiment, stopwords):
    tweets = df[df.sentiment == sentiment]
    string = []
    for t in tweets.tweet:
        string.append(t)
    string = pd.Series(string).str.cat(sep=' ')

    wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(string)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot()