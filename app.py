import pandas as pd 
import numpy  as np 
import os
import sys
import pickle
import streamlit as st
from PIL import Image
import time
#import base64
import re

@st.cache
def clean_article(article):
    art = re.sub("[^A-Za-z0-9' ]", '', str(article))
    art2 = re.sub("[( ' )(' )( ')]", ' ', str(art))
    art3 = re.sub("\s[A-Za-z]\s", ' ', str(art2))
    return art3.lower()

@st.cache
def load_image(img):
	im =Image.open(os.path.join(img))
	return im
# To Improve speed and cache data
@st.cache(persist=True)
def explore_data(dataset):
	df = pd.read_csv(os.path.join(dataset))
	return df 



@st.cache
def loadData():
    df = explore_data('train.csv')
    return df




bow = pickle.load(open("bow.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# def load_model(var):

#     load_model = pickle.load(open('optimze_logR_model.sav', 'rb'))
#     prediction  = load_model.predict([var])
#     prob= load_model.predict_proba([var])
#     return prediction[0], prob[0][1]

    
def main():
    st.title('FAKE NEWS PREDICTION WEB APP!')
    st.subheader('News Classifier')
    st.subheader('“ What Fake news” is… ?')
    st.write("""\n\n
“Fake news” is a term used to refer to fabricated news. Fake news is an invention – a lie created out of nothing – that takes the appearance of real news with the aim of deceiving people. This is what is important to remember: the information is false, but it seems true.

That’s logical! If it is too obvious that it is a lie, it won’t have any impact. Fake news is a little like a false rumour, but on a large scale…""")

    data = loadData()

    our_image = load_image('Fighting Fake News.jpg')
    with st.spinner("Waiting.."):
        time.sleep(5)
    st.success("Finished Loading!!")

    # file_ = open("/home/rzwitch/Desktop/giphy.gif", "rb")
    # contents = file_.read()
    # data_url = base64.b64encode(contents).decode("utf-8")
    # file_.close()   

    # st.markdown(
    # f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
    # unsafe_allow_html=True,
    # )
    #st.image(our_image)
    
    if st.checkbox('Show Our Training Data'):
        st.subheader('Showing Training Data')

        st.write(data.head(100))
    news_text =  st.text_area("Insert Article here","eg: Obama is running for president 2021")
    if st.button("Predict"):
        st.subheader("Original Text:: {}".format(news_text))
        news_text0 = [news_text]
        news_text1 = clean_article(news_text0)
        news_text2 = [news_text1]
        vect = bow.transform(news_text2)
        vect.columns = bow.get_feature_names()
        predict = model.predict(vect)
        maxProba =  model.predict_proba(vect)

        if predict == 1:
            st.error("Oop's probably this is a FAKE news")
        else:
            st.success("GREAT this is a REAL news")
            st.success("The truth probability score is :: {}".format(maxProba))
            




    st.sidebar.image(our_image, use_column_width=True)
    st.sidebar.header('About')
    st.sidebar.info('Data is been collected from  the LAIR  Dataset.\n\n'+'My model is still learning\n\n'+'A Web App Developed by MOHD FAHAD.\n\n' + \
        '(c) 2020.')
    st.sidebar.markdown('---')
    footer=""" 
               <link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='styles.css') }}">
               <div class="contact">
               <a target="_blank" href="https://github.com/mofahad"><i class="fab fa-github fa-lg contact-icon"></i></a>
               <a target="_blank" href="https://www.linkedin.com/in/mohammedfahad"><i class="fab fa-linkedin fa-lg contact-icon"></i></a>
               </div>"""
            
    st.sidebar.markdown(footer,unsafe_allow_html=True)              
    





if __name__ =='__main__':
	main()


