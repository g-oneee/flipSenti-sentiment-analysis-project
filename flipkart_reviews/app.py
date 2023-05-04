import re
import os
import nltk
import joblib
import requests
import numpy as np
from bs4 import BeautifulSoup
import urllib.request as urllib
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud,STOPWORDS
from flask import Flask,render_template,request
import time
import pandas as pd


from pathlib import Path
import pickle 
from sentiment_analyzer import SentimentAnalyzer
from data_visualizer import DataVisualizer
from utilities import Utility


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


def clean(x):
    x = re.sub(r'[^a-zA-Z ]', ' ', x)
    x = re.sub(r'\s+', ' ', x)
    x = re.sub(r'READ MORE', '', x)
    x = x.lower()
    x = x.split()
    y = []
    for i in x:
        if len(i) >= 3:
            if i == 'osm':
                y.append('awesome')
            elif i == 'nyc':
                y.append('nice')
            elif i == 'thanku':
                y.append('thanks')
            elif i == 'superb':
                y.append('super')
            else:
                y.append(i)
    return ' '.join(y)

def extract_amazon(url, clean_reviews, org_reviews,customernames,commentheads,ratings):
    # headers = {
    #     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36'}
    # page = requests.get(url, headers=headers)
    # print(url)
    # page_html = BeautifulSoup(page.content, "html.parser")

    page_num = 1
    while True:
            headers = {
            # 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36'}
            "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0"}
            # Make a request to the URL and get the HTML content
            response = requests.get(f'{url}?pageNumber={page_num}',headers=headers)
            html_content = response.content
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')

            # Find all the review containers
            review_containers = soup.find_all('div', {'data-hook': 'review'})

            # If there are no more reviews, break out of the loop
            if len(review_containers) == 0:
                break



            # Loop through the review containers and extract the relevant information
            for review in review_containers:
                # Extract the rating
                rating = review.find('i', {'data-hook': 'review-star-rating'}).text.strip()
                ratings.append(int(rating[0]))
                print(rating)
                # Extract the title of the review
                title = review.find('a', {'data-hook': 'review-title'}).text.strip()
                commentheads.append(title)
                # Extract the customer name
                customer_name = review.find('span', {'class': 'a-profile-name'}).text.strip()
                customernames.append(customer_name)
                
                # Extract the review body
                review_body = review.find('span', {'data-hook': 'review-body'}).text.strip()
                clean_reviews.append(clean(review_body))
                org_reviews.append(clean(review_body))
                # Add the information to the worksheet
                # ws.append([rating, title, text])

            page_num += 1


# def extract_all_reviews(url, clean_reviews, org_reviews,customernames,commentheads,ratings):
#     headers = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36'}
#     page = requests.get(url, headers=headers)
#     print(url)
#     page_html = BeautifulSoup(page.content, "html.parser")
#     # reviews = page_html.find_all('span', {'data-hook': 'review-body'})
#     # commentheads_ = page_html.find_all('a',{'class':'review-title-content'})
#     # customernames_ = page_html.find_all('span',{'class':'a-profile-name'})
#     # ratings_ = page_html.find_all('i', {'data-hook': 'review-star-rating'})


#     # reviews = page_html.find_all('div', {'class': 'a-section review aok-relative'})
#     # commentheads_ = page_html.find_all('a',{'class':'review-title-content'})
#     # customernames_ = page_html.find_all('span',{'class':'a-profile-name'})
#     # ratings_ = page_html.find_all('div',{'class':'a-section celwidget'})

#     reviews = page_html.find_all('span', {'data-hook': 'review-body'})
#     commentheads_ = page_html.find_all('a', {'data-hook': 'review-title'})
#     customernames_ = page_html.find_all('span', {'class': 'a-profile-name'})
#     ratings_ = page_html.find_all('span', {'class': 'a-icon-alt'})
    
#     with open('reviews1.html', 'w', encoding='utf-8') as f:
#         f.write(str(reviews))
#     for review in reviews:
#         x = review.find('div', {'class': 'a-section a-text-center a-text-normal'})
#         if x:
#             x = x.get_text()
#             org_reviews.append(re.sub(r'READ MORE', '', x))
#             clean_reviews.append(clean(x))
    
#     for cn in customernames_:
#         customernames.append('~'+cn.get_text())
  
#     for ch in commentheads_:
#         # ch =  ch.replace("\n","")
#         # ch = ch.replace(".","")
#         commentheads.append(ch.get_text())
#     # commentheads.replace("\n","")
#     # print(commentheads)
#     ra = []
#     for r in ratings_:
#         try:
#             if float(r.get_text()) in [1.0,2.0,3.0,4.0,5.0]:
#                 ra.append(int(float(r.get_text())))
#             else:
#                 ra.append(0)
#             print(ra)
#         except:
#             ra.append(r.get_text())
        
#     ratings += ra
#     print(ratings)


def extract_all_reviews(url, clean_reviews, org_reviews,customernames,commentheads,ratings):
    with urllib.urlopen(url) as u:
        page = u.read()
        page_html = BeautifulSoup(page, "html.parser")
    reviews = page_html.find_all('div', {'class': 't-ZTKy'})
    commentheads_ = page_html.find_all('p',{'class':'_2-N8zT'})
    customernames_ = page_html.find_all('p',{'class':'_2sc7ZR _2V5EHH'})
    ratings_ = page_html.find_all('div',{'class':['_3LWZlK _1BLPMq','_3LWZlK _32lA32 _1BLPMq','_3LWZlK _1rdVr6 _1BLPMq']})

    for review in reviews:
        x = review.get_text()
        org_reviews.append(re.sub(r'READ MORE', '', x).lower())
        clean_reviews.append(clean(x))
    
    for cn in customernames_:
        customernames.append('~'+cn.get_text())
    
    for ch in commentheads_:
        commentheads.append(ch.get_text())
    
    ra = []
    for r in ratings_:
        try:
            if int(r.get_text()) in [1,2,3,4,5]:
                ra.append(int(r.get_text()))
            else:
                ra.append(0)
        except:
            ra.append(r.get_text())
        
    ratings += ra
    # print(ratings)
  
  
    

def tokenizer(s):
    s = s.lower()      # convert the string to lower case
    tokens = nltk.tokenize.word_tokenize(s) # make tokens ['dogs', 'the', 'plural', 'for', 'dog']
    tokens = [t for t in tokens if len(t) > 2] # remove words having length less than 2
    tokens = [t for t in tokens if t not in stop_words] # remove stop words like is,and,this,that etc.
    return tokens

def tokens_2_vectors(token):
    X = np.zeros(len(word_2_int)+1)
    for t in token:
        if t in word_2_int:
            index = word_2_int[t]
        else:
            index = 0
        X[index] += 1
    X = X/X.sum()
    return X

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/results',methods=['GET'])
def result():    
    url = request.args.get('url')

    nreviews = int(request.args.get('num'))
    clean_reviews = []
    org_reviews = []
    customernames = []
    commentheads = []
    ratings = []

    with urllib.urlopen(url) as u:
        page = u.read()
        page_html = BeautifulSoup(page, "html.parser")
    if(url[12]=="a"):
        proname = page_html.find('span', {'class': 'a-size-large product-title-word-break'}).get_text()
        price = page_html.find('span', {'class': 'a-price-whole'}).get_text()
        # getting the link of see all reviews button
        all_reviews_url = page_html.find_all('a', {'data-hook': 'see-all-reviews-link-foot'})[0]
        all_reviews_url = "https://www.amazon.in"+all_reviews_url.get('href')
        url2 = all_reviews_url
        extract_amazon(url2, clean_reviews, org_reviews,customernames,commentheads,ratings)



    elif(url[12]=="f"):
        proname = page_html.find_all('span', {'class': 'B_NuCI'})[0].get_text()
        price = page_html.find_all('div', {'class': '_30jeq3 _16Jk6d'})[0].get_text()
        
        # getting the link of see all reviews button
        all_reviews_url = page_html.find_all('div', {'class': 'col JOpGWq'})[0]
        all_reviews_url = all_reviews_url.find_all('a')[-1]
        all_reviews_url = 'https://www.flipkart.com'+all_reviews_url.get('href')
        url2 = all_reviews_url+'&page=1'
        

        # start reading reviews and go to next page after all reviews are read 
        while True:
            x = len(clean_reviews)
            # extracting the reviews
            extract_all_reviews(url2, clean_reviews, org_reviews,customernames,commentheads,ratings)
            url2 = url2[:-1]+str(int(url2[-1])+1)
            if x == len(clean_reviews) or len(clean_reviews)>=nreviews:break


    org_reviews = org_reviews[:nreviews]
    clean_reviews = clean_reviews[:nreviews]
    customernames = customernames[:nreviews]
    commentheads = commentheads[:nreviews]
    ratings = ratings[:nreviews]


    # building our wordcloud and saving it
    for_wc = ' '.join(clean_reviews)
    wcstops = set(STOPWORDS)
    wc = WordCloud(width=1400,height=800,stopwords=wcstops,background_color='white').generate(for_wc)
    plt.figure(figsize=(20,10), facecolor='k', edgecolor='k')
    plt.imshow(wc, interpolation='bicubic') 
    plt.axis('off')
    plt.tight_layout()
    CleanCache(directory='static/images')
    plt.savefig('static/images/woc.png')
    plt.close()



# main.py
    do_pickle = False
    do_train_data = False
    do_fetch_data = False
    do_preprocess_data = False
    do_cross_validation_strategy = False
    do_holdout_strategy = False
    do_analyze_visualize = False

    # Create 'pickled' and 'plots' directories if not exists
    Path('./pickled').mkdir(exist_ok = True)
    Path('./plots').mkdir(exist_ok = True)
    if do_fetch_data or do_preprocess_data or do_cross_validation_strategy or do_holdout_strategy or do_analyze_visualize:
        sentiment = SentimentAnalyzer()
    if do_fetch_data:
        sentiment.getInitialData('datasets/product_reviews.json', do_pickle)
    if do_preprocess_data:
        reviews_df = pd.read_pickle('pickled/product_reviews.pickle')
        sentiment.preprocessData(reviews_df, do_pickle)
    if do_cross_validation_strategy or do_holdout_strategy:
        reviews_df_preprocessed = pd.read_pickle('pickled/product_reviews_preprocessed.pickle')
        print(reviews_df_preprocessed.isnull().values.sum()) # Check for any null values
    if do_cross_validation_strategy:
        sentiment.crossValidationStrategy(reviews_df_preprocessed, do_pickle)

    if do_holdout_strategy: 
        sentiment.holdoutStrategy(reviews_df_preprocessed, do_pickle, do_train_data)
    if do_analyze_visualize:
        analyzeVisualize(sentiment)

    with open('pickled/model_holdout.pickle', 'rb') as model_holdout:
        model = pickle.load(model_holdout)
    model_svc = model[1][1] # Using LinearSVC classifier

    # print('\nEnter your review:')
    d = []
    for i in range(len(org_reviews)):
        x = {}
        x['review'] = org_reviews[i]
        # x['sent'] = predictions[i]
        x['cn'] = customernames[i]
        x['ch'] = commentheads[i]
        x['stars'] = ratings[i]
        d.append(x)
    for i in d:
        # user_review = input()
        if model_svc.predict([i['review']]) == 1:
            i['sent'] = 'POSITIVE'
        else:
           i['sent'] = 'NEGATIVE'
    np,nn =0,0
    for i in d:
        if i['sent']=='NEGATIVE':nn+=1
        else:np+=1

    # verdict = 'Positive' if model_svc.predict([user_review]) == 1 else 'Negative'
    # print('\nPredicted sentiment: '+ verdict)




    # predictions = []
    # 
    #     vector = tokens_2_vectors(tokenizer(clean_reviews[i]))
    #     vector = vector[:-1]
    #     if model.predict([vector])[0] == 1:
    #         predictions.append('POSITIVE')
    #     else:
    #         predictions.append('NEGATIVE')
    


    # making a dictionary of product attributes and saving all the products in a list
    # d = []
    # for i in range(len(org_reviews)):
    #     x = {}
    #     x['review'] = org_reviews[i]
    #     # x['sent'] = predictions[i]
    #     x['cn'] = customernames[i]
    #     x['ch'] = commentheads[i]
    #     x['stars'] = ratings[i]
    #     d.append(x)
    

    # for i in d:
    #     if i['stars']!=0:
    #         if i['stars'] in [1,2]:
    #             i['sent'] = 'NEGATIVE'
    #         else:
    #             i['sent'] = 'POSITIVE'
    # np,nn =0,0
    # for i in d:
    #     if i['sent']=='NEGATIVE':nn+=1
    #     else:np+=1

    return render_template('result.html',dic=d,n=len(clean_reviews),nn=nn,np=np,proname=proname,price=price)


@app.route('/wc')
def wc():
    return render_template('wc.html')

class CleanCache:
	'''
	this class is responsible to clear any residual csv and image files
	present due to the past searches made.
	'''
	def __init__(self, directory=None):
		self.clean_path = directory
		# only proceed if directory is not empty
		if os.listdir(self.clean_path) != list():
			# iterate over the files and remove each file
			files = os.listdir(self.clean_path)
			for fileName in files:
				print(fileName)
				os.remove(os.path.join(self.clean_path,fileName))
		print("cleaned!")


if __name__ == '__main__':
    app.run(debug=True)

























# url = 'https://www.amazon.in/dp/B08XJG8J4P' # URL of the product page on amazon.in
# nreviews = 100 # Number of reviews to scrape
# clean_reviews = []
# org_reviews = []
# customernames = []
# commentheads = []
# ratings = []

# url = url+'/ref=cm_cr_getr_d_paging_btm_prev_1?ie=UTF8&reviewerType=all_reviews&pageNumber=1'
# while True:
#     x = len(clean_reviews)
#     extract_all_reviews(url, clean_reviews, org_reviews, customernames, commentheads, ratings)
#     url = url+'/ref=cm_cr_getr_d_paging_btm_prev_'+str(int(url.split('=')[-1])+1)+'?ie=UTF8&reviewerType=all_reviews&pageNumber='+str(int(url.split('=')[-1])+1)
#     if x == len(clean_reviews) or len(clean_reviews) >= nreviews:
#         break

# org_reviews = org_reviews[:nreviews]
# clean_reviews = clean_reviews[:nreviews]
# customernames = customernames[:nreviews]
# commentheads
