# ### Stock Market Crawler: Yahoo Finance

from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from bs4 import BeautifulSoup
import requests
import os
from transformers import pipeline
import re
import csv

os.environ["HF_MIRROR"] = "https://huggingface.co/mirrors"  # Using different mirror URL to escape bugs

# Defining Model
name = "human-centered-summarization/financial-summarization-pegasus"
tokenizer = PegasusTokenizer.from_pretrained(name)
model = PegasusForConditionalGeneration.from_pretrained(name)


url = "https://finance.yahoo.com/news/term-labs-launches-fixed-rate-130117305.html"
r = requests.get(url)
soup = BeautifulSoup(r.text, 'html.parser')
paragraphs = soup.find_all('p')


print("No. of Paragraphs:", len(paragraphs))
print(paragraphs[3])


text = [paragraph.text for paragraph in paragraphs]
words = ' '.join(text).split(' ')[:400]
full_article = ' '.join(words)
print(full_article)


# Tokenization and summary generation
input_ids = tokenizer.encode(full_article, return_tensors='pt')
generation = model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True)
summary = tokenizer.decode(generation[0], skip_special_tokens=True)
print(summary)


# ### Pipeline


stock_tickers = ['PFE', 'TSLA', 'BTC', 'ZI']


def crawl_to_retrieve_urls(ticker):
    search_url = "https://www.google.com/search?q=yahoo+finance+{}&tbm=nws".format(ticker)
    r = requests.get(search_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    atags = soup.find_all('a')
    hrefs = [link['href'] for link in atags]
    return hrefs 


urls = {ticker:crawl_to_retrieve_urls(ticker) for ticker in stock_tickers}
print(urls)


bad_words = ['preferences', 'accounts', 'support', 'maps', 'policies']


def parse_urls(urls, bad_words):
    val = []
    for url in urls: 
        if 'https://' in url and not any(bad_word in url for bad_word in bad_words):
            res = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
            val.append(res)
    return list(set(val))


cleaned_urls = {ticker:parse_urls(urls[ticker], bad_words) for ticker in stock_tickers}
print(cleaned_urls)


def scrape_and_process(URLs):
    articles = []
    for url in URLs: 
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = [paragraph.text for paragraph in paragraphs]
        words = ' '.join(text).split(' ')[:300]
        article = ' '.join(words)
        articles.append(article)
    return articles


articles = {ticker:scrape_and_process(cleaned_urls[ticker]) for ticker in stock_tickers}
print(articles)


def NLP_summarize(articles):
    summaries = []
    for article in articles:
        input_ids = tokenizer.encode(article, return_tensors='pt')
        output = model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True)
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries


summaries = {ticker:NLP_summarize(articles[ticker]) for ticker in stock_tickers}
print(summaries)


sentiment = pipeline('sentiment-analysis')


# !pip install xformers

sentiment(summaries['BTC'])

scores = {ticker:sentiment(summaries[ticker]) for ticker in stock_tickers}
print(scores)
print(summaries['PFE'][3], scores['PFE'][3]['label'], scores['PFE'][3]['score'])

scores['BTC'][0]['score']
print(summaries)

print(cleaned_urls)


def format_data(summaries, scores, urls):
    output = []
    for ticker in stock_tickers:
        for counter in range(len(summaries[ticker])):
            output_this = [
                ticker,
                summaries[ticker][counter],
                scores[ticker][counter]['label'],
                scores[ticker][counter]['score'],
                urls[ticker][counter]
            ]
            output.append(output_this)
    return output


formatted_data = format_data(summaries, scores, cleaned_urls)
formatted_data


titles = ["Ticker Name", "News Summary", "Sentiment", "Sentiment Rating", "News URL"]
# Write data to the CSV file
with open('stocksummaries.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    # Write the titles first
    csv_writer.writerow(titles)
    # Write the data rows
    csv_writer.writerows(formatted_data)