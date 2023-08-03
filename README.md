# Investron
Investron is a mini-tool that automates daily stock research by summarizing news and sentiment for each stock. It utilizes the Pegasus transformer model for text summarization and the Hugging Face pipeline for sentiment analysis to provide valuable insights for informed investment decisions.

### Data Pipeline Overview
* The tool identifies and scrapes relevant URLs from Google search results for the specified stock tickers using the "yahoo finance" query. It filters out irrelevant URLs to retrieve only related to news articles.
* The cleaned URLs are used to fetch the corresponding articles from Yahoo Finance. The articles are cleaned, tokenized and summarized using transformers.
* The generated summaries are analyzed for sentiment using the Hugging Face pipeline. The sentiment scores provide insights into the positive, neutral, or negative nature of the news related to each stock.
* The collected data, including ticker names, summaries, sentiment, sentiment ratings, and article URLs, is organized and formatted into a CSV file for easy access and further analysis.
### Results
Below is a snippet of the output CSV for a few stocks (Pfizer Inc, Tesla, Bitcoin, ZoomInfo Technologies). 
<br/>
<img width="836" alt="Stock Summary Image" src="https://github.com/anirudhk33/Investron/assets/114661218/1cdeb63b-b9fe-4207-81f6-f173ebf37fe9">

