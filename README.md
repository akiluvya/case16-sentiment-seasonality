# case16-sentiment-seasonality

This is a project that I jsut statrted for one of our Company's client. 

The client is interested in understanding the fractuations in the orders placed every day. Her main concern is to understad if her product has a hidden seasonality, if social media comments about the product affect the orders placed, and if the artwork hsared on social media has impact on orders placed since she is looking to reduce the amount spent in advertising in social media.

## Seasonality Check
The first part is in done in Jupyter notebook, to check if the product has any sign of seasonality. She is specifically interested to see if there should be atime she should reduce distribution as well as spending on social media.
This part is done using statsmodels library, pandas, numpy and matplotlib for simple visualizations.

## Fetching Tweets
The second part is fetching all all tweets that mentions any keyword that describes her product, business name or company name. This is done using Tweepy, although it is not suitable for production due to reliability in broken connections, it is sufficient to for proof of concept part.

## Natural Language Processing
The third part is perfoming Natural Language Processing to identify the sentiment of a ery single tweet. This is implemented from scratch uing naive bayes classifier. The classifier is trained in a publicly available dataset. The main challenge is the training set is not product/ busiiness oriented, rather general tweets oriented. It can even be used as Political dataset. This classifier can achive accuracy of 73%. SVM or deep learning could definetely perform better, but for the proof of concept purpose, this works great.

## Data storage
The fouth part is Data storage, I use MongoDB for storage due to its flexibility in accommodating horizontal scalability.

## Future Work
After collecting the sentiments, the next stage is checking for correlation between sentiments and orders.
Also, Computer vision (object detection and recognition) classifier will be built to see if the sentiments are a results of the artworks posted on social media.
