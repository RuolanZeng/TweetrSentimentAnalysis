import pandas as pd  # data processing, CSV file I/O
from sklearn.model_selection import train_test_split  # function for splitting data to train and test sets

import nltk
from nltk.corpus import stopwords
# from wordcloud import WordCloud, STOPWORDS
# import matplotlib.pyplot as plt
import time
start_time = time.time()

data = pd.read_csv('Tweets.csv', nrows=2000)
data = data[['text', 'airline_sentiment']]
data = data[data.airline_sentiment != "neutral"]
# data = data[1:2000]
train, test = train_test_split(data, test_size=0.4)

train_pos = train[train['airline_sentiment'] == 'positive']
train_pos = train_pos['text']
train_neg = train[train['airline_sentiment'] == 'negative']
train_neg = train_neg['text']


# def wordcloud_draw(data, color='black'):
#     words = ' '.join(data)
#     cleaned_word = " ".join([word for word in words.split()
#                              if 'http' not in word
#                              and not word.startswith('@')
#                              and not word.startswith('#')
#                              and word != 'RT'
#                              ])
#     wordcloud = WordCloud(stopwords=STOPWORDS,
#                           background_color=color,
#                           width=2500,
#                           height=2000
#                           ).generate(cleaned_word)
#     plt.figure(1, figsize=(13, 13))
#     plt.imshow(wordcloud)
#     plt.axis('off')
#     plt.show()


#print("Positive words")
#wordcloud_draw(train_pos, 'white')
#print("Negative words")
#wordcloud_draw(train_neg)

tweets = []
stopwords_set = set(stopwords.words("english"))

for index, row in train.iterrows():
    words_filtered = [e.lower() for e in row.text.split() if len(e) >= 3]
    words_cleaned = [word for word in words_filtered
                     if 'http' not in word
                     and not word.startswith('@')
                     and not word.startswith('#')
                     and word != 'RT']
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    tweets.append((words_without_stopwords, row.airline_sentiment))

test_pos = test[test['airline_sentiment'] == 'positive']
test_pos = test_pos['text']
test_neg = test[test['airline_sentiment'] == 'negative']
test_neg = test_neg['text']


def get_words_in_tweets(tweets):
    all = []
    for (words, sentiment) in tweets:
        all.extend(words)
    return all


def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    return features


w_features = get_word_features(get_words_in_tweets(tweets))


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


#print("features")
#wordcloud_draw(w_features)


training_set = nltk.classify.apply_features(extract_features, tweets)

# Training the Naive Bayes classifier
classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier = nltk.Cl

# # Training the Decision Tree classifier
# classifier = nltk.DecisionTreeClassifier.train(training_set)


neg_cnt = 0
pos_cnt = 0
for obj in test_neg:
    res = classifier.classify(extract_features(obj.split()))
    if res == 'negative':
        neg_cnt = neg_cnt + 1
for obj in test_pos:
    res = classifier.classify(extract_features(obj.split()))
    if res == 'positive':
        pos_cnt = pos_cnt + 1

print('[negative]: %s/%s ' % (neg_cnt, len(test_neg)))
print(neg_cnt/len(test_neg))
print('[positive]: %s/%s ' % (pos_cnt, len(test_pos)))
print(pos_cnt/len(test_pos))

accuracy = (neg_cnt + pos_cnt) / (len(test_neg) + len(test_pos))
print('[accuracy]')
print(accuracy)

print("--- %s seconds ---" % (time.time() - start_time))
