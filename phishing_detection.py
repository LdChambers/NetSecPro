import mailbox
import gensim
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import *
import eml_parser, json, datetime
import email
import pandas as pd
import numpy as np
import os
import joblib

# NLP with NLTK
import nltk
nltk.download("stopwords")
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

# For visualization
from wordcloud import WordCloud
#import pyLDAvis
#import pyLDAvis.gensim

# Sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

# Doc2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Keras
import tensorflow as tf
from tensorflow.keras import layers
# From https://stackoverflow.com/questions/7166922/extracting-the-body-of-an-email-from-mbox-file-decoding-it-to-plain-text-regard

def get_charsets(msg):
    charsets = set({})
    for c in msg.get_charsets():
        if c is not None:
            charsets.update([c])
    return charsets

def handle_error(errmsg, emailmsg,cs):
    '''print()
    print(errmsg)
    print("This error occurred while decoding with ",cs," charset.")
    print("These charsets were found in the one email.",get_charsets(emailmsg))
print("This is the subject:",emailmsg['subject'])
    print("This is the sender:",emailmsg['From'])'''
    pass

def get_body_from_email(msg):
    body = None
    #Walk through the parts of the email to find the text body.
    if msg.is_multipart():
        for part in msg.walk():

            # If part is multipart, walk through the subparts.
            if part.is_multipart():

                for subpart in part.walk():
                    if subpart.get_content_type() == 'text/plain':
                        # Get the subpart payload (i.e the message body)
                        body = subpart.get_payload(decode=True)
                        #charset = subpart.get_charset()

            # Part isn't multipart so get the email body
            elif part.get_content_type() == 'text/plain':
                body = part.get_payload(decode=True)
                #charset = part.get_charset()
 # If this isn't a multi-part message then get the payload (i.e the message body)
    #elif msg.get_content_type() == 'text/plain':
        #body = msg.get_payload(decode=True)

    # Uncomment this to also include html and other formats
    else:
        body = msg.get_payload(decode=True)

    return body
phishingBoxFilenames = ["/home/datasets/monkey/phishing3.mbox","/home/datasets/monkey/phishing2.mbox", "/home/datasets/monkey/phishing1.mbox", "/home/datasets/monkey/phishing0.mbox",
                        "/home/datasets/monkey/phishing2022.mbox", "/home/datasets/monkey/phishing2021.mbox", "/home/datasets/monkey/phishing2020.mbox", "/home/datasets/monkey/phishing2019.mbox", "/home/datasets/monkey/phishing2018.mbox",
                       "/home/datasets/monkey/phishing2017.mbox", "/home/datasets/monkey/phishing2016.mbox", "/home/datasets/monkey/phishing2015.mbox"]
phishingBoxes = [mailbox.mbox(f) for f in phishingBoxFilenames]
phishingMessages = [m[1] for phishingBox in phishingBoxes for m in phishingBox.items()]

phishingMessageBodies = []
for p in phishingMessages:
    body = get_body_from_email(p)
    if body is not None and len(body) > 0:
        phishingMessageBodies.append(body)

print(len(phishingMessages), len(phishingMessageBodies))
# Clair Fraud Email Database
added = []
with open("/home/datasets/Clair/fraudulent_emails.txt", 'r', errors="ignore") as f:
    body = ""
    inBody = False
    for line in f:
        if line.startswith("Status: O"):
            inBody = True

        elif line.startswith("From r") and len(body) > 0:
            inBody = False
            added.append(body)
            body = ""

        elif inBody:
            body += line

phishingMessageBodies = list(set(phishingMessageBodies + [a for a in added if len(a) > 0]))
print(len(phishingMessageBodies))
# SpamAssassin Spam (not exactly phishing, but NVIDIA article used it as phishing so attempting it)
ep = eml_parser.EmlParser(include_raw_body=True)

spamDir = "/home/datasets/SpamAssassin/spam_2/spam_2/"
spamFilenames = [os.path.join(spamDir, f) for f in os.listdir(spamDir)]

added = []
for filename in spamFilenames:
    with open(filename, "rb") as f:
        b = f.read()

    m = ep.decode_email_bytes(b)
    if len(m["body"]) >= 1:
        added.append(m["body"][0]["content"])

phishingMessageBodies = list(set(phishingMessageBodies + added))
print(len(phishingMessageBodies))
ep = eml_parser.EmlParser(include_raw_body=True)

easyHamDir = "/home/datasets/SpamAssassin/easy_ham/easy_ham/"
hardHamDir = "/home/datasets/SpamAssassin/hard_ham/hard_ham/"
hamFilenames = [os.path.join(easyHamDir, f) for f in os.listdir(easyHamDir)] + [os.path.join(hardHamDir, f) for f in os.listdir(hardHamDir)]

benignMessageBodies = []
for filename in hamFilenames:
    with open(filename, "rb") as f:
        b = f.read()
 m = ep.decode_email_bytes(b)
    if len(m["body"]) >= 1:
        benignMessageBodies.append(m["body"][0]["content"])

print(len(benignMessageBodies))
# Custom stop words and preprocessing filters

stopWords = nltk.corpus.stopwords
stopWords = stopWords.words("english")
stopWords.extend(["nbsp", "font", "sans", "serif", "bold", "arial", "verdana", "helvetica", "http", "https", "www", "html", "enron", "margin", "spamassassin"])

def remove_custom_stopwords(p):
    return remove_stopwords(p, stopwords=stopWords)

CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_custom_stopwords, remove_stopwords, strip_short, stem_text]
# Decode to utf-8 as needed

# Phishing
phishingDecoded = []
for b in phishingMessageBodies:
    try:
        p = b.decode("utf-8", errors="ignore")
except AttributeError:
        p = b
    benignDecoded.append(p)
benignMessageBodies = benignDecoded
# Phishing emails
phishingPreprocessed = []
for b in phishingMessageBodies:
    p = preprocess_string(b, filters=CUSTOM_FILTERS)
    #p = gensim.parsing.preprocessing.remove_stopwords(p, stopwords=stopWords

    phishingPreprocessed.append(p)
print(len(phishingPreprocessed))
# Benign emails
benignPreprocessed = []
for b in benignMessageBodies:
    p = preprocess_string(b, filters=CUSTOM_FILTERS)
    benignPreprocessed.append(p)
print(len(benignPreprocessed))
# Train the model on all messages
model = Word2Vec(phishingPreprocessed + benignPreprocessed, min_count=1, workers=3, window=5)
model.wv.most_similar("dollar", topn=20)
model.wv["dollar"]
#cloudString = ','.join(','.join(list(phishingPreprocessed + benignPreprocessed)))
'''cloudString = ""
for doc in phishingPreprocessed + benignPreprocessed:
cloudString += ",".join(doc)'''
'''wordCloud = WordCloud(max_words=5000, background_color="white")
wordCloud.generate(cloudString)
wordCloud.to_image()'''
numTopics = 1024
# Create dictionary and corpus
dictionary = gensim.corpora.Dictionary(phishingPreprocessed + benignPreprocessed)
corpus = [dictionary.doc2bow(text) for text in phishingPreprocessed + benignPreprocessed] # Term document frequency
joblib.dump(dictionary, "/home/datasets/models/ldaDictionary.joblib")
# Create model
ldaModel = gensim.models.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=numTopics)
# Print keyword for the topics
print(ldaModel.print_topics())
joblib.dump(ldaModel, "/home/datasets/models/ldaModel.joblib")
'''# Visualize LDA
pyLDAvis.enable_notebook()

ldavis = pyLDAvis.gensim.prepare(ldaModel, corpus, dictionary)
pyLDAvis.save_html(ldavis, "/home/datasets/results/ldavis_both.html")
#ldavis'''
'''# Create dictionary and corpus
dictionaryPhishing= gensim.corpora.Dictionary(phishingPreprocessed)
corpusPhishing = [dictionaryPhishing.doc2bow(text) for text in phishingPreprocessed] # Term document frequency'''
'''# Create model
ldaModelPhishing = gensim.models.LdaMulticore(corpus=corpusPhishing, id2word=dictionaryPhishing, num_topics=numTopics)'''
'''# Visualize LDA
pyLDAvis.enable_notebook()

ldavisPhishing = pyLDAvis.gensim.prepare(ldaModelPhishing, corpusPhishing, dictionaryPhishing)
pyLDAvis.save_html(ldavisPhishing, "/home/datasets/results/ldavis_phishing.html")
#ldavisPhishing'''
'''# Create dictionary and corpus
dictionaryBenign = gensim.corpora.Dictionary(benignPreprocessed)
corpusBenign = [dictionaryBenign.doc2bow(text) for text in benignPreprocessed] # Term document frequency'''
'''# Create model
ldaModelBenign = gensim.models.LdaMulticore(corpus=corpusBenign, id2word=dictionaryBenign, num_topics=numTopics)'''
'''# Visualize LDA
pyLDAvis.enable_notebook()

ldavisBenign = pyLDAvis.gensim.prepare(ldaModelBenign, corpusBenign, dictionaryBenign)
pyLDAvis.save_html(ldavisBenign, "/home/datasets/results/ldavis_benign.html")
#ldavisBenign'''
taggedDocs = [TaggedDocument(doc, [i]) for i, doc in enumerate(phishingPreprocessed + benignPreprocessed)]
doc2VecModel = Doc2Vec(taggedDocs, vector_size=20, window=2, min_count=1, workers=4)
joblib.dump(doc2VecModel, "home/datasets/models/doc2VecModel.joblib")
allPreprocessed = phishingPreprocessed + benignPreprocessed
allBodies = phishingMessageBodies + benignMessageBodies
# Blacklisted words and phrases pulled form spam wordlist
# Will be a list of lists. Each phrase/words is a list of preprocessed words If a given email has all of the stemmed/preprocessed words in a given list, it's a hit.

blackListWords = []
with open("/home/datasets/results/spam_wordlist.txt", 'r') as f:
    for line in f:
        blackListWords.append(line)

blackList = []
for b in blackListWords:
    blackListEntry = preprocess_string(b, filters=CUSTOM_FILTERS)
    if len(blackListEntry) > 0 and blackListEntry not in blackList:
        blackList.append(blackListEntry)
        # TF-IDF for the top set amount of words
maximumTermCount = 6

def preprocessor_func(s):
    sentence = " ".join(preprocess_string(s, filters=CUSTOM_FILTERS))
    return sentence

tfIDFVectorizer = TfidfVectorizer(max_features=maximumTermCount, preprocessor=preprocessor_func, sublinear_tf=True)
tfIDF = tfIDFVectorizer.fit_transform(allBodies).toarray()
print(tfIDFVectorizer.get_feature_names_out())
print(tfIDF.shape)
joblib.dump(tfIDF, "/home/datasets/models/tfIDF.joblib")
def count_all_upper_word(text):
    count = 0
    for word in text.split():
        if word.upper() == word:
            count += 1

    return count
allVectors = []
for i in range(len(allBodies)):
    topTopics = ldaModel.get_document_topics(corpus[i], minimum_probability=0.0)

    # Can extend this array with other stuff later
    vec = [topTopics[i][1] for i in range(numTopics)] # Topics

    for v in doc2VecModel.infer_vector(allPreprocessed[i]): # Doc2Vec
        vec.append(v)

    # Sentiment analysis of polarity
    sia = SentimentIntensityAnalyzer()
    sentence = " ".join(allPreprocessed[i])
    polarity = sia.polarity_scores(sentence)
    for s in polarity:
        vec.append(polarity[s])

    # Contains HTML
    if "<html>" in allBodies[i].lower():
        vec.append(1)
    else:
        vec.append(0)

    # Contains a link (how many)
    if "http://" in allBodies[i].lower() or "https://" in allBodies[i].lower():
        vec.append(1)
    else:
        vec.append(0)

    # How many blacklisted phrases/words appear in this email
    for b in blackList:
        count = 0
        for word in b:
            if word in allPreprocessed[i]:
                count += 1
        vec.append(count)

    # TF-IDF for top terms
    for w in tfIDF[i]:
        vec.append(w)
# Has all caps word?
    vec.append(count_all_upper_word(allBodies[i]))

    # Has exclamation marks?
    vec.append(allBodies[i].count("!"))

    # Total length
    vec.append(len(allBodies[i]))

    # Num words
    vec.append(len(allPreprocessed[i]))

    allVectors.append(vec)
print(np.array(allVectors).shape)
allLabels = []
for i in range(len(phishingPreprocessed)):
    allLabels.append(1)
for i in range(len(benignPreprocessed)):
    allLabels.append(0)
# Scale and split data
#scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(allVectors)
X_train, X_test, y_train, y_test = train_test_split(scaler.transform(allVectors), allLabels, test_size=0.2, shuffle=True)
rf = RandomForestClassifier()
#rf = make_pipeline(StandardScaler(), RandomForestClassifier())
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

rfcAccuracy = accuracy_score(y_test, y_pred)
rfcPrecision = precision_score(y_test, y_pred)
rfcRecall = recall_score(y_test, y_pred)

print("Accuracy:", rfcAccuracy)
print("Precision:", rfcPrecision)
print("Recall:", rfcRecall)
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot();
joblib.dump(rf, "/home/datasets/models/random_forest_model.joblib")
joblib.dump(scaler, "/home/datasets/models/rfcStandardScaler.joblib")
def create_vectors_from_messages(messages, messagesPreprocessed):
    corpus = [dictionary.doc2bow(text) for text in messagesPreprocessed] # Term document frequency
    allPreprocessed = messagesPreprocessed
    allBodies = messages
    allVectors = []
    for i in range(len(allBodies)):
        topTopics = ldaModel.get_document_topics(corpus[i], minimum_probability=0.0)

 # Can extend this array with other stuff later
        vec = [topTopics[i][1] for i in range(numTopics)] # Topics

        for v in doc2VecModel.infer_vector(allPreprocessed[i]): # Doc2Vec
            vec.append(v)

        # Sentiment analysis of polarity
        sia = SentimentIntensityAnalyzer()
        sentence = " ".join(allPreprocessed[i])
        polarity = sia.polarity_scores(sentence)
        for s in polarity:
            vec.append(polarity[s])

        # Contains HTML
        if "<html>" in allBodies[i].lower():
            vec.append(1)
        else:
            vec.append(0)

        # Contains a link
        if "http://" in allBodies[i].lower() or "https://" in allBodies[i].lower():
            vec.append(1)
        else:
            vec.append(0)

        # How many blacklisted phrases/words appear in this email
        for b in blackList:
            count = 0
            for word in b:
                if word in allPreprocessed[i]:
                    count += 1
            vec.append(count)

        # TF-IDF for top terms
        for w in tfIDF[i]:
            vec.append(w)

        # Has all caps word?
        vec.append(count_all_upper_word(allBodies[i]))

        # Has exclamation marks?
        vec.append(allBodies[i].count("!"))

        # Total length
        vec.append(len(allBodies[i]))

        # Num words
        vec.append(len(allPreprocessed[i]))

        allVectors.append(vec)

    return allVectors

#box = mailbox.mbox("/home/datasets/monkey/phishing1.mbox")
box = mailbox.mbox("benign.mbox")
boxMessages = [m[1] for m in box.items()]

messages = []
for p in boxMessages:
    body = get_body_from_email(p)
    if body is not None and len(body) > 0:
        messages.append(body)

messagesDecoded = []
for m in messages:
    try:
        d = m.decode("utf-8", errors="ignore")
    except AttributeError:
        d = m
    messagesDecoded.append(d)
messages = messagesDecoded

# Apply preprocessing function to emails
messagesPreprocessed = []
for b in messages:
    p = preprocess_string(b, filters=CUSTOM_FILTERS)

    messagesPreprocessed.append(p)

allVectorsTest = create_vectors_from_messages(messages, messagesPreprocessed)
print("Shape of messages: {}".format(np.array(allVectorsTest).shape))

# RFC Prediction
#scaler = StandardScaler()
#scaler.fit(allVectors)
X = scaler.transform(allVectors)
y_pred = rf.predict(X)
numPhishes = 0
for i in y_pred:
    if i == 1:
        numPhishes += 1
print("Phishes: {}. Benign: {}.".format(numPhishes, len(messages)-numPhishes))
gmailLabels = [0 for i in range(len(y_pred))]
gmailAccuracy = accuracy_score(gmailLabels, y_pred)
print(gmailAccuracy)
#svc = make_pipeline(StandardScaler(), SVC(gamma='auto'))
svc = SVC(gamma="auto")
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot();
#Disable GPU
print(tf.config.list_physical_devices('GPU'))
tf.config.set_visible_devices([], '')
print(tf.config.list_physical_devices('GPU'))
input_shape = (1, len(X_train[0]), 1) # Batch size is the first number
print(input_shape)

X_train = allVectors

# Scale data
scaler = StandardScaler()
scaler.fit(X_train)
joblib.dump(scaler, "/home/datasets/models/cnnScaler.joblib")
X_train = scaler.transform(X_train)

X_train = np.reshape(X_train, (len(X_train), input_shape[1], input_shape[2]))
#print(X_train[30])
print(X_train.shape)
y_train = allLabels
y_train = np.reshape(y_train, (len(X_train), 1))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
embedding_dim = 1024
num_layers = 3


inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32) # Input vector
x = layers.Conv1D(embedding_dim,2, padding="valid", activation="relu", strides=3)(inputs)
x = layers.Dropout(0.5)(x)
x = layers.Conv1D(embedding_dim, 2, padding="valid", activation="relu", strides=3)(x)
x = layers.Dropout(0.5)(x)
x = layers.Conv1D(embedding_dim, 2, padding="valid", activation="relu", strides=3)(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(embedding_dim, activation="relu")(x)
x = layers.Dropout(0.5)(x)
# We project onto a single unit output layer, and squash it with a sigmoid:
predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)


model = tf.keras.Model(inputs, predictions)

# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"])
epochs = 4

# Fit the model using the train and test datasets.
#model.fit([X_train, y_train], validation_data=[X_test, y_test], epochs=epochs)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs)
# Test on my gmail inbox
def create_vectors_from_messages(messages, messagesPreprocessed):
    corpus = [dictionary.doc2bow(text) for text in messagesPreprocessed] # Term document frequency
    allPreprocessed = messagesPreprocessed
    allBodies = messages
allVectors = []
    for i in range(len(allBodies)):
        topTopics = ldaModel.get_document_topics(corpus[i], minimum_probability=0.0)

        # Can extend this array with other stuff later
        vec = [topTopics[i][1] for i in range(numTopics)] # Topics

        for v in doc2VecModel.infer_vector(allPreprocessed[i]): # Doc2Vec
            vec.append(v)

        # Sentiment analysis of polarity
        sia = SentimentIntensityAnalyzer()
        sentence = " ".join(allPreprocessed[i])
        polarity = sia.polarity_scores(sentence)
        for s in polarity:
            vec.append(polarity[s])

        # Contains HTML
        if "<html>" in allBodies[i].lower():
            vec.append(1)
        else:
            vec.append(0)

        # Contains a link
if "http://" in allBodies[i].lower() or "https://" in allBodies[i].lower():
            vec.append(1)
        else:
            vec.append(0)

        # How many blacklisted phrases/words appear in this email
        for b in blackList:
            count = 0
            for word in b:
                if word in allPreprocessed[i]:
                    count += 1
            vec.append(count)

        # TF-IDF for top terms
        for w in tfIDF[i]:
            vec.append(w)

        # Has all caps word?
        vec.append(count_all_upper_word(allBodies[i]))

        # Has exclamation marks?
        vec.append(allBodies[i].count("!"))

        # Total length
vec.append(len(allBodies[i]))

        # Num words
        vec.append(len(allPreprocessed[i]))

        allVectors.append(vec)

    return allVectors

#box = mailbox.mbox("/home/datasets/monkey/phishing1.mbox")
box = mailbox.mbox("benign.mbox")
boxMessages = [m[1] for m in box.items()]

messages = []
for p in boxMessages:
    body = get_body_from_email(p)
    if body is not None and len(body) > 0:
        messages.append(body)

messagesDecoded = []
for m in messages:
    try:
        d = m.decode("utf-8", errors="ignore")
    except AttributeError:
        d = m
    messagesDecoded.append(d)
messages = messagesDecoded

# Apply preprocessing function to emails
messagesPreprocessed = []
for b in messages:
    p = preprocess_string(b, filters=CUSTOM_FILTERS)

    messagesPreprocessed.append(p)

allVectorsTest = create_vectors_from_messages(messages, messagesPreprocessed)
print("Shape of messages: {}".format(np.array(allVectorsTest).shape))

# RFC Prediction
#scaler = StandardScaler()
#scaler.fit(allVectors)
X = scaler.transform(allVectors)
#y_pred = rf.predict(X)
y_pred = model.predict(X)
y_pred_bin = []
numPhishesCNN = 0
for i in y_pred:
    if i > 0.5:
        y_pred_bin.append(1)
        numPhishesCNN += 1
    else:
        y_pred_bin.append(0)
print("Phishes: {}. Benign: {}.".format(numPhishesCNN, len(messages)-numPhishesCNN))
gmailAccuracyCNN = accuracy_score(gmailLabels, y_pred_bin)
print(gmailAccuracyCNN)

statsFilename = "/home/datasets/results/stats.csv"
if os.path.isfile(statsFilename):
    outData = {"Num Topics" : [], "TF-IDF Count" : [], "Num Training" : [], "Num Test" : [], "RFC Test Accuracy" : [], "RFC Test Recall" : [], "RFC GMail Accuracy" : [], "CNN Dim" : [], "CNN Num Layers" : [],
               "CNN Accuracy" : [], "CNN Validation Accuracy" : [], "CNN GMail Accuracy" : [] }
    outData["Num Topics"].append(numTopics)
    outData["TF-IDF Count"].append(maximumTermCount)
    outData["Num Training"].append(len(X_train))
    outData["Num Test"].append(len(X_test))
    outData["RFC Test Accuracy"].append(rfcAccuracy)
    outData["RFC Test Recall"].append(rfcRecall)
    outData["RFC GMail Accuracy"].append(gmailAccuracy)
    outData["CNN Dim"].append(embedding_dim)
    outData["CNN Num Layers"].append(num_layers)
    outData["CNN Accuracy"].append(history.history["binary_accuracy"][-1])
    outData["CNN Validation Accuracy"].append(history.history["val_binary_accuracy"][-1])
    outData["CNN GMail Accuracy"].append(gmailAccuracyCNN)
    outDataFrame = pd.DataFrame(outData)

    df = pd.read_csv(statsFilename)
    print(df)
    #df = pd.concat([df, pd.DataFrame(outData)], ignore_index=True)
    outDataFrame["ID"] = len(df)
    outDataFrame.set_index("ID")
    print(outDataFrame)
    #df.loc[len(df)] = outDataFrame
    df = pd.concat([df, pd.DataFrame(outData)], ignore_index=True)
    df.to_csv(statsFilename, index=False)
else:
    outData = {"Num Topics" : [], "TF-IDF Count" : [], "Num Training" : [], "Num Test" : [], "RFC Test Accuracy" : [], "RFC Test Recall" : [], "RFC GMail Accuracy" : [], "CNN Dim" : [], "CNN Num Layers" : [],
               "CNN Accuracy" : [], "CNN Validation Accuracy" : [], "CNN GMail Accuracy" : [] }
    outData["Num Topics"].append(numTopics)
    outData["TF-IDF Count"].append(maximumTermCount)
    outData["Num Training"].append(len(X_train))
    outData["Num Test"].append(len(X_test))
    outData["RFC Test Accuracy"].append(rfcAccuracy)
    outData["RFC Test Recall"].append(rfcRecall)
    outData["RFC GMail Accuracy"].append(gmailAccuracy)
    outData["CNN Dim"].append(embedding_dim)
    outData["CNN Num Layers"].append(num_layers)
    outData["CNN Accuracy"].append(history.history["binary_accuracy"][-1])
    outData["CNN Validation Accuracy"].append(history.history["val_binary_accuracy"][-1])
    outData["CNN GMail Accuracy"].append(gmailAccuracyCNN)
    df = pd.DataFrame(outData)
    df.to_csv(statsFilename)

