
# coding: utf-8

# In[2]:


from nltk.tokenize import sent_tokenize, word_tokenize


# In[3]:


SAMPLE_TEXT = "This is an example text? Let's see how it works."


# In[4]:


sent_tokenize(SAMPLE_TEXT)


# In[11]:


words = word_tokenize(SAMPLE_TEXT)


# In[6]:


from nltk.corpus import stopwords


# In[7]:


stops = set(stopwords.words('english'))


# In[10]:


stops


# In[19]:


lower_text = SAMPLE_TEXT.lower()
lower_text = word_tokenize(lower_text)


# In[20]:


clean_wrods = [w for w in lower_text if not w in stops]


# In[21]:


clean_wrods


# stem_words

# In[22]:


stem_words = ["play", "played", "playing", "player"]
from nltk.stem import  PorterStemmer
ps = PorterStemmer()
for w in stem_words:
    print(ps.stem(w))


# In[23]:


words = ["imperfect", "perfect"]
ps = PorterStemmer()
for w in words:
    print(ps.stem(w))


# In[24]:


from nltk import pos_tag
from nltk.corpus import state_union


# In[25]:


text = state_union.raw("2006-GWBush.txt")


# In[26]:


text


# In[27]:


pos = pos_tag(word_tokenize(text))


# In[28]:


pos[1:50]

POS tag list:

CC	coordinating conjunction
CD	cardinal digit
DT	determiner
EX	existential there (like: "there is" ... think of it like "there exists")
FW	foreign word
IN	preposition/subordinating conjunction
JJ	adjective	'big'
JJR	adjective, comparative	'bigger'
JJS	adjective, superlative	'biggest'
LS	list marker	1)
MD	modal	could, will
NN	noun, singular 'desk'
NNS	noun plural	'desks'
NNP	proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	predeterminer	'all the kids'
POS	possessive ending	parent's
PRP	personal pronoun	I, he, she
PRP$	possessive pronoun	my, his, hers
RB	adverb	very, silently,
RBR	adverb, comparative	better
RBS	adverb, superlative	best
RP	particle	give up
TO	to	go 'to' the store.
UH	interjection	errrrrrrrm
VB	verb, base form	take
VBD	verb, past tense	took
VBG	verb, gerund/present participle	taking
VBN	verb, past participle	taken
VBP	verb, sing. present, non-3d	take
VBZ	verb, 3rd person sing. present	takes
WDT	wh-determiner	which
WP	wh-pronoun	who, what
WP$	possessive wh-pronoun	whose
WRB	wh-abverb	where, when
# In[29]:


from nltk.stem import WordNetLemmatizer


# In[30]:


lemmatizer = WordNetLemmatizer()


# In[31]:


lemmatizer.lemmatize("good", pos = "a")


# In[32]:


lemmatizer.lemmatize("better", pos = "a")


# In[34]:


lemmatizer.lemmatize("excellent", pos = "n")


# In[35]:


lemmatizer.lemmatize("painting", pos = "v")


# In[36]:


lemmatizer.lemmatize("painting", pos = "n")


# In[37]:


lemmatizer.lemmatize("studies", pos = "v")

lemmatizer.lemmatize("studies", pos = "n")
# f
# 

# In[39]:


from nltk.corpus import wordnet


# In[41]:


syns = wordnet.synsets("can")
syns


# In[42]:


syns = wordnet.synsets("regard")
syns


# In[43]:


syns[4].definition()


# In[44]:


syns[2].examples()


# In[45]:


syns[5].examples()


# In[46]:


w1 = wordnet.synset('boat.n.01')
w2 = wordnet.synset('car.n.01')
print(w1.wup_similarity(w2))


# In[47]:


w1 = wordnet.synset('boat.n.01')
w2 = wordnet.synset('ship.n.01')
print(w1.wup_similarity(w2))


# In[48]:


from nltk import ne_chunk


# In[51]:


sentence = "Sergey Brin, the manager of Google Inc. is walking down the street in London."
sentence =  sentence.lower()


# In[52]:


converted = ne_chunk(pos_tag(word_tokenize(sentence)))
converted

ORGANIZATION - Georgia-Pacific Corp., WHO
PERSON - Eddy Bonte, President Obama
LOCATION - Murray River, Mount Everest
DATE - June, 2008-06-29
TIME - two fifty a m, 1:30 p.m.
MONEY - 175 million Canadian Dollars, GBP 10.40
PERCENT - twenty pct, 18.75 %
FACILITY - Washington Monument, Stonehenge
GPE - South East Asia, Midlothian
# In[55]:


from nltk.corpus import movie_reviews


# movie_reviews.categories()

# In[60]:


movie_reviews.words(movie_reviews.fileids()[5])


# In[62]:


documents = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((movie_reviews.words(fileid), category))


# In[63]:


len(documents)


# In[65]:


documents[2][1]


# In[66]:


import random
random.shuffle(documents)


# In[74]:


all_words = []
stops.update(".", "?", "(", ")", ",", "-", "'", '"')
for w in movie_reviews.words():
    if w.lower() not in stops:
        all_words.append(w.lower())


# In[75]:


len(all_words)


# In[76]:


import nltk


# In[77]:


all_words = nltk.FreqDist(all_words)
all_words.most_common(15)


# In[86]:


word_features = list(all_words.most_common(3000)[:,0])


# In[90]:


word_features = []
most_common = all_words.most_common()


# In[91]:


for i in range(3000):
    word_features.append(most_common[i][0])


# In[92]:


len(word_features)


# In[93]:


word_features[1:10]


# In[94]:


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


# In[96]:


features = find_features(movie_reviews.words(movie_reviews.fileids()[2]))


# In[98]:


featuresets = [(find_features(doc), category) for (doc,category) in documents]


# In[99]:


featuresets[0]


# In[100]:


training_set = featuresets[:1500]
testing_set = featuresets[1500:]


# In[102]:


classifier = nltk.NaiveBayesClassifier.train(training_set)


# In[104]:


nltk.classify.accuracy(classifier, testing_set)


# In[105]:


classifier.show_most_informative_features(15)


# In[106]:


from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.svm import SVC


# In[108]:


SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
nltk.classify.accuracy(SVC_classifier, testing_set)


# In[109]:


import pickle


# In[110]:


save_classifier = open("naivebayes.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()


# In[111]:


classifier_new = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_new)
classifier_new.close()


# In[112]:


train_set = ("The sky is blue.", "The sun is bright")


# In[113]:


from sklearn.feature_extraction.text import CountVectorizer


# In[114]:


count_vec = CountVectorizer()


# In[116]:


count_vec.fit_transform(train_set)


# In[117]:


count_vec.get_feature_names()


# In[120]:


test_sen = ("The sky is bright and blue", "Bright day")


# In[121]:


freq_term = count_vec.transform(test_sen)


# In[122]:


freq_term.todense()


# In[123]:


reviews = []
categories = []
for (rev, category) in documents:
    reviews.append(rev)
    categories.append(category)


# In[124]:


from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


# In[125]:


lemmatizer = WordNetLemmatizer()


# In[133]:


def get_simple_pos(tag):
    
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# In[134]:


def clean_data(data):
    meaningful_words = []
    for w in data:
        if w.lower() not in stops:
            pos = pos_tag([w])
           # print(pos)
         #  if get_simple_tag(pos)
            n_word = lemmatizer.lemmatize(w, pos = get_simple_pos(pos[0][1]))
            meaningful_words.append(n_word.lower())
    return (" ".join(meaningful_words))


# In[135]:


clean_review = clean_data(reviews[0])


# In[136]:


clean_review


# In[137]:


clean_reviews = [clean_data(review) for review in reviews]


# In[138]:


from sklearn.cross_validation import train_test_split


# In[139]:


x_train, x_test, y_train, y_test = train_test_split(clean_reviews, categories, test_size = 0.2)


# In[140]:


count_vect = CountVectorizer(analyzer = "word", max_features = 4500)


# In[142]:


train_transformed = count_vect.fit_transform(x_train)


# In[144]:


train_transformed.shape


# In[145]:


test_transformed = count_vect.transform(x_test)

