import pandas as pd

df = pd.read_csv(r"C:\Users\Amitttt\Desktop\NLP Final Project\WikiQACorpus\Umassfaq - Sheet1.csv");
df.columns=["questions","answers"];

print(df)

#************************************************************************************************************************************************

import re
import gensim 
from gensim.parsing.preprocessing import remove_stopwords

#from nltk.stem.lancaster import LancasterStemmer
#st = LancasterStemmer()

def clean_sentence(sentence, stopwords=False):
    
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
    #sentence = re.sub(r'\s{2,}', ' ', sentence)
    
    if stopwords:
         sentence = remove_stopwords(sentence)
    
    #sent_stemmed='';
    #for word in sentence.split():
    #    sent_stemmed+=' '+st.stem(word) 
    #sentence=sent_stemmed
    
    return sentence
                    
def get_cleaned_sentences(df,stopwords=False):    
    sents=df[["questions"]];
    cleaned_sentences=[]

    for index,row in df.iterrows():
        #print(index,row)
        cleaned=clean_sentence(row["questions"],stopwords);
        cleaned_sentences.append(cleaned);
    return cleaned_sentences;

cleaned_sentences=get_cleaned_sentences(df,stopwords=True)
print(cleaned_sentences);

print("\n")

cleaned_sentences_with_stopwords=get_cleaned_sentences(df,stopwords=False)
print(cleaned_sentences_with_stopwords);

#*************************************************************************************************************************************************

import numpy

sentences=cleaned_sentences_with_stopwords
#sentences=cleaned_sentences

# Split it by white space 
sentence_words = [[word for word in document.split() ]
         for document in sentences]

from gensim import corpora

dictionary = corpora.Dictionary(sentence_words)
for key, value in dictionary.items():
    print(key, ' : ', value)

import pprint
bow_corpus = [dictionary.doc2bow(text) for text in sentence_words]
for sent,embedding in zip(sentences,bow_corpus):
    print(sent)
    print(embedding)


#question_orig="How does COVID-19 testing work"
#question_orig="How much does testing cost"
question_orig="what is the dubois library schedule"
question=clean_sentence(question_orig,stopwords=False);
question_embedding = dictionary.doc2bow(question.split())


print("\n\n",question,"\n",question_embedding)

#888888888888888888********************************************************************************************************************************

import sklearn
from sklearn.metrics.pairwise import cosine_similarity;
def retrieveAndPrintFAQAnswer(question_embedding,sentence_embeddings,FAQdf,sentences):
    max_sim=-1;
    index_sim=-1;
    for index,faq_embedding in enumerate(sentence_embeddings):
        #sim=cosine_similarity(embedding.reshape(1, -1),question_embedding.reshape(1, -1))[0][0];
        sim=cosine_similarity(faq_embedding,question_embedding)[0][0];
        print(index, sim, sentences[index])
        if sim>max_sim:
            max_sim=sim;
            index_sim=index;
       
    print("\n")
    print("Question: ",question)
    print("\n");
    print("Our output: ",FAQdf.iloc[index_sim,0]) 
    print(FAQdf.iloc[index_sim,1])        
    
retrieveAndPrintFAQAnswer(question_embedding,bow_corpus,df,sentences);

#*************************************************************************************************************************************************

from gensim.models import Word2Vec 
import gensim.downloader as api


glove_model=None;
try:
    glove_model = gensim.models.KeyedVectors.load("./glovemodel.mod")
    print("Loaded glove model")
except:            
    glove_model = api.load('glove-twitter-25')
    glove_model.save("./glovemodel.mod")
    print("Saved glove model")
    
v2w_model=None;
try:
    v2w_model = gensim.models.KeyedVectors.load("./w2vecmodel.mod")
    print("Loaded w2v model")
except:            
    v2w_model = api.load('word2vec-google-news-300')
    v2w_model.save("./w2vecmodel.mod")
    print("Saved glove model")

w2vec_embedding_size=len(v2w_model['computer']);
glove_embedding_size=len(glove_model['computer']);

#*********************************88

def getWordVec(word,model):
        samp=model['computer'];
        vec=[0]*len(samp);
        try:
                vec=model[word];
        except:
                vec=[0]*len(samp);
        return (vec)


def getPhraseEmbedding(phrase,embeddingmodel):
                       
        samp=getWordVec('computer', embeddingmodel);
        vec=numpy.array([0]*len(samp));
        den=0;
        for word in phrase.split():
            #print(word)
            den=den+1;
            vec=vec+numpy.array(getWordVec(word,embeddingmodel));
        #vec=vec/den;
        #return (vec.tolist());
        return vec.reshape(1, -1)

sent_embeddings=[];
for sent in cleaned_sentences:
    sent_embeddings.append(getPhraseEmbedding(sent,v2w_model));

question_embedding=getPhraseEmbedding(question,v2w_model);

retrieveAndPrintFAQAnswer(question_embedding,sent_embeddings,df, cleaned_sentences);

#***************************************************************************************888

#With Glove

sent_embeddings=[];
for sent in cleaned_sentences:
    sent_embeddings.append(getPhraseEmbedding(sent,glove_model));
    
question_embedding=getPhraseEmbedding(question,glove_model);

retrieveAndPrintFAQAnswer(question_embedding,sent_embeddings,df, cleaned_sentences);