import pandas as pd

df = pd.read_csv(r"C:\Users\Amitttt\Desktop\NLP Final Project\WikiQACorpus\Umassfaq - Sheet1.csv");
df.columns=["questions","answers"];

print(df)

#************************************************************************************************************************************************

import re #regular expressions
import gensim 
from gensim.parsing.preprocessing import remove_stopwords

def clean_sentence(sentence, stopwords=False):
    
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence) #https://docs.python.org/2/library/re.html
    
    if stopwords:
         sentence = remove_stopwords(sentence) #https://radimrehurek.com/gensim/parsing/preprocessing.html
    return sentence
                    
def get_cleaned_sentences(df,stopwords=False):    
    sents=df[["questions"]];
    cleaned_sentences=[]

    for index,row in df.iterrows():
        cleaned=clean_sentence(row["questions"],stopwords);
        cleaned_sentences.append(cleaned);
    return cleaned_sentences;

cleaned_sentences=get_cleaned_sentences(df,stopwords=True)  #sentences without stopwords
print(cleaned_sentences);

print("\n")
print("\n")
print("\n")

cleaned_sentences_with_stopwords=get_cleaned_sentences(df,stopwords=False) #sentences with stopwords
print(cleaned_sentences_with_stopwords);

#*************************************************************************************************************************************************

import numpy


#using sentences with stopwords to better understand the meaning of the question asked and compare to our dataset.
#I have tried using sentences without stopwords and we got better results with stopwords.



sentences=cleaned_sentences_with_stopwords
#sentences=cleaned_sentences

# Split it by white space 
sentence_words = [[word for word in document.split() ]
         for document in sentences]

print("\n")
print("\n")
print("\n")
print("\n")
print("\n")
print(sentence_words) # we get the questions as a list of lists with each word as an element in the list and words are split based on whitespace


from gensim import corpora #https://radimrehurek.com/gensim/auto_examples/core/run_corpora_and_vector_spaces.html

dictionary = corpora.Dictionary(sentence_words)
print("\n")
print("\n")
print("\n")
print("\n")
print(dictionary)
print("\n")
print("\n")
print("\n")
print("\n")
print("\n")
for key, value in dictionary.items():
    #print("\n")
    print(key, ' : ', value) #assigning serial number to tokens to all unique words(tokens stored in the dictionary) 

import pprint #pretty print
bow_corpus = [dictionary.doc2bow(text) for text in sentence_words] #The function doc2bow() simply counts the number of occurrences of each distinct word, 
for sent,embedding in zip(sentences,bow_corpus):                   #converts the word to its integer word id and returns the result as a sparse vector.
    print(sent)
    print(embedding)



#question_orig="How does COVID-19 testing work"
#question_orig="How much does testing cost"
question_orig="what is the dubois library schedule"
question=clean_sentence(question_orig,stopwords=False);
question_embedding = dictionary.doc2bow(question.split())


print("\n\n",question,"\n",question_embedding)
