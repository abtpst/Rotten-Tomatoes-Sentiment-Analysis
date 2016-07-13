'''
Created on Sep 22, 2015

@author: atomar
'''
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re

def Sentiment_to_wordlist(Sentiment):
    '''
    Meant for converting each of the IMDB Sentiments into a list of words.
    '''
    # First remove the HTML.
    Sentiment_text = BeautifulSoup(Sentiment).get_text()

    smileys = """:-) :) :o) :] :3 :c) :> =] 8) =) :} :^)
                :D 8-D 8D x-D xD X-D XD =-D =D =-3 =3 B^D :( :/ :-( :'( :D :P""".split()
    smiley_pattern = "|".join(map(re.escape, smileys))
    
    # Use regular expressions to only include words.
    #Sentiment_text = re.sub("[^a-zA-Z]"," ", Sentiment_text)
    
    Sentiment_text = re.sub("[^a-zA-Z0-9" + smiley_pattern + "]", " ", Sentiment_text)
    # Convert words to lower case and split them into separate words.
    words = Sentiment_text.lower().split()

    stops = set(stopwords.words("english"))
    # remove stop words from the list
    words = [w for w in words if w not in stops]

    # Return a list of words
    return(words)

def review_to_words(raw_review, remove_stopwords=False, remove_numbers=False, remove_smileys=False):
    # use BeautifulSoup library to remove the HTML/XML tags (e.g., <br />)
    review_text = BeautifulSoup(raw_review).get_text()

    # emotional symbols may affect the meaning of the review
    smileys = """:-) :) :o) :] :3 :c) :> =] 8) =) :} :^)
                :D 8-D 8D x-D xD X-D XD =-D =D =-3 =3 B^D :( :/ :-( :'( :D :P""".split()
    smiley_pattern = "|".join(map(re.escape, smileys))

    # [^] matches a single character that is not contained within the brackets
    # re.sub() replace the pattern by the desired character/string
    '''
    if remove_numbers and remove_smileys:
        # any character that is not in a to z and A to Z (non text)
        review_text = re.sub("[^a-zA-Z]", " ", review_text)
    elif remove_smileys:
         # numbers are also included
        review_text = re.sub("[^a-zA-Z0-9]", " ", review_text)
    elif remove_numbers:
        review_text = re.sub("[^a-zA-Z" + smiley_pattern + "]", " ", review_text)
    else:
        review_text = re.sub("[^a-zA-Z0-9" + smiley_pattern + "]", " ", review_text)
    '''

    # split in to a list of words
    words = review_text.lower().split()

    if remove_stopwords:
        # create a set of all stop words
        stops = set(stopwords.words("english"))
        # remove stop words from the list
        words = [w for w in words if w not in stops]

    # for bag of words, return a string that is the concatenation of all the meaningful words
    # for word2Vector, return list of words
    # return " ".join(words)
               
    return words

def review_to_sentences(review, tokenizer, sentiment="",remove_stopwords=False, remove_numbers=False, remove_smileys=False):
    """
    This function splits a review into parsed sentences
    :param review:
    :param tokenizer:
    :param remove_stopwords:
    :return: sentences, list of lists
    """
    # review.strip()remove the white spaces in the review
    # use tokenizer to separate review to sentences
    raw_sentences = tokenizer.tokenize(review.strip())

    #cleaned_review = [review_to_words(sentence, remove_stopwords, remove_numbers, remove_smileys) for sentence
    #                  in raw_sentences if len(sentence) > 0]
    # generic form equals append
    cleaned_review = []
    for sentence in raw_sentences:
        
        if len(sentence) > 0:
            cleaned_review += review_to_words(sentence, remove_stopwords, remove_numbers, remove_smileys)

    if(sentiment != ""):
        cleaned_review.append(sentiment)
              
    return cleaned_review

def clean_data(data,revCol):
    """
    clean the training and test data and return a list of words
    :param data:
    :return:
    """
    # raise an error if there is no review column
    try:
        reviewsSet = data[revCol]
    except ValueError:
        print('No "review" column!')
        raise

    cleaned_data = [review_to_words(review, False, False, False) for review in reviewsSet]
    #cleaned_data = [review_to_words(review, True, True, False) for review in reviewsSet]
    # for review in reviewsSet:
    #  cleaned_data.append(review_to_words(review, True, True, False))
   
    return cleaned_data