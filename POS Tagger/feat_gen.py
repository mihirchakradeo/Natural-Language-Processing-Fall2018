#!/bin/python
from nltk.stem import PorterStemmer
import inflect
import string

inflect = inflect.engine()
stemmer = PorterStemmer()
possible_dt = set(['the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'our', 'their'])
brown_cluster_dict = {}

def preprocess_corpus(train_sents):
    """Use the sentences to do whatever preprocessing you think is suitable,
    such as counts, keeping track of rare features/words to remove, matches to lexicons,
    loading files, and so on. Avoid doing any of this in token2features, since
    that will be called on every token of every sentence.

    Of course, this is an optional function.

    Note that you can also call token2features here to aggregate feature counts, etc.
    """
    
    # Uncomment to enable brown clustering
    # make sure the parsed_sorted.txt file is present

    # Reading brown cluster features in a dictionary
    # Dictionary format: {word: bitstring}
    brown_file = "parsed_sorted.txt"
    with open(brown_file) as f:
        line = f.read().splitlines()
        for item in line:
            word, bs, _ = item.split("\t")
            brown_cluster_dict[word] = int(bs, 2)

    pass

def token2features(sent, i, add_neighs = True):
    """Compute the features of a token.

    All the features are boolean, i.e. they appear or they do not. For the token,
    you have to return a set of strings that represent the features that *fire*
    for the token. See the code below.

    The token is at position i, and the rest of the sentence is provided as well.
    Try to make this efficient, since it is called on every token.

    One thing to note is that it is only called once per token, i.e. we do not call
    this function in the inner loops of training. So if your training is slow, it's
    not because of how long it's taking to run this code. That said, if your number
    of features is quite large, that will cause slowdowns for sure.

    add_neighs is a parameter that allows us to use this function itself in order to
    recursively add the same features, as computed for the neighbors. Of course, we do
    not want to recurse on the neighbors again, and then it is set to False (see code).
    """
    ftrs = []
    # bias
    ftrs.append("BIAS")
    # position features
    if i == 0:
        ftrs.append("SENT_BEGIN")
    if i == len(sent)-1:
        ftrs.append("SENT_END")

    # the word itself
    word = unicode(sent[i])
    # word = stemmer.stem(word)
    ftrs.append("WORD=" + word)
    ftrs.append("LCASE=" + word.lower())

    # some features of the word
    if word.isalnum():
        ftrs.append("IS_ALNUM")
    if word.isnumeric():
        ftrs.append("IS_NUMERIC")
    if word.isdigit():
        ftrs.append("IS_DIGIT")
    if word.isupper():
        ftrs.append("IS_UPPER")
    if word.islower():
        ftrs.append("IS_LOWER")

    # CUSTOM FEATURES
    
    # 1 check singular and plural
    if inflect.singular_noun(word.lower()) is False:
        ftrs.append("IS_SINGULAR")

    # 2 check for punctuations
    if word.lower() in string.punctuation:
        ftrs.append("IS_PUNCTUATION")

    # 3 check for # or @
    if "#" in word or "@" in word or "RT" or "rt" in word:
        ftrs.append("IS_X")

    # 4 check adverb
    if word[-2:].lower() == "ly":
        ftrs.append("IS_ADVERB")

    # 5 first caps
    if word[0].isupper():
        ftrs.append("IS_1_UPPER")

    # 6 has hyphen
    if "-" in word:
        ftrs.append("HAS_HYPHEN")

    # 7 possible adj
    if word.lower().endswith("able") or word.lower().endswith("ible") or word.lower().endswith("ent") or word.lower().endswith("er") or word.lower().endswith("ous") or word.lower().endswith("est"):
        ftrs.append("IS_ADJECTIVE") 

    # 8 possible verbs
    if word.lower().endswith("ing") or word.lower().endswith("ed"):
        ftrs.append("IS_VERB")

    # 9 possible determiners
    if word.lower() in possible_dt:
        ftrs.append("IS_DT")


    # brown clustering, uncomment to enable
    if word.lower() in brown_cluster_dict:
        cluster_id = brown_cluster_dict[word.lower()]
        feat = "IS_CLUSTER_"+`cluster_id`
        ftrs.append(feat)

    # previous/next word feats
    if add_neighs:
        if i > 0:
            for pf in token2features(sent, i-1, add_neighs = False):
                ftrs.append("PREV_" + pf)
        if i < len(sent)-1:
            for pf in token2features(sent, i+1, add_neighs = False):
                ftrs.append("NEXT_" + pf)

    # return it!
    return ftrs

if __name__ == "__main__":
    sents = [
    [ "I", "love", "food" ]
    ]
    preprocess_corpus(sents)
    for sent in sents:
        for i in xrange(len(sent)):
            print sent[i], ":", token2features(sent, i)
