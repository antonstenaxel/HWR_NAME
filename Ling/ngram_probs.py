from collections import defaultdict, Counter
import csv
import pprint
import operator

def print_probs(lm, history):
    probs = sorted(lm[history],key=lambda x:(-x[1],x[0]))
    pp = pprint.PrettyPrinter()
    pp.pprint(probs)
    

def gen_characters(corpus):
    with open(corpus, 'r') as f:
        data = csv.reader(f, delimiter=',')
        for line in data:
            ngram, transcr, freq = line
            yield ngram


def train_char_lm(data, order=2, add_k=1):
    ''' Trains a language model.
    This code was borrowed from 
    http://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139
    Inputs:
    f  name: Path to a text corpus.
    order: The length of the n-grams.
    add_k: k value for add-k smoothing. NOT YET IMPLMENTED
    Returns:
    A dictionary mapping from n-grams of length n to a list of tuples.
    Each tuple consists of a possible net character and its probability.
    '''

    #data = open(fname).read()
    lm = defaultdict(Counter)
    pad = "~" * order
    data = pad + data
    for i in range(len(data)-order):
        history, char = data[i:i+order], data[i+order]
        lm[history][char]+=1
  
    def normalize(counter):
        s = float(sum(counter.values()))
        return [(c,cnt/s) for c,cnt in counter.items()]
  
    outlm = {hist:normalize(chars) for hist, chars in lm.items()}
    return outlm


if __name__ == '__main__':
    characters = gen_characters('ngrams_frequencies.csv')
    data = ' '.join(characters)
    lm = train_char_lm(data, order=2)
    print_probs(lm, "אר")
  
