import json
from tqdm import tqdm
from datasets import load_dataset
from collections import defaultdict
from collections import Counter
import math

# Get the unique token in corpus
# Corpus is a list of sentences
def get_lexicon(corpus):
  word_counts = defaultdict(int)
  for sentence in corpus:
      for word in sentence: 
          word_counts[word] += 1
  return set(word for word in word_counts if word_counts[word] > 1)

def get_ngrams(inp, n):
  """
  Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
  This should work for arbitrary values of 1 <= n < len(sequence).
  """
  if type(inp) == str:
      sequence = inp.split()
  else:
      sequence = inp.copy()
      
  end = len(sequence)
  result = []
  start = 0

  
  sequence.insert(0,'START')
  sequence.append('STOP')
  end+=2
  
  if n==1:
      return sequence
  
  else:
      while start+n<end+1:
          result.append(tuple(sequence[start:start+n]))
          start+=1
      return result
    
class TrigramModel(object):
  def __init__(self, train_set):
    self.total_words = 0
    # Iterate through the corpus once to build a lexicon 
    self.lexicon = get_lexicon(train_set)
    # self.lexicon.add("UNK")
    self.lexicon.add("START")
    self.lexicon.add("STOP")
    print("constructing unigram, bigram, trigram counts on train set...")
    self.count_ngrams(train_set)
    
  def count_ngrams(self, corpus):
    """
    Given a corpus iterator, populate dictionaries of unigram, bigram,
    and trigram counts. 
    """
    one_g = []
    two_g = []
    three_g = []
    for sequence in corpus:
        self.total_words += len(sequence)
        one_g.extend(get_ngrams(sequence,1))
        two_g.extend(get_ngrams(sequence,2))
        three_g.extend(get_ngrams(sequence,3))
        
    self.unigramcounts = Counter(one_g)
    self.bigramcounts = Counter(two_g)
    self.trigramcounts = Counter(three_g)
    print("done constructing unigram, bigram, trigram counts")
    return None
  
  def raw_trigram_probability(self,trigram):
    """
    Returns the raw (unsmoothed) trigram probability
    """
    assert len(trigram)==3, "Input should be 3 words"
    if self.bigramcounts[trigram[:2]] == 0:
        return 0
    else:
        return self.trigramcounts[trigram]/self.bigramcounts[trigram[:2]]

  def raw_bigram_probability(self, bigram):
    """
    Returns the raw (unsmoothed) bigram probability
    """
    assert len(bigram)==2, "Input should be 2 words"
    if self.unigramcounts[bigram[0]] == 0:
        return 0
    else:
        return self.bigramcounts[bigram]/self.unigramcounts[bigram[0]]
        
  def raw_unigram_probability(self, unigram):
    """
    Returns the raw (unsmoothed) unigram probability.
    """
    uni = []
    uni.append(unigram)
    assert len(uni)==1, "Input should be only 1 word"
    return self.unigramcounts[unigram]/self.total_words
  
  def smoothed_trigram_probability(self, trigram):
    """
    Returns the smoothed trigram probability (using linear interpolation). 
    """
    assert len(trigram)==3, "Input should be 3 words"
    lambda1 = 1/3.0
    lambda2 = 1/3.0
    lambda3 = 1/3.0
    u,v,w = trigram[0],trigram[1],trigram[2]
    prob =  (lambda1*self.raw_unigram_probability(w))+\
    (lambda2*self.raw_bigram_probability((v,w)))+\
    (lambda3*self.raw_trigram_probability((u,v,w)))
    return prob
  
  def sentence_logprob_perp(self, sentence):
    """
    Returns the perplexity entire sequence.
    Also we have the log probability if needed
    """
    from math import log2
    if type(sentence) == str:
        sentence = sentence.split()
    tri_g = get_ngrams(sentence,3)
    sent_prob = 0.0
    for tri_tuple in tri_g:
        sent_prob += log2(self.smoothed_trigram_probability(tri_tuple)) 

    pp = math.exp((-1 / len(sentence)) * sent_prob)   
    return pp
  
  def output_list(self, train_set, premise, hypothesis):
    final_ret=[]
    for i in tqdm(range(len(train_set))):
      sentence = train_set[i]
      pp = self.sentence_logprob_perp(sentence)
      entry = {'premise':premise[i], 'hypothesis': hypothesis[i], 'perplexity': pp}
      final_ret.append(entry)
    return final_ret
  
  def output_jsonl(self, out_name, train_set, premise, hypothesis):
    final_rest = self.output_list(train_set, premise, hypothesis)
    with open(out_name, 'w') as outfile:
      for entry in final_rest:
          json.dump(entry, outfile)
          outfile.write('\n')
    

# TODO: adding support for gsm8k dataset
# 549366
def main():
  # Load the data
  dataset = load_dataset("snli", split="train")
  # Exclude entries where label=-1, concatenate
  dataset.filter(lambda sample: sample["label"] != -1)
  
  premise = dataset['premise']
  hypothesis = dataset['hypothesis']
  train_set = [i + " " + j for i, j in zip(premise, hypothesis)]
  
  print(len(train_set))
  

  trigram_model = TrigramModel(train_set=train_set)
  trigram_model.output_jsonl('interpolated_ngram_perplexity.jsonl', train_set, premise, hypothesis)
    

if __name__ == "__main__":
    main()
  