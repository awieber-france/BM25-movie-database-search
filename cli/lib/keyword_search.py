from lib.search_utils import (
    load_database,
    load_stopwords,
    DEFAULT_SEARCH_LIMIT,
    CACHE_PATH,
    BM25_K1,
    BM25_B)
from pathlib import Path
import os
import math
import pickle
import string
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
#from nltk.stem.wordnet import WordNetLemmatizer
#lmtzr = WordNetLemmatizer()
stop = load_stopwords()

# ---------------------
# SEARCH FUNCTIONS CALLED BY KEYWORD SEARCH CLI
# ---------------------
def search_command(query: str, limit: int | None = DEFAULT_SEARCH_LIMIT) -> list[str]:
    # Format the query
    query = preprocess_text(query)
    query = stem_and_tokenize(query)
    # Import word index and docmap of documents
    docmap, index, term_frequencies, doc_lengths = InvertedIndex().load()
    matches = set()
    for word in query:
        ids = index.get(word)
        if ids is not None:
            matches = matches | set(ids)
    # Order the results
    matches = sorted(matches, reverse=False)
    # Retrieve top results (exit early if no matches)
    top_results = []
    if len(matches) <= 0:
        return top_results
    for i, id in enumerate(matches):
        if i == DEFAULT_SEARCH_LIMIT:
            break
        top_results.append(docmap.get(id))
    return top_results

def bm25search_command(query: str, **kwargs) -> float:
    k1 = kwargs.get('k1', BM25_K1)
    b = kwargs.get('b', BM25_B)
    idx = InvertedIndex()
    top_bm25_scores = idx.get_bm25_score(query, k1, b)
    return top_bm25_scores

# ---------------------
# OTHER COMMANDS CALLED BY KEYWORD SEARCH CLI
# ---------------------

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
    # Test code
    id_test = idx.get_documents('merida')
    if len(id_test) > 0:
        print(f"First document for token 'merida' = {id_test[0]}")

def tf_command(doc_id: int, term: str) -> None:
    idx = InvertedIndex()
    tf = idx.get_tf(doc_id, term)
    print(f"Frequency of '{term}' in document {doc_id}: {tf}")

def idf_command(term: str) -> None:
    idx = InvertedIndex()
    idf = idx.get_idf(term)
    #print('{0:.2f}'.format(idf))
    print(f"IDF score of '{term}' in index: {idf:.2f}")

def tfidf_command(doc_id: int, term: str) -> None:
    idx = InvertedIndex()
    tf = idx.get_tf(doc_id, term)
    idf = idx.get_idf(term)
    tf_idf = tf * idf
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")

def bm25_idf_command(term: str) -> None:
    idx = InvertedIndex()
    bm25idf = idx.get_bm25_idf(term)
    print(f"BM25 IDF score of '{term}': {bm25idf:.2f}")

def bm25_tf_command(doc_id: int, term: str, **kwargs) -> None:
    k1 = kwargs.get('k1', BM25_K1)
    b = kwargs.get('b', BM25_B)
    idx = InvertedIndex()
    bm25_tf = idx.get_bm25_tf(doc_id, term, k1)
    print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25_tf:.2f}")

# ---------------------
# TEXT PREPROCESSING FUNCTIONS
# ---------------------

def preprocess_text(text: str) -> str:
    # Set to lower case
    text = text.lower()
    # Remove punctuation via a translation table 
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    return text

def stem_and_tokenize(text: str) -> list[str]:
    # Split into word tokens
    #word_list = text.split(' ')
    word_list = text.split()
    word_list = [word for word in word_list if word is not None]
    # Remove stopwords
    word_list = [word for word in word_list if word not in stop]
    # Stem words
    stemmer = PorterStemmer()
    word_list = [stemmer.stem(word) for word in word_list]
    # Lematize words
    #word_list = [lmtzr.lemmatize(word, 'v') for word in word_list]
    return word_list

# ---------------------
# INVERTED INDEX CLASS IMPLEMENTATION
# ---------------------
class InvertedIndex():
    def __init__(self):
        self.document_list = load_database()
        self.cache_path = CACHE_PATH
        self.index_path = self.cache_path / "index.pkl"
        self.docmap_path = self.cache_path / "docmap.pkl"
        self.tf_path = self.cache_path / "term_frequencies.pkl"
        self.doc_lengths_path = self.cache_path / "doc_lengths.pkl"
        try:
            self.docmap, self.index, self.term_frequencies, self.doc_lengths = self.load()
            print("Loading previously built cache")
        except:
            self.docmap = dict() # a dictionary mapping document IDs to their full document objects
            self.index = dict() # a dictionary mapping tokens (strings) to sets of document IDs (integers)
            self.term_frequencies = dict() # a dictionary mapping document IDs to word counter objects
            self.doc_lengths = dict()
            print("Index must be built before use. Use the build command.")

    #Tokenize the input text, then add each token to the index with the document ID.
    def __add_document(self, doc_id, text):
        # Tokenize input text
        text = self.__preprocessing(text)
        # Count nb of occurrences of each word in the document
        self.term_frequencies[doc_id] = Counter(text)
        # Create word index (each time a token appears, add the document id)
        # Format key: value is token: (doc_id, doc_id, doc_id, ...)
        for token in text:
            if token not in self.index.keys():
                self.index[str(token)] = set()
            self.index[token] = self.index[token] | {doc_id}
    
    # Use external functions to preprocess text: lower case, remove punctuation, stem, and tokenize
    def __preprocessing(self, text):
        text = preprocess_text(text)
        text = stem_and_tokenize(text)
        return text
    
    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        else:
            return sum(self.doc_lengths.values())/len(self.doc_lengths)

    # Iterate over all the documents and add them to both the index and the docmap
    def build(self):
        print("Building new index....")
        for m in self.document_list:
            doc_id = m["id"]
            text = f"{m['title']} {m['description']}"
            self.__add_document(doc_id, text) # adds words to the index
            self.docmap[doc_id] = m["title"]
        self.doc_lengths = {doc_id: sum(self.term_frequencies.get(doc_id).values()) for doc_id in self.term_frequencies.keys()}
        print("Done")

    #  Cache the index files to disk
    def save(self):
        print("Saving new index files to cache...")
        if not os.path.isdir(self.cache_path):
            os.mkdir(self.cache_path)
        with open(self.docmap_path, "wb") as handle:
            pickle.dump(self.docmap, handle)
        with open(self.index_path, "wb") as handle:
            pickle.dump(self.index, handle)
        with open(self.tf_path, "wb") as handle:
            pickle.dump(self.term_frequencies, handle)
        with open(self.doc_lengths_path, "wb") as handle:
            pickle.dump(self.doc_lengths, handle)
        print("Done")
    
    #  Load the index caches from disk
    def load(self):
        try:
            with open(self.docmap_path, "rb") as handle:
                docmap = pickle.load(handle)
        except:
            raise FileExistsError()
        try:
            with open(self.index_path, "rb") as handle:
                index = pickle.load(handle)
        except:
            raise FileExistsError()
        try:
            with open(self.tf_path, "rb") as handle:
                term_frequencies = pickle.load(handle)
        except:
            raise FileExistsError()
        try:
            with open(self.doc_lengths_path, "rb") as handle:
                doc_lengths = pickle.load(handle)
        except:
            FileExistsError()
        return docmap, index, term_frequencies, doc_lengths

    # Retrieve list of document ids sorted in increasing order
    def get_documents(self, term):
        return sorted(list(self.index.get(term.lower())))
    
    # Retrieve a document title by its document id
    def get_document(self, doc_id: int):
        return self.docmap.get(doc_id)
    
    # Basic term frequency for a document
    def get_tf(self, doc_id: int, term: str, preprocessed: bool | None = False) -> int:
        if not preprocessed:
            term: list[str] = self.__preprocessing(term)
            if len(term) > 1:
                raise Exception("Error: more than one term was supplied in term frequency request.")
            term = term[0] # Convert to text
        if len(term) <= 0:
            raise Exception("Error: no term was supplied in term frequency request.")
        else:
            return self.term_frequencies.get(doc_id)[term]
    
    # Basic inverse document frequency for a term
    def get_idf(self, term: str, preprocessed: bool | None = False) -> float:
        if not preprocessed:
            term: list[str] = self.__preprocessing(term)
            if len(term) > 1:
                raise Exception("Error: more than one term was supplied in inverse document frequency request.")
            term = term[0] # Convert to text
        if len(term) <= 0:
            raise Exception("Error: no term was supplied in inverse document frequency request.")
        # N = total number of documents
        # df = document frequency
        N = len(self.docmap)
        term_docs = self.index.get(term) #document ids containing the term
        if term_docs is None:
            df = 0
        else:
            df = len(term_docs)
        score = math.log((N + 1) / (df + 1))
        return score

    # BM25 term frequency for a document
    def get_bm25_tf(self, doc_id: int, term: str, k1: float | None = BM25_K1, b: float | None = BM25_B, preprocessed: bool | None = False) -> float:
        if not preprocessed:
            term: list[str] = self.__preprocessing(term)
            if len(term) > 1:
                raise Exception("Error: more than one term was supplied in bm25 term frequency request.")
            term = term[0]
        if len(term) <= 0:
            raise Exception("Error: no term was supplied in bm25 term frequency request.")
        doc_length : int =  self.doc_lengths.get(doc_id)
        avg_doc_length : float = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        tf = self.get_tf(doc_id, term)
        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm) # term saturation formula
        return tf_component
    
    # BM25 inverse document frequency for a term
    def get_bm25_idf(self, term: str, preprocessed: bool | None = False) -> float:
        if not preprocessed:
            term: list[str] = self.__preprocessing(term)
            if len(term) > 1:
                raise Exception("Error: more than one term was supplied in inverse document frequency request.")
            term = term[0]
        if len(term) <= 0:
            raise Exception("Error: no term was supplied in inverse document frequency request.")
        # N = total number of documents
        # df = document frequency
        N = len(self.docmap)
        term_docs = self.index.get(term) #document ids containing the term
        if term_docs is None:
            df = 0
        else:
            df = len(term_docs)
        idf_component = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return idf_component
    
    # Full BM25 score
    def get_bm25_score(self, query: str, k1: float | None = BM25_K1, b: float | None = BM25_B) -> float:
        # NOTA: tokenization performed within get_bm25_tf & get_bm25_idf
        bm25_scores = dict()
        query_terms = self.__preprocessing(query)
        for doc_id in self.docmap.keys():
            bm25_scores[doc_id] = sum([self.get_bm25_tf(doc_id, term, k1, b, preprocessed=True) * self.get_bm25_idf(term, preprocessed=True) for term in query_terms])
        # find highest scores
        bm25_sorted = sorted(bm25_scores.items(), key=lambda item: item[1], reverse=True)
        top_5_titles = [(doc_id, self.docmap.get(doc_id), score) for doc_id, score in bm25_sorted[0:DEFAULT_SEARCH_LIMIT]]
        return top_5_titles
    
# ---------------------
# TEST CODE
# ---------------------
"""
# Check movie importation
movie_list = load_database()
# Create a new inverted index
inv_index = InvertedIndex()
result = inv_index.get_documents("bear")
result
inv_index.get_document(5)
inv_index.build()
inv_index.save()
# Import inverted index from cache
docmap, index, term_frequencies, doc_lengths = InvertedIndex().load()
print(term_frequencies.get(5)['turkey'])
# Test basic search command
result = search_command('magic charlie')
print(result)
result = search_command('nonsensetoken assault')
print(result)
# Test basic calculations of BM25
tf_command(424, 'bear')
idf_command('grizzly')
bm25_tf_command(1, 'police')
# Test final BM25 score
result = inv_index.get_bm25_score('The Adventures of the Galaxy Rangers')
print(result)
"""