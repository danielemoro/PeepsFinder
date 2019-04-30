# IMPORTANT: Change the following variable to be the path of the location of the relationship extraction model
# You can find the repository here https://github.com/UKPLab/emnlp2017-relation-extraction
root = "D:/Relationship Extraction/emnlp2017-relation-extraction/"

# for parsing
from googlesearch import search
from bs4 import BeautifulSoup
from bs4 import SoupStrainer
from urllib.request import urlopen
import re, string
import html2text
import collections
from pprint import pprint
html2text = html2text.HTML2Text()
html2text.ignore_links = False

# for cleaning
import re, string
import collections
from pprint import pprint
from textblob import TextBlob
from tqdm import tqdm
import contextlib
import io
import sys
import json
import os
import time
import collections

# nltk
from nltk.tokenize import word_tokenize
import nltk.data
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

#sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#spacey
import spacy
from spacy import displacy
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")

# relationship extraction
import sys
sys.path.insert(0, root+"relation_extraction/")
from core.parser import RelParser
from core import keras_models
from core import entity_extraction
keras_models.model_params['wordembeddings'] = root+"/embeddings/glove/glove.6B.50d.txt"
relparser = RelParser("model_ContextWeighted", models_folder=root+"/trainedmodels/")

# most common words
# Source: https://github.com/first20hours/google-10000-english
def most_common_words(topn):
    with open("google-10000-english-usa.txt", 'r') as f:
        return [i.replace("\n", "") for i in f.readlines()][:topn]
common_words = most_common_words(topn=500)


class PersonSearcher():
    def __init__(self):
        self.meta_file = "./data/person_searcher_metadata.json"
        metadata = {}
        if os.path.exists(self.meta_file):
            metadata = json.load(open(self.meta_file, 'r'))
        self.metadata = metadata

    def add_search(self, name, search_term, topn=20):
        file_name = "./data/" + search_term.lower().replace(" ", "_") + ".txt"
        if name in self.metadata:
            self.metadata[name].append((file_name, time.time()))
        else:
            self.metadata[name] = [(file_name, time.time())]
        json.dump(self.metadata, open(self.meta_file, 'w'), indent=4)

        self.crawl_data(search_term, file_name, topn=topn)

    def get_most_recent_data(self, name, search=None):
        if name not in self.metadata.keys():
            return None
        most_recent = sorted(self.metadata[name], key=lambda x: x[1], reverse=True)[0]
        return most_recent[0]

    def get_html(self, url):
        try:
            return urlopen(url, timeout=5).read().decode("utf8")
        except Exception as e:
            # print(f'ERROR: Can\'t parse {url} because of {e}')
            return None

    def find_other_pages(self, url):
        links = set([url])
        response = self.get_html(url)
        if response is None:
            return links

        soup = BeautifulSoup(response, 'lxml')
        for link in soup.findAll('a', attrs={'href': re.compile("^(/)")}):  # |^("+url+")
            if link.get('href').startswith("/"):
                links.add(url[:-1] + link.get('href'))
            else:
                links.add(link.get('href'))
        return list(links)[:10]

    def parse_link(self, url, file, tags_to_search=['p']):
        # print(f'grabbing data from {url}')
        response = self.get_html(url)
        if response is None: return

        soup = BeautifulSoup(response, 'lxml')
        text_bucket = []
        for tag in soup.findAll(tags_to_search):
            # print(repr(tag.contents))
            text = html2text.handle(tag.text)
            text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
            text = re.sub(r"[^a-zA-Z0-9.!?,;:@()'’]", ' ', text)
            text_bucket.append(text.strip())

        # print(repr("\n".join(text_bucket)))
        file.write("\n===\n")
        file.write("\n".join(text_bucket))

    def crawl_data(self, search_term, file_name, topn=10):
        with open(file_name, 'w') as file:
            for url in tqdm([i for i in search(search_term, stop=topn)], "Crawling the internet for {}".format(search_term)):
                pages = self.find_other_pages(url)
                [self.parse_link(u, file) for u in pages]

class PeepsFinder:
    def  __init__(self):
        self.blacklist = ["view export record dblp key ask others share record short URL", "export record", "url",
                          "Loading","Loading...", "Loading playlists...", "Working...", "...", "id", "zfn", 'loading',
                          'javascript', 'JavaScript', 'twitter', 'tweet', 'try', 'ob', 'tweeted', 'hmm', 'twitter developer agreement']
        self.ner_types = {"PERSON": 'knows', "NORP": 'nationality', "FAC": 'workplace', "ORG": 'organization',
                          "GPE": 'country', "LOC": 'place', "PRODUCT": 'product', "EVENT": 'event',
                          "WORK_OF_ART": 'known for', "LAW": 'known for', "LANGUAGE": 'speaks',
                          "DATE": 'important date', "TIME": 'important time', "PERCENT": 'percent',
                          "MONEY": 'value', "QUANTITY": 'number', "ORDINAL": 'place', "CARDINAL": 'number'}
        self.cache = {}

    def retrieve_person_data(self, name, search=None, topn=20):
        if (name,search) in self.cache:
            return self.cache[(name,search)]
        ps = PersonSearcher()
        recent_data = ps.get_most_recent_data(name)
        if recent_data is None or search is not None:
            search = name if search is None else search
            ps.add_search(name, search, topn=topn)
            recent_data = ps.get_most_recent_data(name)
        text, clean_text, docs, sentences = self.parse_data(recent_data)

        sentences = sentences[:10000]
        docs = docs[:1000]
        text = text[:100000]
        clean_text = clean_text[:100000]

        rel_res = self.extract_relations(sentences, name)
        rel_dict = collections.defaultdict(list)
        for i in [j[1:] for i in rel_res for j in i]:
            if i[1] not in rel_dict[i[0]]:
                rel_dict[i[0]].append(i[1])
        rel_clean = [(k, v) for k in rel_dict.keys() for v in rel_dict[k]]

        emails = [i for i in self.find_emails(text)]
        phones = [i for i in self.find_phones(text)]

        named_ent = [tuple(i[0].split(': ') + [i[1]]) for i in self.grab_named_entities(clean_text)
                     if len(i[0].split(': ')[1].strip())>0]
        tfidf = list(self.tfidf_keywords(docs, topn=100))

        noun_phrases = [i[0] for i in self.grab_noun_phrases(clean_text)
                        if 50 > len(i[0]) > 0 and i[0] not in self.blacklist and i[0] not in common_words][:20]
        result = {'email': emails,
                'phone': phones,
                'rel_extr': rel_clean,
                'named_entities': named_ent,
                'noun_phrases': noun_phrases,
                'tfidf': tfidf}
        self.cache[(name,search)] = result
        return result

    def parse_data(self, file_name):
        with open(file_name, 'r') as file:
            text = file.read()
            docs = text.split("\n===\n")
            clean_text = re.sub(r"[^a-zA-Z\.']+", " ", text, 0, re.MULTILINE)
            for black in self.blacklist: clean_text = clean_text.replace(black, "")

        sentences = []
        for doc in docs:
            cleaned = re.sub(r"[^a-zA-Z\.\=']+", " ", doc, 0, re.MULTILINE)
            for black in self.blacklist: cleaned = cleaned.replace(black, "")
            curr_sents = sent_detector.tokenize(cleaned.strip())
            for s in curr_sents:
                if len(s.strip()) > 10:
                    sentences.append(s)

        return text, clean_text, docs, sentences

    def extract_relations(self, sentences, name, replace_i = True):
        def get_tagged_text(input_text):
            doc = nlp(input_text)
            result = []
            for tok in doc:
                if len(str(tok.text).strip()) == 0:
                    continue
                label = 'O'
                for ent in doc.ents:
                    if tok.idx >= ent.start_char and tok.idx <= ent.end_char:
                        label = ent.label_
                result.append((tok.text, label, tok.tag_))
            return result

        def gen_rel_graph(tagged):
            entity_fragments = entity_extraction.extract_entities(tagged)
            edges = entity_extraction.generate_edges(entity_fragments)
            non_parsed_graph = {'tokens': [t for t, _, _ in tagged],
                                'edgeSet': edges}
            parsed_graph = relparser.classify_graph_relations([non_parsed_graph])
            return parsed_graph

        def clean_graph(parsed_graph):
            if parsed_graph is None:
                return
            results = []
            tokens = parsed_graph[0]['tokens']
            for edge in parsed_graph[0]['edgeSet']:
                if edge['lexicalInput'] == "ALL_ZERO": continue
                left = " ".join([tokens[i] for i in edge['left']])
                right = " ".join([tokens[i] for i in edge['right']])
                if left.lower().strip() in names or right.lower().strip() in names:
                    results.append((left, edge['lexicalInput'], right))
            return results

        names = name.lower().strip().split(" ") + [name.lower().strip()]
        #print(names)
        results = []
        for s in tqdm(sentences, desc="Extracting relationships"):
            if replace_i:
                s = s.replace("I ", names[0].capitalize() + " ")
            tagged = get_tagged_text(s)
            parsed_graph = gen_rel_graph(tagged)
            rel = clean_graph(parsed_graph)
            if rel is not None and len(rel) > 0:
                results.append(rel)
        return results

    def grab_named_entities(self, text, topn=30):
        doc = nlp(text)
        ne = collections.defaultdict(lambda: 0)
        for ent in doc.ents:
            ne[self.ner_types[ent.label_] + ': ' + ent.text] += 1
        ne = sorted([(k, v) for k, v in ne.items()], key=lambda x: x[1], reverse=True)
        return ne[:topn]

    def grab_noun_phrases(self, text):
        blob = TextBlob(text)
        np = collections.defaultdict(lambda: 0)
        for e in blob.noun_phrases:
            np[e] += 1
        np = sorted([(k, v) for k, v in np.items()], key=lambda x: x[1], reverse=True)
        return np

    # Source: https://medium.freecodecamp.org/how-to-extract-keywords-from-text-with-tf-idf-and-pythons-scikit-learn-b2a0f3d7e667
    def tfidf_keywords(self, docs, topn=50):
        cv = CountVectorizer()
        word_count_vector = cv.fit_transform(docs)
        tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        tfidf_transformer.fit(word_count_vector)

        feature_names = cv.get_feature_names()
        tf_idf_vector = tfidf_transformer.transform(cv.transform([" ".join(docs)]))
        tuples = zip(tf_idf_vector.tocoo().col, tf_idf_vector.tocoo().data)
        sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

        sorted_items = sorted_items[:topn]
        score_vals = []
        feature_vals = []
        for idx, score in sorted_items:
            score_vals.append(round(score, 5))
            feature_vals.append(feature_names[idx])

        results = {}
        for idx in range(len(feature_vals)):
            if feature_vals[idx] not in common_words + [i.lower().strip().split() for i in self.blacklist]:
                results[feature_vals[idx]] = score_vals[idx]
        return results

    # Source: https://www.regextester.com/19
    def find_emails(self, text):
        return list(re.findall(r"[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*", text))

    # Source: http://regexlib.com/REDetails.aspx?regexp_id=45
    def find_phones(self, text):
        return ["-".join(i) for i in set(re.findall(r"\D?(\d{3})\D?\D?(\d{3})\D?(\d{4})", text))]
