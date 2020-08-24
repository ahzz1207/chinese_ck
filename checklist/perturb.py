import numpy as np
import collections
import re
import os
import json
import pattern
import requests
import random
import execjs
import sys
cur_folder = os.path.dirname(__file__)
sys.path.append(os.path.join(cur_folder, 'data', 'Generator'))
from pattern.en import tenses
from .editor import recursive_apply, MunchWithAdd


class HttpRequest(object):
    """不记录任何的请求方法"""

    @classmethod
    def request(cls, method, url, data=None, headers=None): # 这里是要传入的参数，请求方法、接口地址、传参、头文件
        method = method.upper() # 这里将传入的请求方法统一大写，然后进行判断采用什么方法
        if method == 'POST':
            return requests.post(url=url, data=data, headers=headers)
        elif method == 'GET':
            return requests.get(url=url, params=data, headers=headers)


class HttpSession(object):
    """记录Session的方法"""
    def __init__(self):
        self.session = requests.session() # 初始化一个保存session的方法

    def request(self, method, url, data=None, headers=None):
        method = method.upper()
        if method == 'POST':
            return self.session.post(url=url, data=data, headers=headers)
        elif method == 'GET':
            return self.session.get(url=url, params=data, headers=headers)

    def close(self):
        """断开session连接的方法"""
        self.session.close()


def load_data():
    cur_folder = os.path.dirname(__file__)
    basic = json.load(open(os.path.join(cur_folder, 'data', 'lexicons', 'basic_zh.json')))
    names = json.load(open(os.path.join(cur_folder, 'data', 'names_zh.json')))
    intent = json.load(open(os.path.join(cur_folder, 'data', 'lexicons', 'intent_zh.json')))
    name_set = { x:set(names[x]) for x in names }
    pos2neg = json.load(open(os.path.join(cur_folder, 'data', 'pos2neg.json')))
    neg2pos = json.load(open(os.path.join(cur_folder, 'data', 'neg2pos.json')))
    contrac2full = json.load(open(os.path.join(cur_folder, 'data', 'contrac2full.json')))
    full2contrac = json.load(open(os.path.join(cur_folder, 'data', 'full2contrac.json')))
    date = json.load(open(os.path.join(cur_folder, 'data', 'lexicons', 'date.json')))
    data = {
        'name': names,
        'contrac2full': contrac2full,
        'full2contrac': full2contrac,
        'neg2pos': neg2pos,
        'pos2neg': pos2neg,
        'date': date,
    }
    # 更新字典
    data.update(basic)
    data = dict(data, **intent)
    return data


def process_ret(ret, ret_m=None, meta=False, n=10):
    if ret:
        if len(ret) > n:
            idxs = np.random.choice(len(ret), n, replace=False)
            ret = [ret[i] for i in idxs]
            if ret_m:
                ret_m = [ret_m[i] for i in idxs]
        if meta:
            ret = (ret, ret_m)
        return ret
    return None

class Perturb:
    data = load_data()

    @staticmethod
    def perturb(data, perturb_fn, keep_original=True, nsamples=None, *args, **kwargs):
        """Perturbs data according to some function

        Parameters
        ----------
        data : list
            List of examples, could be strings, tuples, dicts, spacy docs, whatever
        perturb_fn : function
            Arguments: (example, *args, **kwargs)
            Returns: list of examples, or (examples, meta) if meta=True in **kwargs.
            Can also return None if perturbation does not apply, and it will be ignored.
        keep_original : bool
            if True, include original example (from data) in output
        nsamples : int
            number of examples in data to perturb
        meta : bool
            if True, perturb_fn returns (examples, meta), and meta is added to ret.meta

        Returns
        -------
        MunchWithAdd
            will have .data and .meta (if meta=True in **kwargs)

        """
        ret = MunchWithAdd()
        use_meta = kwargs.get('meta', False)
        ret_data = []
        meta = []
        order = list(range(len(data)))
        samples = 0
        if nsamples:
            np.random.shuffle(order)
        for i in order:
            d = data[i]
            t = []
            add = []
            if keep_original:
                org = recursive_apply(d, str)
                t.append(org)
                add.append(None)
            p = perturb_fn(d, *args, **kwargs)
            a = []
            x = []
            if not p or all([not x for x in p]):
                continue
            if use_meta:
                p, a = p
            if type(p) in [np.array, list]:
                t.extend(p)
                add.extend(a)
            else:
                t.append(p)
                add.append(a)
            ret_data.append(t)
            meta.append(add)
            samples += 1
            if nsamples and samples == nsamples:
                break
        ret.data = ret_data
        if use_meta:
            ret.meta = meta
        return ret

    @staticmethod
    def strip_punctuation(doc):
        """Removes punctuation

        Parameters
        ----------
        doc : spacy.tokens.Doc
            spacy doc

        Returns
        -------
        string
            With punctuation stripped

        """
        # doc is a spacy doc
        while len(doc) and doc[-1].pos_ == 'PUNCT':
            doc = doc[:-1]
        return doc.text

    @staticmethod
    def punctuation(doc):
        """Perturbation function which adds / removes punctuations

        Parameters
        ----------
        doc : spacy.tokens.Doc
            spacy doc

        Returns
        -------
        list(string)
            With punctuation removed and / or final stop added.

        """
        # doc is a spacy doc
        s = Perturb.strip_punctuation(doc)
        ret = []
        if s != doc.text:
            ret.append(s)
        if s + '.' != doc.text:
            ret.append(s + '.')
        return ret

    @staticmethod
    def trans_num(sentence, mode):
        """Perturbation function which adds / removes punctuations
        
        Parameters
        ----------
        doc : spacy.tokens.Doc
            spacy doc
        number: numbert to be transferd
        mode: 'number' or 'money'
        
        Returns
        -------
        list(string)
            With digits transferd to number or money.

        """
        rex = re.compile(r"([1-9]\d*\.?\d*)|(0\.\d*[1-9])")
        ret = []
        numbers = rex.findall(sentence)
        cur_folder = os.path.dirname(__file__)
        js = execjs.compile(open(os.path.join(cur_folder, 'data', 'Generator', 'index.js')).read())
        if numbers:
            for number in numbers:
                ret = js.call('transNumber', number[0], mode)
                print(ret, number)
                sentence = sentence.replace(number[0], ret)
            return sentence
        else:
            return sentence        
        
    @staticmethod
    def add_typos(string, typos=1):
        """Perturbation functions, swaps random characters with their neighbors

        Parameters
        ----------
        string : str
            input string
        typos : int
            number of typos to add

        Returns
        -------
        list(string)
            perturbed strings

        """
        string = list(string)
        swaps = np.random.choice(len(string) - 1, typos)
        for swap in swaps:
            tmp = string[swap]
            string[swap] = string[swap + 1]
            string[swap + 1] = tmp
        return ''.join(string)

    @staticmethod
    def remove_things(string, types):
        def remove_number(string):
            par = r"\d+\.?\d*|[一,二,三,四,五,六,七,八,九,十,千,百,万]+"
            ps = re.findall(par, string)
            for p in ps:
                string = string.replace(p, "")
            return string
        def remove_digit(string):
            par = r"([1-9]\d*\.?\d*)|(0\.\d*[1-9])"
            ps = re.findall(par, string)
            for p in ps:
                string = string.replace(p[0], "")
            return string
        def remove_puncuation(string):
            par = r"[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）“”]"
            ps = re.findall(par, string)
            for p in ps:
                string = string.replace(p, "")
            return string
        def remove_date(string):
            par = r"[一,二,三,四,五,六,七,八,九,十,去,上]+[日,天,周,月,年]+[前,后]?|[1-9]+[日,天,周,月,年]+[前,后]?"
            ps = re.findall(par, string)
            for p in ps:
                string = string.replace(p, "")
            return string
        
        fn = {'number':remove_number, 
              'digit':remove_digit,
              'puncuation':remove_puncuation,
              'date':remove_date}
        return fn[types](string)        

    @staticmethod
    def remove_negation(doc):
        """Removes negation from doc.
        This is experimental, may or may not work.

        Parameters
        ----------
        doc : string
            input

        Returns
        -------
        string
            With all negations removed

        """
        for k in Perturb.data['neg2pos']:
            if doc.find(k) > -1:
                doc = doc.replace(k, Perturb.data['neg2pos'][k])
                break
        return doc

    @staticmethod
    def add_negation(doc):
        """Adds negation to doc
        This is experimental, may or may not work. It also only works for specific parses.

        Parameters
        ----------
        doc : string
            input

        Returns
        -------
        string
            With negations added

        """
        for k in Perturb.data['pos2neg']:
            if doc.find(k) > -1:
                doc = doc.replace(k, random.choice(Perturb.data['pos2neg'][k]))
                break
        return doc
        
    @staticmethod
    def contractions(sentence, **kwargs):
        """Perturbation functions, contracts and expands contractions if present

        Parameters
        ----------
        sentence : str
            input

        Returns
        -------
        list
            List of strings with contractions expanded or contracted, or []

        """
        expanded = [Perturb.expand_contractions(sentence), Perturb.contract(sentence)]
        return [t for t in expanded if t != sentence]

    @staticmethod
    def expand_contractions(sentence, **kwargs):
        """Expands contractions in a sentence (if any)

        Parameters
        ----------
        sentence : str
            input string

        Returns
        -------
        string
            String with contractions expanded (if any)

        """
        contraction_map = {
            "ain't": "is not", "aren't": "are not", "can't": "cannot",
            "can't've": "cannot have", "could've": "could have", "couldn't":
            "could not", "didn't": "did not", "doesn't": "does not", "don't":
            "do not", "hadn't": "had not", "hasn't": "has not", "haven't":
            "have not", "he'd": "he would", "he'd've": "he would have",
            "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y":
            "how do you", "how'll": "how will", "how's": "how is",
            "I'd": "I would", "I'll": "I will", "I'm": "I am",
            "I've": "I have", "i'd": "i would", "i'll": "i will",
            "i'm": "i am", "i've": "i have", "isn't": "is not",
            "it'd": "it would", "it'll": "it will", "it's": "it is", "ma'am":
            "madam", "might've": "might have", "mightn't": "might not",
            "must've": "must have", "mustn't": "must not", "needn't":
            "need not", "oughtn't": "ought not", "shan't": "shall not",
            "she'd": "she would", "she'll": "she will", "she's": "she is",
            "should've": "should have", "shouldn't": "should not", "that'd":
            "that would", "that's": "that is", "there'd": "there would",
            "there's": "there is", "they'd": "they would",
            "they'll": "they will", "they're": "they are",
            "they've": "they have", "wasn't": "was not", "we'd": "we would",
            "we'll": "we will", "we're": "we are", "we've": "we have",
            "weren't": "were not", "what're": "what are", "what's": "what is",
            "when's": "when is", "where'd": "where did", "where's": "where is",
            "where've": "where have", "who'll": "who will", "who's": "who is",
            "who've": "who have", "why's": "why is", "won't": "will not",
            "would've": "would have", "wouldn't": "would not",
            "you'd": "you would", "you'd've": "you would have",
            "you'll": "you will", "you're": "you are", "you've": "you have"
            }
        contraction_map = Perturb.data['contrac2full']
        # self.reverse_contraction_map = dict([(y, x) for x, y in self.contraction_map.items()])
        contraction_pattern = re.compile(r'({})'.format('|'.join(contraction_map.keys())),
            flags=re.IGNORECASE|re.DOTALL)

        def expand_match(contraction):
            match = contraction.group(0)
            print(match)
            # expanded_contraction = contraction_map.get(match, contraction_map.get(match.lower()))
            expanded_contraction = contraction_map.get(match, contraction_map.get(match)) 
            # expanded_contraction = first_char + expanded_contraction[1:]
            return random.choice(expanded_contraction)
        contraction = contraction_pattern.search(sentence)
        if contraction:
            return contraction_pattern.sub(expand_match(contraction), sentence)
        
    @staticmethod
    def contract(sentence, **kwargs):
        """Contract expanded contractions in a sentence (if any)

        Parameters
        ----------
        sentence : str
            input string

        Returns
        -------
        string
            String with contractions contracted (if any)

        """
        reverse_contraction_map = {
            'is not': "isn't", 'are not': "aren't", 'cannot': "can't",
            'could not': "couldn't", 'did not': "didn't", 'does not':
            "doesn't", 'do not': "don't", 'had not': "hadn't", 'has not':
            "hasn't", 'have not': "haven't", 'he is': "he's", 'how did':
            "how'd", 'how is': "how's", 'I would': "I'd", 'I will': "I'll",
            'I am': "I'm", 'i would': "i'd", 'i will': "i'll", 'i am': "i'm",
            'it would': "it'd", 'it will': "it'll", 'it is': "it's",
            'might not': "mightn't", 'must not': "mustn't", 'need not': "needn't",
            'ought not': "oughtn't", 'shall not': "shan't", 'she would': "she'd",
            'she will': "she'll", 'she is': "she's", 'should not': "shouldn't",
            'that would': "that'd", 'that is': "that's", 'there would':
            "there'd", 'there is': "there's", 'they would': "they'd",
            'they will': "they'll", 'they are': "they're", 'was not': "wasn't",
            'we would': "we'd", 'we will': "we'll", 'we are': "we're", 'were not':
            "weren't", 'what are': "what're", 'what is': "what's", 'when is':
            "when's", 'where did': "where'd", 'where is': "where's",
            'who will': "who'll", 'who is': "who's", 'who have': "who've", 'why is':
            "why's", 'will not': "won't", 'would not': "wouldn't", 'you would':
            "you'd", 'you will': "you'll", 'you are': "you're",
        }
        reverse_contraction_map = Perturb.data['full2contrac']
        reverse_contraction_pattern = re.compile(r'({})'.format('|'.join(reverse_contraction_map.keys())),
            )
        def cont(possible):
            match = possible.group(1)
            first_char = match
            # expanded_contraction = reverse_contraction_map.get(match, reverse_contraction_map.get(match.lower()))
            expanded_contraction = reverse_contraction_map.get(match, reverse_contraction_map.get(match))
            # expanded_contraction = first_char + expanded_contraction[1:] + ' '
            return random.choice(expanded_contraction)
        possible = reverse_contraction_pattern.search(sentence)
        if possible:
            return reverse_contraction_pattern.sub(cont(possible), sentence)
        
    @staticmethod
    def change_names(doc, meta=False, n=10, first_only=False, last_only=False, seed=None):
        """Replace names with other names

        Parameters
        ----------
        doc : spacy.token.Doc
            input
        meta : bool
            if True, will return list of (orig_name, new_name) as meta
        n : int
            number of names to replace original names with
        first_only : bool
            if True, will only replace first names
        last_only : bool
            if True, will only replace last names
        seed : int
            random seed

        Returns
        -------
        list(str)
            if meta=True, returns (list(str), list(tuple))
            Strings with names replaced.

        """
        if seed is not None:
            np.random.seed(seed)
        ents = [x.text for x in doc.ents if np.all([a.ent_type_ == 'PERSON' for a in x])]
        print(ents)
        ret = []
        ret_m = []
        for x in ents:
            sex = None
            if x in Perturb.data['name_set']['female']:
                sex = 'female'
            if x in Perturb.data['name_set']['male']:
                sex = 'male'
            
            names = Perturb.data['name'][sex][:90+n]
            to_use = np.random.choice(names, n)
            
            # if not first_only:
            #     f = x
            #     if len(x.split()) > 1:
            #         last = Perturb.data['name']['last'][:90+n]
            #         last = np.random.choice(last, n)
            #         to_use = ['%s %s' % (x, y) for x, y in zip(names, last)]
            #         if last_only:
            #             to_use = last
            #             f = x.split()[1]
            for y in to_use:
                # ret.append(re.sub(r'\b%s\b' % re.escape(f), y, doc.text))
                ret.append(doc.text.replace(x, y))
                ret_m.append((x, y))
        return process_ret(ret, ret_m=ret_m, n=n, meta=meta)

    @staticmethod
    def change_city(doc, meta=False, seed=None, n=10):
        """Change city and country names

        Parameters
        ----------
        doc : spacy.token.Doc
            input
        meta : bool
            if True, will return list of (orig_loc, new_loc) as meta
        seed : int
            random seed
        n : int
            number of locations to replace original locations with

        Returns
        -------
        list(str)
            if meta=True, returns (list(str), list(tuple))
            Strings with locations replaced.

        """
        if seed is not None:
            np.random.seed(seed)
        print(doc.ents[0], doc.ents[0].text, doc.ents[0][0].ent_type_)
        ents = [x.text for x in doc.ents if np.all([a.ent_type_ == 'GPE' for a in x])]
        print(ents)
        ret = []
        ret_m = []
        for x in ents:
            if x in Perturb.data['city']:
                names = Perturb.data['city']
            elif x in Perturb.data['country']:
                names = Perturb.data['country']
            else:
                continue
            sub_re = re.compile(r'%s' % x)
            to_use = np.random.choice(names, n)
            ret.extend([doc.text.replace(x, n) for n in to_use])
            # ret.extend([sub_re.sub(n, doc.text) for n in to_use])
            ret_m.extend([(x, n) for n in to_use])
        return process_ret(ret, ret_m=ret_m, n=n, meta=meta)

    @staticmethod
    def change_number(doc, meta=False, seed=None, n=10):
        """Change integers to other integers within 20% of the original integer
        Does not change '2' or '4' to avoid abbreviations (this is 4 you, etc)

        Parameters
        ----------
        doc : spacy.token.Doc
            input
        meta : bool
            if True, will return list of (orig_number, new_number) as meta
        seed : int
            random seed
        n : int
            number of numbers to replace original locations with

        Returns
        -------
        list(str)
            if meta=True, returns (list(str), list(tuple))
            Strings with numbers replaced.

        """
        if seed is not None:
            np.random.seed(seed)
        nums = [x.text for x in doc if x.text.isdigit()]
        ret = []
        ret_m = []
        for x in nums:
            # e.g. this is 4 you
            if x == '2' or x == '4':
                continue
            sub_re = re.compile(r'\b%s\b' % x)
            try:
                change = int(int(x) * .2) + 1
            except:
                continue
            to_sub = np.random.randint(-min(change, int(x) - 1), change + 1, n * 3)
            to_sub = ['%s' % str(int(x) + t) for t in to_sub if str(int(x) + t) != x][:n]
            # ret.extend([sub_re.sub(n, doc.text) for n in to_sub])
            ret.extend([doc.text.replace(x, n) for n in to_sub])
            ret_m.extend([(x, n) for n in to_sub])
        return process_ret(ret, ret_m=ret_m, n=n, meta=meta)

    @staticmethod
    def change_entity(doc, entity, func, meta=False, seed=None, n=10):
        ret = []
        ret_m = []
        
        options = Perturb.data[func]
        if seed is not None:
            np.random.seed(seed)
        
        # sub_re = re.compile(r'%s' % entity)
        to_use = np.random.choice(options, n)
        ret.extend([doc.text.replace(entity, n) for n in to_use])
        # ret.extend([sub_re.sub(n, doc.text) for n in to_use])
        ret_m.extend([(entity, n) for n in to_use])   
        return process_ret(ret, ret_m=ret_m, n=n, meta=meta)  
    
    