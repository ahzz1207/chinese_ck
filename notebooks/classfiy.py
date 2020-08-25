# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import checklist
import spacy
import itertools
import checklist.editor
import checklist.text_generation
import numpy as np
import random
import json
import pkuseg
from checklist.model_api import test_model
from checklist.test_types import MFT, INV, DIR
from checklist.expect import Expect
from checklist.test_suite import TestSuite
from checklist.perturb import Perturb
from checklist.pred_wrapper import PredictorWrapper


# %%
editor = checklist.editor.Editor()
seg = pkuseg.pkuseg()
nlp = spacy.load('zh_core_web_sm')
editor.tg


# %%
examples, labels, texts = [], [], []
label2id = json.load(open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/data/intent/new_label2id_680.json"))
for line in open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/data/intent/dev_680_train.txt").readlines()[:500]:
    line = line.strip('\n')
    lines = line.split('|')
    texts.append(line)
    examples.append(lines[0])
    labels.append(label2id[lines[1]])
labels = np.array(labels).astype(int)


# %%
parsed_examples = list(nlp.pipe(examples))
spacy_map = dict([(x, y) for x, y in zip(examples, parsed_examples)])
parsed_qs = [(spacy_map[q]) for q in examples]
parsed_qs[:2]

# %% [markdown]
# ## Vocabulary

# %%
template = "我是要{v:mask}张伟。他是工程师"
verbs = editor.suggest(templates=template, )[:20]
print(', '.join(verbs))


# %%
t = editor.template('我是要{verb}张伟,他是工程师',
                verb=verbs,
                remove_duplicates=True, 
                nsamples=20)
test = MFT(**t, labels=563, name='动词测试', capability='Vocabulary', 
          description = '填充不同动词，测试分类是否依然准确')
suite = TestSuite()
suite.add(test)


# %%
t = editor.template('我是要找张伟,她是{post}',
                remove_duplicates=True, 
                nsamples=100)
print(t.data[:3])
test = MFT(**t, labels=563, name='职位测试', capability='Vocabulary', 
          description = '填充不同职位，测试分类是否依然准确')
suite.add(test, overwrite=True)


# %%
mod = ['真的', '确实', '绝对', '毫无疑问', '确确实实']
t = editor.template('我{mod}不是{company}的员工!', mod=mod, remove_duplicates=True, nsamples=50)
test = MFT(**t, labels=361, name='修饰语测试', capability='Vocabulary', 
          description = '增加不同的修饰语，测试分类是否依然准确')
suite.add(test)


# %%
adj = editor.suggest('我好{a:mask}啊，去休息室怎么走？')[:20]
t = editor.template('我好{adj}啊，去休息室怎么走？', adj=adj, remove_duplicates=True, nsamples=50)
test = MFT(**t, labels=495, name='形容词测试', capability='Vocabulary', 
          description = '使用不同的形容词，测试分类是否依然准确')
suite.add(test, overwrite=True)


# %%
followup = ['会伤身吗？', '会伤肾吗?','伤身体吗？', '可以一直吃吗?']
t = editor.template('蛋白粉吃多了会怎么样?{followup}', followup=followup, remove_duplicates=True, nsamples=20)
test = MFT(**t, labels=523, name='增加追问', capability='Vocabulary', 
          description = '增加一段追问，测试分类是否依然准确')
suite.add(test, overwrite=True)

# %% [markdown]
# ### Taxonomy

# %%
syn = []
x = editor.suggest('什么空气净化器比较{mask}?')
x += editor.suggest('什么空气净化器不太{mask}?')
for a in set(x):
    e = editor.synonyms('什么空气净化器%s?' % a, a)
    if e:
        syn.append([a] + e)
print(',\n'.join([str(tuple(x)) for x in syn]))


# %%
ops = []
for a in set(x):
    e = editor.antonyms('什么空气净化器%s?' % a, a)
    if e:
        ops.append([a] + e)
print(',\n'.join([str(tuple(x)) for x in ops]))


# %%
temp = [y for x in syn for y in x] + [y for x in ops for y in x]
data = editor.template('什么空气净化器{syn}?', syn=temp, remove_duplicates=True, nsamples=50)
test = MFT(**t, labels=655, name='测试同义词和反义词', capability='Taxonomy', 
          description = '填充同义词或反义词，测试分类是否依然准确')
suite.add(test)


# %%
examples = []
for line in open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/checklist/checklist-master/checklist/data/test_classify/classify.txt").readlines()[:10]:
    line = line.strip('\n')
    examples.append(line)
parsed_examples = list(nlp.pipe(examples))
spacy_map = dict([(x, y) for x, y in zip(examples, parsed_examples)])
parsed_qs = [(spacy_map[q]) for q in examples]
parsed_qs[:2]


# %%
synonyms = []
antonyms = []
for e in examples:
    syns = []
    es = seg.cut(e)
    for word in es:
        for s in editor.synonyms(e, word)[1]:
            synonyms.append((e, s))
        for s in editor.antonyms(e, word)[1]:
            antonyms.append((e, s))
        # syns.append(editor.synonyms(e, word))
    # synonyms.append(zip(es, syns))
synonyms[:2]
antonyms[:2]            


# %%
import re
def replace_pairs(pairs):
    def replace_z(text):
        ret = []
        for x, y in pairs:
            t = text.replace(x, y)
            if t != text:
                ret.append(t)
            t = text.replace(y, x)
            if t != text:
                ret.append(t)
        return list(set(ret))
    return replace_z
def apply_and_pair(fn):
    def ret_fn(text):
        ret = fn(text)
        return [(text, r) for r in ret]
    return ret_fn   


# %%
t = Perturb.perturb(list(examples), apply_and_pair(replace_pairs(synonyms)), nsamples=100, keep_original=False)
test = INV(t.data, threshold=0.1, name='同义词替换', description='替换近义词，分类是否会受影响', capability='Taxonomy')
suite.add(test)


# %%
t = Perturb.perturb(list(examples), apply_and_pair(replace_pairs(antonyms)), nsamples=100, keep_original=False)
test = INV(t.data, threshold=0.1, name='反义词替换', description='替换反义词，分类是否会受影响', capability='Taxonomy')
suite.add(test)

# %% [markdown]
# ## Robustness
# %% [markdown]
# Typos

# %%
def wrapper_fn(fn):
    def real_fn(text):
        ret = fn(text)
        if ret == text:
            return 
        return [(text, ret)]
    return real_fn

def wrapper_change(fn):
    def apply_change(text):
        seed = np.random.randint(100)
        c = fn(text)
        if not c:
            return
        if type(c) == list and c:
            return [(text, c1) for c1 in c if c1]
        elif c == text:
            return
        else: 
            return [(text, c)]
    return apply_change


# %%
t = Perturb.perturb(examples, wrapper_fn(Perturb.add_typos), nsamples=100)
test = INV(t.data, name='随机替换', capability='Robustness', description='对文本随机替换1个字符，分类结果应尽量不变')
suite.add(test, overwrite=True)

# %% [markdown]
# Conctraction

# %%
t = Perturb.perturb(examples, wrapper_change(Perturb.contractions), nsamples=10000)
test = INV(**t, name='缩写转换', capability='Robustness', description='对文本中的缩写简写进行转换，分类结果应保持不变')
suite.add(test, overwrite=True)

# %% [markdown]
# ## Ner
# %% [markdown]
# ### 改变句子中的姓名，数字，地点等。
# %% [markdown]
# #### 姓名替换

# %%
male = random.choices(list(editor.data['names']['male']), k=1000)
t = editor.template('我是要找{male},他是工程师',
                male=male,
                remove_duplicates=True, 
                nsamples=100)
test = MFT(**t, labels=563, name='姓名测试1', capability='NER', 
          description = '填充不同男性姓名，测试分类是否依然准确')
suite.add(test)


# %%
female = random.choices(list(editor.data['names']['female']), k=1000)
t = editor.template('我是要找{female},她是工程师',
                male=female,
                remove_duplicates=True, 
                nsamples=100)
test = MFT(**t, labels=563, name='姓名测试2', capability='NER', 
          description = '填充不同女性姓名，测试分类是否依然准确')
suite.add(test)


# %%
data = editor.template('我不是{company}的员工!', remove_duplicates=True, nsamples=50)
test = MFT(**t, labels=361, name='公司测试', capability='NER', 
          description = '填充不同公司，测试分类是否依然准确')
suite.add(test)


# %%
t = Perturb.perturb(parsed_examples, wrapper_change(Perturb.change_names), nsamples=100)
test = INV(**t, name='姓名替换', capability='NER',
          description='对所有句子中的姓名随机替换')
suite.add(test)


# %%
t = Perturb.perturb(examples, wrapper_fn(Perturb.trans_num), nsamples=100)
test = INV(**t, name='数字替换', capability='NER',
          description='对所有句子中的小写数字转大写数字')
suite.add(test)


# %%
t = Perturb.perturb(parsed_examples, wrapper_change(Perturb.change_location), nsamples=100)
test = INV(**t, name='地点替换', capability='NER',
          description='对所有句子中的城市，国家，地名进行对应替换')
suite.add(test)

# %% [markdown]
# ## Temporal
# %% [markdown]
# #### 实体随机填充，替换句式

# %%



# %%
data1 = editor.template('我不是{male1}，我是{company}的{post}，我叫{male2}。', remove_duplicates=True, nsamples=100)
data2 = editor.template('我叫{male2}，我不是{male1}，我是{company}的{post}', remove_duplicates=True, nsamples=100)
data = zip(data1, data2)
test = INV(data, name='句式替换', capability='Temporal',
          description='两种句式是否结果一致')
suite.add(test)

# %% [markdown]
# #### 颠倒句子顺序

# %%
def change_list(text):
    splits = text.split('，')
    if len(splits) > 1:
        n = random.choice(range(len(splits)))
        swap = splits[n]
        y = random.choice(range(len(splits)).remove(n))
        splits[n] = splits[y]
        splits[y] = swap
        return [(text, '，'.join(splits))]
    else:
        return 


# %%
t = Perturb.perturb(examples, change_list, keep_original=False, nsamples=100)
test = INV(**t, name='顺序替换', capability='Temporal',
          description='两种句式是否结果一致')
suite.add(test)

# %% [markdown]
# ## Negation

# %%
t = Perturb.perturb(examples, wrapper_change(Perturb.add_negation), nsamples=100)
test = INV(t.data, name='肯定否定替换', capability='Negation', description='对文本中肯定转否定，否定转肯定')
suite.add(test, overwrite=True)


# %%
males = zip(male[:100], female[:100])
data = []
for m, f in males:
    data.append((f"我的名字是{m}，而不是{f}。", f"我的名字是{f}，而不是{m}。"))
test = INV(data, name='名字肯否定替换', capability='Negation', description='名字转换，句式相同，句意改变')
suite.add(test, overwrite=True)


# %%
suite.to_raw_file("./classfiy.text")


# %%



