import checklist
import spacy
import itertools
import checklist.editor
import checklist.text_generation
import numpy as np
import random
import json
from checklist.model_api import test_model
from checklist.test_types import MFT, INV, DIR
from checklist.expect import Expect
from checklist.test_suite import TestSuite
from checklist.perturb import Perturb
from checklist.pred_wrapper import PredictorWrapper

editor = checklist.editor.Editor()
nlp = spacy.load('zh_core_web_sm')
editor.tg

examples, labels, texts = [], [], []
label2id = json.load(open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/data/intent/new_label2id_680.json"))
for line in open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/data/intent/dev_680_train.txt").readlines()[:10]:
    line = line.strip('\n')
    lines = line.split('|')
    texts.append(line)
    examples.append(lines[0])
    labels.append(label2id[lines[1]])
labels = np.array(labels).astype(int)

parsed_examples = list(nlp.pipe(examples))
spacy_map = dict([(x, y) for x, y in zip(examples, parsed_examples)])
suite = TestSuite()
parsed_qs = [(spacy_map[q]) for q in examples]
parsed_qs[:2]

template = "我是要{v:mask}黄庄泽。"
verbs = editor.suggest(templates=template)[:10]
print(', '.join(verbs))
