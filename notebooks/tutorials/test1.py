import checklist
import spacy
import itertools
import random
import checklist.editor
import checklist.text_generation
import numpy as np
import spacy
from checklist.model_api import circle_model
from checklist.test_types import MFT, INV, DIR
from checklist.expect import Expect
from checklist.test_suite import TestSuite
from checklist.perturb import Perturb
from checklist.pred_wrapper import PredictorWrapper

test_model = circle_model()
pred_fn = test_model.eval
wrapped_pp = PredictorWrapper.wrap_softmax(pred_fn)
editor = checklist.editor.Editor(language='chinese')
suite = TestSuite()

## 
ret = editor.template('去{company}是地铁几号线,我该怎么去', labels=468, nsamples=200)
ret += editor.template('坐什么公交可以到{company}', labels=467, nsamples=200)
print(ret.data[:5])
test = MFT(ret.data, labels=ret.labels, name='simple vocabulary',
        capability='vocabulary', description='test any company would not confuse model')
suite.add(test)
# test.run(wrapped_pp)
# test.summary()


def change_v(data):
    if data.find('去') > -1:
        new_data=data.replace('去', random.choice(v))
    else:
        new_data=data.replace('坐', random.choice(v2))
    return [new_data]

data = ret.data
print(data[:3])
v = editor.suggest('去{company}是地铁几号线？我应该怎么{mask}呢？')[:20]
v2 = editor.suggest('我应该{mask}什么公交可以到{company}')[:20]
t = Perturb.perturb(data, change_v)
inv = INV(**t, name='simple pos',
        capability='pos', description='test change verb would not confuse model')
suite.add(inv)
# inv.run(wrapped_pp)
# inv.summary()

# 我们期望采取否定语法后，所有样本不改变预测结果
def high_confidence(x, pred, conf, label=None, meta=None):
    return conf.max() < 0.6
expect_fn = Expect.single(high_confidence)
# 继续沿用INV的data
t = Perturb.perturb(data, Perturb.add_negation)
print(t.data[:3])
dir_ = DIR(**t, expect=expect_fn, name='simple negation',
        capability='negation', description='test add negation would not confuse model')
# dir_.run(wrapped_pp)
# dir_.summary()
suite.add(dir_)

def changed_pred(orig_pred, pred, orig_conf, conf, labels=None, meta=None):
    return pred != orig_pred
expect_fn = Expect.pairwise(changed_pred)
dir_ = dir_.set_expect(expect_fn)
suite.add(dir_)
suite.run(wrapped_pp)
suite.summary()