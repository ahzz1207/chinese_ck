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
from checklist.model_api import test_model
from checklist.test_types import MFT, INV, DIR
from checklist.expect import Expect
from checklist.test_suite import TestSuite
from checklist.perturb import Perturb
from checklist.pred_wrapper import PredictorWrapper


# %%
# 需要写一个接口版一个写文件版
# 用户定义的接口
test_model = test_model()
wrapped_pp = PredictorWrapper.wrap_softmax(test_model.eval)
editor = checklist.editor.Editor(language='chinese')


# %%
suite = TestSuite()

# %% [markdown]
# 测试分类模型对POS+Vocabulary的能力，构造MFT INV DIR三种测试方案
# 首先构造MFT测试
# 我们首先利用词库提供一些地名，构造一些简单样本进行分类准确率的测试

# %%
ret = editor.template('去{company}是地铁几号线,我该怎么去', labels=468, nsamples=200)
ret += editor.template('坐什么公交可以到{company}', labels=467, nsamples=200)
print(ret.data[:5])
test = MFT(ret.data, labels=ret.labels, name='vocabluary',
           capability='vocab', description='test model vocab ')
suite.add(test)
# test.run(wrapped_pp)

# %% [markdown]
# 构造INV测试, 使用模型生成一些动词，利用这些动词对样本随机替换
# 模型应该保持预测结果不变

# %%
def change_v(data):
    if data.find('去') > -1:
        new_data=data.replace('去', random.choice(v))
    else:
        new_data=data.replace('坐', random.choice(v2))
    return [new_data]


# %%
# 这里复用mft单元生成的数据，由于INV测试只比较模型是否对扰动前后样本预测结果相同，因此INV数据不需要label标签
data = ret.data
print(data[:3])
v = editor.suggest('去{company}是地铁几号线？我应该怎么{mask}呢？')[:20]
v2 = editor.suggest('我应该{mask}什么公交可以到{company}')[:20]
t = Perturb.perturb(data, change_v)
inv = INV(**t, name='verb test', capability='vocab')
suite.add(inv)
# 将测试数据重新写入文件中
# examples = inv.to_raw_file('/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/data/checklist/dev_468_inv.txt')

# %% [markdown]
# 最后生成DIR测试，我们希望肯定转为否定能降低模型对类别判断的置信度(概率)
# 我们首先需要编写一个自定义的期望函数，即认为概率下降/上升多少该case为pass/fail
# 

# %%
# 我们期望采取否定语法后，所有样本不改变预测结果
def high_confidence(x, pred, conf, label=None, meta=None):
    return conf.max() < 0.6
expect_fn = Expect.single(high_confidence)


# %%
# 继续沿用INV的data
t = Perturb.perturb(data, Perturb.add_negation)
print(t.data[:3])
dir = DIR(**t, expect=expect_fn, name="negation test", capability='vocab')
suite.add(dir)
# 将测试数据重新写入文件中
# examples = inv.to_raw_file('/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/data/checklist/dev_468_dir.txt')


# %%
suite.run(wrapped_pp)


# %%
suite.summary()


# %%
a, b = suite.visual_summary_table()


# %%
from checklist.viewer.suite_summarizer import SuiteSummarizer
SuiteSummarizer(a, b)


# %%



