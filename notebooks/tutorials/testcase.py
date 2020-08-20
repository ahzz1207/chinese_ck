import checklist
from checklist.editor import Editor
from checklist.perturb import Perturb
import spacy
from checklist.test_types import MFT, INV, DIR

# editor = Editor()
data = ['我愿意去那里', '我喜欢这个东西', '我记得清他的事']
t = Perturb.perturb(data, Perturb.add_negation)
print(t)
# nlp = spacy.load('zh_core_web_sm')
# data = ['我不叫丁长春。', '我的一个账户不在福建省?', '这是一个数字5。']
# pdata = list(nlp.pipe(data))
# print(pdata[0].ents)
# editor = Editor(language='chinese')
# print(editor.synonyms('我现在能做啥', '能'))
# ret = editor.suggest('{mask}去{city}看望{male}。', 
#                       male=editor.lexicons.male_from['China'])
# print(ret[:5])
# ret = editor.synonyms('这很热', '热')
# print(ret)
# print(len(ret.data))
# ret = editor.template('这是个好{mask}.', remove_duplicates=True)
# ret = Perturb.strip_punctuation(pdata[0])
# print(ret)
# ret = Perturb.change_location(pdata[1])
# # ret = Perturb.perturb(pdata[1], Perturb.change_location, nsamples=1)
# print(ret)
# ret = Perturb.change_names(pdata[0])
# print(ret)
# ret = Perturb.change_number(pdata[2])
# print(ret)
