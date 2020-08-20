import json
import math
import pandas as pd

def deal_atons():
    aton = open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/checklist/checklist-master/checklist/data/resource/antonyms.txt").readlines()
    aton2 = open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/checklist/checklist-master/checklist/data/resource/antonym2.txt").readlines()
    atons = {}
    for a in aton:
        a = a.strip('\n')
        reg = ['--', '——', '──', '―', '—']
        for r in reg:
            if a.find(r) > 0:
                ab = a.split(r)
                if ab[0] in atons:
                    atons[ab[0]].append(ab[1])
                else:
                    atons[ab[0]] = [ab[1]]
                if ab[1] in atons:
                    atons[ab[1]].append(ab[0])
                else:
                    atons[ab[1]] = [ab[0]]
                break
    for a in aton2:
        a = a.strip('\n')
        a = a.split(':')
        if a[0] not in atons:
            atons[a[0]] = a[1].split(';')
        else:
            atons[a[0]] += a[1].split(';')
    for k in atons:
        atons[k] = list(set(atons[k]))
    print(len(atons))            
    json.dump(atons, open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/checklist/checklist-master/checklist/data/antonyms.json", 'w'), ensure_ascii=False)
# deal_atons()
def deal_contrac():
    lines = open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/checklist/checklist-master/checklist/data/contrac2.txt").readlines()
    kvm = {}
    vkm = {}
    for l in lines:
        l = l.strip('\n')
        kv = l.split(': ')
        k = kv[0]
        v = ''.join([x.split('/')[0] for x in kv[1].split(' ')])
        if k not in kvm:
            kvm[k] = [v]
        else:
            kvm[k].append(v)
        
        if v not in vkm:
            vkm[v] = [k]
        else:
            vkm[v].append(k)
        
    for s in kvm:
        kvm[s] = list(set(kvm[s]))
    for s in vkm:
        vkm[s] = list(set(vkm[s]))
        
    json.dump(kvm, open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/checklist/checklist-master/checklist/data/contrac2full.json", 'w'), ensure_ascii=False)
    json.dump(vkm, open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/checklist/checklist-master/checklist/data/full2contrac.json", 'w'), ensure_ascii=False)

def basic():
    nation = json.load(open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/checklist/checklist-master/checklist/data/resource/nation.json"))
    nations = []
    for item in nation.items():
        nations.append(item[-1]['zh'])
    print(len(nations))

    diming = open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/checklist/checklist-master/checklist/data/resource/city.txt").readlines()
    city = set()
    for d in diming:
        city.add(d.split('#')[0])
    print(len(city))

    name = open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/checklist/checklist-master/checklist/data/resource/Chinese_Names_Corpus_Gender（120W）.txt").readlines()
    male, female = [], []
    web, locate, email, post, call, company, phone_num, fax, ss, depart = [], [], [], [], [], [], [], [], [], []
    person, person_rare = [], []
    for n in name:
        ns = n.strip('\n').split(',')
        if ns[-1] == '男':
            male.append(ns[0])
        if ns[-1] == '女':
            female.append(ns[0])

    all_value = json.load(open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/checklist/checklist-master/checklist/data/lexicons/all_value_dict_v6.json"))
    for k in all_value:
        if k == 'PER':
            person = all_value['PER']['ch'] + all_value['PER']['ch_corpus']
            person_rare = all_value['PER']['rare']
        elif k == 'MISC_POST':
            post = all_value['MISC_POST']['ch'] + all_value['MISC_POST']['ch_corpus']
        elif k == 'ORG_COMP':
            company = all_value['ORG_COMP']['ch'] + all_value['ORG_COMP']['ch_corpus']
        elif k == 'ORG_DEPART':
            depart = all_value['ORG_DEPART']['ch_corpus']
        elif k == 'LOC':
            locate = all_value['LOC']['ch'] + all_value['LOC']['ch_corpus']
        elif k == 'NUM_PHONE':
            phone_num = all_value['NUM_PHONE']['own'] + all_value['NUM_PHONE']['corpus']
        elif k == 'NUM_CELL':
            call = all_value['NUM_CELL']['own'] + all_value['NUM_CELL']['corpus']
        elif k == 'NUM_FAX':
            fax =  all_value['NUM_FAX']['own'] + all_value['NUM_FAX']['corpus']
        elif k == 'MISC_WEB':
            web = all_value['MISC_WEB']['own'] + all_value['MISC_WEB']['corpus']
        elif k == 'MISC_MAIL':
            email = all_value['MISC_MAIL']['own'] + all_value['MISC_MAIL']['corpus']
        elif k == 'MISC_SS':
            ss = all_value['MISC_SS']['own'] + all_value['MISC_SS']['corpus']

    chengyu = open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/checklist/checklist-master/checklist/data/resource/chengyu.txt").readlines()
    cy = []
    for c in chengyu:
        cy.append(c.strip('\n'))
    
    animals = open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/checklist/checklist-master/checklist/data/resource/animal.txt").readlines()
    animal = []
    for a in animals:
        animal.append(a.split('	')[0])
        
    foods = open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/checklist/checklist-master/checklist/data/resource/food.txt").readlines()
    food = []
    for f in foods:
        food.append(f.split('	')[0])
    
    medical = open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/checklist/checklist-master/checklist/data/resource/medical.txt").readlines()
    med = []
    for m in medical:
        med.append(m.split('	')[0])
        
    his_name = open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/checklist/checklist-master/checklist/data/resource/history_name.txt").readlines() 
    hisn = []
    for h in his_name:
        hisn.append(h.split('\t')[0])  
        
    time = json.load(open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/checklist/checklist-master/checklist/data/resource/TM.json"))
    tm = []
    
    number = open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/checklist/checklist-master/checklist/data/resource/number.txt").readlines()
    numbers = []
    for n in number:
        numbers.append(n.strip('\n'))
        
    money = open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/checklist/checklist-master/checklist/data/resource/money.txt").readlines()
    moneys = []
    for n in money:
        moneys.append(n.strip('\n'))
     
    basic = {'male':male, 'female':female, 'city':list(city), 'country':nations, 'person':person, 'person_rare':person_rare,
             'post':post, 'company':company, 'depart':depart, 'locate':locate, 'phone_num':phone_num, 'cellphone':call,
             'fax':fax, 'web':web, 'email':email, 'social_id':ss, 'chengyu':cy, 'animal':animal, 'food':food, 'medical':med,
             'history_name':hisn, 'time':time, 'money':moneys, 'number':numbers}
    names = {'male':male, 'female':female}
    for k in basic:
        print(f"键值{k}含有数据{len(basic[k])}条，如{basic[k][:3]}")
    json.dump(basic, open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/checklist/checklist-master/checklist/data/lexicons/basic_zh2.json", 'w'), ensure_ascii=False)
    json.dump(names, open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/checklist/checklist-master/checklist/data/names_zh2.json", 'w'), ensure_ascii=False)

def excel():
    excel_file = "/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/checklist/checklist-master/checklist/data/resource/实体_正例.xlsx"
    data = pd.read_excel(excel_file)
    dic = {}
    for d in data:
        dic[d] = []
        for x in data[d]:
            if type(x) == float and math.isnan(x):
                continue
            else:
                dic[d].append(x)
    json.dump(dic, open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/checklist/checklist-master/checklist/data/lexicons/intent_zh.json", 'w'), ensure_ascii=False)
    for k in dic:
        print(f"键值{k}含有数据{len(dic[k])}条，如{dic[k][:3]}")
# basic()
# excel()
def neg():
    t = open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/checklist/checklist-master/checklist/data/resource/neg.txt").readlines()
    negdic = {}
    posdic = {}
    for l in t:
        ls = l.strip('\n').split('\t')
        if len(ls) > 1:
            negdic[ls[0]] = ls[1]
            if ls[1] in posdic:
                posdic[ls[1]].append(ls[0])
            else:
                posdic[ls[1]] = [ls[0]]
        else:
            negdic[ls[0]] = ''
    nn = {}
    for n in sorted(negdic.keys(),reverse=True,key=lambda x:len(x)):
        nn[n] = negdic[n]
    pp = {}    
    for n in sorted(posdic, reverse=True,key=lambda x:len(x)):
        pp[n] = posdic[n]
    print(len(nn), len(pp))
    json.dump(nn, open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/checklist/checklist-master/checklist/data/neg2pos.json", 'w'), ensure_ascii=False)
    json.dump(pp, open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/checklist/checklist-master/checklist/data/pos2neg.json", 'w'), ensure_ascii=False)

import random     
def make_some_time():
    
    def get_year(danwei=True):
        pre = random.choice(['199', '200', '201'])
        year = pre + str(random.choice(range(0, 10)))
        if danwei:
            return year + '年'
        else:
            return year

    def get_moth(danwei=True):
        month = random.choice(range(1, 13))
        if danwei:
            return month + '月'
        else:
            return month + '月'
    
    def get_day(danwei=True):
        day = random.choice(range(1, 31))
        if danwei:
            return day + '日'
        else:
            return day
    
    def get_week(daxie=True):
        if daxie:
            return '星期' + random.choice(['一', '二', '三', '四', '五', '六', '天'])
        else:    
            return '星期' + random.choice(['1', '2', '3', '4', '5', '6', '天'])

    years = [get_year() for _ in range(1000)]
    month = [str(i)+'月' for i in range(1, 13)]
    day = [str(i)+'日' for i in range(1, 31)]
    
    
    def get_year_month():
        return random.choice(years) + random.choice(month) + random.choice(day)
    
    def get_year_month_day():
        return random.choice(years) + random.choice(month) + random.choice(day)
    
    def get_month_day():
        return random.choice(month) + random.choice(day)
    
    dic = {}
    dic['year'] = years
    dic['month'] = month
    dic['day'] = day
    dic['year_month'] = [get_year_month() for _ in range(1000)]
    dic['year_month_day'] = [get_year_month_day() for _ in range(1000)]
    dic['month_day'] = [get_month_day() for _ in range(1000)]
    json.dump(dic, open('/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/checklist/checklist-master/checklist/data/lexicons/date.json', 'w'), ensure_ascii=False)    
# make_some_time()    

def make_data():
    def get_money_low(number):
        i = random.choice(range(10))
        if i == 0 or i == 1:
            num = random.choice(range(1, 1000))
        elif 2 <= i <= 5:
            num = random.choice(range(1001, 10000))
        elif 6 <= i <= 9:
            num = random.choice(range(10001, 100000))
        else:
            num = random.choice(range(100001, 10000000))
        i = random.choice(range(10))
        if 0 <= i <= 2:
            danwei = '元'
        elif 3 <= i <= 5:
            danwei = '块'
        elif 6 <= i <= 8:
            danwei = '圆'
        else:
            danwei = '美元'
        return str(num)+danwei

    def get_period(number):
        i = random.choice(range(10))
        if i == 8:
            num = '半'
        elif i == 9:
            num = '两'
        else:
            num = str(random.choice(range(1, 30)))
        i = random.choice(range(10))
        if i == 1 or i == 0:
            danwei = '年'
        elif i == 2 or i == 3:
            danwei = '个月'
        elif i == 4 or i == 5:
            danwei = '周'
        elif i == 6 or i == 7:
            danwei = '天'
        elif i == 8:
            danwei = '个星期'
        elif i == 9:
            danwei = '个礼拜'
        return num+danwei

    def get_punctuation(number):
        for i in range(20):
            if 0 <= i <= 2:
                p = ','
            elif 3 <= i <= 5:
                p = '，'
            elif 6 <= i <= 7:
                p = '.'
            elif 8 <= i <= 10:
                p = '。'
            elif 11 <= i <= 12:
                p = ':'
            elif 13 <= i <= 14:
                p = '：'
            elif i == 15:
                p = ';'
            elif i == 16:
                p = '；'
            elif i == 17:
                p = '、'
            else:
                p = ' '
        return p
    
    dic = {'period':[], 'money_low':[], 'punctuation':[]}
    for _ in range(100000):
        dic['period'].append(get_period(1))
        dic['money_low'].append(get_money_low(1))
        dic['punctuation'].append(get_punctuation(1))
    json.dump(dic, open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/checklist/checklist-master/checklist/data/resource/period_moeny_punc.json", 'w'), ensure_ascii=False)
# make_data()    