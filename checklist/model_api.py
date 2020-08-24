import collections
import gc
import json
import logging
import os
import random
import shelve
import time
import warnings
import tqdm
import torch
import re
import sys
import pdb
# import _locale
# _locale._getdefaultlocale = (lambda *args: ['en_US', 'utf8'])
sys.path.append("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1")
cur_floder = "/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1"
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.dataset import TensorDataset
from config.arguments import parser
from config.hyper_parameters import HyperParams
from model.modeling_bert import BertForIntentClassificationCircle
from model.optimization import AdamWeightDecayOptimizer
from common_utils.utils import restore_from_checkpoint
from common_utils import tokenizer as tokenization
from torch.nn import functional as F
from torch.nn import Parameter
import torch.nn as nn
warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def device_config(gpu=1):
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    # torch.distributed.init_process_group(backend='nccl')
    return device, n_gpu


def random_seed_config(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_masks, label = None, text = None):
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.label = label
        self.text_id = text


class IntentProcessor():
    """Processor for the Classification data set (GLUE version)."""
    def replace_date(self, txt):
        par = r"[一,二,三,四,五,六,七,八,九,十,去,上,前,昨]+[日,天,周,月,年]+[前,后]?|[1-9]+[日,天,周,月,年]+[前,后]?"
        ps = re.findall(par, txt)
        for p in ps:
            txt = txt.replace(p, "")
        return txt

    def replace_num(self, txt):
        par = r"\d+\.?\d*|[一,二,三,四,五,六,七,八,九,十,千,百,万]+[一,二,三,四,五,六,七,八,九,十,千,百,万]+"
        ps = re.findall(par, txt)
        for p in ps:
            txt = txt.replace(p, "")
        return txt

    def replace_punc(self, txt):
        par = r"[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）“”]"
        ps = re.findall(par, txt)
        for p in ps:
            txt = txt.replace(p, "")
        return txt

    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        
        for (i, line) in enumerate(lines):
        
            guid = "%s" % (i)
            text_a = tokenization.convert_to_unicode(lines[i])
            label = "0"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=i, label=label))
        return examples


class test_model():

    def __init__(self):
        # device config
        self.device, _ = device_config()
        parsed_args = parser.parse_known_args()[0]
        self.hp = HyperParams.init_from_parsed_args(parsed_args)
        self.hp.do_lower_case = True
        self.hp.pooler = 'mean'
        self.hp.dir_init_checkpoint = cur_floder + "/checkpoint/intent/circle_r128_mean_706/pytorch_model_3880.pt"
        self.hp.vocab_file = cur_floder + "/resources/vocab.txt"
        self.hp.num_labels = 706
        self.hp.max_seq_length = 64
        self.hp.dev_batch_size
        # prepare model
        self.model = BertForIntentClassificationCircle(self.hp)
        state_dict = torch.load(self.hp.dir_init_checkpoint, map_location='cpu')    
        self.model.load_state_dict(state_dict, strict=False)

        self.processor = IntentProcessor()

        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.hp.vocab_file)

        self.model.to(self.device)               

    def convert_examples(self, examples, max_seq_length, tokenizer, training=True):
        """Converts a single `InputExample` into a single `InputFeatures`."""
        features = []
        
        for idx, example in enumerate(examples):
            tokens_a = tokenizer.tokenize(example.text_a)
            tokens_b = None

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

            tokens = []
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            label_id = 0

            features.append(InputFeatures(
                input_ids=input_ids,
                input_masks=input_mask,
                label=label_id,
                text=example.text_b))

        return features

    def eval(self, inputs):
        self.model.eval()
        
        time_start = time.time()
        
        dev_examples = self.processor._create_examples(inputs)
        
        dev_features = self.convert_examples(dev_examples, self.hp.max_seq_length, self.tokenizer)
        dev_steps = int(len(dev_features) / self.hp.dev_batch_size) + 1
        logger.info("***** Running devaluation *****")
        logger.info("  Num examples = %d", len(dev_examples))
        all_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_masks for f in dev_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label for f in dev_features], dtype=torch.long)
        all_text_id = torch.tensor([f.text_id for f in dev_features], dtype=torch.long)
        dev_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_text_id)
        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=self.hp.dev_batch_size)
        prob = []
        # start dev
        logging.info("Start dev!")
        with torch.no_grad():
            for step, batch in enumerate(tqdm.tqdm(dev_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, label, text_id = batch
                # forward propagation
                loss, intent_out, pool_out = self.model(input_ids, label, input_mask)
                logist = torch.softmax(intent_out, dim=-1)
                predictions = torch.argmax(logist, dim=-1)
                for i, l in enumerate(label.cpu().tolist()):
                    prob.append(intent_out[i].cpu().numpy())       
        print(f"本轮eval耗时共{time.time() - time_start}")                    
        # json.dump(prob, open("/work/QA_task/roberta-1.1/BERTCN/bertcn-pytorch-r1.1/checklist/checklist-master/notebooks/tutorials/prob_dir.json", 'w'), ensure_ascii=False)
        return prob
    
    
def main():
    model = test_model()
    print(model.eval(['做个测试']))

