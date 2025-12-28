from utils import append_new_line, save_json
import os, time, json
from datetime import datetime
import ast

def get_list_from_text(text):
    text = text.strip()

    l_id = text.find('[') 
    r_id = text.rfind(']') + 1

    # text = text[1:-1]
    return text[l_id :r_id]



def parse_quards_for_true(true_seq):

    true_seq = json.loads(true_seq)
    
    true_tuple = []
    for item in true_seq:
        true_tuple.append(tuple([item['holder'], item['target'], item['sentiment_expression'], item['sentiment_polarity']]))


    print('True:', true_tuple)

    return true_tuple

def fix_quotes(corrupted_text):
        
    try:


        tt = corrupted_text.replace("'", "\"")

        a = [item.strip() for item in tt.split(',')]

        new_a = []
        for item in a:

            l_index = item.find('"') 
            r_index = item.rfind('"') 

            inner = item[l_index + 1:r_index ].replace('"', "'")

            
            item = item[:item.find('"') + 1] + inner + item[item.rfind('"'):] 


            new_a.append(f'{item}')

        new_a = ','.join(new_a)

        print(new_a)

        

        return new_a
    
    except:

        return corrupted_text


def parse_quards_for_predict(quards_seq, sentence):




    valid_flag = True

    try:
        quards_seq = json.loads(quards_seq)
    except:
        print('error:', quards_seq)


        try:
            quards_seq = get_list_from_text(quards_seq)

            quards_seq = json.loads(quards_seq)

            print('after cleanup:', quards_seq)
        except:
            
            print('Failed to parse JSON even after cleanup')
            return quards_seq, [], False


    def parse_seq(item):
        if len(item) != 4:
            return False

        if 'holder' not in item or 'target' not in item or 'sentiment_expression' not in item or 'sentiment_polarity' not in item:
            return False

        holder, target, expression, sentiment = item['holder'], item['target'], item['sentiment_expression'], item['sentiment_polarity']
        holder = holder.strip()
        target = target.strip()
        expression = expression.strip()
        if type(sentiment) != str:
            
            return False
        sentiment = sentiment.strip()

        if sentiment not in ('positive', 'neutral', 'negative',):
            return False

        return  holder, target, expression, sentiment

    quards_list = []
    for sub_seq in quards_seq:
        quards = parse_seq(sub_seq)
        if not quards:
            valid_flag = False
        else:
            quards_list.append(quards)


    print('prediction:', quards_list)

    return quards_seq, quards_list, valid_flag


class F1Measure:
    def __init__(self):
        self.pred_list = []  # List to store predictions
        self.true_list = []  # List to store ground truths

    def add_predictions(self, idx, preds):
        """Adds a batch of predictions for a specific index."""
        self.pred_list.extend((idx, pred) for pred in preds)

    def add_ground_truths(self, idx, trues):
        """Adds a batch of ground truths for a specific index."""
        self.true_list.extend((idx, true) for true in trues)

    def report(self):
        """Calculates and returns the F1 score."""
        self.f1, self.precision, self.recall = self.calculate_f1()
        return self.f1

    def __getitem__(self, key):
        """Allows retrieval of attributes like a dictionary."""
        if hasattr(self, key):
            return getattr(self, key)
        raise AttributeError(f"{key} is not a valid attribute of F1Measure.")

    def calculate_f1(self):
        """Calculates F1 score along with precision and recall."""
        n_tp = sum(pred in self.true_list for pred in self.pred_list)
        precision = n_tp / len(self.pred_list) if self.pred_list else 1

        n_tp = sum(true in self.pred_list for true in self.true_list)
        recall = n_tp / len(self.true_list) if self.true_list else 1

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        return f1, precision, recall

from nltk.tokenize.simple import SpaceTokenizer

tk = SpaceTokenizer()

def convert_char_offsets_to_token_idxs(char_offsets, token_offsets):

    token_idxs = []
    #
    for char_offset in char_offsets:
        bidx, eidx = char_offset.split(":")
        bidx, eidx = int(bidx), int(eidx)
        intoken = False
        for i, (b, e) in enumerate(token_offsets):
            if b == bidx:
                intoken = True
            if intoken:
                token_idxs.append(i)
            if e == eidx:
                intoken = False
    return frozenset(token_idxs)

def convert_opinion_to_tuple(sentence):
    text = sentence["sentence"]
    opinions = sentence["opinions"]
    opinion_tuples = []
    token_offsets = list(tk.span_tokenize(text))

    #
    if len(opinions) > 0:
        for opinion in opinions:
            holder_char_idxs = opinion["Source"][1]
            target_char_idxs = opinion["Target"][1]
            exp_char_idxs = opinion["Polar_expression"][1]
            polarity = opinion["Polarity"]
            #
            holder = convert_char_offsets_to_token_idxs(holder_char_idxs, token_offsets)
            target = convert_char_offsets_to_token_idxs(target_char_idxs, token_offsets)
            exp = convert_char_offsets_to_token_idxs(exp_char_idxs, token_offsets)
            opinion_tuples.append((holder, target, exp, polarity))
    return opinion_tuples

def parse_to_origin(sentence, label_seq):
    if len(label_seq) == 0:
        return []

    constructed_opinions = []
    for opinion in label_seq:
        dict = {}
        if len(opinion) != 4:
            return []
        holder, target, expression, sentiment = opinion


        if holder == 'NULL':
            dict['Source'] = [[], []]
        else:
            begin = sentence.find(holder)
            if begin == -1:
                dict['Source'] = [[], []]
            else:
                end = begin + len(holder)
                dict['Source'] = [[holder], [f"{str(begin)}:{str(end)}"]]

        dict['Target'] = [[], []]
        for item in target.split('AND'):
            item = item.strip()
            begin = sentence.find(item)
            if begin == -1:
                continue
            else:
                end = begin + len(item)
                dict['Target'][0].append(item)
                dict['Target'][1].append(f"{str(begin)}:{str(end)}")

        dict['Polar_expression'] = [[], []]
        for item in expression.split('AND'):
            item = item.strip()
            begin = sentence.find(item)
            if begin == -1:
                continue
            else:
                end = begin + len(item)
                dict['Polar_expression'][0].append(item)
                dict['Polar_expression'][1].append(f"{str(begin)}:{str(end)}")

        dict['Polarity'] = sentiment.capitalize()

        constructed_opinions.append(dict)
    return constructed_opinions

def weighted_score(sent_tuple1, list_of_sent_tuples):
    best_overlap = 0
    holder1, target1, exp1, pol1 = sent_tuple1
    if len(holder1) == 0:
        holder1 = frozenset(["_"])
    if len(target1) == 0:
        target1 = frozenset(["_"])
    for holder2, target2, exp2, pol2 in list_of_sent_tuples:
        if len(holder2) == 0:
            holder2 = frozenset(["_"])
        if len(target2) == 0:
            target2 = frozenset(["_"])
        if (
            len(holder2.intersection(holder1)) > 0
            and len(target2.intersection(target1)) > 0
            and len(exp2.intersection(exp1)) > 0
        ):
            holder_overlap = len(holder2.intersection(holder1)) / len(holder1)
            target_overlap = len(target2.intersection(target1)) / len(target1)
            exp_overlap = len(exp2.intersection(exp1)) / len(exp1)
            overlap = (holder_overlap + target_overlap + exp_overlap) / 3
            if overlap > best_overlap:
                best_overlap = overlap
    return best_overlap

def sent_tuples_in_list(sent_tuple1, list_of_sent_tuples, keep_polarity=True):
    holder1, target1, exp1, pol1 = sent_tuple1
    if len(holder1) == 0:
        holder1 = frozenset(["_"])
    if len(target1) == 0:
        target1 = frozenset(["_"])
    for holder2, target2, exp2, pol2 in list_of_sent_tuples:
        if len(holder2) == 0:
            holder2 = frozenset(["_"])
        if len(target2) == 0:
            target2 = frozenset(["_"])
        if (
            len(holder1.intersection(holder2)) > 0
            and len(target1.intersection(target2)) > 0
            and len(exp1.intersection(exp2)) > 0
        ):
            if keep_polarity:
                if pol1 == pol2:
                    # print(holder1, target1, exp1, pol1)
                    # print(holder2, target2, exp2, pol2)
                    return True
                else:
                    # print('sentiment polarity not match')
                    pass
            else:
                # print(holder1, target1, exp1, pol1)
                # print(holder2, target2, exp2, pol2)
                return True
    return False

def tuple_precision(gold, pred, keep_polarity=True, weighted=True):
    """
    Weighted true positives / (true positives + false positives)
    """
    weighted_tp = []
    tp = []
    fp = []
    #
    for sent_idx in pred.keys():
        ptuples = pred[sent_idx]
        gtuples = gold[sent_idx]
        for stuple in ptuples:
            if sent_tuples_in_list(stuple, gtuples, keep_polarity):
                if weighted:
                    #sc = weighted_score(stuple, gtuples)
                    #if sc != 1:
                        #print(sent_idx)
                        #print(sc)
                        #print()
                    weighted_tp.append(weighted_score(stuple, gtuples))
                    # print('precision weighted score:', weighted_score(stuple, ptuples))

                    tp.append(1)
                else:
                    weighted_tp.append(1)
                    tp.append(1)
            else:
                #print(sent_idx)
                fp.append(1)
    #print("weighted tp: {}".format(sum(weighted_tp)))
    #print("tp: {}".format(sum(tp)))
    #print("fp: {}".format(sum(fp)))
    return sum(weighted_tp) / (sum(tp) + sum(fp) + 0.0000000000000001)


def tuple_recall(gold, pred, keep_polarity=True, weighted=True):
    """
    Weighted true positives / (true positives + false negatives)
    """
    weighted_tp = []
    tp = []
    fn = []
    #
    assert len(gold) == len(pred)
    #
    for sent_idx in pred.keys():
        ptuples = pred[sent_idx]
        gtuples = gold[sent_idx]
        for stuple in gtuples:
            if sent_tuples_in_list(stuple, ptuples, keep_polarity):
                if weighted:
                    weighted_tp.append(weighted_score(stuple, ptuples))
                    # print('recall weighted score:', weighted_score(stuple, ptuples))
                    tp.append(1)
                else:
                    weighted_tp.append(1)
                    tp.append(1)
            else:
                fn.append(1)
    return sum(weighted_tp) / (sum(tp) + sum(fn) + 0.0000000000000001)

def tuple_f1(gold, pred, keep_polarity=True, weighted=True):
    prec = tuple_precision(gold, pred, keep_polarity, weighted)
    rec = tuple_recall(gold, pred, keep_polarity, weighted)
    #print("prec: {}".format(prec))
    #print("rec: {}".format(rec))
    return prec, rec, 2 * (prec * rec) / (prec + rec + 0.00000000000000001)

class Result_ssa:
    def __init__(self, data):
        self.data = data

    def __ge__(self, other):
        return self.monitor >= other.monitor

    def __gt__(self, other):
        return self.monitor > other.monitor

    @classmethod
    def parse_from(cls, outputs):

        data = {}
        ID = 0
        for example, prediction in outputs:


            sentence = example['sentence']

            ssa_list_true = parse_quards_for_true(example['label_seq'])



            # if ssa_list_true == [('NULL', 'NULL', 'NULL', 'NULL')]:
            #     print('all null, skip')
            #     continue



            # print('prediction:', prediction)

            original_prediction, ssa_list_pred = parse_quards_for_predict(prediction, sentence)[:2]

            opinions = parse_to_origin(sentence, ssa_list_pred)


            # if example['dataset'] == 'absa/dsunis':
            #     print('1'*100)
            #     print(example['opinions'])
            #     print('2' * 100)
            #     print(opinions)
            #     print('3' * 100)
            #     print(original_prediction)


            data[ID] = {
                'ID': example.get('ID', ID),
                'sentence': sentence,
                'prompts':example['prompts'],
                'label_seq':example['label_seq'],
                'golden_label_parsed':ssa_list_true,
                'prediction_parsed': ssa_list_pred,
                'original_output_of_model': original_prediction,
                'gold':example,
                'pred': {
                    'sent_id':example['sent_id'],
                    'opinions':opinions,
                    'sentence': example['sentence']},
                'dataset': example['dataset']
                
            }
            ID += 1
        return cls(data)

    def cal_metric(self):
        golds = []
        preds = []
        for ID in self.data:
            example = self.data[ID]
            golds.append(example['gold'])
            preds.append(example['pred'])

        # print('golds:', golds)
        # print('preds:', preds)
        # print('+'*10)


        golds = dict([(s["sent_id"], convert_opinion_to_tuple(s)) for s in golds])

        preds = dict([(s["sent_id"], convert_opinion_to_tuple(s)) for s in preds])


        # print('golds:', golds)
        # print('preds:', preds)

        prec, rec, f1 = tuple_f1(golds, preds)

        self.detailed_metrics = {
            'f1': f1,
            'precision':prec,
            'recall':rec
        }

        self.monitor = self.detailed_metrics['f1']

    def save_prediction(self, output_dir, model_name_or_path, dataset, subname, seed, rank, alpha, target_module, dropout, lr, experiment_name):

        now = datetime.now()
        now = now.strftime("%Y-%m-%d")
        file_name = os.path.join(output_dir, 'result', f'{experiment_name}_{dataset}_{subname}_seed:{seed}_rank:{rank}_alpha:{alpha}_target_module:{target_module}_dropout:{dropout}_lr:{lr}.json')

        print('save prediction to', file_name)
        save_json(
            {
                'data': self.data,
                'meta': (model_name_or_path, subname, dataset, seed, lr, now)
            },
            file_name
        )



    def save_metric(self, output_dir, model_name_or_path, dataset, subname, seed, rank, alpha, target_module, dropout, lr,experiment_name):

        now = datetime.now()
        now = now.strftime("%Y-%m-%d")
        performance_file_name = os.path.join(output_dir, 'performance', 'performance.txt')

        print('save performace to', performance_file_name)
        append_new_line(performance_file_name, json.dumps({
            'experiment_name':experiment_name,
            'metric': self.detailed_metrics['f1'],
            'dataset': dataset,
            'subname': subname,
            'seed': seed,
            'time': time.strftime('%Y-%m-%d %H_%M_%S', time.localtime()),
            'model_name_or_path': model_name_or_path,
            'lr': lr,
            'rank':rank,
            'alpha': alpha,
            'target_module': target_module,
            'dropout': dropout,
            
        }))

    def report(self):
        for metric_names in (('precision', 'recall', 'f1'),):
            for metric_name in metric_names:
                value = self.detailed_metrics[metric_name] if metric_name in self.detailed_metrics else 0
                print(f'{metric_name}: {value:.4f}', end=' | ')
            print()
