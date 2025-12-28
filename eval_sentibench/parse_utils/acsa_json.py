from utils import append_new_line, save_json
import os, time, json
from datetime import datetime
import ast
import re


def get_list_from_text(text):
    text = text.strip()

    l_id = text.find('[') 
    r_id = text.rfind(']') + 1

    # text = text[1:-1]
    return text[l_id :r_id]

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


def parse_category_polarity_for_true(acsa_seq, sentence):

    acsa_seq = json.loads(acsa_seq)


    category_polarity_list = []
    valid_flag = True


    for sub_seq in acsa_seq:

        category_polarity_list.append(tuple([sub_seq['aspect_category'], sub_seq['sentiment']]))

    return category_polarity_list, valid_flag

# TODO: invalid_string
def parse_aspect_polarity(atsa_seq):
    aspect_polarity_list = []

    def parse_seq(category_polarity):

        if 'aspect_category' not in category_polarity or 'sentiment' not in category_polarity:
            return False

        category = category_polarity['aspect_category']
        polarity = category_polarity['sentiment']


        category = category.strip()
        polarity = polarity.strip()

        if polarity not in ('positive', 'neutral', 'negative'):
            return False

        return category, polarity


    try:
        aspect_polarity_list = json.loads(atsa_seq)

    except:
        print('error:', atsa_seq)

        # Add cleanup for markdown code blocks
        try:
            atsa_seq = get_list_from_text(atsa_seq)

            aspect_polarity_list = json.loads(atsa_seq)

            print('after cleanup:', aspect_polarity_list)
        except:
            print('Failed to parse JSON even after cleanup')
            return []


   
    try:
        aspect_polarity_list = [
            parse_seq(aspect_polarity)
            for aspect_polarity in aspect_polarity_list if aspect_polarity
        ]
    except Exception as e:
        print(e)
        print('string:', atsa_seq)
        print('parsed:', aspect_polarity_list)

    return aspect_polarity_list




class Result_acsa:
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


            category_polarity_list_true = parse_category_polarity_for_true(example['label_seq'], sentence)[0]
            category_polarity_list_pred = parse_aspect_polarity(prediction)


            print('sentence:',sentence)
            print('original prediction:',prediction)

            print('true: ',category_polarity_list_true)
            print('pred: ',category_polarity_list_pred)
            print('-'*100)


            data[ID] = {
                'ID': example.get('ID', ID),
                'sentence': sentence,
                'prompts':example['prompts'],
                'label_seq':example['label_seq'],
                'golden_label_parsed': category_polarity_list_true,
                'prediction_parsed': category_polarity_list_pred,
                'dataset': example['dataset']
            }
            ID += 1

        return cls(data)

    def cal_metric(self):
        f1 = F1Measure()

        for ID in self.data:
            example = self.data[ID]
            f1.add_ground_truths(ID, example['golden_label_parsed'])
            f1.add_predictions(ID, example['prediction_parsed'])

        f1.report()

        self.detailed_metrics = {
            'f1': f1['f1'],
            'recall': f1['recall'],
            'precision': f1['precision'],
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
        performance_file_name = os.path.join(output_dir, 'performance' , now, 'performance.txt')

        print('save performace to', performance_file_name)
        append_new_line(performance_file_name, json.dumps({
            'experiment_name':experiment_name,
            'time': time.strftime('%Y-%m-%d %H_%M_%S', time.localtime()),
            'model_name_or_path': model_name_or_path,
            'subname': subname,
            'dataset': dataset,
            'seed': seed,
            'lr': lr,
            'rank':rank,
            'alpha': alpha,
            'target_module': target_module,
            'dropout': dropout,
            'metric': self.detailed_metrics['f1']
        }))

    def report(self):
        for metric_names in (('precision', 'recall', 'f1'),):
            for metric_name in metric_names:
                value = self.detailed_metrics[metric_name] if metric_name in self.detailed_metrics else 0
                print(f'{metric_name}: {value:.4f}', end=' | ')
            print()
