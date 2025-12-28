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

    # def add_predictions(self, idx, preds):
    #     """Adds a batch of predictions for a specific index."""
    #     self.pred_list.append(preds)

    # def add_ground_truths(self, idx, trues):
    #     """Adds a batch of ground truths for a specific index."""
    #     self.true_list.append(trues)


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

        print('pred_list:', self.pred_list)
        print('true_list:', self.true_list)

        # for item in self.true_list:
        #     index, true = item
        #     print('true:', )
        #     print(true)
        #     print('pred:')
        #     for pred in self.pred_list:
        #         if index == pred[0]:
        #             print(pred[1])
        #     print('-'*100)

        n_tp = sum(pred in self.true_list for pred in self.pred_list)
        precision = n_tp / len(self.pred_list) if self.pred_list else 1

        n_tp = sum(true in self.pred_list for true in self.true_list)
        recall = n_tp / len(self.true_list) if self.true_list else 1

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        # micro_f1 = self.process_tuple_f1(self.true_list, self.pred_list)

        return f1, precision, recall


    def process_tuple_f1(self, labels, predictions, verbose=False):
        tp, fp, fn = 0, 0, 0
        epsilon = 1e-7
        for i in range(len(labels)):
            gold = set(labels[i])
            try:
                pred = set(predictions[i])
            except Exception:
                pred = set()

            print('gold:', gold)
            print('pred:', pred)
            print('-'*100)
            tp += len(gold.intersection(pred))
            fp += len(pred.difference(gold))
            fn += len(gold.difference(pred))
        if verbose:
            print('-'*100)
            print(gold, pred)
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        micro_f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        return micro_f1


def parse_quards_for_true(quards_seq, sentence):


    quards_seq = json.loads(quards_seq)


    quards_tuple = []
    for item in quards_seq:
        quards_tuple.append(tuple([item['aspect_term'], item['aspect_category'], item['opinion'], item['sentiment']]))



    return quards_tuple


def parse_quards_for_predict(quards_seq, sentence):




    valid_flag = True
    print("before parsed :", quards_seq)

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

        if 'aspect_category' not in item or 'aspect_term' not in item or 'opinion' not in item or 'sentiment' not in item:
            return False

        category, aspect , opinion, sentiment = item['aspect_category'], item['aspect_term'], item['opinion'], item['sentiment']
        category = category.strip()
        aspect = aspect.strip()
        opinion = opinion.strip()
        if type(sentiment) != str:
            return False
        
        sentiment = sentiment.strip()


        return aspect, category , opinion, sentiment

    quards_list = []

    for sub_seq in quards_seq:
        quards = parse_seq(sub_seq)
        if not quards:
            valid_flag = False
        else:
            quards_list.append(quards)

    print('-'*100)
    print('final_list:', quards_list)
    print('-'*100)

    return quards_seq, quards_list, valid_flag


class Result_asqp:
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
        for example, prediction  in outputs:


            
            sentence = example['sentence']

            asqp_list_true = parse_quards_for_true(example['label_seq'], sentence)
            original_prediction, asqp_list_pred = parse_quards_for_predict(prediction,
                                                                                               sentence)[:2]

            # print(asqp_list_pred)
            # print(asqp_list_true)

            data[ID] = {
                'ID': example.get('ID', ID),
                'sentence': sentence,
                'prompts':example['prompts'],
                'label_seq':example['label_seq'],
                'golden_label_parsed': asqp_list_true,
                'prediction_parsed': asqp_list_pred,
                'original_output_of_model': original_prediction,
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

            
            # print('a'*100)
            # print('golden_label:')
            # print(example['golden_label'])
            
            # print('a' * 100)
            # print('prediction:')
            # print(example['prediction'])


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
