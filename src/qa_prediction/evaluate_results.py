import argparse
import glob
import json
import os
import re
import string
from sklearn.metrics import precision_score

def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1

def eval_acc(prediction, answer):
    matched = 0.
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)

def eval_hit(prediction, answer):
    for a in answer:
        if match(prediction, a):
            return 1
    return 0

def eval_hits1(prediction_list, answer):
    if not prediction_list:
        return 0
    top1 = str(prediction_list[0])
    for a in answer:
        if match(top1, a):
            return 1
    return 0

def eval_em(prediction_list, answer):
    pred_set = {normalize(str(p)) for p in prediction_list if normalize(str(p))}
    ans_set = {normalize(str(a)) for a in answer if normalize(str(a))}
    if not pred_set and not ans_set:
        return 1
    return 1 if pred_set == ans_set else 0

def eval_f1(prediction, answer):
    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0
    prediction_str = ' '.join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
    precision = matched / len(prediction)
    recall = matched / len(answer)
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return 2 * precision * recall / (precision + recall), precision, recall

def extract_topk_prediction(prediction, k=-1):
    results = {}
    for p in prediction:
        if p in results:
            results[p] += 1
        else:
            results[p] = 1
    if k > len(results) or k < 0:
        k = len(results)
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:k]]

def eval_result(predict_file, cal_f1=True, topk = -1):
    # predict_file = os.path.join(result_path, 'predictions.jsonl')
    eval_name = "detailed_eval_result_top_{topk}.jsonl" if topk > 0 else 'detailed_eval_result.jsonl'
    detailed_eval_file = predict_file.replace('predictions.jsonl', eval_name)
    # Load results
    acc_list = []
    hit_list = []
    hits1_list = []
    em_list = []
    f1_list = []
    precission_list = []
    recall_list = []
    with open(predict_file, 'r') as f, open(detailed_eval_file, 'w') as f2:
        for line in f:
            try:
                data = json.loads(line)
            except:
                print(line)
                continue
            id = data['id']
            prediction = data['prediction']
            answer = data['ground_truth']
            if cal_f1:
                if not isinstance(prediction, list):
                    prediction = prediction.split("\n")
                else:
                    prediction = extract_topk_prediction(prediction, topk)
                f1_score, precision_score, recall_score = eval_f1(prediction, answer)
                f1_list.append(f1_score)
                precission_list.append(precision_score)
                recall_list.append(recall_score)
                prediction_str = ' '.join(prediction)
                acc = eval_acc(prediction_str, answer)
                hit = eval_hit(prediction_str, answer)
                hits1 = eval_hits1(prediction, answer)
                em = eval_em(prediction, answer)
                acc_list.append(acc)
                hit_list.append(hit)
                hits1_list.append(hits1)
                em_list.append(em)
                f2.write(json.dumps({'id': id, 'prediction': prediction, 'ground_truth': answer, 'acc': acc, 'hit': hit, 'hits1': hits1, 'em': em, 'f1': f1_score, 'precission': precision_score, 'recall': recall_score}) + '\n')
        
            else:
                acc = eval_acc(prediction, answer)
                hit = eval_hit(prediction, answer)
                hits1 = hit
                acc_list.append(acc)
                hit_list.append(hit)
                hits1_list.append(hits1)
                f2.write(json.dumps({'id': id, 'prediction': prediction, 'ground_truth': answer, 'acc': acc, 'hit': hit, 'hits1': hits1}) + '\n')
    
    if len(f1_list) > 0:
        result_str = (
            "Hit: " + str(sum(hit_list) * 100 / len(hit_list)) +
            " F1: " + str(sum(f1_list) * 100 / len(f1_list)) +
            " Precision: " + str(sum(precission_list) * 100 / len(precission_list)) +
            " Recall: " + str(sum(recall_list) * 100 / len(recall_list)) +
            " Hits@1: " + str(sum(hits1_list) * 100 / len(hits1_list)) +
            " EM: " + str(sum(em_list) * 100 / len(em_list)) +
            " Accuracy: " + str(sum(acc_list) * 100 / len(acc_list))
        )
    else:
        result_str = (
            "Hit: " + str(sum(hit_list) * 100 / len(hit_list)) +
            " Hits@1: " + str(sum(hits1_list) * 100 / len(hits1_list)) +
            " Accuracy: " + str(sum(acc_list) * 100 / len(acc_list))
        )
    print(result_str)
    result_name = "eval_result_top_{topk}.txt" if topk > 0 else 'eval_result.txt'
    eval_result_path = predict_file.replace('predictions.jsonl', result_name)
    with open(eval_result_path, 'w') as f:
        f.write(result_str)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d', type=str, default='results/KGQA/csqa/alpaca_default/test')
    argparser.add_argument('--cal_f1', action="store_true")
    argparser.add_argument('--top_k', type=int, default=-1)
    args = argparser.parse_args()
    
    eval_result(args.d, args.cal_f1, args.top_k)