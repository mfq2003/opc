"""对移动点随机森林做固定网格的按版图留一参数敏感性实验。

输入沿用 v4 Excel，先删除 result=0，再删除剩余样本中含空值的特征列。所有候选使用
完全相同的五折版图划分，按 macro-F1、balanced accuracy、accuracy、模型复杂度依次排序。
搜索折同时用于选参和报告，只能作为训练版图内部敏感性结果，不能替代独立测试或 Golden 回放。
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from .recipe_v2_vision_tree import LABELS, load_data, scores, write_csv, write_json


DEPTHS=(5,8,12,None)
LEAVES=(1,2,5,10)
FEATURES=('sqrt',.5,1.0)
WEIGHTS=('balanced','balanced_subsample',None)


def conflict_report(X,y,names):
    groups=defaultdict(list)
    for index,row in enumerate(X): groups[tuple(int(v) for v in row)].append(index)
    conflicts=[];irreducible_correct=0
    for vector,indices in groups.items():
        counts=Counter(int(y[i]) for i in indices);irreducible_correct+=max(counts.values())
        if len(counts)>1:
            conflicts.append({'feature_vector':dict(zip(names,vector)),'sample_count':len(indices),
                              'label_counts':dict(sorted(counts.items()))})
    conflicts.sort(key=lambda r:(-r['sample_count'],str(r['label_counts'])))
    conflicted=sum(r['sample_count'] for r in conflicts)
    return {'unique_feature_vectors':len(groups),'conflicting_vectors':len(conflicts),
        'samples_in_conflicting_vectors':conflicted,'conflicting_sample_fraction':conflicted/len(y),
        'maximum_training_accuracy_from_exact_feature_vector':irreducible_correct/len(y),
        'groups':conflicts}


def run(args):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneGroupOut
    import sklearn
    X,y,groups,ids,names,audit=load_data(args.input,drop_stay=True)
    labels=[k for k in LABELS if k!=0]
    out=Path(args.output)
    if out.exists() and any(out.iterdir()): raise ValueError('输出目录须为空，避免覆盖搜索结果')
    if args.n_estimators<1: raise ValueError('n-estimators 必须为正')
    folds=list(LeaveOneGroupOut().split(X,y,groups));rows=[]
    candidates=list(itertools.product(DEPTHS,LEAVES,FEATURES,WEIGHTS))
    started=time.monotonic()
    for number,(depth,leaf,max_features,weight) in enumerate(candidates,1):
        pred=np.empty_like(y);fold_scores=[]
        for tr,te in folds:
            model=RandomForestClassifier(n_estimators=args.n_estimators,max_depth=depth,
                min_samples_leaf=leaf,max_features=max_features,class_weight=weight,
                bootstrap=True,random_state=0,n_jobs=-1,criterion='gini').fit(X[tr],y[tr])
            pred[te]=model.predict(X[te])
            fold_scores.append(scores(y[te],pred[te],labels))
        metric=scores(y,pred,labels)
        key='macro_f1_fixed_four_classes'
        rows.append({'candidate':number,'max_depth':'None' if depth is None else depth,
            'min_samples_leaf':leaf,'max_features':max_features,'class_weight':str(weight),
            'accuracy':metric['accuracy'],'macro_f1':metric[key],
            'balanced_accuracy':metric['balanced_accuracy_present_classes'],
            'worst_layout_macro_f1':min(x[key] for x in fold_scores),
            'mean_layout_macro_f1':float(np.mean([x[key] for x in fold_scores])),
            'elapsed_seconds':time.monotonic()-started})
        if number==1 or number%12==0:
            print(f'候选 {number}/{len(candidates)}：macro-F1={metric[key]:.4f}',flush=True)
    depth_order={5:0,8:1,12:2,'None':3}
    rows.sort(key=lambda r:(-r['macro_f1'],-r['balanced_accuracy'],-r['accuracy'],
                            depth_order[r['max_depth']],r['min_samples_leaf'],str(r['max_features']),r['class_weight']))
    for rank,row in enumerate(rows,1): row['rank']=rank
    conflict=conflict_report(X,y,names)
    report={'status':'diagnostic_only','selection_scope':'same_leave_one_layout_out_folds_used_for_selection',
        'source':str(Path(args.input).resolve()),'audit':audit,'samples':len(y),'features':names,
        'evaluation_classes':labels,'layouts':sorted(set(groups)),'candidate_count':len(candidates),
        'n_estimators':args.n_estimators,'grid':{'max_depth':list(DEPTHS),'min_samples_leaf':list(LEAVES),
            'max_features':list(FEATURES),'class_weight':['balanced','balanced_subsample','None']},
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},
        'best':rows[0],'top_10':rows[:10],'feature_label_conflicts':conflict,
        'elapsed_seconds':time.monotonic()-started,
        'limitation':'同一组训练版图留一折用于比较参数，最佳分数有选择偏差；需独立版图或 Golden 回放确认。'}
    out.mkdir(parents=True,exist_ok=True)
    header=['rank','candidate','max_depth','min_samples_leaf','max_features','class_weight','accuracy',
            'macro_f1','balanced_accuracy','worst_layout_macro_f1','mean_layout_macro_f1','elapsed_seconds']
    write_csv(out/'grid_results.csv',header,[[r[k] for k in header] for r in rows])
    write_json(out/'search.json',report)
    write_json(out/'feature_label_conflicts.json',conflict)
    if hashlib.sha256(Path(args.input).read_bytes()).hexdigest()!=audit['source_xlsx_sha256']:
        raise RuntimeError('搜索期间源 Excel 发生变化')
    print(json.dumps({'output':str(out),'best':rows[0],'conflicts':{k:v for k,v in conflict.items() if k!='groups'},
                      'elapsed_seconds':report['elapsed_seconds']},ensure_ascii=False,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='runs/v2-vision-dataset-004/training.xlsx')
    parser.add_argument('--output',required=True)
    parser.add_argument('--n-estimators',type=int,default=500)
    run(parser.parse_args())


if __name__=='__main__': main()
