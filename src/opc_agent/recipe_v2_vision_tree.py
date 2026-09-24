"""从 v2 视觉标注 Excel 训练探索性 EPE 决策树或随机森林，支持过滤不移动点后四分类。

只读取 samples、provenance 和 summary，不修改工作簿，不补入失败点。含空值的特征整列
删除，布尔值严格转成 0/1；编号和版图来源只供追溯、按版图交叉验证，不进入模型。
固定浅树参数，提供全 stay 对照、逐类指标、六图留一预测、树 JSON/文本和重要性 CSV/PNG。
本模块不调用 API、OpenILT、PPO 或 Golden；分类结果不能替代完整 Recipe 的光刻回放。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import posixpath
import sys
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from .recipe_v2_vision_prompt import FEATURES

LABELS = [-2, -1, 0, 1, 2]
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def read_sheets(content):
    """读取普通 OOXML 数值、布尔、字符串与空单元格，拒绝公式和错误单元格。"""
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            strings=[''.join(x.itertext()) for x in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        rels={r.attrib['Id']:r.attrib['Target'] for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        result={}
        for sheet in ET.fromstring(z.read('xl/workbook.xml')).find('s:sheets',NS):
            name=sheet.attrib['name']
            if name not in ('samples','provenance','summary'): continue
            rid=sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
            target=rels[rid]
            path=target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/'+target)
            rows=[]
            for row in ET.fromstring(z.read(path)).findall('s:sheetData/s:row',NS):
                values=[]
                for cell in row.findall('s:c',NS):
                    ref=cell.attrib['r'];col=0
                    for ch in ref:
                        if ch.isalpha(): col=26*col+ord(ch.upper())-64
                    while len(values)<col: values.append(None)
                    if cell.find('s:f',NS) is not None or cell.attrib.get('t')=='e':
                        raise ValueError('输入数据含公式或 Excel 错误：'+name+'!'+ref)
                    kind=cell.attrib.get('t');value=cell.find('s:v',NS)
                    text=value.text if value is not None else None
                    if kind=='inlineStr': v=''.join(t.text or '' for t in cell.findall('s:is//s:t',NS))
                    elif text is None: v=None
                    elif kind=='s': v=strings[int(text)]
                    elif kind=='b':
                        if text not in ('0','1'): raise ValueError('非法 Excel 布尔值')
                        v=text=='1'
                    elif kind=='str': v=text
                    else: v=float(text)
                    values[col-1]=v
                if any(v is not None for v in values): rows.append(values)
            result[name]=rows
        return result


def records(rows):
    if not rows: raise ValueError('工作表为空')
    header=rows[0]
    if None in header or len(header)!=len(set(header)): raise ValueError('列名缺失或重复')
    out=[]
    for row in rows[1:]:
        if len(row)>len(header): raise ValueError('数据超出表头')
        out.append(dict(zip(header,row+[None]*(len(header)-len(row)))))
    return out


def is_null(value):
    return value is None or (isinstance(value,float) and math.isnan(value)) or (
        isinstance(value,str) and value.strip().lower() in ('','null','none','nan'))


def binary(value):
    if type(value) is bool: return int(value)
    if type(value) in (int,float) and value in (0,1): return int(value)
    if isinstance(value,str) and value.strip().lower() in ('true','false'):
        return int(value.strip().lower()=='true')
    raise ValueError('特征必须为布尔值或 0/1，不能把非空字符串直接转 bool')


def integer(value,name):
    if type(value) not in (int,float) or not math.isfinite(value) or int(value)!=value:
        raise ValueError(name+' 必须是整数')
    return int(value)


def load_data(path,drop_stay=False):
    content=Path(path).read_bytes();sheets=read_sheets(content)
    samples=records(sheets['samples']);provenance=records(sheets['provenance'])
    summary={r['field']:r['value'] for r in records(sheets['summary'])}
    if summary.get('teacher')!='coordinate_search_not_ppo': raise ValueError('教师来源不符')
    if not samples: raise ValueError('没有可训练样本')
    header=sheets['samples'][0]
    if set(header)!={'epe_id','result'}|set(FEATURES): raise ValueError('samples 列名与 v2 特征协议不一致')
    ids=[integer(r['epe_id'],'epe_id') for r in samples]
    if len(set(ids))!=len(ids): raise ValueError('样本编号重复')
    pids=[integer(r['epe_id'],'provenance.epe_id') for r in provenance]
    if len(set(pids))!=len(pids) or set(pids)!=set(ids): raise ValueError('来源编号不唯一或与样本不匹配')
    by_id=dict(zip(pids,provenance))
    y=np.array([integer(r['result'],'result') for r in samples],dtype=int)
    if not set(y)<=set(LABELS): raise ValueError('标签不属于五个移动类别')
    groups=[]
    for eid,label in zip(ids,y):
        p=by_id[eid];layout=p['layout_parent']
        if not isinstance(layout,str) or not layout: raise ValueError('版图来源为空')
        if p['normal_offset_nm']!=int(label)*10: raise ValueError('移动类别与来源位移不一致')
        groups.append(layout)
    source_samples=len(samples)
    if integer(summary['training_rows'],'training_rows')!=source_samples: raise ValueError('汇总样本数不一致')
    removed_ids=[eid for eid,label in zip(ids,y) if drop_stay and label==0]
    removed_layouts=sorted(set(groups)-{g for g,label in zip(groups,y) if not drop_stay or label!=0})
    keep=[i for i,label in enumerate(y) if not drop_stay or label!=0]
    samples=[samples[i] for i in keep];ids=[ids[i] for i in keep]
    groups=[groups[i] for i in keep];y=y[keep]
    if not samples: raise ValueError('过滤不移动点后没有可训练样本')
    null_counts={k:sum(is_null(r[k]) for r in samples) for k in FEATURES}
    dropped=[k for k,n in null_counts.items() if n]
    names=[k for k in FEATURES if k not in dropped]
    if not names: raise ValueError('删除空值列后没有特征')
    X=np.array([[binary(r[k]) for k in names] for r in samples],dtype=np.int8)
    if len(set(groups))<2: raise ValueError('按版图验证至少需要两张版图')
    return X,y,np.array(groups),ids,names,{
        'source_xlsx_sha256':hashlib.sha256(content).hexdigest(),
        'source_samples':source_samples,'drop_stay':drop_stay,'removed_stay_count':len(removed_ids),
        'removed_stay_epe_ids':removed_ids,'layouts_without_retained_samples':removed_layouts,
        'source_summary':summary,'null_counts':null_counts,'dropped_null_features':dropped,
        'feature_count_before':len(FEATURES),'feature_count_after':len(names),
        'constant_features':[k for i,k in enumerate(names) if len(set(X[:,i]))==1]}


def scores(y,pred,labels=None):
    labels=LABELS if labels is None else labels
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
    precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=labels,zero_division=0)
    present=support>0
    return {'accuracy':float(accuracy_score(y,pred)),
        'macro_f1_fixed_'+('four' if len(labels)==4 else 'five')+'_classes':float(f1_score(y,pred,labels=labels,average='macro',zero_division=0)),
        'balanced_accuracy_present_classes':float(np.mean(recall[present])),
        'per_class':{str(k):{'precision':float(precision[i]),'recall':float(recall[i]),
            'f1':float(f1[i]),'support':int(support[i])} for i,k in enumerate(labels)},
        'confusion_matrix':confusion_matrix(y,pred,labels=labels).tolist()}


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def write_csv(path,header,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(header);writer.writerows(rows)


def tree_json(model,names):
    t=model.tree_;nodes=[]
    for i in range(t.node_count):
        leaf=t.children_left[i]==t.children_right[i]
        nodes.append({'node_id':i,'left':int(t.children_left[i]),'right':int(t.children_right[i]),
            'feature':None if leaf else names[t.feature[i]],'threshold':None if leaf else float(t.threshold[i]),
            'class':int(model.classes_[np.argmax(t.value[i][0])]),'samples':int(t.n_node_samples[i]),
            'class_probabilities':(t.value[i][0]/t.value[i][0].sum()).tolist()})
    return {'feature_names':names,'classes':model.classes_.tolist(),'nodes':nodes}


def predict_tree(artifact,X):
    index={name:i for i,name in enumerate(artifact['feature_names'])};out=[]
    for row in X:
        node=artifact['nodes'][0]
        while node['feature'] is not None:
            side='left' if row[index[node['feature']]]<=node['threshold'] else 'right'
            node=artifact['nodes'][node[side]]
        out.append(node['class'])
    return np.array(out)


def forest_json(model,names):
    """将子树内部编码恢复成真实移动类别；森林按概率平均决策，不按硬投票决策。"""
    trees=[]
    for estimator in model.estimators_:
        tree=tree_json(estimator,names)
        if len(tree['classes'])!=len(model.classes_): raise ValueError('子树类别与森林类别不一致')
        for node in tree['nodes']: node['class']=int(model.classes_[node['class']])
        tree['classes']=model.classes_.tolist();trees.append(tree)
    return {'feature_names':names,'classes':model.classes_.tolist(),'aggregation':'mean_class_probabilities','trees':trees}


def predict_forest(artifact,X):
    index={name:i for i,name in enumerate(artifact['feature_names'])}
    probabilities=np.zeros((len(X),len(artifact['classes'])))
    for tree in artifact['trees']:
        for i,row in enumerate(X):
            node=tree['nodes'][0]
            while node['feature'] is not None:
                side='left' if row[index[node['feature']]]<=node['threshold'] else 'right'
                node=tree['nodes'][node[side]]
            probabilities[i]+=node['class_probabilities']
    return np.array(artifact['classes'])[probabilities.argmax(axis=1)]


def train(args):
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import make_scorer, f1_score
    import sklearn
    import joblib
    drop_stay=getattr(args,'drop_stay',False)
    labels=[k for k in LABELS if k!=0] if drop_stay else LABELS
    X,y,groups,ids,names,audit=load_data(args.input,drop_stay=drop_stay)
    out=Path(args.output)
    if out.exists() and any(out.iterdir()): raise ValueError('输出目录须为空，避免覆盖模型')
    if (args.max_depth is not None and args.max_depth<1) or args.min_samples_leaf<1: raise ValueError('树深和叶节点样本数须为正或 None')
    class_weight=getattr(args,'class_weight','balanced')
    if class_weight=='none': class_weight=None
    if class_weight not in ('balanced','balanced_subsample',None): raise ValueError('class-weight 非法')
    max_features=getattr(args,'max_features','sqrt')
    if max_features not in ('sqrt',.5,1.0): raise ValueError('max-features 非法')
    params={'max_depth':args.max_depth,'min_samples_leaf':args.min_samples_leaf,
            'class_weight':class_weight,'random_state':0,'criterion':'gini'}
    model_type=getattr(args,'model_type','tree')
    if model_type not in ('tree','forest'): raise ValueError('模型类型必须是 tree 或 forest')
    forest=model_type=='forest'
    constructor=RandomForestClassifier if forest else DecisionTreeClassifier
    if forest:
        count=getattr(args,'n_estimators',300)
        if count<1: raise ValueError('n-estimators 必须为正')
        params.update(n_estimators=count,max_features=max_features,bootstrap=True,n_jobs=1)
    prediction=np.empty_like(y);baseline_prediction=np.empty_like(y)
    folds=[];permutations=[];seen=np.zeros(len(y),dtype=int)
    baseline_name='training_fold_majority' if drop_stay else 'all_stay'
    scorer=make_scorer(f1_score,labels=labels,average='macro',zero_division=0)
    for fold_index,(tr,te) in enumerate(LeaveOneGroupOut().split(X,y,groups),1):
        print(f'折 {fold_index}: 留出版图 {groups[te][0]}，训练 {len(tr)}，验证 {len(te)}',flush=True)
        model=constructor(**params).fit(X[tr],y[tr]);prediction[te]=model.predict(X[te]);seen[te]+=1
        assert not set(groups[tr])&set(groups[te])
        classes,counts=np.unique(y[tr],return_counts=True)
        baseline_class=int(classes[np.argmax(counts)]) if drop_stay else 0
        baseline_prediction[te]=baseline_class
        importance=permutation_importance(model,X[te],y[te],scoring=scorer,n_repeats=10,random_state=0,n_jobs=1)
        permutations.append(importance.importances)
        folds.append({'held_out_layout':str(groups[te][0]),'train_layouts':sorted(set(groups[tr])),
            'train_count':len(tr),'test_count':len(te),'model_type':model_type,'model':scores(y[te],prediction[te],labels),
            'baseline_class':baseline_class,'baseline':scores(y[te],baseline_prediction[te],labels)})
        if not forest: folds[-1]['tree']=folds[-1]['model']
    assert np.all(seen==1)
    model=constructor(**params).fit(X,y)
    artifact=forest_json(model,names) if forest else tree_json(model,names)
    infer=predict_forest if forest else predict_tree
    assert np.array_equal(model.predict(X),infer(artifact,X))
    perm=np.concatenate(permutations,axis=1)
    ranking=[{'feature':k,'definition':FEATURES[k],'gini_importance':float(model.feature_importances_[i]),
              'held_out_permutation_macro_f1_mean':float(perm[i].mean()),
              'held_out_permutation_macro_f1_std':float(perm[i].std())} for i,k in enumerate(names)]
    ranking.sort(key=lambda r:(-r['gini_importance'],r['feature']))
    for rank,item in enumerate(ranking,1): item['rank']=rank
    counts=Counter(y.tolist());layout_counts=Counter(groups.tolist())
    report={'status':'diagnostic_only','teacher':'coordinate_search_not_ppo','golden_replay_performed':False,
        'source':str(Path(args.input).resolve()),'audit':audit,'samples':len(y),'features':names,'parameters':params,'model_type':model_type,
        'task':'move_only_four_class' if drop_stay else 'five_class_with_stay','evaluation_classes':labels,
        'requires_external_move_selection':drop_stay,'baseline_name':baseline_name,
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},
        'environment':{'python_executable':sys.executable,'numpy_module':np.__file__,'sklearn_module':sklearn.__file__},
        'class_counts':dict(sorted(counts.items())),'layout_counts':dict(sorted(layout_counts.items())),
        'training_fit':scores(y,model.predict(X),labels),'leave_one_layout_out':scores(y,prediction,labels),
        'baseline':scores(y,baseline_prediction,labels),'folds':folds,'importance':ranking,
        'evaluation_note':f'固定参数、{len(set(groups))} 张有保留样本的训练版图内部留一验证；未使用独立 validation/test 或进行超参数搜索。',
        'importance_note':'Gini 为全样本最终模型的不纯度重要性，森林聚合各树贡献；置换重要性为各留出版图 macro-F1 下降等权均值，负值保留。相关特征会分摊重要性。'}
    if forest:
        report['forest_structure']={'tree_count':len(model.estimators_),
            'max_actual_depth':max(int(t.get_depth()) for t in model.estimators_),
            'mean_leaves':float(np.mean([t.get_n_leaves() for t in model.estimators_]))}
    else: report.update(actual_depth=int(model.get_depth()),leaf_count=int(model.get_n_leaves()))
    out.mkdir(parents=True,exist_ok=True)
    if not drop_stay: report['all_stay_baseline']=report['baseline']
    artifact.update(task=report['task'],requires_external_move_selection=drop_stay)
    model_file='random_forest.joblib' if forest else 'decision_tree.joblib'
    write_json(out/'metrics.json',report);write_json(out/('forest.json' if forest else 'tree.json'),artifact)
    joblib.dump({'model':model,'feature_names':names,'source_sha256':audit['source_xlsx_sha256'],
                 'status':'diagnostic_only','task':report['task'],'requires_external_move_selection':drop_stay,
                 'model_type':model_type,'class_offsets_nm':{k:k*10 for k in labels}},out/model_file)
    assert np.array_equal(joblib.load(out/model_file)['model'].predict(X),model.predict(X))
    if not forest:
        (out/'tree_rules.txt').write_text(export_text(model,feature_names=names,max_depth=args.max_depth),encoding='utf-8')
    header=['rank','feature','definition','gini_importance','held_out_permutation_macro_f1_mean','held_out_permutation_macro_f1_std']
    write_csv(out/'feature_importance.csv',header,[[r[k] for k in header] for r in ranking])
    write_csv(out/'training_binary.csv',['epe_id']+names+['result'],[[eid]+row.tolist()+[int(label)] for eid,row,label in zip(ids,X,y)])
    write_csv(out/'out_of_fold_predictions.csv',['epe_id','layout_parent','result','prediction',baseline_name],
              zip(ids,groups,y.tolist(),prediction.tolist(),baseline_prediction.tolist()))
    # 固定样本来源哈希；下载覆盖源文件时不把两个快照混成同一次实验。
    if hashlib.sha256(Path(args.input).read_bytes()).hexdigest()!=audit['source_xlsx_sha256']:
        raise RuntimeError('训练期间源 Excel 发生变化，请在新目录重新运行')
    if not args.no_plots: plots(out,model,names,ranking,forest=forest)
    baseline_title='各折训练集多数类对照' if drop_stay else '全 stay 对照'
    macro_key='macro_f1_fixed_'+('four' if drop_stay else 'five')+'_classes'
    model_name='随机森林' if forest else '决策树'
    lines=[f'# v2 EPE {model_name}训练结果','',f'样本 {len(y)}，特征 {len(names)}，删除空值列：{audit["dropped_null_features"]}。',
        f'从 {audit["source_samples"]} 条样本中排除 {audit["removed_stay_count"]} 条不移动点；评估类别 {labels}。',
        f'固定 max_depth={args.max_depth}，min_samples_leaf={args.min_samples_leaf}，class_weight={class_weight}，seed=0。',
        '',f'| 指标 | 训练集拟合 | 按版图留一验证 | {baseline_title} |','| --- | ---: | ---: | ---: |']
    for title,key in [('Accuracy','accuracy'),(f'Macro-F1（固定 {len(labels)} 类）',macro_key),('Balanced accuracy','balanced_accuracy_present_classes')]:
        lines.append(f'| {title} | {report["training_fit"][key]:.4f} | {report["leave_one_layout_out"][key]:.4f} | {report["baseline"][key]:.4f} |')
    if forest: lines+=['',f'森林共 {params["n_estimators"]} 棵树，bootstrap=True，max_features=sqrt，按各树类别概率平均预测。']
    lines+=['','## 特征重要性（全样本最终模型 Gini）','','| 排名 | 特征 | 重要性 | 留出版图置换重要性 |','| --- | --- | ---: | ---: |']
    lines += [f'| {r["rank"]} | {r["feature"]} | {r["gini_importance"]:.4%} | {r["held_out_permutation_macro_f1_mean"]:.5f} |' for r in ranking]
    lines+=['','重要性为当前模型的贡献，不是因果效应。类型和方向等相关列可能分摊或替代彼此。',
        '这是坐标搜索教师的探索性蒸馏。未出现在 samples 的失败/待复核记录没有补入训练。',
        '按版图留一验证仍限于保留样本所在训练版图，分类指标不能代替完整 Recipe 的 Golden 光刻回放。']
    if drop_stay:
        lines+=['此模型只在已知需要移动的点中分类，不具备判断是否移动的能力；不能直接替代全点五分类树。',
                '没有保留样本的版图：'+str(audit['layouts_without_retained_samples'])+'。',
                '过滤改变了评价样本与类别空间，指标不能直接与此前包含 stay 的五分类结果比较。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(out),'samples':len(y),'features':len(names),
        'validation':report['leave_one_layout_out'],'baseline':report['baseline'],
        'top_features':ranking[:8]},ensure_ascii=False,indent=2))
    return report


def plots(out,model,names,ranking,forest=False):
    import re
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree
    fig,ax=plt.subplots(figsize=(11,9))
    ax.barh([r['feature'] for r in ranking[::-1]],[r['gini_importance'] for r in ranking[::-1]],color='#416b99')
    ax.set_xlabel('Gini importance (full-data fitted model)');ax.set_title('Random forest feature importance' if forest else 'EPE feature importance')
    fig.tight_layout();fig.savefig(out/'feature_importance.png',dpi=170);plt.close(fig)
    if forest: return
    fig,ax=plt.subplots(figsize=(36,14))
    artists=plot_tree(model,feature_names=names,class_names=[str(k) for k in model.classes_],
              filled=True,rounded=True,impurity=False,fontsize=9,ax=ax)
    # 省去很长的加权类别计数，避免相邻节点遮挡；完整概率保留在 tree.json。
    for artist in artists:
        artist.set_text(re.sub(r'value = \[.*?\]\n?','',artist.get_text(),flags=re.S))
    classes='/'.join(str(k) for k in model.classes_)
    offsets='/'.join(str(k*10) for k in model.classes_)
    ax.set_title(f'EPE decision tree | class {classes} = {offsets} nm',fontsize=16)
    fig.tight_layout();fig.savefig(out/'decision_tree.svg');fig.savefig(out/'decision_tree.png',dpi=150);plt.close(fig)


def main():
    def depth(value):
        return None if value.lower()=='none' else int(value)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='runs/v2-vision-dataset-004/training.xlsx')
    parser.add_argument('--output',required=True)
    parser.add_argument('--max-depth',type=depth,default=5,help='正整数或 none')
    parser.add_argument('--min-samples-leaf',type=int,default=10)
    parser.add_argument('--drop-stay',action='store_true',help='先删除 result=0 样本，再清理空值列并训练四分类树')
    parser.add_argument('--model-type',choices=['tree','forest'],default='tree')
    parser.add_argument('--n-estimators',type=int,default=300,help='随机森林树数量，默认 300')
    parser.add_argument('--max-features',choices=['sqrt','0.5','1.0'],default='sqrt')
    parser.add_argument('--class-weight',choices=['balanced','balanced_subsample','none'],default='balanced')
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    if args.max_features!='sqrt': args.max_features=float(args.max_features)
    train(args)


if __name__=='__main__': main()
