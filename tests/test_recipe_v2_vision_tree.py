"""验证 Excel 决策树的数据清洗、标签隔离、分组评估和模型导出。

使用合成布尔样本和临时 OOXML 工作簿，不调用 API、GPU 或光刻求解器；验证缺失值按列
删除、false 不丢失、编号/来源不进入训练、错误标签与重复主键被拒绝及树 JSON 可复现预测。
"""
from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pytest

from opc_agent.recipe_v2_vision import write_excel
from opc_agent.recipe_v2_vision_prompt import FEATURES
from opc_agent import recipe_v2_vision_tree as tree


def workbook(tmp_path,change=None):
    samples=[['epe_id']+list(FEATURES)+['result']]
    provenance=[['epe_id','layout_parent','point_id','normal_offset_nm','recipe_sha256']]
    for eid in range(30):
        label=tree.LABELS[eid%5]
        samples.append([eid]+[bool((eid+j)%2) for j in range(len(FEATURES))]+[label])
        provenance.append([eid,'L'+str(eid//10),'p'+str(eid),label*10,'source'])
    sheets={'samples':samples,'provenance':provenance,
            'summary':[['field','value'],['teacher','coordinate_search_not_ppo'],['training_rows',30],['total',32]]}
    if change: change(sheets)
    path=tmp_path/'training.xlsx';write_excel(path,sheets)
    return path


def test_null_column_dropped_false_preserved_and_ids_excluded(tmp_path):
    path=workbook(tmp_path,lambda s:s['samples'][1].__setitem__(2,'null'))
    X,y,g,ids,names,audit=tree.load_data(path)
    assert X.shape==(30,27)
    assert audit['dropped_null_features']==[list(FEATURES)[1]]
    assert X[0,0]==0 and X[1,0]==1
    assert 'result' not in names and 'epe_id' not in names and 'layout_parent' not in names
    assert y.tolist()==tree.LABELS*6
    assert len(set(g))==3 and len(ids)==30


@pytest.mark.parametrize('value',[None,'null','NULL','nan','None','',float('nan')])
def test_missing_values(value): assert tree.is_null(value)


@pytest.mark.parametrize('value',[False,True,0,1,'false','true'])
def test_boolean_conversion(value):
    assert tree.binary(value) in (0,1)
    assert not tree.is_null(value)


@pytest.mark.parametrize('value',['wrong','0',2,-1])
def test_invalid_binary_rejected(value):
    with pytest.raises(ValueError): tree.binary(value)


def test_duplicate_sample_id_rejected(tmp_path):
    path=workbook(tmp_path,lambda s:s['samples'][2].__setitem__(0,0))
    with pytest.raises(ValueError,match='编号重复'):tree.load_data(path)


def test_provenance_mismatch_rejected(tmp_path):
    path=workbook(tmp_path,lambda s:s['provenance'][1].__setitem__(3,99))
    with pytest.raises(ValueError,match='位移不一致'):tree.load_data(path)


def test_full_training_preserves_input_and_holds_out_layouts(tmp_path):
    path=workbook(tmp_path);before=hashlib.sha256(path.read_bytes()).hexdigest()
    out=tmp_path/'tree'
    args=SimpleNamespace(input=str(path),output=str(out),max_depth=3,min_samples_leaf=2,no_plots=True)
    report=tree.train(args)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    assert report['status']=='diagnostic_only' and not report['golden_replay_performed']
    assert len(report['folds'])==3
    assert sum(f['test_count'] for f in report['folds'])==30
    for fold in report['folds']: assert fold['held_out_layout'] not in fold['train_layouts']
    assert len(report['importance'])==28
    X,y,*_=tree.load_data(path)
    artifact=json.loads((out/'tree.json').read_text(encoding='utf-8'))
    prediction=tree.predict_tree(artifact,X)
    assert len(prediction)==len(y) and set(prediction)<=set(tree.LABELS)
    assert (out/'feature_importance.csv').exists()
    assert (out/'decision_tree.joblib').exists()
    with pytest.raises(ValueError,match='输出目录'):tree.train(args)


def test_drop_stay_before_null_column_cleanup(tmp_path):
    # 第三个样本 result=0，其 null 不应导致移动样本中的有效特征被删除。
    path=workbook(tmp_path,lambda s:s['samples'][3].__setitem__(2,'null'))
    X,y,g,ids,names,audit=tree.load_data(path,drop_stay=True)
    assert X.shape==(24,28) and set(y)=={-2,-1,1,2}
    assert audit['removed_stay_count']==6 and audit['source_samples']==30
    assert set(ids).isdisjoint(audit['removed_stay_epe_ids'])
    assert audit['dropped_null_features']==[]


def test_layout_with_only_stay_is_not_a_validation_fold(tmp_path):
    def change(s):
        for row in s['samples'][21:]: row[-1]=0
        for row in s['provenance'][21:]: row[3]=0
    path=workbook(tmp_path,change)
    _,y,g,_,_,audit=tree.load_data(path,drop_stay=True)
    assert len(y)==16 and set(g)=={'L0','L1'}
    assert audit['layouts_without_retained_samples']==['L2']


def test_all_stay_rejected_after_filter(tmp_path):
    def change(s):
        for row in s['samples'][1:]: row[-1]=0
        for row in s['provenance'][1:]: row[3]=0
    with pytest.raises(ValueError,match='没有可训练样本'):
        tree.load_data(workbook(tmp_path,change),drop_stay=True)


def test_four_class_metrics_do_not_include_zero():
    labels=[-2,-1,1,2]
    metrics=tree.scores(np.array(labels),np.array(labels),labels)
    assert metrics['macro_f1_fixed_four_classes']==1
    assert set(metrics['per_class'])=={'-2','-1','1','2'}
    assert len(metrics['confusion_matrix'])==4


def test_move_only_training_exports_no_zero(tmp_path):
    import csv
    import joblib
    path=workbook(tmp_path);before=path.read_bytes();out=tmp_path/'move-only'
    report=tree.train(SimpleNamespace(input=str(path),output=str(out),max_depth=3,
        min_samples_leaf=2,no_plots=True,drop_stay=True))
    assert path.read_bytes()==before
    assert report['samples']==24 and report['evaluation_classes']==[-2,-1,1,2]
    assert report['requires_external_move_selection'] and report['baseline_name']=='training_fold_majority'
    assert 'all_stay_baseline' not in report
    saved=joblib.load(out/'decision_tree.joblib')
    assert set(saved['model'].classes_)=={-2,-1,1,2}
    rows=list(csv.DictReader((out/'training_binary.csv').open(encoding='utf-8-sig')))
    assert len(rows)==24 and all(int(r['result'])!=0 for r in rows)
    for fold in report['folds']:
        assert fold['baseline_class']==-2  # 此合成数据的每折训练集四类同频，确定性选最小类。
        assert '0' not in fold['tree']['per_class']


def test_random_forest_keeps_protocol_and_portable_class_mapping(tmp_path):
    import joblib
    path=workbook(tmp_path);before=path.read_bytes();out=tmp_path/'forest'
    report=tree.train(SimpleNamespace(input=str(path),output=str(out),max_depth=3,
        min_samples_leaf=2,no_plots=True,drop_stay=True,model_type='forest',n_estimators=7))
    assert path.read_bytes()==before
    assert report['samples']==24 and report['evaluation_classes']==[-2,-1,1,2]
    assert report['model_type']=='forest' and report['forest_structure']['tree_count']==7
    assert len(report['folds'])==3
    for fold in report['folds']:
        assert fold['held_out_layout'] not in fold['train_layouts']
        assert '0' not in fold['model']['per_class']
    saved=joblib.load(out/'random_forest.joblib')
    artifact=json.loads((out/'forest.json').read_text(encoding='utf-8'))
    assert artifact['classes']==[-2,-1,1,2] and artifact['aggregation']=='mean_class_probabilities'
    assert all(t['classes']==artifact['classes'] for t in artifact['trees'])
    assert all(n['class'] in {-2,-1,1,2} for t in artifact['trees'] for n in t['nodes'])
    X=np.random.RandomState(0).randint(0,2,(64,len(FEATURES)))
    assert np.array_equal(tree.predict_forest(artifact,X),saved['model'].predict(X))
    assert not (out/'tree_rules.txt').exists()
    assert abs(sum(r['gini_importance'] for r in report['importance'])-1)<1e-10


def test_invalid_forest_tree_count_rejected(tmp_path):
    path=workbook(tmp_path)
    with pytest.raises(ValueError,match='n-estimators'):
        tree.train(SimpleNamespace(input=str(path),output=str(tmp_path/'bad'),max_depth=3,
            min_samples_leaf=2,no_plots=True,drop_stay=True,model_type='forest',n_estimators=0))


def test_unlimited_depth_and_balanced_subsample_supported(tmp_path):
    path=workbook(tmp_path);out=tmp_path/'tuned'
    report=tree.train(SimpleNamespace(input=str(path),output=str(out),max_depth=None,
        min_samples_leaf=1,no_plots=True,drop_stay=True,model_type='forest',n_estimators=5,
        max_features=.5,class_weight='balanced_subsample'))
    assert report['parameters']['max_depth'] is None
    assert report['parameters']['max_features']==.5
    assert report['parameters']['class_weight']=='balanced_subsample'
