"""验证随机森林固定网格与特征标签冲突统计的确定性和数据边界。"""
import numpy as np
from opc_agent.recipe_v2_vision_forest_search import conflict_report


def test_conflict_report_counts_exact_vectors_and_accuracy_ceiling():
    X=np.array([[0,0],[0,0],[0,0],[1,0],[1,1]],dtype=np.int8)
    y=np.array([-2,-2,1,2,-1])
    report=conflict_report(X,y,['a','b'])
    assert report['unique_feature_vectors']==3
    assert report['conflicting_vectors']==1
    assert report['samples_in_conflicting_vectors']==3
    assert report['conflicting_sample_fraction']==.6
    assert report['maximum_training_accuracy_from_exact_feature_vector']==.8
    assert report['groups'][0]['label_counts']=={-2:2,1:1}


def test_no_conflict_has_unit_accuracy_ceiling():
    report=conflict_report(np.eye(4,dtype=np.int8),np.array([-2,-1,1,2]),list('abcd'))
    assert report['conflicting_vectors']==0
    assert report['maximum_training_accuracy_from_exact_feature_vector']==1
