"""验证 EPE 视觉数据链路的源标签校验、双图标注缓存、异常处理及 Excel 布尔导出。

使用合成几何和替代 API，不读取密钥、不发送网络请求、不依赖 GPU；这些测试不代表云端视觉准确率。
"""
import json
import sys
import types
import zipfile
from pathlib import Path
from types import SimpleNamespace as NS
from xml.etree import ElementTree as ET
import numpy as np
import pytest
from opc_agent import recipe_v2_vision as v
from opc_agent.recipe_v2_vision_prompt import FEATURES, point_prompt, validate_response


def response(eid):
    features=dict.fromkeys(FEATURES,False)
    features.update(type_H=True,on_horizontal_edge=True)
    return json.dumps({'epe_id':eid,'features':features,'uncertain_features':[]})


@pytest.fixture
def dataset(tmp_path):
    root=tmp_path/'dataset';root.mkdir()
    rows=[]
    for eid in range(3):
        paths=[]
        for name in ('local','context'):
            p=root/f'{eid}-{name}.png';p.write_bytes(b'fake-image')
            paths.append({'path':p.name,'sha256':v.digest(p)})
        rows.append({'epe_id':eid,'point_id':f'p{eid}','layout_parent':f'L{eid}',
                     'images':paths,'result':eid-2,'normal_offset_nm':(eid-2)*10,'recipe_sha256':'abc',
                     'normal_xy':[0,-1],
                     'geometry_evidence':{'segment_start_xy':[0,0],'segment_end_xy':[16,0],
                                          'start_corner_type':0,'end_corner_type':0},
                     'geometry_features':{k:json.loads(response(eid))['features'][k] for k in v.GEOMETRY_FEATURES},
                     'corner_evidence':{'clockwise_start_corner_type':0,'clockwise_end_corner_type':0}})
    manifest={'teacher':'coordinate_search_not_ppo','rows':rows,'visual_protocol':v.PROMPT_VERSION}
    manifest['content_sha256']=v.identity(manifest)
    v.save(root/'manifest.json',manifest)
    return root


def args(root,**kwargs):
    return NS(dataset=str(root),model='fake',base_url='https://example.invalid/v1',limit=60,retries=0,retry_uncertain=False,timeout_seconds=300.0,**kwargs)


def fake(monkeypatch,handler):
    class Client:
        def __init__(self,*a,**kwargs):
            assert kwargs['timeout_seconds']==300.0
        def extract_point_features(self,paths,prompt):
            assert len(paths)==2
            assert 'normal_offset_nm' not in prompt and 'result' not in prompt
            return handler(int(prompt.rsplit('：',1)[1]))
    monkeypatch.setitem(sys.modules,'opc_agent.qwen',types.SimpleNamespace(SiliconFlowQwen=Client))


def test_schema_rejects_string_bool_and_wrong_id():
    obj=json.loads(response(1));obj['features']['near_jog']='false'
    with pytest.raises(ValueError):validate_response(json.dumps(obj),1)


def test_prompt_exposes_segment_direction_without_label_or_action():
    prompt=point_prompt(8)
    assert '蓝色箭头' in prompt and '顺时针方向' in prompt
    assert 'action_index' not in prompt and 'normal_offset_nm' not in prompt
    with pytest.raises(ValueError):validate_response(response(1),2)
    obj=json.loads(response(1));obj['features']['on_vertical_edge']=True
    with pytest.raises(ValueError):validate_response(json.dumps(obj),1)


def test_cache_budget_and_export(dataset,monkeypatch):
    calls=[]
    fake(monkeypatch,lambda eid:(calls.append(eid) or response(eid)))
    a=args(dataset);a.limit=1
    v.annotate(a); assert calls==[0]
    attempt=v.read(dataset/'annotations/00000.json')['attempts'][-1]
    assert attempt['elapsed_seconds']>=0 and attempt['timeout_seconds']==300.0
    v.export(a);assert v.read(dataset/'training.json')['complete'] is False
    a.limit=3;v.annotate(a);v.annotate(a);assert calls==[0,1,2]
    v.export(a);obj=v.read(dataset/'training.json')
    assert obj['complete'] and len(obj['samples'])==3
    assert [r['result'] for r in obj['samples']]==[-2,-1,0]
    with zipfile.ZipFile(dataset/'training.xlsx') as z:
        for name in z.namelist():ET.fromstring(z.read(name))
        sheet=ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        cells=sheet.findall('.//s:c[@t="b"]',ns)
        assert len(cells)==3*len(FEATURES)
        actual=[int(c.find('s:v',ns).text) for c in cells]
        expected=[int(r['features'][k]) for r in obj['samples'] for k in FEATURES]
        assert actual==expected
    a.model='other'
    with pytest.raises(ValueError):v.annotate(a)


def test_unknown_and_failure_excluded(dataset,monkeypatch):
    def handler(eid):
        if eid==1:return 'bad json'
        obj=json.loads(response(eid));obj['features']['near_jog']=None;obj['uncertain_features']=['near_jog']
        return json.dumps(obj)
    fake(monkeypatch,handler);a=args(dataset)
    v.annotate(a);v.export(a)
    assert v.read(dataset/'training.json')['training_rows']==0
    assert v.read(dataset/'annotations/00000.json')['status']=='needs_review'
    assert v.read(dataset/'annotations/00001.json')['status']=='failed'
    assert v.read(dataset/'annotations/00001.json')['attempts'][-1]['elapsed_seconds']>=0
    (dataset/'0-local.png').write_bytes(b'changed')
    with pytest.raises(ValueError):v.export(a)


def test_retry_budget(dataset,monkeypatch):
    calls=[];fake(monkeypatch,lambda eid:(calls.append(eid) or 'bad'))
    a=args(dataset);a.limit=2;a.retries=2
    monkeypatch.setattr(v.time,'sleep',lambda x:None)
    v.annotate(a);assert calls==[0,0]


def test_prepare_and_point_mismatch(tmp_path,monkeypatch):
    import yaml
    source=tmp_path/'run';source.mkdir()
    v.save(source/'recipe-v2-search.json',{'test':True})
    config={'data':{'train_parents':['L']},'recipe_v2':{'fragment_parameters_nm':{'corner':16,'uniform':32}}}
    (source/'config.snapshot.yaml').write_text(yaml.safe_dump(config))
    labels={'source_run':'run','source_summary_sha256':v.digest(source/'recipe-v2-search.json'),
            'teacher':'coordinate_search_not_ppo','action_offsets_nm':[-20,-10,0,10,20],
            'fragment_parameters_nm':{'corner':16,'uniform':32},
            'recipes':[{'layout_parent':'L','recipe_sha256':'hash','labels':[{'point_id':'p','action_index':4,'normal_offset_nm':20}]}]}
    p=tmp_path/'labels.json';v.save(p,labels)
    v.save(source/'L/seed-0/coordinate/result.json',{'final_replay_equal':True,'best':{'recipe_sha256':'hash','actions':{'p':4}}})
    point=NS(point_id='p',base_xy=(32,32),segment_start_xy=(16,32),segment_end_xy=(48,32),normal_xy=(0,-1),
             start_corner_type=1,end_corner_type=1)
    target=np.zeros((64,64));target[32:50,16:49]=1
    monkeypatch.setattr(v,'rebuild',lambda *x:(target,[point],'glphash'))
    a=NS(labels=str(p),source_run=str(source),output=str(tmp_path/'out'),openilt_dir=None,iccad13_dir=None,local_window=32,context_window=64)
    v.prepare(a);m=v.read(Path(a.output)/'manifest.json')
    assert m['rows'][0]['result']==2 and len(m['rows'][0]['images'])==2
    with pytest.raises(ValueError):v.prepare(a)
    point.point_id='wrong';a.output=str(tmp_path/'bad')
    with pytest.raises(ValueError,match='点集'):v.prepare(a)


def test_qwen_sends_two_images_without_labels(tmp_path,monkeypatch):
    import importlib.util
    captured={}
    class OpenAI:
        def __init__(self,**kwargs):
            captured['init']=kwargs
            self.chat=NS(completions=NS(create=self.create))
        def create(self,**kwargs):
            captured['request']=kwargs
            raw=response(8)
            return [NS(choices=[NS(delta=NS(content=raw[:20]))]),
                    NS(choices=[NS(delta=NS(content=raw[20:]))])]
    monkeypatch.setitem(sys.modules,'openai',NS(OpenAI=OpenAI))
    monkeypatch.setenv('SILICONFLOW_API_KEY','test-placeholder')
    spec=importlib.util.spec_from_file_location('opc_agent._vision_transport_test',Path(v.__file__).with_name('qwen.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    paths=[tmp_path/'a.png',tmp_path/'b.png']
    for p in paths:p.write_bytes(b'png')
    client=module.SiliconFlowQwen('Qwen/Qwen3.8-27B')
    raw=client.extract_point_features(paths,point_prompt(8))
    assert validate_response(raw,8)
    content=captured['request']['messages'][1]['content']
    assert len(content)==3 and all(x['type']=='image_url' for x in content[1:])
    assert captured['init']['max_retries']==0
    assert captured['init']['timeout']==300.0
    module.SiliconFlowQwen('Qwen/Qwen3.8-27B',timeout_seconds=420.0)
    assert captured['init']['timeout']==420.0
    assert captured['request']['model']=='Qwen/Qwen3.8-27B'
    assert captured['request']['stream'] is True
    assert captured['request']['max_tokens']==1024
    assert captured['request']['extra_body']=={'enable_thinking':False}
    assert all(item['image_url']['detail']=='high' for item in content[1:])
    assert client.last_stream_diagnostics['content_chunk_count']==2
    with pytest.raises(ValueError):client.extract_point_features(paths[:1],point_prompt(8))


def test_manifest_tampering_rejected(dataset):
    m=v.read(dataset/'manifest.json');m['rows'][0]['result']=2;v.save(dataset/'manifest.json',m)
    with pytest.raises(ValueError,match='manifest'):v.export(args(dataset))


@pytest.mark.parametrize('timeout',[0,-1,float('nan'),float('inf')])
def test_invalid_timeout_rejected_before_request(dataset,timeout):
    a=args(dataset);a.timeout_seconds=timeout
    with pytest.raises(ValueError,match='timeout-seconds'):v.annotate(a)


@pytest.mark.parametrize('normal,start,end',[
    ((0,-1),(0,0),(10,0)), ((1,0),(10,0),(10,10)),
    ((0,1),(10,10),(0,10)), ((-1,0),(0,10),(0,0))])
def test_clockwise_independent_of_vertex_order(normal,start,end):
    for a,b in [(start,end),(end,start)]:
        p=NS(normal_xy=normal,segment_start_xy=a,segment_end_xy=b)
        assert v.clockwise_segment(p)==(start,end)


def test_recover_preserves_source_and_rejects_empty(dataset,tmp_path):
    before={}
    for eid in range(3):
        parsed=json.loads(response(eid))
        raw=json.dumps({str(eid): parsed['features'] if eid==0 else (parsed if eid==1 else {})})
        p=dataset/'annotations'/f'{eid:05d}.json'
        v.save(p,{'attempts':[{'raw_response':raw}],'prompt_version':'old'})
        before[p]=p.read_bytes()
    out=tmp_path/'recovery.json'
    v.recover(NS(dataset=str(dataset),output=str(out)))
    report=v.read(out)
    assert report['counts']=={'ok':2,'failed':1}
    assert all(p.read_bytes()==data for p,data in before.items())
    assert report['rows'][0]['source_prompt_version']=='old'


def test_selected_points_only(dataset,monkeypatch):
    calls=[];fake(monkeypatch,lambda eid:(calls.append(eid) or response(eid)))
    a=args(dataset);a.point_ids='2';v.annotate(a)
    assert calls==[2]


@pytest.mark.parametrize('normal,start,end',[
    ((0,-1),(0,0),(16,0)),((1,0),(16,0),(16,16)),
    ((0,1),(16,16),(0,16)),((-1,0),(0,16),(0,0))])
def test_geometry_corner_orientation_independent_of_winding(normal,start,end):
    for a,b,kinds in [(start,end,(1,0)),(end,start,(0,1))]:
        point=NS(normal_xy=normal,segment_start_xy=a,segment_end_xy=b,
                 start_corner_type=kinds[0],end_corner_type=kinds[1])
        features,evidence=v.geometry_metadata(point)
        assert features['on_start_corner_seg'] and not features['on_end_corner_seg']
        assert features['type_CH']==(start[1]==end[1])
        assert features['type_CV']==(start[0]==end[0])
        assert evidence=={'clockwise_start_corner_type':1,'clockwise_end_corner_type':0}


def test_both_corners_and_concave_corner_metadata():
    p=NS(normal_xy=(0,-1),segment_start_xy=(0,0),segment_end_xy=(16,0),start_corner_type=-1,end_corner_type=1)
    fixed,evidence=v.geometry_metadata(p)
    assert fixed['on_start_corner_seg'] and fixed['on_end_corner_seg'] and fixed['type_CH']
    assert evidence['clockwise_start_corner_type']==-1


def test_geometry_conflict_preserved_and_excluded(dataset,monkeypatch):
    def handler(eid):
        obj=json.loads(response(eid));obj['features']['on_start_corner_seg']=True
        return json.dumps(obj)
    fake(monkeypatch,handler)
    a=args(dataset);v.annotate(a);v.export(a)
    record=v.read(dataset/'annotations/00000.json')
    assert record['model_features']['on_start_corner_seg'] is True
    assert record['parsed']['features']['on_start_corner_seg'] is False
    assert record['review_reasons']==['geometry_mismatch:on_start_corner_seg']
    assert record['status']=='needs_review' and not record['parsed']['uncertain_features']
    assert v.read(dataset/'training.json')['training_rows']==0
    assert v.read(dataset/'annotation-summary.json')['counts']=={'needs_review':3}


def test_attached_corner_false_is_review_not_silently_repaired(dataset):
    row=v.read(dataset/'manifest.json')['rows'][0]
    row['corner_evidence']['clockwise_start_corner_type']=1
    result=v.merge_response(response(0),row)
    assert result['status']=='needs_review'
    assert result['parsed']['features']['near_convex_corner'] is False
    assert 'attached_corner_requires_true:near_convex_corner' in result['review_reasons']


def test_normalize_online_only_exact_complete_wrapper(dataset):
    row=v.read(dataset/'manifest.json')['rows'][0]
    obj=json.loads(response(0))
    for wrapped in [obj,obj['features']]:
        result=v.merge_response(json.dumps({'0':wrapped}),row)
        assert result['status']=='ok'
        assert result['model_features']==obj['features']
    for raw in ['{"0":{}}','{"1":{}}','[]']:
        with pytest.raises(ValueError): v.merge_response(raw,row)


def test_all_images_preflight_before_any_paid_call(dataset,monkeypatch):
    calls=[];fake(monkeypatch,lambda eid:(calls.append(eid) or response(eid)))
    (dataset/'2-context.png').unlink()
    with pytest.raises(ValueError,match='图片缺失'):v.annotate(args(dataset))
    assert calls==[]


def test_run_all_exports_and_resumes_failed_only(dataset,monkeypatch):
    calls=[]
    fake(monkeypatch,lambda eid:(calls.append(eid) or ('{}' if eid==1 else response(eid))))
    a=args(dataset);v.run_all(a)
    assert calls==[0,1,2]
    assert v.read(dataset/'training.json')['training_rows']==2
    assert v.read(dataset/'annotation-summary.json')['failure_types']=={'ValueError':1}
    fake(monkeypatch,lambda eid:(calls.append(eid) or response(eid)))
    v.run_all(a)
    assert calls==[0,1,2,1]
    assert v.read(dataset/'training.json')['complete']


def test_run_all_exports_on_interrupt(dataset,monkeypatch):
    def handler(eid):
        if eid==1: raise KeyboardInterrupt()
        return response(eid)
    fake(monkeypatch,handler)
    with pytest.raises(KeyboardInterrupt):v.run_all(args(dataset))
    assert v.read(dataset/'training.json')['training_rows']==1
    assert v.read(dataset/'annotation-summary.json')['counts']=={'ok':1,'missing':2}


def test_export_rejects_tampered_merged_features(dataset,monkeypatch):
    fake(monkeypatch,response);v.annotate(args(dataset))
    path=dataset/'annotations/00000.json';record=v.read(path)
    record['parsed']['features']['near_jog']=True;v.save(path,record)
    with pytest.raises(ValueError,match='原始模型响应'):v.export(args(dataset))


def test_geometry_evidence_validated(dataset):
    path=dataset/'manifest.json';manifest=v.read(path)
    manifest['rows'][0]['geometry_features']['on_start_corner_seg']=True
    manifest.pop('content_sha256');manifest['content_sha256']=v.identity(manifest);v.save(path,manifest)
    with pytest.raises(ValueError,match='分段证据'):v.load_manifest(dataset)


@pytest.mark.parametrize('interrupt',[False,True])
def test_stream_empty_or_interrupted_saves_diagnostics_and_closes(tmp_path,monkeypatch,interrupt):
    import importlib.util
    class Stream:
        closed=False
        def __iter__(self):
            if interrupt:
                yield NS(choices=[NS(delta=NS(content='{"epe_id":',reasoning_content='private reasoning'))])
                raise TimeoutError('transport timeout')
            return
        def close(self): self.closed=True
    stream=Stream()
    class OpenAI:
        def __init__(self,**kwargs): self.chat=NS(completions=NS(create=lambda **kw:stream))
    monkeypatch.setitem(sys.modules,'openai',NS(OpenAI=OpenAI))
    monkeypatch.setenv('SILICONFLOW_API_KEY','test-placeholder')
    spec=importlib.util.spec_from_file_location('opc_agent._vision_stream_test',Path(v.__file__).with_name('qwen.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    path=tmp_path/'test.png';path.write_bytes(b'png')
    client=module.SiliconFlowQwen('fake')
    with pytest.raises(TimeoutError if interrupt else module.QwenResponseError) as caught:
        client.extract_point_features([path,path],point_prompt(0))
    assert stream.closed
    assert caught.value.stream_diagnostics['chunk_count']==int(interrupt)
    assert 'private reasoning' not in json.dumps(caught.value.stream_diagnostics)
    assert caught.value.raw_response==('{"epe_id":' if interrupt else '')


def test_authentication_error_stops_full_run_and_closes(dataset,monkeypatch):
    calls=[];closed=[]
    class AuthError(Exception): status_code=401
    class Client:
        def __init__(self,*args,**kwargs): pass
        def close(self): closed.append(True)
        def extract_point_features(self,*args):
            calls.append(True)
            raise AuthError('credential must not be logged')
    monkeypatch.setitem(sys.modules,'opc_agent.qwen',NS(SiliconFlowQwen=Client))
    with pytest.raises(RuntimeError,match='认证错误'):v.run_all(args(dataset))
    assert len(calls)==1 and closed==[True]
    raw=(dataset/'annotations/00000.json').read_text(encoding='utf-8')
    assert 'credential must not be logged' not in raw
    assert json.loads(raw)['attempts'][-1]['http_status']==401
    assert v.read(dataset/'training.json')['training_rows']==0


def test_crop_keeps_geometry_above_legend(tmp_path):
    import cv2
    target=np.zeros((128,128));target[32:96,32:120]=1
    point=NS(base_xy=(32,40),normal_xy=(-1,0),segment_start_xy=(32,32),segment_end_xy=(32,48))
    for context,width in [(None,512),(64,1024)]:
        path=tmp_path/f'{width}.png'
        v.crop(target,point,128,path,context)
        pixels=cv2.imdecode(np.frombuffer(path.read_bytes(),dtype=np.uint8),cv2.IMREAD_COLOR)
        assert pixels.shape==(584,width,3)
        assert np.any(pixels[512:]!=255)


def test_fixed_features_and_corner_evidence_in_prompt_without_search_labels(dataset):
    row=v.read(dataset/'manifest.json')['rows'][0]
    prompt=v.row_prompt(row)
    assert 'geometry_features' in prompt and 'corner_evidence' in prompt
    assert 'layout_parent' not in prompt and 'normal_offset_nm' not in prompt
    assert 'action_index' not in prompt and row['recipe_sha256'] not in prompt
