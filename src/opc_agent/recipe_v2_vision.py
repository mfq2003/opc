"""本模块提供云端 EPE 双图数据集制作、硅基流动标注和 JSON/Excel 导出入口。

从冻结搜索配置及完整标签重建原始 target 点，不运行 OPC 或 PPO；每点图像与响应均保存哈希。
标注支持有预算的串行调用、失败记录和缓存续跑；未知特征不进入训练表。Excel 用标准 OOXML
写出布尔单元格，无需额外表格依赖。程序不读取 .env；密钥仅由既有 Qwen 客户端读取环境变量。
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import sys
import time
import zipfile
from collections import Counter
from contextlib import ExitStack
from types import SimpleNamespace
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np
import yaml
from .recipe_v2_vision_prompt import (FEATURES, GEOMETRY_FEATURES, PROMPT_VERSION,
    point_prompt, validate_response, normalize_response)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rebuild(config, layout):
    """复用锁定 OpenILT 的 GLP、栅格化及项目分段算法，只加载几何。"""
    from .recipe_v2_openilt import _validate_openilt
    from .recipe_v2 import dissect_global_fragments
    from .recipe_v2_contract import FragmentParameters
    root = Path(config['data']['openilt_dir']).resolve()
    _validate_openilt(root, config['openilt']['commit'])
    sys.path.insert(0, str(root))
    glp = importlib.import_module('pycommon.glp')
    polygon = importlib.import_module('utils.polygon')
    for module, relative in [(glp, 'pycommon/glp.py'), (polygon, 'utils/polygon.py')]:
        if Path(module.__file__).resolve() != root / relative:
            raise RuntimeError('OpenILT 模块来源不匹配')
    recipe = config['recipe_v2']
    size = recipe['solver']['image_size']
    path = Path(config['data']['iccad13_dir']) / (layout + '.glp')
    design = glp.Design(str(path.resolve()), down=1)
    design.center(size[0], size[1], 0, 0)
    polygons = tuple(tuple((int(x),int(y)) for x,y in poly) for poly in design.polygons)
    target = np.asarray(polygon.poly2img([list(p) for p in polygons], *size, scale=1), dtype=np.float32) / 255.0
    frag = recipe['fragment_parameters_nm']
    geo = recipe['geometry_adapter']
    points = dissect_global_fragments(polygons, target, FragmentParameters(frag['corner'],frag['uniform']),
        recipe['nm_per_coordinate'], polygon.dissect, normal_probe_coordinate=geo['normal_probe_coordinate'],
        raster_scale=geo['raster_scale'], raster_offset_xy=geo['raster_offset_xy']).epe_points
    return target, points, digest(path)


def clockwise_segment(point):
    """屏幕右 x、下 y 坐标中使实体位于行进右侧，独立于原始顶点顺序。"""
    nx, ny = point.normal_xy
    tangent = (-ny, nx)
    start, end = point.segment_start_xy, point.segment_end_xy
    dot = (end[0]-start[0])*tangent[0]+(end[1]-start[1])*tangent[1]
    if dot == 0:
        raise ValueError('分段与外法线不正交')
    return (start, end) if dot > 0 else (end, start)


def geometry_metadata(point):
    """只用原始分段及角点拓扑确定八列；不从搜索动作或视觉预测推断。"""
    start,end=clockwise_segment(point)
    kinds=(int(point.start_corner_type),int(point.end_corner_type))
    if start != point.segment_start_xy: kinds=kinds[::-1]
    if any(k not in (-1,0,1) for k in kinds): raise ValueError('非法角点类型')
    horizontal=start[1]==end[1]
    corner=any(kinds)
    features={'type_CV':bool(corner and not horizontal),'type_CH':bool(corner and horizontal),
        'type_H':bool(not corner and horizontal),'type_V':bool(not corner and not horizontal),
        'on_horizontal_edge':horizontal,'on_vertical_edge':not horizontal,
        'on_start_corner_seg':kinds[0]!=0,'on_end_corner_seg':kinds[1]!=0}
    evidence={'clockwise_start_corner_type':kinds[0],'clockwise_end_corner_type':kinds[1]}
    return features,evidence


def row_prompt(row):
    return point_prompt(row['epe_id'],row['geometry_features'],row['corner_evidence'])


def merge_response(raw,row):
    """保留模型原值，合并几何真值；矛盾只标待复核，不静默放入训练集。"""
    obj,normalization=normalize_response(raw,row['epe_id'])
    model=validate_response(json.dumps(obj),row['epe_id'],check_semantics=False)
    features=dict(model['features']); reasons=[]
    for name,value in row['geometry_features'].items():
        if features[name] is not value: reasons.append('geometry_mismatch:'+name)
        features[name]=value
    for name,kind in [('near_convex_corner',1),('near_concave_corner',-1)]:
        if kind in row['corner_evidence'].values() and features[name] is not True:
            reasons.append('attached_corner_requires_true:'+name)
    parsed={'epe_id':row['epe_id'],'features':features,
            'uncertain_features':[k for k,v in features.items() if v is None]}
    try: validate_response(json.dumps(parsed),row['epe_id'])
    except ValueError as exc: reasons.append('semantic_conflict:'+str(exc))
    return {'parsed':parsed,'model_features':model['features'],
            'review_reasons':reasons,'normalization':normalization,
            'feature_sources':{k:('geometry' if k in GEOMETRY_FEATURES else 'vision') for k in FEATURES},
            'status':'needs_review' if reasons or parsed['uncertain_features'] else 'ok'}


def crop(target, point, window, output, local_window=None):
    """保留栅格坐标方向，以未知灰色补边，绘制不遮住点中心的标记。"""
    import cv2
    x,y = point.base_xy
    left,top = int(x-window//2),int(y-window//2)
    canvas = np.full((window,window,3), 180, dtype=np.uint8)
    h,w = target.shape
    x0,y0,x1,y1 = max(left,0),max(top,0),min(left+window,w),min(top+window,h)
    if x1>x0 and y1>y0:
        gray = np.where(target[y0:y1,x0:x1] > 0.5, 0, 255).astype(np.uint8)
        canvas[y0-top:y1-top,x0-left:x1-left] = gray[:,:,None]
    canvas = cv2.resize(canvas,(512,512),interpolation=cv2.INTER_NEAREST)
    def xy(p):
        return (round((p[0]-left)*512/window),round((p[1]-top)*512/window))
    if local_window:
        a = local_window/2
        cv2.rectangle(canvas,xy((x-a,y-a)),xy((x+a,y+a)),(0,165,255),1)
    start, end = clockwise_segment(point)
    cv2.arrowedLine(canvas,xy(start),xy(end),
                    (255,100,0),2,tipLength=.18)
    c = xy((x,y)); nx,ny = point.normal_xy
    cv2.arrowedLine(canvas,c,(c[0]+nx*22,c[1]+ny*22),(0,150,0),1,tipLength=.3)
    cv2.circle(canvas,c,5,(0,0,255),2)
    if not local_window:
        for name,p in [('SEG START',start),('SEG END',end)]:
            px,py=xy(p)
            pos=(min(max(px+10,4),402),min(max(py,14),498))
            cv2.putText(canvas,name,pos,cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),3)
            cv2.putText(canvas,name,pos,cv2.FONT_HERSHEY_SIMPLEX,.35,(160,60,0),1)
    if local_window:
        # 全图等比例缩放并留白，避免覆盖上下文；两面板仍作为一张图发送。
        scale = 512 / max(h,w)
        gh,gw = max(1,round(h*scale)),max(1,round(w*scale))
        overview = np.full((512,512,3),180,dtype=np.uint8)
        gray = np.where(target > .5,0,255).astype(np.uint8)
        small = cv2.resize(gray,(gw,gh),interpolation=cv2.INTER_AREA)
        ox,oy = (512-gw)//2,(512-gh)//2
        overview[oy:oy+gh,ox:ox+gw] = small[:,:,None]
        def global_xy(p):
            return (round(p[0]*scale)+ox,round(p[1]*scale)+oy)
        cv2.rectangle(overview,global_xy((left,top)),global_xy((left+window,top+window)),(0,165,255),1)
        cv2.circle(overview,global_xy((x,y)),5,(0,0,255),2)
        canvas = np.concatenate((canvas,overview),axis=1)
        cv2.putText(canvas,'CONTEXT',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(100,100,100),1)
        cv2.putText(canvas,'FULL TARGET',(520,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(100,100,100),1)
    # 图例放在图像下方独立留白区，避免覆盖版图；裁剪坐标与缩放不变。
    footer=np.full((72,canvas.shape[1],3),255,dtype=np.uint8)
    legend=[('RED: current EPE point',(0,0,210)),('BLUE: clockwise segment direction',(200,70,0)),
            ('GREEN: outward normal (+)',(0,120,0)),('ORANGE: crop window; BLACK: target',(70,70,70))]
    for i,(label,color) in enumerate(legend):
        cv2.putText(footer,label,(8,15+i*17),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
    canvas=np.concatenate((canvas,footer),axis=0)
    ok, encoded = cv2.imencode('.png',canvas)
    if not ok:
        raise RuntimeError('PNG 写入失败')
    Path(output).write_bytes(encoded.tobytes())


def prepare(args):
    labels = read(args.labels)
    source = Path(args.source_run)
    if source.name != labels['source_run'] or digest(source/'recipe-v2-search.json') != labels['source_summary_sha256']:
        raise ValueError('搜索源目录或总表哈希不匹配')
    config_path = source/'config.snapshot.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if labels['action_offsets_nm'] != [-20,-10,0,10,20] or labels['teacher'] != 'coordinate_search_not_ppo':
        raise ValueError('只支持 v2 坐标搜索五动作标签')
    if labels['fragment_parameters_nm'] != config['recipe_v2']['fragment_parameters_nm']:
        raise ValueError('FRAG 与搜索配置不一致')
    if {r['layout_parent'] for r in labels['recipes']} != set(config['data']['train_parents']):
        raise ValueError('必须包含源配置的全部训练版图')
    if args.openilt_dir: config['data']['openilt_dir'] = args.openilt_dir
    if args.iccad13_dir: config['data']['iccad13_dir'] = args.iccad13_dir
    if not 32 <= args.local_window < args.context_window <= 4096:
        raise ValueError('窗口必须满足 32 <= local < context <= 4096')
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('prepare 输出目录须为空，避免覆盖已有数据')
    out.mkdir(parents=True,exist_ok=True)
    (out/'images').mkdir(exist_ok=True)
    rows=[]; seen=set()
    for recipe in labels['recipes']:
        layout=recipe['layout_parent']
        if layout not in config['data']['train_parents'] or layout in seen:
            raise ValueError('版图重复或不在训练划分')
        seen.add(layout)
        result=read(source/layout/'seed-0'/'coordinate'/'result.json')
        if not result['final_replay_equal'] or result['best']['recipe_sha256'] != recipe['recipe_sha256']:
            raise ValueError('源回放或 Recipe 身份不一致')
        target,points,glp_hash=rebuild(config,layout)
        by_id={p.point_id:p for p in points}
        entries=recipe['labels']
        if len(entries)!=len(by_id) or {x['point_id'] for x in entries}!=set(by_id):
            raise ValueError('重建点集与完整标签不一致')
        for label in entries:
            pid=label['point_id']; action=label['action_index']
            if type(action) is not int or not 0<=action<5 or label['normal_offset_nm'] != labels['action_offsets_nm'][action] or result['best']['actions'][pid]!=action:
                raise ValueError('标签与源动作不一致')
            eid=len(rows); point=by_id[pid]; paths=[]
            geometry_features,corner_evidence=geometry_metadata(point)
            for name,window in [('local',args.local_window),('context',args.context_window)]:
                path=Path('images')/f'{eid:05d}-{name}.png'
                crop(target,point,window,out/path,args.local_window if name=='context' else None)
                paths.append({'path':path.as_posix(),'sha256':digest(out/path)})
            rows.append({'epe_id':eid,'point_id':pid,'layout_parent':layout,'split':'train',
                'base_xy':list(point.base_xy),'normal_xy':list(point.normal_xy),'images':paths,
                'geometry_features':geometry_features,'corner_evidence':corner_evidence,
                'geometry_evidence':{'segment_start_xy':list(point.segment_start_xy),
                    'segment_end_xy':list(point.segment_end_xy),'start_corner_type':int(point.start_corner_type),
                    'end_corner_type':int(point.end_corner_type)},
                'layout_sha256':glp_hash,'recipe_sha256':recipe['recipe_sha256'],
                'action_index':action,'normal_offset_nm':label['normal_offset_nm'],'result':action-2})
        print(f'{layout}: {len(entries)} 点已裁图',flush=True)
    manifest={'schema_version':'v2-vision-dataset-v1','teacher':labels['teacher'],'status':'diagnostic_only',
        'labels_sha256':digest(args.labels),'source_summary_sha256':labels['source_summary_sha256'],
        'config_sha256':digest(config_path),'source_run':source.name,'local_window':args.local_window,
        'context_window':args.context_window,'visual_protocol':PROMPT_VERSION,'rows':rows}
    manifest['content_sha256']=identity(manifest)
    save(out/'manifest.json',manifest)
    save(out/'feature_dictionary.json',FEATURES)
    print(f'完成 {len(rows)} 点、{len(rows)*2} 张图；未调用 API 或 solver')


def load_manifest(root):
    """检查数据集身份及样本编号，避免后续修改标签或图片索引后继续使用缓存。"""
    obj=read(root/'manifest.json')
    expected=obj.pop('content_sha256')
    if identity(obj)!=expected: raise ValueError('manifest 内容哈希不一致')
    obj['content_sha256']=expected
    if [r['epe_id'] for r in obj['rows']] != list(range(len(obj['rows']))):
        raise ValueError('样本编号不连续或重复')
    for row in obj['rows']:
        if len(row['images'])!=2 or type(row['result']) is not int or row['result'] not in (-2,-1,0,1,2):
            raise ValueError('图片数量或动作类别错误')
        for im in row['images']:
            try: (root/im['path']).resolve().relative_to(root.resolve())
            except ValueError: raise ValueError('图片必须位于数据集目录内')
        if obj.get('visual_protocol') == PROMPT_VERSION:
            actual=row.get('geometry_features',{})
            if set(actual)!=set(GEOMETRY_FEATURES) or any(type(v) is not bool for v in actual.values()):
                raise ValueError('几何特征缺失或类型错误，请重新 prepare')
            evidence=row['geometry_evidence']
            point=SimpleNamespace(normal_xy=tuple(row['normal_xy']),
                segment_start_xy=tuple(evidence['segment_start_xy']),segment_end_xy=tuple(evidence['segment_end_xy']),
                start_corner_type=evidence['start_corner_type'],end_corner_type=evidence['end_corner_type'])
            fixed,corners=geometry_metadata(point)
            if actual!=fixed or row['corner_evidence']!=corners:
                raise ValueError('几何特征与原始分段证据不一致')
    return obj


def annotate(args):
    """关闭客户端并无论正常结束或中断都保存当前汇总。"""
    root=Path(args.dataset)
    manifest=load_manifest(root)
    if manifest.get('visual_protocol') != PROMPT_VERSION:
        raise ValueError('图片协议与当前 Prompt 不匹配，请在新目录运行 prepare')
    with ExitStack() as resources:
        try: _annotate(args,resources)
        finally: summarize(root,manifest)


def _annotate(args,resources):
    """串行有界请求；缓存身份包括图片、模型、端点和完整提示词，失败可续跑。"""
    if not np.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        raise ValueError('timeout-seconds 必须是有限正数')
    root=Path(args.dataset); manifest=load_manifest(root)
    if manifest.get('visual_protocol') != PROMPT_VERSION:
        raise ValueError('图片协议与当前 Prompt 不匹配，请在新目录运行 prepare')
    rows=manifest['rows']; selected=[]
    # 轮转取样确保小预算覆盖全部版图。
    groups={}
    for row in rows: groups.setdefault(row['layout_parent'],[]).append(row)
    for i in range(max(len(g) for g in groups.values())):
        selected.extend(g[i] for g in groups.values() if i<len(g))
    ids = getattr(args,'point_ids',None)
    if ids:
        wanted={int(value) for value in ids.split(',')}
        if not wanted <= {r['epe_id'] for r in rows}: raise ValueError('point-ids 存在未知编号')
        selected=[r for r in selected if r['epe_id'] in wanted]
    if args.limit<1 or args.retries<0 or args.retries>5: raise ValueError('limit 必须为正，retries 为 0..5')
    # 在第一笔付费调用前检查所选图片及缓存身份，避免运行数小时后才发现缺图。
    for row in selected:
        for im in row['images']:
            if not (root/im['path']).is_file(): raise ValueError('图片缺失：'+im['path'])
            if digest(root/im['path'])!=im['sha256']: raise ValueError('图片哈希变化，停止标注')
        path=root/'annotations'/f'{row["epe_id"]:05d}.json'
        key=identity({'images':row['images'],'prompt':row_prompt(row),'model':args.model,'base_url':args.base_url})
        if path.exists() and read(path)['request_sha256']!=key:
            raise ValueError('缓存模型/提示词/图片发生变化，请使用新的数据集目录')
    calls=0; client=None
    for row in selected:
        eid=row['epe_id']; prompt=row_prompt(row)
        for im in row['images']:
            if digest(root/im['path'])!=im['sha256']: raise ValueError('图片哈希变化，停止标注')
        key=identity({'images':row['images'],'prompt':prompt,'model':args.model,'base_url':args.base_url})
        path=root/'annotations'/f'{eid:05d}.json'
        if path.exists():
            old=read(path)
            if old['request_sha256']!=key: raise ValueError('缓存模型/提示词/图片发生变化，请使用新的数据集目录')
            if old['status']=='ok' or (old['status']=='needs_review' and not args.retry_uncertain): continue
        if calls>=args.limit: break
        if client is None:
            from .qwen import SiliconFlowQwen
            client=SiliconFlowQwen(args.model,args.base_url,timeout_seconds=args.timeout_seconds)
            if hasattr(client,'close'): resources.callback(client.close)
        record={'epe_id':eid,'request_sha256':key,'model':args.model,'base_url':args.base_url,
                'prompt_version':PROMPT_VERSION,'prompt':prompt,'status':'failed','attempts':[]}
        if path.exists(): record['attempts']=read(path).get('attempts',[])
        for attempt in range(args.retries+1):
            if calls>=args.limit: break
            calls+=1
            started=time.monotonic()
            print(f'点 {eid}: 开始请求 {calls}/{args.limit}，超时设置 {args.timeout_seconds:g} 秒',flush=True)
            try:
                raw=client.extract_point_features([root/im['path'] for im in row['images']],prompt)
                record['attempts'].append({'raw_response':raw,'response_sha256':hashlib.sha256(raw.encode()).hexdigest(),
                    'elapsed_seconds':round(time.monotonic()-started,3),'timeout_seconds':args.timeout_seconds})
                diagnostics=getattr(client,'last_stream_diagnostics',None)
                if diagnostics is not None: record['attempts'][-1]['stream_diagnostics']=diagnostics
                record.update(merge_response(raw,row))
                save(path,record)
                break
            except Exception as exc:
                # 不记录 SDK 异常全文，避免潜在请求凭据泄漏；本项目自己的流式计数可安全保留。
                failure={'error_type':type(exc).__name__,
                    'elapsed_seconds':round(time.monotonic()-started,3),'timeout_seconds':args.timeout_seconds}
                stream_diagnostics=getattr(exc,'stream_diagnostics',None)
                if stream_diagnostics is not None:
                    failure['stream_diagnostics']=stream_diagnostics
                status_code=getattr(exc,'status_code',None)
                if status_code is not None: failure['http_status']=status_code
                if isinstance(exc,ValueError) and type(exc).__module__=='builtins':
                    failure['validation_error']=str(exc)
                raw_response=getattr(exc,'raw_response',None)
                if raw_response is not None: failure['raw_response']=raw_response
                record['attempts'].append(failure)
                save(path,record)
                if getattr(exc,'status_code',None) in (400,401,403,404):
                    raise RuntimeError('API 模型/参数/认证错误，已停止；请检查配置和账户权限') from None
                if attempt<args.retries and calls<args.limit: time.sleep(min(2**attempt,8))
        print(f'点 {eid}: {record["status"]}; 最近一次耗时 {record["attempts"][-1]["elapsed_seconds"]:.1f} 秒; 本次请求 {calls}/{args.limit}',flush=True)
    print('标注结束；运行 export 查看完整/缺失/待复核数量')


def summarize(root,manifest):
    """按每点最新状态统计，失败类型每点只计一次，汇总不代表人工验收。"""
    counts=Counter(); layouts={}; errors=Counter(); uncertain=Counter(); reasons=Counter(); times=[]
    for row in manifest['rows']:
        path=root/'annotations'/f'{row["epe_id"]:05d}.json'
        record=read(path) if path.exists() else {}
        status=record.get('status','missing');counts[status]+=1
        layouts.setdefault(row['layout_parent'],Counter())[status]+=1
        uncertain.update(record.get('parsed',{}).get('uncertain_features',[]))
        reasons.update(record.get('review_reasons',[]))
        attempts=record.get('attempts',[])
        if attempts:
            times.append(attempts[-1].get('elapsed_seconds',0))
            if status=='failed': errors[attempts[-1].get('error_type','unknown')]+=1
    report={'feature_version':PROMPT_VERSION,'total_points':len(manifest['rows']),
        'counts':dict(counts),'by_layout':{k:dict(v) for k,v in layouts.items()},
        'failure_types':dict(errors),'uncertain_features':dict(uncertain),'review_reasons':dict(reasons),
        'latest_attempt_seconds_sum':round(sum(times),3),'human_verified':False}
    save(root/'annotation-summary.json',report)
    return report


def run_all(args):
    """在云端准备或续跑同版本数据集，每点一次请求，结束或中断后导出已完成结果。"""
    root=Path(args.dataset)
    if not (root/'manifest.json').exists():
        args.output=args.dataset
        prepare(args)
    manifest=load_manifest(root)
    if manifest.get('visual_protocol')!=PROMPT_VERSION:
        raise ValueError('run-all 需要新的 v4 目录，不能混入旧版标注')
    args.limit=len(manifest['rows']);args.retries=0;args.point_ids=None;args.retry_uncertain=False
    try: annotate(args)
    finally: export(args)


def write_excel(path, sheets):
    """用标准 OOXML 写工作簿，训练布尔值使用 Excel 原生布尔类型。"""
    ns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    def col(n):
        s=''
        while n: n,r=divmod(n-1,26); s=chr(65+r)+s
        return s
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        types='<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        rels='<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        book=f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
        for i,(name,rows) in enumerate(sheets.items(),1):
            book+=f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>'
            rels+=f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
            types+=f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            xml=f'<worksheet xmlns="{ns}"><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews><sheetData>'
            for ri,row in enumerate(rows,1):
                xml+=f'<row r="{ri}">'
                for ci,v in enumerate(row,1):
                    ref=f'{col(ci)}{ri}'
                    if type(v) is bool: cell=f'<c r="{ref}" t="b"><v>{int(v)}</v></c>'
                    elif type(v) in (int,float): cell=f'<c r="{ref}"><v>{v}</v></c>'
                    else: cell=f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{escape(str(v))}</t></is></c>'
                    xml+=cell
                xml+='</row>'
            xml+='</sheetData>'
            if rows: xml+=f'<autoFilter ref="A1:{col(len(rows[0]))}{len(rows)}"/>'
            z.writestr(f'xl/worksheets/sheet{i}.xml',xml+'</worksheet>')
        z.writestr('[Content_Types].xml',types+'</Types>')
        z.writestr('xl/workbook.xml',book+'</sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels',rels+'</Relationships>')
        z.writestr('_rels/.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')


def recover(args):
    """只解包与当前点完全匹配的 JSON 外层；独立报告保留旧语义和所有原始响应。"""
    root=Path(args.dataset); manifest=load_manifest(root)
    report={'source_manifest_sha256':digest(root/'manifest.json'),
            'status':'diagnostic_only','semantics':'source_annotations_unchanged',
            'rows':[],'counts':{}}
    for row in manifest['rows']:
        path=root/'annotations'/f'{row["epe_id"]:05d}.json'
        if not path.exists(): continue
        record=read(path); eid=row['epe_id']
        raw=next((a['raw_response'] for a in reversed(record['attempts']) if 'raw_response' in a),None)
        item={'epe_id':eid,'source_annotation_sha256':digest(path),'raw_response':raw,
              'source_prompt_version':record.get('prompt_version'),'normalization':'none'}
        try:
            if raw is None: raise ValueError('没有原始响应')
            obj=json.loads(raw)
            if isinstance(obj,dict) and set(obj)=={str(eid)}:
                obj=obj[str(eid)]
                item['normalization']='unwrap_exact_point_id'
                if isinstance(obj,dict) and set(obj)==set(FEATURES):
                    obj={'epe_id':eid,'features':obj,'uncertain_features':[k for k,v in obj.items() if v is None]}
                    item['normalization']='unwrap_exact_point_id_and_derive_null_list'
            parsed=validate_response(json.dumps(obj),eid)
            item['parsed']=parsed
            item['result']=row['result']
            item['status']='needs_review' if parsed['uncertain_features'] else 'ok'
        except (ValueError,TypeError,KeyError) as exc:
            item['status']='failed';item['error']=str(exc)
        report['rows'].append(item)
        status=item['status'];report['counts'][status]=report['counts'].get(status,0)+1
    destination=Path(args.output)
    if destination.exists(): raise ValueError('恢复报告已存在，请使用新文件名')
    try:
        destination.resolve().relative_to(root.resolve())
    except ValueError:
        pass
    else:
        raise ValueError('恢复报告须放在旧数据集目录之外')
    save(destination,report)
    print(json.dumps(report['counts'],ensure_ascii=False))


def export(args):
    root=Path(args.dataset); manifest=load_manifest(root); accepted=[]; review=[]; provenance=[]; request_configs=set()
    if manifest.get('visual_protocol')!=PROMPT_VERSION:
        raise ValueError('export 需要当前版本数据集，旧数据请使用 recover 独立报告')
    for row in manifest['rows']:
        eid=row['epe_id']; path=root/'annotations'/f'{eid:05d}.json'
        for im in row['images']:
            if digest(root/im['path'])!=im['sha256']: raise ValueError('图片哈希变化')
        status='missing'; reasons=[]
        if path.exists():
            record=read(path); status=record['status']
            key=identity({'images':row['images'],'prompt':row_prompt(row),'model':record['model'],'base_url':record['base_url']})
            if key!=record['request_sha256']: raise ValueError('标注缓存身份不一致')
            request_configs.add((record['model'],record['base_url']))
            if len(request_configs)>1: raise ValueError('不允许混用模型或 API 端点')
            if status in ('ok','needs_review'):
                raw=next(a['raw_response'] for a in reversed(record['attempts']) if 'raw_response' in a)
                derived=merge_response(raw,row)
                if any(record.get(k)!=v for k,v in derived.items()):
                    raise ValueError('标注内容与原始模型响应/几何证据不一致')
                p=derived['parsed'];status=derived['status'];reasons=derived['review_reasons']
                if status=='ok':
                    accepted.append({'epe_id':eid,'features':p['features'],'result':row['result']})
                    provenance.append([eid,row['layout_parent'],row['point_id'],row['normal_offset_nm'],row['recipe_sha256']])
                    continue
        review.append([eid,row['layout_parent'],status,'; '.join(reasons)])
    payload={'schema_version':'v2-vision-training-v1','teacher':manifest['teacher'],'status':'diagnostic_only',
        'manifest_sha256':digest(root/'manifest.json'),'feature_version':PROMPT_VERSION,'feature_names':list(FEATURES),
        'complete':not review,'total_points':len(manifest['rows']),'training_rows':len(accepted),
        'excluded_rows':len(review),'class_offsets_nm':{'-2':-20,'-1':-10,'0':0,'1':10,'2':20},'samples':accepted}
    payload['feature_sources']={k:('geometry' if k in GEOMETRY_FEATURES else 'vision') for k in FEATURES}
    save(root/'training.json',payload)
    sheets={'samples':[['epe_id']+list(FEATURES)+['result']]+[[r['epe_id']]+[r['features'][k] for k in FEATURES]+[r['result']] for r in accepted],
        'features':[['feature','definition']]+list(FEATURES.items()),
        'provenance':[['epe_id','layout_parent','point_id','normal_offset_nm','recipe_sha256']]+provenance,
        'review':[['epe_id','layout_parent','status','review_reasons']]+review,
        'summary':[['field','value'],['complete',not review],['total',len(manifest['rows'])],['training_rows',len(accepted)],['excluded',len(review)],['teacher',manifest['teacher']]]}
    tmp=root/'training.tmp.xlsx';write_excel(tmp,sheets);tmp.replace(root/'training.xlsx')
    summarize(root,manifest)
    print(f'已导出 JSON/Excel：可训练 {len(accepted)}，缺失/失败/待复核 {len(review)}，完整={not review}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare')
    p.add_argument('--labels',default='docs/v2_search_20260915_recipes.json')
    p.add_argument('--source-run',default='runs/20260915T014128Z-v2-search-5697a08c')
    p.add_argument('--output',required=True)
    p.add_argument('--openilt-dir');p.add_argument('--iccad13-dir')
    p.add_argument('--local-window',type=int,default=128);p.add_argument('--context-window',type=int,default=512)
    p.set_defaults(func=prepare)
    p=commands.add_parser('annotate');p.add_argument('--dataset',required=True)
    p.add_argument('--model',default='Qwen/Qwen3.8-27B');p.add_argument('--base-url',default='https://api.siliconflow.cn/v1')
    p.add_argument('--timeout-seconds',type=float,default=300.0,help='SDK 网络超时秒数，默认 300；不是整个请求的严格墙钟上限')
    p.add_argument('--limit',type=int,default=60,help='本次 API 请求总上限，包含重试')
    p.add_argument('--point-ids',help='仅标注这些编号，逗号分隔；仍遵守 limit 总预算')
    p.add_argument('--retries',type=int,default=2);p.add_argument('--retry-uncertain',action='store_true');p.set_defaults(func=annotate)
    p=commands.add_parser('export');p.add_argument('--dataset',required=True);p.set_defaults(func=export)
    p=commands.add_parser('recover');p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);p.set_defaults(func=recover)
    p=commands.add_parser('run-all',help='新建或续跑全量 v4，每点一次请求并自动导出')
    p.add_argument('--dataset',default='runs/v2-vision-dataset-004')
    p.add_argument('--labels',default='docs/v2_search_20260915_recipes.json')
    p.add_argument('--source-run',default='runs/20260915T014128Z-v2-search-5697a08c')
    p.add_argument('--openilt-dir');p.add_argument('--iccad13-dir')
    p.add_argument('--local-window',type=int,default=128);p.add_argument('--context-window',type=int,default=512)
    p.add_argument('--model',default='Qwen/Qwen3.8-27B');p.add_argument('--base-url',default='https://api.siliconflow.cn/v1')
    p.add_argument('--timeout-seconds',type=float,default=300.0);p.set_defaults(func=run_all)
    args=parser.parse_args();args.func(args)


if __name__=='__main__':
    main()
