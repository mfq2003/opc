"""本模块定义论文附录 A.2 的视觉特征词典和双图标注提示词。

输入为同一 EPE 点的局部图、上下文图和匿名编号；输出仅为视觉布尔特征，不含搜索动作。
使用定性 near/far、long/short；types 展开为四个布尔列，原文矛盾字段按名称解释并显式记录。
"""
import json

FEATURES = {
    'type_CV': '位于角点段且所在边垂直（CV）',
    'type_CH': '位于角点段且所在边水平（CH）',
    'type_H': '位于水平边且不属于角点段（H）',
    'type_V': '位于垂直边且不属于角点段（V）',
    'near_jog': '当前边附近有台阶状折转 jog',
    'face_jog': '有台阶状折转面对当前点',
    'on_jog_long_edge': '当前点位于台阶的长边',
    'on_jog_short_edge': '当前点位于台阶的短边',
    'on_start_corner_seg': '按多边形顺时针遍历，位于边的起始角点段',
    'on_end_corner_seg': '按多边形顺时针遍历，位于边的结束角点段',
    'near_hor_dir_has_polygon': '水平方向近处有面对当前边、但不与该边连接的多边形',
    'far_hor_dir_has_polygon': '水平方向远处有面对当前边、但不与该边连接的多边形',
    'near_ver_dir_has_polygon': '垂直方向近处有面对当前边、但不与该边连接的多边形',
    'far_ver_dir_has_polygon': '垂直方向远处有面对当前边、但不与该边连接的多边形；原文解释出现 no，与字段名矛盾，本协议按 has 的肯定含义标注',
    'on_horizontal_edge': '当前点位于水平边',
    'on_vertical_edge': '当前点位于垂直边',
    'near_convex_corner': '附近有与当前边连接的凸角',
    'near_concave_corner': '附近有与当前边连接的凹角',
    'face_convex_corner': '有面对当前点、但不与当前边连接的凸角',
    'face_concave_corner': '有面对当前点、但不与当前边连接的凹角',
    'near_horizontal_edge': '当前点附近有水平边',
    'near_vertical_edge': '当前点附近有垂直边',
    'far_horizontal_edge': '当前点远处有水平边',
    'far_vertical_edge': '当前点远处有垂直边',
    'at_long_path_end': '当前点位于长路径的端部',
    'at_short_path_end': '当前点位于短路径的端部',
    'at_long_path_side': '当前点位于长路径的侧面',
    'at_short_path_side': '当前点位于短路径的侧面',
}
PROMPT_VERSION = 'appendix-a2-hybrid-v4'
GEOMETRY_FEATURES = ('type_CV','type_CH','type_H','type_V',
    'on_horizontal_edge','on_vertical_edge','on_start_corner_seg','on_end_corner_seg')
PROMPT = '''你是半导体版图几何特征标注员。只标注当前 EPE 点的几何特征，不预测动作或优化效果。
图1为局部细节。图2左面板为上下文放大图，右面板为完整 target 概览；所有红圈标记同一个测量点。
黑色为 target 图形，白色为背景，灰色为版图外未知区域。蓝色箭头是当前分段的顺时针方向（沿箭头行进时黑色实体在右侧），绿色箭头是外法线。
上下文图中的橙色框表示局部图取景范围。彩色标记均不属于几何，不可将其当作边或角点。
两图保持相同方向：屏幕右为 +x，下为 +y；放大倍数不同。不要根据显示尺寸混淆长短关系。
使用完整特征词典，不新增、删除或改名。near/far 和 long/short 按视觉结构关系判断，不计算或编造阈值。
区分连接在当前边上的邻近结构与面对当前点但不连接的结构。on_start_corner_seg/on_end_corner_seg 中的 start/end
以蓝色箭头给出的顺时针方向为准：所在原始边的起始角点一侧为 start，结束角点一侧为 end。
蓝色箭头端点只是分段端点，不一定是原始边的角点；必须看到实际转角才能判断角点段。
上下文左面板橙框表示局部窗口，右面板橙框表示上下文窗口。忽略文字、箭头和边框。
near/far 特征只在左面板上下文窗口内判断：按相对距离作定性区分，不把整个版图上的边都算作 far。
若完整可见的观察区域内没有符合条件的结构，返回 false；若遮挡、模糊或截断妨碍判断，返回 null。
jog 必须看到台阶的连续折转，不得把单个直角或矩形端部直接当作 jog；确认不存在 jog 时四个 jog 特征均为 false。
长短路径特征需结合全图概览看到路径延伸和端部，并按实际几何比例比较；不得因局部裁剪把长路径当短路径。
全图过小无法辨认相关细节时仍返回 null。不要为了填满布尔字段而猜测。
图外结构不可见不能直接判 false；无法判断返回 null，并在 uncertain_features 中列出对应字段。
图片底部图例和 SEG START/SEG END 为辅助标记，指当前分段的顺时针起止端，不一定是原始边的角点。
输入提供的 geometry_features 是程序从原始分段确定的八个特征，必须原样复制，不得凭图修改。
corner_evidence 给出当前分段顺时针两端的真实角点类型（1 凸、-1 凹、0 非角点）。
若当前分段端点就是凸角/凹角，near_convex_corner/near_concave_corner 应分别为 true。
端点没有角点不代表附近不存在角点，仍需看图判断。不要把截图边缘、红圈、箭头当作转角。
四个 type 字段在确定时恰好一个 true；水平/垂直字段在确定时恰好一个 true。
只返回 JSON：{"epe_id": 输入编号, "features": {所有字段: true/false/null}, "uncertain_features": [无法判断的字段名]}。
'''


def point_prompt(epe_id, geometry_features=None, corner_evidence=None):
    """仅加入匿名编号，不暴露版图名、搜索标签或最终质量。"""
    context = ''
    if geometry_features is not None:
        context = '\ngeometry_features：' + json.dumps(geometry_features, ensure_ascii=False)
        context += '\ncorner_evidence：' + json.dumps(corner_evidence, ensure_ascii=False)
    return PROMPT + '\n特征词典：' + json.dumps(FEATURES, ensure_ascii=False) + context + '\n输入编号：' + str(epe_id)


def normalize_response(raw, epe_id):
    """只恢复唯一且匹配的编号外层，绝不补造缺失特征。"""
    obj = json.loads(raw)
    normalization = 'none'
    if isinstance(obj, dict) and set(obj) == {str(epe_id)}:
        obj = obj[str(epe_id)]
        normalization = 'unwrap_exact_point_id'
        if isinstance(obj, dict) and set(obj) == set(FEATURES):
            obj = {'epe_id': epe_id, 'features': obj,
                   'uncertain_features': [k for k,v in obj.items() if v is None]}
            normalization += '_and_derive_null_list'
    return obj, normalization


def validate_response(raw, epe_id, check_semantics=True):
    """严格校验字段、布尔类型、未知项和基本互斥关系，拒绝字符串布尔值。"""
    obj = json.loads(raw)
    if not isinstance(obj, dict) or set(obj) != {'epe_id', 'features', 'uncertain_features'} or type(obj['epe_id']) is not int or obj['epe_id'] != epe_id:
        raise ValueError('响应编号或顶层字段错误')
    f = obj['features']
    if not isinstance(f, dict) or set(f) != set(FEATURES):
        raise ValueError('特征字段不完整或有额外字段')
    if any(v is not None and type(v) is not bool for v in f.values()):
        raise ValueError('特征必须为 true/false/null')
    unknown = obj['uncertain_features']
    if not isinstance(unknown, list) or any(type(x) is not str for x in unknown) or len(set(unknown)) != len(unknown) or set(unknown) != {k for k,v in f.items() if v is None}:
        raise ValueError('未知字段列表不一致')
    if not check_semantics:
        return obj
    for group in [('type_CV','type_CH','type_H','type_V'), ('on_horizontal_edge','on_vertical_edge')]:
        values = [f[k] for k in group]
        if None not in values and sum(values) != 1:
            raise ValueError('方向或类型互斥关系错误')
    for kind, direction in [('type_CV','on_vertical_edge'),('type_V','on_vertical_edge'),('type_CH','on_horizontal_edge'),('type_H','on_horizontal_edge')]:
        if f[kind] is True and f[direction] is False:
            raise ValueError('类型与方向矛盾')
    if f['on_jog_long_edge'] is True and f['on_jog_short_edge'] is True:
        raise ValueError('台阶长短边矛盾')
    return obj
