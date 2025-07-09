from flask import Flask, render_template, request, redirect, url_for, jsonify
import random
import json
import os
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 确保数据目录存在
os.makedirs('data', exist_ok=True)

# 风格数据
STYLES = {
    "人名": {
        "奇幻": [
            "艾尔", "索林", "莱戈", "阿尔温", "吉姆利", "阿拉贡", "甘道夫", "萨鲁曼",
            "伊露恩", "菲欧娜", "卡兰德", "希尔维斯", "德鲁伊", "艾瑞克", "米拉娜", "塞西莉亚",
            "艾斯特", "洛兰", "维拉", "塔利亚", "艾德里安", "西格蒙德", "莉莉丝", "奥菲利亚"
        ],
        "科幻": [
            "尼奥", "崔妮蒂", "墨菲斯", "史密斯", "佐伊", "沃伦", "赛博", "量子",
            "阿尔法", "贝塔", "伽马", "德尔塔", "艾文", "赛琳娜", "泰坦", "泽塔",
             "奥米伽", "艾丽卡", "诺亚", "莱昂", "伊芙", "赛拉", "艾伦", "维克特"
        ],
        "古风": [
            "云轩", "凌霜", "墨尘", "青岚", "瑶光", "寒星", "竹影", "宸风",
            "子墨", "清漪", "若尘", "玉衡", "南歌", "北辰", "沐雪", "流苏",
            "明玥", "夜白", "青璃", "楚然", "洛川", "苏陌", "锦瑟", "无双"
        ],
        "蒸汽朋克": [
            "维克多", "艾达", "霍华德", "伊莎贝拉", "阿尔伯特", "玛格丽特", "尼古拉", "奥利维亚",
            "西蒙", "艾米丽", "查尔斯", "艾琳", "亨利", "艾格尼丝", "菲利克斯", "艾德加",
             "阿黛尔", "奥斯卡", "玛蒂尔达", "艾文", "艾伦", "艾玛", "西尔维娅", "阿尔弗雷德"
        ],
        "赛博朋克": [
            "银手", "露西", "大卫", "瑞贝卡", "法拉第", "基努", "奥特", "荒坂",
            "零号", "赛琳", "艾什", "杰克", "莉兹", "艾达", "赛博", "艾文",
             "艾米", "莱拉", "艾利克斯", "诺瓦", "艾尔莎", "艾伦", "艾瑞斯", "艾利奥特"
        ]
    },
    "地名": {
        "奇幻": ["银月城", "风暴谷", "龙脊山", "精灵森林", "矮人矿坑", "遗忘之地", "光明圣殿", "黑暗深渊"],
        "科幻": ["新星城", "赛博空间", "量子基地", "星际港口", "矩阵核心", "轨道站", "月球殖民地", "火星前哨"],
        "古风": ["长安", "洛阳", "姑苏", "金陵", "临安", "幽州", "青城", "云梦泽"],
        "蒸汽朋克": ["齿轮城", "飞艇港", "蒸汽堡", "发条镇", "铜管巷", "黄铜塔", "烟囱区", "维多利亚站"],
        "赛博朋克": ["霓虹街", "数据港", "黑客区", "企业广场", "贫民窟", "天空城", "地下城", "网络空间"]
    }
}

MEANINGS = {
    "人名": {
        "奇幻": [
            "光明使者", "森林守护者", "龙之友", "精灵王子", "矮人战士", "人类国王", "白袍巫师", "黑暗法师",
            "月之女祭司", "森林精灵", "勇敢的骑士", "银发魔法师", "自然之子", "北境猎人", "月影女巫", "圣堂女武神",
             "星辰守望者", "流浪诗人", "治愈者", "预言家", "龙裔", "雷霆之锤", "黑暗女王", "命运编织者"
        ],
        "科幻": [
            "救世主", "黑客", "船长", "特工", "工程师", "科学家", "机器人", "量子物理学家",
            "智能AI", "太空探险家", "能量操控者", "数据分析师", "虚拟现实设计师", "星舰指挥官", "机械战士", "纳米专家",
             "终极武器", "生化专家", "未来预言家", "星际外交官", "仿生人", "时空旅者", "系统管理员", "异星移民者"
        ],
        "古风": [
            "高洁之士", "冷峻剑客", "文雅书生", "山间隐士", "美玉佳人", "寒夜孤星", "竹下君子", "皇家贵胄",
            "墨香才子", "清丽佳人", "云游道人", "星河诗人", "琴心剑魄", "北地将军", "雪中行者", "江南才女",
             "明月公子", "夜色游侠", "青衣隐者", "楚地才俊", "洛水仙子", "陌上公子", "琴瑟佳人", "绝世无双"
        ],
        "蒸汽朋克": [
            "发明家", "飞艇船长", "机械师", "贵族小姐", "工业大亨", "探险家", "钟表匠", "秘密特工",
            "机械工程师", "蒸汽科学家", "齿轮设计师", "动力专家", "能源大亨", "机械贵族", "飞行员", "侦探",
            "女伯爵", "冒险家", "机械公主", "动力骑士", "工程师", "机械师", "女发明家", "蒸汽领主"
        ],
        "赛博朋克": [
            "网络黑客", "街头佣兵", "公司特工", "义体医生", "媒体人", "帮派成员", "AI实体", "赛博忍者",
            "数据幽灵", "虚拟偶像", "义体改造人", "黑市商人", "赛博女巫", "机械少女", "网络战士", "信息猎人",
            "黑客少女", "街头舞者", "系统入侵者", "未来特工", "虚拟女王", "义体工程师", "数据分析师", "未来侦探"
        ]
    },
    "地名": {
        "奇幻": ["精灵的故乡", "风暴聚集之地", "巨龙栖息处", "远古森林", "矮人的家园", "被遗忘的王国", "神圣殿堂", "邪恶深渊"],
        "科幻": ["未来之城", "虚拟世界", "科研中心", "星际枢纽", "计算机核心", "太空站", "月球基地", "火星殖民地"],
        "古风": ["永恒之都", "牡丹之城", "江南水乡", "六朝古都", "南宋都城", "北方重镇", "道教圣地", "云梦大泽"],
        "蒸汽朋克": ["齿轮与蒸汽之城", "飞艇交通枢纽", "工业革命中心", "发条机械之都", "黄铜管道网络", "维多利亚风格建筑区", "烟雾缭绕的工厂区", "蒸汽动力车站"],
        "赛博朋克": ["霓虹灯照耀的街道", "数据流交汇处", "黑客活动中心", "企业权力象征", "社会底层聚集地", "悬浮建筑群", "地下反抗军基地", "虚拟现实空间"]
    }
}

def load_data(filename):
    """加载JSON数据"""
    try:
        with open(f'data/{filename}', 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_data(filename, data):
    """保存数据到JSON文件"""
    with open(f'data/{filename}', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def generate_detail(name_type, style, name, meaning):
    """生成详细信息"""
    if name_type == "人名":
        return f"""名称: {name}
风格: {style}
含义: {meaning}

背景故事:
{name}是一位{random.choice(['著名的', '神秘的', '年轻的', '年长的'])}{meaning}。
{random.choice(['他', '她'])}来自{random.choice(['一个贵族家庭', '偏远山村', '神秘组织', '流浪民族'])},
以{random.choice(['勇敢', '智慧', '狡诈', '忠诚'])}著称。"""
    else:  # 地名
        return f"""名称: {name}
风格: {style}
含义: {meaning}

地点描述:
{name}是一个{random.choice(['繁华的', '荒凉的', '神秘的', '危险的'])}{meaning}。
这里以{random.choice(['独特的建筑', '奇异的风俗', '丰富的资源', '战略位置'])}闻名，
{random.choice(['吸引着众多游客', '被当地人视为圣地', '是兵家必争之地', '隐藏着不为人知的秘密'])}。"""

@app.route('/')
def index():
    """主页"""
    return render_template('index.html', 
                         name_types=list(STYLES.keys()),
                         styles=list(STYLES['人名'].keys()) + list(STYLES['地名'].keys()))

@app.route('/generate', methods=['POST'])
def generate():
    """生成名称"""
    name_type = request.form.get('name_type', '人名')
    style = request.form.get('style', '奇幻')
    
    try:
        # 随机选择名称和含义
        names = STYLES[name_type][style]
        meanings = MEANINGS[name_type][style]
        idx = random.randint(0, len(names) - 1)
        name = names[idx]
        meaning = meanings[idx]
        
        # 生成详细信息
        detail = generate_detail(name_type, style, name, meaning)
        
        # 添加到历史记录
        history = load_data('history.json')
        history.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': name_type,
            'style': style,
            'name': name,
            'meaning': meaning,
            'detail': detail
        })
        save_data('history.json', history)
        
        return jsonify({
            'success': True,
            'name': name,
            'meaning': meaning,
            'detail': detail,
            'type': name_type,
            'style': style
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/add_favorite', methods=['POST'])
def add_favorite():
    """添加到收藏夹"""
    data = request.json
    favorites = load_data('favorites.json')
    
    # 检查是否已收藏
    for item in favorites:
        if item['name'] == data['name'] and item['meaning'] == data['meaning']:
            return jsonify({'success': False, 'message': '该条目已在收藏夹中'})
    
    # 添加到收藏夹
    favorites.append({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'type': data['type'],
        'style': data['style'],
        'name': data['name'],
        'meaning': data['meaning'],
        'detail': data['detail']
    })
    save_data('favorites.json', favorites)
    
    return jsonify({'success': True, 'message': '已添加到收藏夹'})

@app.route('/history')
def history():
    """历史记录页面"""
    history_data = load_data('history.json')
    return render_template('history.html', history=reversed(history_data))

@app.route('/favorites')
def favorites():
    """收藏夹页面"""
    favorites_data = load_data('favorites.json')
    return render_template('favorites.html', favorites=reversed(favorites_data))

@app.route('/delete_history', methods=['POST'])
def delete_history():
    """删除历史记录"""
    data = request.json
    history_data = load_data('history.json')
    
    # 过滤掉要删除的记录
    history_data = [item for item in history_data 
                   if item['timestamp'] != data['timestamp'] or item['name'] != data['name']]
    
    save_data('history.json', history_data)
    return jsonify({'success': True})

@app.route('/remove_favorite', methods=['POST'])
def remove_favorite():
    """移除收藏"""
    data = request.json
    favorites_data = load_data('favorites.json')
    
    # 过滤掉要移除的记录
    favorites_data = [item for item in favorites_data 
                     if item['timestamp'] != data['timestamp'] or item['name'] != data['name']]
    
    save_data('favorites.json', favorites_data)
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)