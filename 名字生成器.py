import random
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import os
import json
from datetime import datetime

class EnhancedWorldGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("高级世界观生成器")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # 初始化数据存储
        self.history = []
        self.favorites = []
        self.load_data()
        
        # 设置样式
        self.setup_styles()
        
        # 扩展的风格选项
        self.styles = {
            "人名": {
                "奇幻": ["艾尔", "索林", "莱戈", "阿尔温", "吉姆利", "阿拉贡", "甘道夫", "萨鲁曼"],
                "科幻": ["尼奥", "崔妮蒂", "墨菲斯", "史密斯", "佐伊", "沃伦", "赛博", "量子"],
                "古风": ["云轩", "凌霜", "墨尘", "青岚", "瑶光", "寒星", "竹影", "宸风"],
                "蒸汽朋克": ["维克多", "艾达", "霍华德", "伊莎贝拉", "阿尔伯特", "玛格丽特", "尼古拉", "奥利维亚"],
                "赛博朋克": ["银手", "露西", "大卫", "瑞贝卡", "法拉第", "基努", "奥特", "荒坂"]
            },
            "地名": {
                "奇幻": ["银月城", "风暴谷", "龙脊山", "精灵森林", "矮人矿坑", "遗忘之地", "光明圣殿", "黑暗深渊"],
                "科幻": ["新星城", "赛博空间", "量子基地", "星际港口", "矩阵核心", "轨道站", "月球殖民地", "火星前哨"],
                "古风": ["长安", "洛阳", "姑苏", "金陵", "临安", "幽州", "青城", "云梦泽"],
                "蒸汽朋克": ["齿轮城", "飞艇港", "蒸汽堡", "发条镇", "铜管巷", "黄铜塔", "烟囱区", "维多利亚站"],
                "赛博朋克": ["霓虹街", "数据港", "黑客区", "企业广场", "贫民窟", "天空城", "地下城", "网络空间"]
            }
        }
        
        # 含义数据
        self.meanings = {
            "人名": {
                "奇幻": ["光明使者", "森林守护者", "龙之友", "精灵王子", "矮人战士", "人类国王", "白袍巫师", "黑暗法师"],
                "科幻": ["救世主", "黑客", "船长", "特工", "工程师", "科学家", "机器人", "量子物理学家"],
                "古风": ["高洁之士", "冷峻剑客", "文雅书生", "山间隐士", "美玉佳人", "寒夜孤星", "竹下君子", "皇家贵胄"],
                "蒸汽朋克": ["发明家", "飞艇船长", "机械师", "贵族小姐", "工业大亨", "探险家", "钟表匠", "秘密特工"],
                "赛博朋克": ["网络黑客", "街头佣兵", "公司特工", "义体医生", "媒体人", "帮派成员", "AI实体", "赛博忍者"]
            },
            "地名": {
                "奇幻": ["精灵的故乡", "风暴聚集之地", "巨龙栖息处", "远古森林", "矮人的家园", "被遗忘的王国", "神圣殿堂", "邪恶深渊"],
                "科幻": ["未来之城", "虚拟世界", "科研中心", "星际枢纽", "计算机核心", "太空站", "月球基地", "火星殖民地"],
                "古风": ["永恒之都", "牡丹之城", "江南水乡", "六朝古都", "南宋都城", "北方重镇", "道教圣地", "云梦大泽"],
                "蒸汽朋克": ["齿轮与蒸汽之城", "飞艇交通枢纽", "工业革命中心", "发条机械之都", "黄铜管道网络", "维多利亚风格建筑区", "烟雾缭绕的工厂区", "蒸汽动力车站"],
                "赛博朋克": ["霓虹灯照耀的街道", "数据流交汇处", "黑客活动中心", "企业权力象征", "社会底层聚集地", "悬浮建筑群", "地下反抗军基地", "虚拟现实空间"]
            }
        }
        
        # 视觉元素
        self.images = {}
        self.load_images()
        
        # 创建界面
        self.create_ui()
    
    def setup_styles(self):
        """设置界面样式"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # 配置颜色
        self.style.configure('.', background='#f0f0f0')
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('微软雅黑', 10))
        self.style.configure('TButton', font=('微软雅黑', 10))
        self.style.configure('Header.TLabel', font=('微软雅黑', 16, 'bold'))
        self.style.configure('Result.TLabel', font=('微软雅黑', 14), foreground='#2c3e50')
        self.style.configure('Meaning.TLabel', font=('微软雅黑', 10), foreground='#7f8c8d')
        
        # 标签页样式
        self.style.configure('TNotebook', background='#f0f0f0')
        self.style.configure('TNotebook.Tab', font=('微软雅黑', 10), padding=[10, 5])
    
    def load_images(self):
        """加载视觉元素图片"""
        try:
            # 尝试加载背景图片（如果没有图片文件会跳过）
            bg_image = Image.open("background.jpg")
            bg_image = bg_image.resize((800, 600), Image.Resampling.LANCZOS)
            self.images['background'] = ImageTk.PhotoImage(bg_image)
            
            # 加载风格图标
            styles = ['奇幻', '科幻', '古风', '蒸汽朋克', '赛博朋克']
            for style in styles:
                try:
                    img = Image.open(f"{style}.png")
                    img = img.resize((32, 32), Image.Resampling.LANCZOS)
                    self.images[style] = ImageTk.PhotoImage(img)
                except:
                    pass
        except:
            pass
    
    def create_ui(self):
        """创建用户界面"""
        # 主框架
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill='both', expand=True)
        
        # 如果有背景图片，添加背景
        if 'background' in self.images:
            bg_label = tk.Label(self.main_frame, image=self.images['background'])
            bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        # 顶部标题
        self.header_frame = ttk.Frame(self.main_frame)
        self.header_frame.pack(pady=20)
        
        self.header_label = ttk.Label(
            self.header_frame, 
            text="高级世界观生成器", 
            style='Header.TLabel'
        )
        self.header_label.pack()
        
        # 使用标签页组织功能
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 创建生成标签页
        self.create_generation_tab()
        
        # 创建历史记录标签页
        self.create_history_tab()
        
        # 创建收藏夹标签页
        self.create_favorites_tab()
        
        # 底部状态栏
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(
            self.main_frame, 
            textvariable=self.status_var,
            relief='sunken', 
            anchor='w'
        )
        self.status_bar.pack(fill='x', padx=5, pady=5)
        self.update_status("就绪")
    
    def create_generation_tab(self):
        """创建生成标签页"""
        gen_frame = ttk.Frame(self.notebook)
        self.notebook.add(gen_frame, text='生成器')
        
        # 生成选项区域
        options_frame = ttk.Frame(gen_frame)
        options_frame.pack(pady=10, fill='x')
        
        # 类型选择
        type_label = ttk.Label(options_frame, text="选择类型:")
        type_label.grid(row=0, column=0, padx=5, pady=5, sticky='e')
        
        self.type_var = tk.StringVar(value='人名')
        self.type_combobox = ttk.Combobox(
            options_frame, 
            textvariable=self.type_var, 
            values=['人名', '地名'], 
            state='readonly', 
            width=15
        )
        self.type_combobox.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        
        # 风格选择
        style_label = ttk.Label(options_frame, text="选择风格:")
        style_label.grid(row=1, column=0, padx=5, pady=5, sticky='e')
        
        self.style_var = tk.StringVar(value='奇幻')
        self.style_combobox = ttk.Combobox(
            options_frame, 
            textvariable=self.style_var, 
            values=list(self.styles['人名'].keys()),  # 初始使用人名风格
            state='readonly', 
            width=15
        )
        self.style_combobox.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        
        # 绑定类型变化事件，更新风格选项
        self.type_var.trace_add('write', self.update_style_options)
        
        # 生成按钮
        button_frame = ttk.Frame(options_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=15)
        
        generate_button = ttk.Button(
            button_frame, 
            text='生成名称', 
            command=self.generate_name
        )
        generate_button.pack(side='left', padx=5)
        
        favorite_button = ttk.Button(
            button_frame, 
            text='添加到收藏', 
            command=self.add_to_favorites
        )
        favorite_button.pack(side='left', padx=5)
        
        # 结果显示区域
        result_frame = ttk.Frame(gen_frame)
        result_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # 如果有风格图片，显示图片
        if self.images:
            self.image_label = tk.Label(result_frame)
            self.image_label.pack()
        
        self.result_var = tk.StringVar()
        self.result_label = ttk.Label(
            result_frame, 
            textvariable=self.result_var,
            style='Result.TLabel'
        )
        self.result_label.pack(pady=10)
        
        self.meaning_var = tk.StringVar()
        self.meaning_label = ttk.Label(
            result_frame, 
            textvariable=self.meaning_var,
            style='Meaning.TLabel',
            wraplength=400
        )
        self.meaning_label.pack(pady=10)
        
        # 详细信息文本框
        detail_label = ttk.Label(result_frame, text="详细信息:")
        detail_label.pack(pady=(20, 5))
        
        self.detail_text = scrolledtext.ScrolledText(
            result_frame, 
            width=60, 
            height=8,
            font=('微软雅黑', 9)
        )
        self.detail_text.pack(fill='both', expand=True)
    
    def create_history_tab(self):
        """创建历史记录标签页"""
        history_frame = ttk.Frame(self.notebook)
        self.notebook.add(history_frame, text='历史记录')
        
        # 历史记录列表
        columns = ('时间', '类型', '风格', '名称', '含义')
        self.history_tree = ttk.Treeview(
            history_frame,
            columns=columns,
            show='headings',
            selectmode='browse'
        )
        
        # 设置列
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=100, anchor='center')
        
        self.history_tree.column('含义', width=200)
        self.history_tree.column('名称', width=150)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(
            history_frame, 
            orient='vertical', 
            command=self.history_tree.yview
        )
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        
        # 布局
        self.history_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # 双击查看详情
        self.history_tree.bind('<Double-1>', self.view_history_detail)
        
        # 右键菜单
        self.history_menu = tk.Menu(self.root, tearoff=0)
        self.history_menu.add_command(
            label="添加到收藏", 
            command=self.add_history_to_favorites
        )
        self.history_menu.add_command(
            label="删除记录", 
            command=self.delete_history
        )
        self.history_tree.bind('<Button-3>', self.show_history_menu)
        
        # 刷新历史记录
        self.refresh_history()
    
    def create_favorites_tab(self):
        """创建收藏夹标签页"""
        fav_frame = ttk.Frame(self.notebook)
        self.notebook.add(fav_frame, text='我的收藏')
        
        # 收藏列表
        columns = ('时间', '类型', '风格', '名称', '含义')
        self.favorites_tree = ttk.Treeview(
            fav_frame,
            columns=columns,
            show='headings',
            selectmode='browse'
        )
        
        # 设置列
        for col in columns:
            self.favorites_tree.heading(col, text=col)
            self.favorites_tree.column(col, width=100, anchor='center')
        
        self.favorites_tree.column('含义', width=200)
        self.favorites_tree.column('名称', width=150)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(
            fav_frame, 
            orient='vertical', 
            command=self.favorites_tree.yview
        )
        self.favorites_tree.configure(yscrollcommand=scrollbar.set)
        
        # 布局
        self.favorites_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # 双击查看详情
        self.favorites_tree.bind('<Double-1>', self.view_favorite_detail)
        
        # 右键菜单
        self.favorites_menu = tk.Menu(self.root, tearoff=0)
        self.favorites_menu.add_command(
            label="查看详情", 
            command=self.view_favorite_detail
        )
        self.favorites_menu.add_command(
            label="移除收藏", 
            command=self.remove_favorite
        )
        self.favorites_tree.bind('<Button-3>', self.show_favorites_menu)
        
        # 刷新收藏夹
        self.refresh_favorites()
    
    def update_style_options(self, *args):
        """更新风格选项"""
        current_type = self.type_var.get()
        styles = list(self.styles[current_type].keys())
        self.style_combobox['values'] = styles
        self.style_var.set(styles[0])
    
    def generate_name(self):
        """生成随机名称"""
        name_type = self.type_var.get()
        style = self.style_var.get()
        
        try:
            # 从数据中随机选择
            name = random.choice(self.styles[name_type][style])
            meaning = random.choice(self.meanings[name_type][style])
            
            # 显示结果
            self.result_var.set(name)
            self.meaning_var.set(f"含义: {meaning}")
            
            # 显示风格图片（如果有）
            if style in self.images:
                self.image_label.config(image=self.images[style])
            
            # 生成详细信息
            detail = self.generate_detail(name_type, style, name, meaning)
            self.detail_text.delete('1.0', 'end')
            self.detail_text.insert('1.0', detail)
            
            # 添加到历史记录
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.history.append({
                'timestamp': timestamp,
                'type': name_type,
                'style': style,
                'name': name,
                'meaning': meaning,
                'detail': detail
            })
            self.refresh_history()
            self.save_data()
            
            self.update_status(f"成功生成 {name_type} - {style} 风格")
            
            # 播放生成音效（可选）
            self.play_sound()
            
        except Exception as e:
            messagebox.showerror("错误", f"生成名称时出错: {str(e)}")
            self.update_status("生成失败")
    
    def generate_detail(self, name_type, style, name, meaning):
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
    
    def add_to_favorites(self):
        """添加当前生成结果到收藏夹"""
        if not self.result_var.get():
            messagebox.showwarning("警告", "没有可收藏的生成结果")
            return
        
        # 检查是否已经收藏
        current_item = {
            'name': self.result_var.get(),
            'meaning': self.meaning_var.get().replace("含义: ", ""),
            'detail': self.detail_text.get('1.0', 'end').strip()
        }
        
        for item in self.favorites:
            if item['name'] == current_item['name'] and item['meaning'] == current_item['meaning']:
                messagebox.showinfo("提示", "该条目已在收藏夹中")
                return
        
        # 添加到收藏夹
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.favorites.append({
            'timestamp': timestamp,
            'type': self.type_var.get(),
            'style': self.style_var.get(),
            'name': current_item['name'],
            'meaning': current_item['meaning'],
            'detail': current_item['detail']
        })
        
        self.refresh_favorites()
        self.save_data()
        self.update_status(f"已收藏: {current_item['name']}")
        messagebox.showinfo("成功", "已添加到收藏夹")
    
    def refresh_history(self):
        """刷新历史记录列表"""
        self.history_tree.delete(*self.history_tree.get_children())
        for item in reversed(self.history):  # 最新的显示在最上面
            self.history_tree.insert('', 'end', values=(
                item['timestamp'],
                item['type'],
                item['style'],
                item['name'],
                item['meaning']
            ))
    
    def refresh_favorites(self):
        """刷新收藏夹列表"""
        self.favorites_tree.delete(*self.favorites_tree.get_children())
        for item in reversed(self.favorites):  # 最新的显示在最上面
            self.favorites_tree.insert('', 'end', values=(
                item['timestamp'],
                item['type'],
                item['style'],
                item['name'],
                item['meaning']
            ))
    
    def view_history_detail(self, event=None):
        """查看历史记录详情"""
        selected = self.history_tree.focus()
        if not selected:
            return
        
        item = self.history_tree.item(selected)
        values = item['values']
        
        # 在历史记录中查找匹配项
        for record in self.history:
            if (record['timestamp'] == values[0] and 
                record['name'] == values[3]):
                self.show_detail_popup(record)
                break
    
    def view_favorite_detail(self, event=None):
        """查看收藏夹详情"""
        selected = self.favorites_tree.focus()
        if not selected:
            return
        
        item = self.favorites_tree.item(selected)
        values = item['values']
        
        # 在收藏夹中查找匹配项
        for record in self.favorites:
            if (record['timestamp'] == values[0] and 
                record['name'] == values[3]):
                self.show_detail_popup(record)
                break
    
    def show_detail_popup(self, record):
        """显示详细信息弹出窗口"""
        popup = tk.Toplevel(self.root)
        popup.title(f"详细信息 - {record['name']}")
        popup.geometry("500x400")
        
        # 标题
        title_frame = ttk.Frame(popup)
        title_frame.pack(pady=10)
        
        ttk.Label(
            title_frame, 
            text=record['name'], 
            font=('微软雅黑', 14, 'bold')
        ).pack()
        
        ttk.Label(
            title_frame, 
            text=f"{record['type']} - {record['style']}风格",
            font=('微软雅黑', 10)
        ).pack()
        
        # 详细信息
        detail_frame = ttk.Frame(popup)
        detail_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(detail_frame)
        scrollbar.pack(side='right', fill='y')
        
        text = tk.Text(
            detail_frame,
            wrap='word',
            yscrollcommand=scrollbar.set,
            font=('微软雅黑', 10)
        )
        text.pack(fill='both', expand=True)
        
        scrollbar.config(command=text.yview)
        
        text.insert('1.0', record['detail'])
        text.config(state='disabled')
        
        # 关闭按钮
        button_frame = ttk.Frame(popup)
        button_frame.pack(pady=10)
        
        ttk.Button(
            button_frame,
            text="关闭",
            command=popup.destroy
        ).pack()
    
    def show_history_menu(self, event):
        """显示历史记录右键菜单"""
        selected = self.history_tree.identify_row(event.y)
        if selected:
            self.history_tree.selection_set(selected)
            self.history_menu.post(event.x_root, event.y_root)
    
    def show_favorites_menu(self, event):
        """显示收藏夹右键菜单"""
        selected = self.favorites_tree.identify_row(event.y)
        if selected:
            self.favorites_tree.selection_set(selected)
            self.favorites_menu.post(event.x_root, event.y_root)
    
    def add_history_to_favorites(self):
        """从历史记录添加到收藏夹"""
        selected = self.history_tree.focus()
        if not selected:
            return
        
        item = self.history_tree.item(selected)
        values = item['values']
        
        # 检查是否已经收藏
        for fav in self.favorites:
            if fav['name'] == values[3] and fav['meaning'] == values[4]:
                messagebox.showinfo("提示", "该条目已在收藏夹中")
                return
        
        # 在历史记录中查找匹配项
        for record in self.history:
            if (record['timestamp'] == values[0] and 
                record['name'] == values[3]):
                # 添加到收藏夹
                self.favorites.append(record)
                self.refresh_favorites()
                self.save_data()
                self.update_status(f"已收藏: {record['name']}")
                messagebox.showinfo("成功", "已添加到收藏夹")
                break
    
    def remove_favorite(self):
        """从收藏夹移除"""
        selected = self.favorites_tree.focus()
        if not selected:
            return
        
        item = self.favorites_tree.item(selected)
        values = item['values']
        
        # 从收藏夹中移除
        for i, fav in enumerate(self.favorites):
            if (fav['timestamp'] == values[0] and 
                fav['name'] == values[3]):
                del self.favorites[i]
                break
        
        self.refresh_favorites()
        self.save_data()
        self.update_status(f"已移除收藏: {values[3]}")
    
    def delete_history(self):
        """删除历史记录"""
        selected = self.history_tree.focus()
        if not selected:
            return
        
        item = self.history_tree.item(selected)
        values = item['values']
        
        # 确认删除
        if not messagebox.askyesno("确认", f"确定要删除历史记录: {values[3]}?"):
            return
        
        # 从历史记录中删除
        for i, hist in enumerate(self.history):
            if (hist['timestamp'] == values[0] and 
                hist['name'] == values[3]):
                del self.history[i]
                break
        
        self.refresh_history()
        self.save_data()
        self.update_status(f"已删除历史记录: {values[3]}")
    
    def load_data(self):
        """加载保存的数据"""
        try:
            if os.path.exists('history.json'):
                with open('history.json', 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
            
            if os.path.exists('favorites.json'):
                with open('favorites.json', 'r', encoding='utf-8') as f:
                    self.favorites = json.load(f)
        except:
            pass
    
    def save_data(self):
        """保存数据到文件"""
        try:
            with open('history.json', 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
            
            with open('favorites.json', 'w', encoding='utf-8') as f:
                json.dump(self.favorites, f, ensure_ascii=False, indent=2)
        except:
            pass
    
    def update_status(self, message):
        """更新状态栏"""
        self.status_var.set(message)
    
    def play_sound(self):
        """播放音效（可选）"""
        try:
            import winsound
            winsound.Beep(1000, 100)
        except:
            pass
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedWorldGenerator(root)
    app.run()