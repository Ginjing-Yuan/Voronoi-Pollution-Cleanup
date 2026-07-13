import numpy as np  # 数值计算库，用于矩阵运算和数学函数
import matplotlib.pyplot as plt  # 绑图库，用于生成热力图和动画
from matplotlib.animation import FuncAnimation  # 动画类，用于逐帧更新可视化
from matplotlib.patches import Circle  # 圆形补丁，用于绘制Agent的清理范围圆圈
import random  # 随机数库，用于生成随机污染源位置和风场

# 稳态污染分布
INNER_MASS = 60 # 稳态污染内圈浓度
MID_MASS = 30 # 稳态污染中圈浓度
OUT_MASS = 10 # 稳态污染外圈浓度

# 清洁Agent参数
CLEANING_EFFICIENCY = 30 # 单位时间清理最大值
CLEANING_RADIUS = 5 # 清洁半径
MOVING_SPEED = 1 # 移动速度

# 场景模拟参数
POLLUTION_SOURCE_NUM = 1 # 污染源数量（暂时仅做了一个）
WIDTH = 100 # 图片宽度
HEIGHT = 100 # 图片高度
AGENTS_NUM = 4 # 清洁装置数量
SENSOR_NUM = 81 # 传感器数量（传感器数量暂时固定为81）

class PollutionEnvironment:
    """污染环境类：管理污染网格、污染源、扩散和清理逻辑"""
    def __init__(self, width=100, height=100, diffusion_rate=50, decay_rate=0,
                 source_mass=1000, diffusion_mass=1,
                 min_concentration=1, diffusion_mode='steady'):
        self.width = width  # 网格宽度（列数）
        self.height = height  # 网格高度（行数）
        self.diffusion_rate = diffusion_rate  # 每圈扩散的总质量（瞬态模式用）
        # self.decay_rate = decay_rate  # 自降解速率：每帧浓度衰减比例
        self.min_concentration = min_concentration  # 最小浓度阈值：低于此值的格子视为无污染
        self.source_mass = source_mass  # 污染源中心质量/浓度（瞬态模式用）

        self.diffusion_mode = diffusion_mode  # 扩散模式：'steady'稳态 / 'transient'瞬态
        self.pollution_grid = np.zeros((height, width))  # 污染浓度网格，初始化全0
        self.wind_x = 0  # 风场X方向分量(风场暂时未生效)
        self.wind_y = 0  # 风场Y方向分量(风场暂时未生效)
        self.wind_change_interval = 50  # 风场变化间隔帧数(风场暂时未生效)
        self.pollution_sources = []  # 污染源列表，每个元素为字典{x, y, concentration, radius, active}
        self.current_ring = 0  # 当前已扩散到的圈数（瞬态模式用）

        self.initialized = False  # 稳态模式是否已完成初始分布的标志

    def add_pollution_source(self, x, y, concentration=INNER_MASS, radius=0):
        """添加污染源：记录源信息并在网格中心点赋初值"""
        self.pollution_sources.append({
            'x': x, 'y': y, 'concentration': concentration,  # 源坐标和浓度
            'radius': radius, 'active': True  # 源半径和是否活跃标志
        })
        self.pollution_grid[y, x] = concentration  # 在网格中心点设置初始浓度

    def update_wind(self, frame):
        """每隔wind_change_interval帧随机更新风场方向"""
        if frame % self.wind_change_interval == 0:  # 到达变化间隔时
            self.wind_x = random.uniform(-2, 2)  # 随机生成X方向风速，范围[-2, 2]
            self.wind_y = random.uniform(-2, 2)  # 随机生成Y方向风速，范围[-2, 2]

    def diffuse(self):
        """扩散入口：根据扩散模式选择对应的扩散方法"""
        if self.diffusion_mode == 'steady':  # 稳态模式
            if not self.initialized:  # 首次执行：生成稳态初始分布
                self._diffuse_steady()
                self.initialized = True  # 标记已初始化，后续不再重新生成
            else:  # 非首次：执行再平衡扩散（让高浓度向低浓度/空洞区域流动）
                self._diffuse_steady_rebalance()
        elif self.diffusion_mode == 'transient':  # 瞬态模式
            self._diffuse_transient()  # 每帧重新计算扩散分布

    # 内嵌函数：Bresenham画圆算法计算环带点
    def get_ring_points(self, r_outer, r_inner, source_x, source_y, width, height):
        inner_points = set()  # 内圈点集合
        outer_points = set()  # 外圈点集合

        x_inner = r_inner  # 内圈当前x
        y_inner = 0  # 内圈当前y
        err_inner = 1 - r_inner  # 内圈决策参数

        x_outer = r_outer  # 外圈当前x
        y_outer = 0  # 外圈当前y
        err_outer = 1 - r_outer  # 外圈决策参数

        while x_inner >= y_inner:  # Bresenham遍历内圈
            pts = [  # 内圈8个对称点
                (source_x + x_inner, source_y + y_inner), (source_x - x_inner, source_y + y_inner),
                (source_x + x_inner, source_y - y_inner), (source_x - x_inner, source_y - y_inner),
                (source_x + y_inner, source_y + x_inner), (source_x - y_inner, source_y + x_inner),
                (source_x + y_inner, source_y - y_inner), (source_x - y_inner, source_y - x_inner)
            ]
            for px, py in pts:  # 收集内圈点
                if 0 <= px < self.width and 0 <= py < self.height:
                    inner_points.add((px, py))
            y_inner += 1
            if err_inner < 0:
                err_inner += 2 * y_inner + 1
            else:
                x_inner -= 1
                err_inner += 2 * (y_inner - x_inner) + 1

        while x_outer >= y_outer:  # Bresenham遍历外圈
            pts = [  # 外圈8个对称点
                (source_x + x_outer, source_y + y_outer), (source_x - x_outer, source_y + y_outer),
                (source_x + x_outer, source_y - y_outer), (source_x - x_outer, source_y - y_outer),
                (source_x + y_outer, source_y + x_outer), (source_x - y_outer, source_y + x_outer),
                (source_x + y_outer, source_y - x_outer), (source_x - y_outer, source_y - x_outer)
            ]

            fill_pts = [  # 内外圈之间的填充点
                (source_x + x_outer - 1, source_y + y_outer), (source_x - x_outer + 1, source_y + y_outer),
                (source_x + x_outer - 1, source_y - y_outer), (source_x - x_outer + 1, source_y - y_outer),
                (source_x + y_outer, source_y + x_outer - 1), (source_x - y_outer, source_y + x_outer - 1),
                (source_x + y_outer, source_y - x_outer + 1), (source_x - y_outer, source_y - x_outer + 1)
            ]
            fill_points = []  # 未使用
            if (source_x + x_outer - 1, source_y + y_outer) not in inner_points:  # 填充点不在内圈时加入
                for px, py in fill_pts:
                    if 0 <= px < width and 0 <= py < height:
                        outer_points.add((px, py))
            for px, py in pts:  # 收集外圈点
                if 0 <= px < width and 0 <= py < height:
                    outer_points.add((px, py))
            y_outer += 1
            if err_outer < 0:
                err_outer += 2 * y_outer + 1
            else:
                x_outer -= 1
                err_outer += 2 * (y_outer - x_outer) + 1
        return list(outer_points)

    def _diffuse_transient(self):
        """
        瞬态扩散：总量守恒，污染物随时间从中心向外摊开
        圆心浓度为source_mass，每帧向外扩散diffusion_mass
        """
        self.current_ring += 1  # 当前扩散圈数递增
        grid = self.pollution_grid  # 引用当前网格

        for source in self.pollution_sources:  # 遍历每个污染源
            cx, cy = int(source['x']), int(source['y'])  # 源中心坐标

            grid[cx, cy] = self.source_mass  # 设置源中心浓度为source_mass

            for current_radius in range(1, self.current_ring + 1):  # 从半径1逐圈向外扩散
                ring_points = self.get_ring_points(current_radius, current_radius - 1, cx, cy, self.width, self.height)  # 获取当前环的点
                if not ring_points:  # 如果环上没有有效点则跳过
                    break

                ring_mass = self.diffusion_rate / len(ring_points) if len(ring_points) > 0 else 0  # 每个格子分到的质量

                for px, py in ring_points:  # 将质量分配到环上每个格子
                    grid[py, px] += ring_mass

        self.pollution_grid = grid  # 更新网格

    def _diffuse_steady(self):
        """
        稳态扩散初始化：源头浓度不变，周围按固定分布生成
        只在首次调用时执行，生成稳态分布
        污染物浓度分布暂时固定
        """
        grid = self.pollution_grid  # 引用当前网格

        for source in self.pollution_sources:  # 遍历每个污染源
            cx, cy = int(source['x']), int(source['y'])  # 源中心坐标

            grid[cx, cy] = self.source_mass  # 设置源中心浓度

            for current_radius in range(1, 30):  # 从半径1扩散到30圈
                ring_points = self.get_ring_points(current_radius, current_radius - 1, cx, cy, self.width, self.height)  # 获取环带点
                if not ring_points:  # 无有效点则停止
                    break

                # 稳态浓度：按距离分段设定固定浓度值
                ring_mass = INNER_MASS  # 默认浓度10
                if current_radius <= 10:  # 0-10圈浓度60
                    ring_mass = INNER_MASS
                elif current_radius <= 20:  # 11-20圈浓度30
                    ring_mass = MID_MASS
                elif current_radius <= 30:  # 21-23圈浓度10
                    ring_mass = OUT_MASS

                for px, py in ring_points:  # 将环上每个格子设为对应浓度
                    grid[py, px] = ring_mass

        self.pollution_grid = grid  # 更新网格

    def get_concentration(self, x, y):
        """查询指定坐标的污染浓度"""
        x, y = int(x), int(y)  # 坐标取整
        if 0 <= x < self.width and 0 <= y < self.height:  # 边界检查
            return self.pollution_grid[y, x]  # 返回网格中的浓度值
        return 0  # 越界返回0

    def reduce_pollution(self, x, y, radius=CLEANING_RADIUS, amount=CLEANING_EFFICIENCY):
        """
        清理污染：在减少清洁Agent附近污染物浓度，最大减少值为cleaning_efficiency
        """
        x, y = int(x), int(y)
        remaining = amount
        cells = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = y + dy, x + dx
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    distance = np.sqrt(dx ** 2 + dy ** 2)
                    if distance <= radius:
                        cells.append((ny, nx, distance))
        cells.sort(key=lambda c: c[2])
        for ny, nx, dist in cells:
            if remaining <= 0:
                break
            avail = self.pollution_grid[ny, nx]
            take = min(avail, remaining)
            self.pollution_grid[ny, nx] -= take
            remaining -= take

    def _diffuse_steady_rebalance(self):
        """
        稳态再平衡扩散：清理后在网格上留下空洞，高浓度区域向低浓度/空洞区域流动
        使用热扩散方程：new = old + diff * (邻居均值 - 自身)
        """
        grid = self.pollution_grid  # 引用当前网格
        diff = 0.25  # 扩散系数，控制每帧再平衡的幅度（0.25为最大稳定值）
        new_grid = grid.copy()  # 复制网格，避免修改时影响原始数据
        for i in range(1, self.height - 1):  # 遍历内部行（跳过边界行）
            for j in range(1, self.width - 1):  # 遍历内部列（跳过边界列）
                neighbors = (grid[i - 1, j] + grid[i + 1, j] +  # 上+下邻居浓度之和
                             grid[i, j - 1] + grid[i, j + 1])  # 左+右邻居浓度之和
                delta = (neighbors - 4 * grid[i, j]) * diff  # 扩散增量：邻居均值与自身的差×扩散系数
                new_val = grid[i, j] + delta  # 新浓度 = 当前浓度 + 扩散增量
                if 0 < new_val < self.min_concentration:  # 浓度在0和阈值之间
                    new_val = 0  # 低于阈值视为无污染，置0
                new_grid[i, j] = max(0, new_val)  # 确保浓度不为负
        for source in self.pollution_sources:  # 遍历所有污染源
            if source['active']:  # 源仍然活跃
                cx, cy = int(source['x']), int(source['y'])  # 源坐标
                new_grid[cy, cx] = max(new_grid[cy, cx], source['concentration'])  # 源点浓度不低于源记录值（持续补充）
        self.pollution_grid = new_grid  # 更新网格


class Sensor:
    """传感器类：部署在固定位置，每帧测量所在位置的污染浓度"""
    def __init__(self, x, y, sensor_id):
        self.x = x  # 传感器X坐标
        self.y = y  # 传感器Y坐标
        self.id = sensor_id  # 传感器编号
        self.readings = []  # 历史浓度读数列表

    def measure(self, environment):
        """测量当前位置的污染浓度并记录"""
        concentration = environment.get_concentration(self.x, self.y)  # 从环境获取浓度
        self.readings.append(concentration)  # 记录到历史读数
        return concentration  # 返回当前浓度


class CleaningAgent:
    """清洁Agent类：在环境中移动并清理污染"""
    def __init__(self, x, y, agent_id, speed=MOVING_SPEED, cleanup_radius=CLEANING_RADIUS, cleanup_rate=CLEANING_EFFICIENCY):
        self.x = x  # 当前X坐标
        self.y = y  # 当前Y坐标
        self.id = agent_id  # Agent编号
        self.speed = speed  # 移动速度：每帧最大移动距离
        self.cleanup_radius = cleanup_radius  # 清理半径
        self.cleanup_rate = cleanup_rate  # 每帧最大清理量
        self.path_x = [x]  # X坐标移动轨迹
        self.path_y = [y]  # Y坐标移动轨迹
        self.total_cleaned = 0  # 累计清理总量

    def move_towards(self, target_x, target_y, environment):
        """向目标点移动，每帧最多移动speed距离"""
        dx = target_x - self.x  # X方向距离
        dy = target_y - self.y  # Y方向距离
        distance = np.sqrt(dx ** 2 + dy ** 2)  # 到目标的欧氏距离
        if distance > 0:  # 未到达目标时
            self.x += (dx / distance) * min(self.speed, distance)  # X方向移动：单位向量×步长
            self.y += (dy / distance) * min(self.speed, distance)  # Y方向移动：单位向量×步长
            self.x = max(0, min(environment.width - 1, self.x))  # X坐标限制在网格范围内
            self.y = max(0, min(environment.height - 1, self.y))  # Y坐标限制在网格范围内
        self.path_x.append(self.x)  # 记录X轨迹
        self.path_y.append(self.y)  # 记录Y轨迹

    def clean(self, environment):
        """在当前位置执行清理，并统计清理量"""
        cleaned_before = np.sum(environment.pollution_grid)  # 清理前环境总污染量
        environment.reduce_pollution(self.x, self.y, self.cleanup_radius, self.cleanup_rate)  # 调用环境清理方法
        cleaned_after = np.sum(environment.pollution_grid)  # 清理后环境总污染量
        self.total_cleaned += (cleaned_before - cleaned_after)  # 累加本次清理量

    def calculate_voronoi_centroid(self, agents, sensors, environment):
        """
        在Voronoi分区内，向浓度最高的sensor移动
        先确定分区，再从分区内的sensor中选浓度最高的作为目标
        """
        agent_positions = [(a.x, a.y) for a in agents]  # 收集所有Agent的坐标，如[(25,25),(75,25),(25,75),(75,75)]
        my_idx = self.id  # 当前Agent的编号，0/1/2/3

        my_sensors = []  # 存放属于当前Agent分区的sensor
        for sensor in sensors:  # 遍历每个sensor
            # 计算该sensor到所有Agent的距离列表，如[30.0, 50.0, 70.0, 86.6]
            distances = [np.sqrt((sensor.x - ax) ** 2 + (sensor.y - ay) ** 2) for ax, ay in agent_positions]
            nearest = int(np.argmin(distances))  # argmin返回最小值的索引，即最近的Agent编号，如0
            if nearest == my_idx:  # 如果最近的Agent就是自己
                my_sensors.append(sensor)  # 这个sensor归我管，加入列表

        if not my_sensors:  # 如果分区内没有sensor（边界情况）
            return self.x, self.y  # 待在原地

        # 从分区内sensor中选浓度最高的：key=lambda指定按readings最后一个值比较
        best_sensor = max(my_sensors, key=lambda s: s.readings[-1] if s.readings else 0)
        target_x, target_y = best_sensor.x, best_sensor.y  # 目标设为该sensor的坐标

        # 如果分区内最高浓度sensor读数都低于0.5，说明分区内已清干净
        if best_sensor.readings and best_sensor.readings[-1] < 0.5:
            # 从所有sensor中找全局浓度最高的
            global_best = max(sensors, key=lambda s: s.readings[-1] if s.readings else 0)
            # 全局最高浓度>0.5才值得去，否则待在原地
            if global_best.readings and global_best.readings[-1] > 0.5:
                target_x, target_y = global_best.x, global_best.y  # 改为向全局最高浓度sensor移动

        return target_x, target_y  # 返回目标坐标

    @staticmethod  # 静态方法，不依赖self，通过CleaningAgent.compute_voronoi_boundaries()调用
    def compute_voronoi_boundaries(agents, environment):
        """
        计算Voronoi分区边界线段，用于可视化
        返回线段列表 [(x1,y1,x2,y2), ...]
        """
        agent_positions = [(a.x, a.y) for a in agents]  # 收集所有Agent坐标
        grid = np.zeros((environment.height, environment.width), dtype=int)  # 创建100×100的整数网格，记录每个格子归属哪个Agent
        step = 1  # 遍历步长，1表示逐格遍历
        for j in range(0, environment.height, step):  # 遍历每一行
            for i in range(0, environment.width, step):  # 遍历每一列
                # 计算格子(i,j)到每个Agent的距离
                distances = [np.sqrt((i - ax) ** 2 + (j - ay) ** 2) for ax, ay in agent_positions]
                grid[j, i] = int(np.argmin(distances))  # 格子归属距离最近的Agent，记录编号0/1/2/3

        segments = []  # 存放边界线段，每条线段格式(x1,y1,x2,y2)
        for j in range(environment.height - 1):  # 遍历行（少1行，因为要和下一行比较）
            for i in range(environment.width - 1):  # 遍历列（少1列，因为要和下一列比较）
                owner = grid[j, i]  # 当前格子的归属Agent编号
                # 右边格子归属不同 → 存在垂直边界线
                if grid[j, i + 1] != owner:
                    # 垂直线段：在i和i+1之间（x=i+0.5），从j-0.5到j+0.5
                    segments.append((i + 0.5, j - 0.5, i + 0.5, j + 0.5))
                # 下方格子归属不同 → 存在水平边界线
                if grid[j + 1, i] != owner:
                    # 水平线段：在j和j+1之间（y=j+0.5），从i-0.5到i+0.5
                    segments.append((i - 0.5, j + 0.5, i + 0.5, j + 0.5))
        return segments  # 返回所有边界线段，用于画图


class MultiAgentSimulation:
    """多Agent仿真类：管理环境、传感器、Agent和可视化"""
    def __init__(self, width=WIDTH, height=HEIGHT, n_agents=AGENTS_NUM, n_sensors=SENSOR_NUM, n_sources=POLLUTION_SOURCE_NUM):
        self.width = width  # 仿真区域宽度
        self.height = height  # 仿真区域高度
        self.env = PollutionEnvironment(width, height)  # 创建污染环境实例

        self.sensors = []  # 传感器列表
        sensor_positions = self._generate_sensor_positions()  # 生成传感器位置，目前位置固定。
        for i, (x, y) in enumerate(sensor_positions):  # 在每个位置创建传感器
            self.sensors.append(Sensor(x, y, i))

        self.agents = []  # Agent列表
        agent_positions = self._generate_agent_positions(n_agents)  # 生成Agent初始位置
        for i, (x, y) in enumerate(agent_positions):  # 在每个位置创建Agent
            self.agents.append(CleaningAgent(x, y, i))

        self._generate_pollution_sources(n_sources)  # 生成污染源

        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))  # 创建2×2子图布局
        self.fig.suptitle('Multi-Agent Pollution Neutralization System', fontsize=14, fontweight='bold')  # 总标题
        self.fig.patch.set_facecolor('#1a1a1a')  # 图表背景色设为深色

    def _generate_sensor_positions(self):
        """生成传感器位置：9×9网格均匀分布（81个点）"""
        positions = []  # 位置列表
        for hori_interval in range(1, 10):  # 水平方向9等分
            for vert_interval in range(1, 10):  # 垂直方向9等分
                positions.append((self.width * hori_interval // 10, self.height * vert_interval // 10))  # 计算网格交点坐标
        return positions  # 返回81个传感器位置

    def _generate_agent_positions(self, n):
        """生成Agent初始位置：分布在四个象限的中心"""
        positions = [(self.width // 4, self.height // 4),  # 左上象限中心
                     (self.width // 4, self.height * 3 // 4),  # 左下象限中心
                     (self.width * 3 // 4, self.height // 4),  # 右上象限中心
                     (self.width * 3 // 4, self.height * 3 // 4)]  # 右下象限中心
        return positions  # 返回4个Agent位置

    def _generate_pollution_sources(self, n):
        """随机生成n个污染源"""
        for i in range(n):  # 循环n次
            x = random.randint(40, self.width - 40)  # 随机X坐标，避开边缘20格
            y = random.randint(40, self.height - 40)  # 随机Y坐标，避开边缘20格
            x = 37
            y = 38
            self.env.add_pollution_source(x, y, concentration=INNER_MASS, radius=1)  # 添加污染源，浓度2000

    def update(self, frame):
        """每帧更新：风场→扩散→传感器测量→Agent移动并清理"""
        self.env.update_wind(frame)  # 更新风场
        self.env.diffuse()  # 执行扩散计算
        for sensor in self.sensors:  # 每个传感器测量浓度
            sensor.measure(self.env)
        for agent in self.agents:  # 每个Agent执行决策和清理
            target_x, target_y = agent.calculate_voronoi_centroid(self.agents, self.sensors, self.env)  # 计算Voronoi分区污染加权质心
            agent.move_towards(target_x, target_y, self.env)  # 向目标移动
            agent.clean(self.env)  # 在当前位置清理
        return frame  # 返回当前帧号

    def visualize(self, frame):
        """可视化当前帧：4个子图展示不同信息"""
        for ax in self.axes.flat:  # 清空所有子图
            ax.clear()  # 清除上帧内容
            ax.set_xlim(0, self.width)  # 设置X轴范围
            ax.set_ylim(0, self.height)  # 设置Y轴范围
            ax.set_aspect('equal')  # 等比例显示
            ax.set_facecolor('#2b2b2b')  # 子图深色背景

        # --- 子图1：污染物浓度热力图 ---
        ax1 = self.axes[0, 0]  # 左上子图
        gamma = 0.3  # 幂次变换系数，<1拉伸低值区域，增强渐变层次感
        display_grid = np.power(self.env.pollution_grid, gamma)  # 对浓度做幂次变换
        vmax_display = np.power(60, gamma)  # 显示最大值，也做幂次变换保持一致
        im = ax1.imshow(display_grid, cmap='inferno', origin='lower',  # 热力图，原点在左下
                        extent=[0, self.width, 0, self.height],  # 坐标范围
                        vmin=0, vmax=vmax_display,  # 颜色映射范围
                        alpha=0.95, interpolation='bilinear')  # 透明度和双线性插值（平滑）
        ax1.set_title('Pollution Diffusion (Inferno Colormap)', color='white')  # 标题
        ax1.tick_params(colors='white')  # 刻度颜色

        if abs(self.env.wind_x) > 0.1 or abs(self.env.wind_y) > 0.1:  # 风速较大时显示风向箭头
            ax1.quiver(self.width * 0.85, self.height * 0.85, self.env.wind_x, self.env.wind_y,  # 箭头位置和方向
                       scale=5, color='cyan', width=0.005)  # 箭头样式
            ax1.text(self.width * 0.70, self.height * 0.9, f'Wind: ({self.env.wind_x:.1f}, {self.env.wind_y:.1f})',  # 风速文字
                      color='cyan', fontsize=8)

        # --- 子图2：Agent位置和路径 ---
        ax2 = self.axes[0, 1]  # 右上子图
        ax2.imshow(display_grid, cmap='inferno', origin='lower',  # 淡淡的污染背景
                   extent=[0, self.width, 0, self.height], vmin=0, vmax=vmax_display, alpha=0.3,
                   interpolation='bilinear')

        colors = ['red', 'lime', 'cyan', 'orange']  # Agent颜色列表
        voronoi_segments = CleaningAgent.compute_voronoi_boundaries(self.agents, self.env)  # 计算Voronoi边界
        for x1, y1, x2, y2 in voronoi_segments:  # 绘制Voronoi分区边界线
            ax2.plot([x1, x2], [y1, y2], color='white', alpha=0.4, linewidth=0.5)
        for i, agent in enumerate(self.agents):  # 遍历每个Agent
            color = colors[i % len(colors)]  # 选取颜色
            ax2.plot(agent.path_x[-40:], agent.path_y[-40:], color=color, alpha=0.6, linewidth=1.5)  # 绘制最近40步路径
            ax2.scatter([agent.x], [agent.y], c=color, s=120, marker='o', edgecolors='white', linewidths=1.5, zorder=5)  # 绘制当前位置
            circle = Circle((agent.x, agent.y), agent.cleanup_radius, fill=False, color=color, linestyle='--',  # 清理范围圆圈
                            alpha=0.8, linewidth=1.5)
            ax2.add_patch(circle)  # 添加圆圈到子图
        ax2.set_title('Agent Paths and Coverage', color='white')  # 标题
        ax2.tick_params(colors='white')  # 刻度颜色
        ax2.legend([f'Agent {i}' for i in range(len(self.agents))], loc='upper right', fontsize='small',  # 图例
                   facecolor='black', labelcolor='white')
        ax2.grid(True, alpha=0.2, color='gray')  # 网格线

        # --- 子图3：传感器位置和读数 ---
        ax3 = self.axes[1, 0]  # 左下子图
        ax3.imshow(display_grid, cmap='inferno', origin='lower',  # 淡淡的污染背景
                   extent=[0, self.width, 0, self.height], vmin=0, vmax=vmax_display, alpha=0.3,
                   interpolation='bilinear')
        sensor_x = [s.x for s in self.sensors]  # 所有传感器X坐标
        sensor_y = [s.y for s in self.sensors]  # 所有传感器Y坐标
        sensor_conc = [s.readings[-1] if s.readings else 0 for s in self.sensors]  # 最新浓度读数
        ax3.scatter(sensor_x, sensor_y, c=sensor_conc, cmap='inferno', s=200, marker='^', edgecolors='white', vmin=0,  # 三角形散点图，颜色映射浓度
                     vmax=100)
        ax3.set_title('Sensor Readings', color='white')  # 标题
        ax3.tick_params(colors='white')  # 刻度颜色
        for i, sensor in enumerate(self.sensors):  # 为每个传感器标注编号和读数
            ax3.annotate(f'S{i}\n{sensor_conc[i]:.0f}', (sensor.x + 2, sensor.y + 2), fontsize=8, color='white',  # 文字标注
                          fontweight='bold')
        ax3.grid(True, alpha=0.2, color='gray')  # 网格线

        # --- 子图4：统计信息 ---
        ax4 = self.axes[1, 1]  # 右下子图
        ax4.axis('off')  # 关闭坐标轴
        total_pollution = np.sum(self.env.pollution_grid)  # 计算当前环境总污染量
        stats_text = f'=== Simulation Statistics ===\n\nFrame: {frame}\nTotal Pollution: {total_pollution:.1f}\n\nWind: X={self.env.wind_x:.2f}, Y={self.env.wind_y:.2f}\n\n=== Agent Performance ===\n'  # 统计文本
        for i, agent in enumerate(self.agents):  # 每个Agent的清理量
            stats_text += f'Agent {i}: Cleaned {agent.total_cleaned:.1f} units\n'
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11, family='monospace',  # 显示统计文本
                  verticalalignment='top', color='white')

    def run(self, frames=200, interval=100, save_gif=False):
        """运行仿真：生成动画并显示或保存为GIF"""
        ani = FuncAnimation(self.fig, lambda frame: self.visualize(self.update(frame)),  # 每帧先update再visualize
                             frames=frames, interval=interval, blit=False, repeat=False)  # 总帧数、帧间隔、不使用blit、不循环
        plt.tight_layout()  # 自动调整子图间距

        if save_gif:  # 保存为GIF
            print("正在保存 GIF，请稍候... (由于插值算法，保存可能需要几分钟)")
            ani.save('pollution_diffusion.gif', writer='pillow', fps=10)  # 使用pillow写入器，10帧/秒
            print("✓ GIF 已保存！")
        else:  # 直接显示动画
            plt.show()
        return ani  # 返回动画对象


if __name__ == "__main__":
    sim = MultiAgentSimulation(width=100, height=100, n_agents=4, n_sensors=17, n_sources=1)  # 创建仿真实例
    ani = sim.run(frames=300, interval=80, save_gif=True)  # 运行150帧，帧间隔80ms，保存为GIF