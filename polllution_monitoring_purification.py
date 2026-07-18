import numpy as np  # 数值计算库，用于矩阵运算和数学函数
import matplotlib.pyplot as plt  # 绑图库，用于生成热力图和动画
from matplotlib.animation import FuncAnimation  # 动画类，用于逐帧更新可视化
import random
from matplotlib.patches import Circle, Polygon as MplPolygon
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# 污染场参数（高斯混合模型）
GAUSSIAN_NUM = 5  # 高斯源数量 N_s
GAUSSIAN_AMPLITUDE_RANGE = (1, 3)  # 每个高斯源的振幅 A_k 范围
GAUSSIAN_SIGMA_RANGE = (30, 30)  # 每个高斯源的扩散半径 σ_k 范围
NOISE_LEVEL = 0.5  # 噪声水平

# 对流-扩散方程参数
DIFFUSION_COEFF = 0.25  # 扩散系数 D
CONVECTION_VX = 0.0  # 对流速度X分量 v_x
CONVECTION_VY = 0.0  # 对流速度Y分量 v_y
SOURCE_STRENGTH = 0.0  # 源项强度 S

# 清洁Agent参数
NEUTRALIZATION_MAX = 1.0
NEUTRALIZATION_DECAY = 0.0003
LINEAR_SPEED = 1.0
ANGULAR_SPEED = 0.15
OVERSPRAY_THRESHOLD = 0.2

# 场景模拟参数
POLLUTION_SOURCE_NUM = GAUSSIAN_NUM
WIDTH = 600 # 图片宽度
HEIGHT = 600 # 图片高度
AGENTS_NUM = 10 # 清洁装置数量

# 11边形顶点坐标（论文指定）
POLYGON_VERTICES = np.array([
    (80, 30), (270, 10), (390, 25), (500, 120), (520, 340),
    (510, 400), (490, 470), (200, 520), (80, 470), (25, 320), (35, 165)
])

# 离散化步长
DX = 2  # Dx=Dy=2m

def point_in_polygon(x, y, vertices):
    """射线法判断点(x,y)是否在多边形内部"""
    n = len(vertices)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def create_region_mask(width, height, vertices):
    """创建多边形区域的掩码矩阵，区域内为True"""
    mask = np.zeros((height, width), dtype=bool)
    for y in range(height):
        for x in range(width):
            if point_in_polygon(x, y, vertices):
                mask[y, x] = True
    return mask


class FieldEstimator:
    """基于高斯过程回归的连续污染场估计器"""

    def __init__(self, length_scale=50.0, noise_level=1.0):
        # 定义 GPR 核函数：常数核 * 径向基核 (RBF)
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2))
        self.gpr = GaussianProcessRegressor(kernel=kernel, alpha=noise_level, normalize_y=True)
        self.estimated_field = None
        self.grid_x = None
        self.grid_y = None

    def update_field(self, sensors, width, height, step=10):
        """
        根据传感器读数更新连续场估计。
        step: 预测网格的步长，为了计算速度，不能是1（600x600太慢），建议10或20。
        """
        # 1. 收集训练数据 (传感器坐标和读数)
        X_train = np.array([[s.x, s.y] for s in sensors])
        y_train = np.array([s.readings[-1] if s.readings else 0 for s in sensors])

        # 如果所有读数都是0，GPR会报错，直接返回全0场
        if np.max(y_train) < 1e-5:
            self.estimated_field = np.zeros((height, width))
            return

        # 2. 拟合 GPR 模型
        self.gpr.fit(X_train, y_train)

        # 3. 生成预测网格 (降采样以加速)
        y_grid, x_grid = np.mgrid[0:height:step, 0:width:step]
        X_test = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

        # 4. 预测连续场
        y_pred = self.gpr.predict(X_test)

        # 将预测结果重塑为网格，并放大回原分辨率 (最近邻插值)
        low_res_field = y_pred.reshape(x_grid.shape)
        self.estimated_field = np.repeat(np.repeat(low_res_field, step, axis=0), step, axis=1)
        # 裁剪到实际宽高
        self.estimated_field = self.estimated_field[:height, :width]


class PollutionEnvironment:
    """污染环境类：管理污染网格、污染源、扩散和清理逻辑"""
    def __init__(self, width=WIDTH, height=HEIGHT, diffusion_mode='steady'):
        self.width = width
        self.height = height

        self.diffusion_mode = diffusion_mode
        self.pollution_grid = np.zeros((height, width))
        self.pollution_sources = []
        self.initialized = False

        # 11边形区域掩码
        print("正在生成11边形区域掩码...")
        self.region_mask = create_region_mask(width, height, POLYGON_VERTICES)
        print(f"区域掩码生成完成，区域内格子数: {np.sum(self.region_mask)}")

    def generate_gmm_field(self):
        """
        用高斯混合模型生成初始污染场：
        φ(x,y) = Σ A_k * exp(-((x-x_k)² + (y-y_k)²) / (2σ_k²)) + Noise
        污染源位置在11边形区域内随机生成
        """
        self.pollution_sources = []
        for _ in range(GAUSSIAN_NUM):
            while True:
                x = random.randint(40, self.width - 40)
                y = random.randint(40, self.height - 40)
                if self.region_mask[y, x]:
                    # 如果不控制污染源点之间的距离可直接break
                    # break
                    # 如果想保证污染源直接不出现重合 可以设置下方条件保证污染源直接的间距
                    if all(np.sqrt((x - s['x'])**2 + (y - s['y'])**2) >= 40 for s in self.pollution_sources):
                        break
            A_k = random.uniform(*GAUSSIAN_AMPLITUDE_RANGE)
            sigma_k = random.uniform(*GAUSSIAN_SIGMA_RANGE)
            self.pollution_sources.append({'x': x, 'y': y, 'amplitude': A_k, 'sigma': sigma_k})

        yy, xx = np.mgrid[0:self.height, 0:self.width]
        field = np.zeros((self.height, self.width))
        for src in self.pollution_sources:
            dx = xx - src['x']
            dy = yy - src['y']
            field += src['amplitude'] * np.exp(-(dx ** 2 + dy ** 2) / (2 * src['sigma'] ** 2))
        field += np.random.normal(0, NOISE_LEVEL, (self.height, self.width))
        field[~self.region_mask] = 0
        field = np.maximum(field, 0)
        self.pollution_grid = field

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

    def _diffuse_transient(self):
        """瞬态模式暂未实现，预留接口"""
        pass

    def _diffuse_steady(self):
        """
        稳态扩散初始化：用高斯混合模型生成初始污染场
        只在首次调用时执行
        """
        self.generate_gmm_field()

    def get_concentration(self, x, y):
        """查询指定坐标的污染浓度"""
        x, y = int(x), int(y)  # 坐标取整
        if 0 <= x < self.width and 0 <= y < self.height:  # 边界检查
            return self.pollution_grid[y, x]  # 返回网格中的浓度值
        return 0  # 越界返回0

    def reduce_pollution(self, x, y, neutralization_max, neutralization_decay, effective_radius):
        """
        距离衰减中和模型：中和效果 = neutralization_max * exp(-neutralization_decay * d²)
        先计算有效半径，再对范围内每个格子按衰减系数降低污染
        """
        x, y = int(x), int(y)
        r = effective_radius
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                ny, nx = y + dy, x + dx
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    d_sq = dx ** 2 + dy ** 2
                    effect = neutralization_max * np.exp(-neutralization_decay * d_sq)
                    if effect < 0.01:
                        continue
                    reduction = effect * self.pollution_grid[ny, nx]
                    self.pollution_grid[ny, nx] = max(0, self.pollution_grid[ny, nx] - reduction)

    def _diffuse_steady_rebalance(self):
        """
        对流-扩散方程求解：
        ∂φ/∂t = D∇²φ - v·∇φ + S(x,y,t) - U(x,y,t)
        D: 扩散系数, v: 对流速度, S: 源项, U: 汇项(清理)
        使用有限差分法离散化，显式时间推进
        """
        grid = self.pollution_grid.copy()
        grid[0, :] = 0
        grid[-1, :] = 0
        grid[:, 0] = 0
        grid[:, -1] = 0
        D = DIFFUSION_COEFF
        vx = CONVECTION_VX
        vy = CONVECTION_VY
        dx = DX
        dt = 0.5 * dx ** 2 / (4 * D + 1e-10)

        laplacian = (np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
                     np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) - 4 * grid) / (dx ** 2)
        grad_x = (np.roll(grid, -1, axis=1) - np.roll(grid, 1, axis=1)) / (2 * dx)
        grad_y = (np.roll(grid, -1, axis=0) - np.roll(grid, 1, axis=0)) / (2 * dx)

        source_term = np.zeros_like(grid)
        if SOURCE_STRENGTH > 0:
            for src in self.pollution_sources:
                sx, sy = int(src['x']), int(src['y'])
                source_term[sy, sx] += SOURCE_STRENGTH

        new_grid = grid + dt * (D * laplacian - vx * grad_x - vy * grad_y + source_term)
        new_grid[~self.region_mask] = 0
        new_grid[0, :] = 0
        new_grid[-1, :] = 0
        new_grid[:, 0] = 0
        new_grid[:, -1] = 0
        new_grid = np.maximum(new_grid, 0)
        self.pollution_grid = new_grid


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
    def __init__(self, x, y, agent_id, linear_speed=LINEAR_SPEED, angular_speed=ANGULAR_SPEED,
                 neutralization_max=NEUTRALIZATION_MAX, neutralization_decay=NEUTRALIZATION_DECAY):
        self.x = x
        self.y = y
        self.id = agent_id
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.heading = 0.0
        self.overspray_on = True
        self.neutralization_max = neutralization_max
        self.neutralization_decay = neutralization_decay
        self.effective_radius = int(np.sqrt(np.log(neutralization_max / 0.01) / neutralization_decay)) if neutralization_decay > 0 else 100
        self.path_x = [x]
        self.path_y = [y]
        self.total_cleaned = 0

    def move_towards(self, target_x, target_y, environment):
        """先转向目标方向，航向对齐后再前进"""
        dx = target_x - self.x
        dy = target_y - self.y
        distance = np.sqrt(dx ** 2 + dy ** 2)
        if distance < 0.5:
            self.path_x.append(self.x)
            self.path_y.append(self.y)
            return
        target_heading = np.arctan2(dy, dx)
        angle_diff = (target_heading - self.heading + np.pi) % (2 * np.pi) - np.pi
        if abs(angle_diff) > self.angular_speed:
            self.heading += np.sign(angle_diff) * self.angular_speed
            self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi
        else:
            self.heading = target_heading
            step = min(self.linear_speed, distance)
            self.x += np.cos(self.heading) * step
            self.y += np.sin(self.heading) * step
            self.x = max(0, min(environment.width - 1, self.x))
            self.y = max(0, min(environment.height - 1, self.y))
        self.path_x.append(self.x)
        self.path_y.append(self.y)

    def clean(self, environment):
        """检测中心浓度，防过喷开关控制是否执行中和"""
        local_conc = environment.get_concentration(self.x, self.y)
        self.overspray_on = local_conc >= OVERSPRAY_THRESHOLD
        if not self.overspray_on:
            return
        cleaned_before = np.sum(environment.pollution_grid)
        environment.reduce_pollution(self.x, self.y, self.neutralization_max, self.neutralization_decay, self.effective_radius)
        cleaned_after = np.sum(environment.pollution_grid)
        self.total_cleaned += (cleaned_before - cleaned_after)

    def calculate_voronoi_centroid(self, agents, sensors, field_estimator):
        """
        基于 GPR 连续场估计计算 Voronoi 广义质心
        """
        agent_positions = np.array([(a.x, a.y) for a in agents])
        my_idx = self.id

        # 获取 GPR 估计的连续场
        est_field = field_estimator.estimated_field
        if est_field is None:
            return self.x, self.y

        # 找出属于当前 Agent Voronoi 分区的所有网格点
        # 为了加速，这里使用步长采样 (例如步长为5)
        step = 5
        y_indices = np.arange(0, est_field.shape[0], step)
        x_indices = np.arange(0, est_field.shape[1], step)
        xx, yy = np.meshgrid(x_indices, y_indices)

        # 计算这些采样点到所有 Agent 的距离
        # 形状: (num_points, num_agents)
        points = np.vstack([xx.ravel(), yy.ravel()]).T
        dists = np.sqrt(np.sum((points[:, None, :] - agent_positions[None, :, :]) ** 2, axis=2))

        # 找出距离当前 Agent 最近的点 (即属于我的 Voronoi 分区)
        nearest_agent_idx = np.argmin(dists, axis=1)
        my_mask = (nearest_agent_idx == my_idx)

        my_points = points[my_mask]
        if len(my_points) == 0:
            return self.x, self.y

        # 获取这些点对应的估计浓度 (质量)
        my_masses = est_field[my_points[:, 1].astype(int), my_points[:, 0].astype(int)]

        total_mass = np.sum(my_masses)
        if total_mass < 0.5:
            # 质量太小，退化为向全局最高浓度点移动
            max_idx = np.unravel_index(np.argmax(est_field), est_field.shape)
            return max_idx[1], max_idx[0]

        # 计算加权质心 (积分近似)
        target_x = np.sum(my_masses * my_points[:, 0]) / total_mass
        target_y = np.sum(my_masses * my_points[:, 1]) / total_mass

        return target_x, target_y

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
    def __init__(self, width=WIDTH, height=HEIGHT, n_agents=AGENTS_NUM, n_sources=POLLUTION_SOURCE_NUM):
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

        self._generate_pollution_sources(n_sources)

        self.avg_conc_history = []
        self.centroid_dist_history = []  # 生成污染源

        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))  # 创建2×2子图布局
        self.fig.suptitle('Multi-Agent Pollution Neutralization System', fontsize=14, fontweight='bold')  # 总标题
        self.fig.patch.set_facecolor('white')

        # --- 修复：在初始化时创建 twinx 轴 ---
        self.ax4 = self.axes[1, 1]
        self.ax4_r = self.ax4.twinx()

        self.field_estimator = FieldEstimator(length_scale=60.0, noise_level=0.5)

    def _generate_sensor_positions(self):
        """在11边形边界1/4处、顶点、以及内部每40像素网格上生成传感器"""
        positions = []
        # 顶点传感器
        for vx, vy in POLYGON_VERTICES:
            positions.append((int(vx), int(vy)))
        # 每条边的1/4、2/4、3/4处放传感器
        n = len(POLYGON_VERTICES)
        for i in range(n):
            x1, y1 = POLYGON_VERTICES[i]
            x2, y2 = POLYGON_VERTICES[(i + 1) % n]
            for j in range(1, 3):
                t = j / 3
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                positions.append((int(x), int(y)))
        # 内部网格传感器：每80像素一个，仅保留在多边形内部的点
        for y in range(20, self.height, 80):
            for x in range(20, self.width, 80):
                if self.env.region_mask[y, x]:
                    positions.append((x, y))
        return positions

    def _generate_agent_positions(self, n):
        """生成Agent初始位置"""
        positions = [(90, 440), (120, 402), (150, 364), (180, 327),
                     (210, 289), (190, 251), (170, 213), (150, 176),
                     (130, 138), (110, 100)]
        return positions

    def _generate_pollution_sources(self, n):
        """污染源由GMM在diffuse()首次调用时自动生成，此处无需手动添加"""
        pass

    def _compute_partition_assignments(self):
        """计算每个传感器属于哪个Agent的Voronoi分区"""
        agent_positions = [(a.x, a.y) for a in self.agents]
        assignments = []
        for sensor in self.sensors:
            distances = [np.sqrt((sensor.x - ax) ** 2 + (sensor.y - ay) ** 2) for ax, ay in agent_positions]
            assignments.append(int(np.argmin(distances)))
        return assignments

    def update(self, frame):
        """每帧更新：扩散→传感器测量→Agent移动并清理→评估指标"""
        self.env.diffuse()
        for sensor in self.sensors:
            sensor.measure(self.env)

        # 每帧（或每隔几帧）更新 GPR 连续场估计
        if frame % 2 == 0:  # 为了性能，每2帧更新一次GPR
            self.field_estimator.update_field(self.sensors, self.width, self.height, step=15)

        for agent in self.agents:
            # 传入 field_estimator
            target_x, target_y = agent.calculate_voronoi_centroid(
                self.agents, self.sensors, self.field_estimator
            )
            agent.move_towards(target_x, target_y, self.env)
            agent.clean(self.env)

        assignments = self._compute_partition_assignments()
        total_avg_conc = 0.0
        total_centroid_dist = 0.0
        for i, agent in enumerate(self.agents):
            my_sensors = [self.sensors[j] for j, a in enumerate(assignments) if a == i]
            if my_sensors:
                avg_conc = np.mean([s.readings[-1] if s.readings else 0 for s in my_sensors] +
                                   [self.env.get_concentration(agent.x, agent.y)])
                total_avg_conc += avg_conc
                total_mass = sum(s.readings[-1] if s.readings else 0 for s in my_sensors)
                if total_mass > 0.5:
                    cx = sum((s.readings[-1] if s.readings else 0) * s.x for s in my_sensors) / total_mass
                    cy = sum((s.readings[-1] if s.readings else 0) * s.y for s in my_sensors) / total_mass
                else:
                    cx, cy = agent.x, agent.y
                total_centroid_dist += np.sqrt((agent.x - cx) ** 2 + (agent.y - cy) ** 2)

        self.avg_conc_history.append(total_avg_conc)
        self.centroid_dist_history.append(total_centroid_dist)
        return frame

    def visualize(self, frame):
        """可视化当前帧：4个子图展示不同信息"""
        # 11边形闭合边界线坐标
        polygon_closed = np.vstack([POLYGON_VERTICES, POLYGON_VERTICES[0]])

        for ax in [self.axes[0, 0], self.axes[0, 1], self.axes[1, 0]]:
            ax.clear()
            ax.set_xlim(0, self.width)
            ax.set_ylim(0, self.height)
            ax.set_aspect('equal')
            ax.set_facecolor('white')

        # --- 子图1：污染物浓度热力图 ---
        ax1 = self.axes[0, 0]
        gamma = 0.3
        display_grid = np.power(self.env.pollution_grid, gamma)
        display_grid[~self.env.region_mask] = np.nan
        vmax_display = np.power(30, gamma)
        im = ax1.imshow(display_grid, cmap='Blues', origin='lower',
                        extent=[0, self.width, 0, self.height],
                        vmin=0, vmax=vmax_display,
                        alpha=0.95, interpolation='bilinear')
        ax1.set_title('Pollution Diffusion', color='black')
        ax1.tick_params(colors='black')
        ax1.plot(polygon_closed[:, 0], polygon_closed[:, 1], 'k-', linewidth=1.5)

        # --- 子图2：Agent位置和路径 ---
        ax2 = self.axes[0, 1]
        ax2.imshow(display_grid, cmap='Blues', origin='lower',
                   extent=[0, self.width, 0, self.height], vmin=0, vmax=vmax_display, alpha=0.3,
                   interpolation='bilinear')
        ax2.plot(polygon_closed[:, 0], polygon_closed[:, 1], 'k-', linewidth=1.5)

        colors = ['red', 'lime', 'blue', 'orange']
        voronoi_segments = CleaningAgent.compute_voronoi_boundaries(self.agents, self.env)
        for x1, y1, x2, y2 in voronoi_segments:
            ax2.plot([x1, x2], [y1, y2], color='gray', alpha=0.5, linewidth=0.5)
        for i, agent in enumerate(self.agents):
            color = colors[i % len(colors)]
            ax2.plot(agent.path_x[-40:], agent.path_y[-40:], color=color, alpha=0.6, linewidth=1.5)
            heading = agent.heading
            size = 10
            tip = (agent.x + size * np.cos(heading), agent.y + size * np.sin(heading))
            left = (agent.x + size * 0.6 * np.cos(heading + 2.5), agent.y + size * 0.6 * np.sin(heading + 2.5))
            right = (agent.x + size * 0.6 * np.cos(heading - 2.5), agent.y + size * 0.6 * np.sin(heading - 2.5))
            triangle = MplPolygon([tip, left, right], closed=True, facecolor=color, edgecolor='black', linewidth=1.5, zorder=5)
            ax2.add_patch(triangle)
            if agent.overspray_on:
                ax2.scatter([agent.x], [agent.y], c='red', s=8, marker='^', zorder=6)
            circle = Circle((agent.x, agent.y), agent.effective_radius, fill=False, color=color, linestyle='--',
                            alpha=0.4, linewidth=1.0)
            ax2.add_patch(circle)
        ax2.set_title('Agent Paths and Coverage', color='black')
        ax2.tick_params(colors='black')
        ax2.legend([f'Agent {i}' for i in range(len(self.agents))], loc='upper right', fontsize='small',
                   facecolor='white', labelcolor='black', edgecolor='gray')
        ax2.grid(True, alpha=0.2, color='gray')

        # --- 子图3：传感器位置和读数 ---
        ax3 = self.axes[1, 0]
        ax3.imshow(display_grid, cmap='Blues', origin='lower',
                   extent=[0, self.width, 0, self.height], vmin=0, vmax=vmax_display, alpha=0.3,
                   interpolation='bilinear')
        ax3.plot(polygon_closed[:, 0], polygon_closed[:, 1], 'k-', linewidth=1.5)
        sensor_x = [s.x for s in self.sensors]
        sensor_y = [s.y for s in self.sensors]
        sensor_conc = [s.readings[-1] if s.readings else 0 for s in self.sensors]
        ax3.scatter(sensor_x, sensor_y, c=sensor_conc, cmap='Blues', s=200, marker='^', edgecolors='black', vmin=0,
                     vmax=100)
        ax3.set_title('Sensor Readings', color='black')
        ax3.tick_params(colors='black')
        for i, sensor in enumerate(self.sensors):
            ax3.annotate(f'S{i}\n{sensor_conc[i]:.0f}', (sensor.x + 2, sensor.y + 2), fontsize=8, color='black',
                          fontweight='bold')
        ax3.grid(True, alpha=0.2, color='gray')

        # --- 子图4：评估曲线 ---
        self.ax4.clear()  # 清空主轴（蓝线）
        self.ax4_r.clear()  # ★ 清空副轴（橙线），而不是重新 twinx()
        self.ax4.set_facecolor('white')

        frames_x = list(range(len(self.avg_conc_history)))
        if frames_x:
            # 蓝线：画在主轴
            self.ax4.plot(frames_x, self.avg_conc_history,
                          color='steelblue', linewidth=1.2, label='Avg Conc Sum')
            # self.ax4.set_ylabel('Avg Concentration Sum', color='steelblue', fontsize=9)
            self.ax4.tick_params(axis='y', labelcolor='steelblue')
            self.ax4.tick_params(axis='x', colors='black')

            # 橙线：画在已有的副轴上（不再 twinx）
            self.ax4_r.plot(frames_x, self.centroid_dist_history,
                            color='orangered', linewidth=1.2, label='Centroid Dist Sum')
            # self.ax4_r.set_ylabel('Centroid Distance Sum', color='orangered', fontsize=9)
            self.ax4_r.tick_params(axis='y', labelcolor='orangered')

            # 合并两个轴的图例
            lines1, labels1 = self.ax4.get_legend_handles_labels()
            lines2, labels2 = self.ax4_r.get_legend_handles_labels()
            self.ax4.legend(lines1 + lines2, labels1 + labels2,
                            loc='upper right', fontsize=8,
                            facecolor='white', labelcolor='black', edgecolor='gray')

        self.ax4.set_title('Evaluation Metrics', color='black', fontsize=10)
        self.ax4.set_xlabel('Frame', color='black', fontsize=9)
        self.ax4.grid(True, alpha=0.2, color='gray')

    def save_gpr_snapshot(self, save_path='gpr_reconstruction.png'):
        """使用当前传感器读数进行GPR场估计，并保存对比图到本地"""
        print("\n正在准备 GPR 场还原图片...")

        # 1. 收集当前所有传感器的位置和读数
        sensor_x = [s.x for s in self.sensors]
        sensor_y = [s.y for s in self.sensors]
        sensor_readings = [s.readings[-1] if s.readings else 0 for s in self.sensors]

        X_train = np.vstack((sensor_x, sensor_y)).T
        y_train = np.array(sensor_readings)

        if np.max(y_train) < 1e-5:
            print("传感器读数全为0，跳过GPR估计。")
            return

        # 2. 配置并训练 GPR 模型
        # length_scale 控制平滑度，alpha 控制对噪声的容忍度
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=40.0, length_scale_bounds=(1e-1, 1e3))
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.5, normalize_y=True)
        gpr.fit(X_train, y_train)

        # 3. 降采样预测全场 (步长设为4，即预测 150x150 的网格，大幅提速)
        step = 4
        y_grid, x_grid = np.mgrid[0:self.height:step, 0:self.width:step]
        X_test = np.vstack((x_grid.ravel(), y_grid.ravel())).T

        print("正在预测全场 (GPR)，请稍候...")
        y_pred = gpr.predict(X_test)
        gpr_field_low = y_pred.reshape(x_grid.shape)

        # 4. 获取真实场并计算低分辨率下的误差
        true_field = self.env.pollution_grid
        true_field_low = true_field[0:self.height:step, 0:self.width:step]
        error_field = np.abs(true_field_low - gpr_field_low)

        # 5. 绘制 1行3列 的对比图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Gaussian Process Regression: Field Reconstruction', fontsize=16, fontweight='bold')

        vmax = np.max(true_field)  # 统一颜色映射的最大值

        # 子图1：真实污染场
        im1 = axes[0].imshow(true_field, cmap='Blues', origin='lower',
                             extent=[0, self.width, 0, self.height], vmin=0, vmax=vmax, interpolation='bilinear')
        axes[0].scatter(sensor_x, sensor_y, c='red', s=20, marker='^', edgecolors='black', label='Sensors')
        axes[0].set_title('True Pollution Field')
        axes[0].set_aspect('equal')
        plt.colorbar(im1, ax=axes[0])

        # 子图2：GPR 估计场 (利用 extent 和 interpolation 自动平滑放大)
        im2 = axes[1].imshow(gpr_field_low, cmap='Blues', origin='lower',
                             extent=[0, self.width, 0, self.height], vmin=0, vmax=vmax, interpolation='bilinear')
        axes[1].scatter(sensor_x, sensor_y, c='red', s=20, marker='^', edgecolors='black')
        axes[1].set_title('GPR Estimated Field')
        axes[1].set_aspect('equal')
        plt.colorbar(im2, ax=axes[1])

        # 子图3：绝对误差场
        im3 = axes[2].imshow(error_field, cmap='hot_r', origin='lower',
                             extent=[0, self.width, 0, self.height], vmin=0, vmax=vmax * 0.2, interpolation='bilinear')
        axes[2].scatter(sensor_x, sensor_y, c='cyan', s=20, marker='^', edgecolors='black')
        axes[2].set_title('Absolute Estimation Error')
        axes[2].set_aspect('equal')
        plt.colorbar(im3, ax=axes[2])

        plt.tight_layout()
        # 保存高清图片
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ GPR 场还原图片已保存至: {save_path}")
        plt.close(fig)  # 关闭图片释放内存

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
    sim = MultiAgentSimulation(width=WIDTH, height=HEIGHT, n_agents=AGENTS_NUM, n_sources=POLLUTION_SOURCE_NUM)
    ani = sim.run(frames=150, interval=80, save_gif=True)

    # 在仿真结束后（此时传感器已经有了丰富的历史读数），保存 GPR 场还原图片
    # sim.save_gpr_snapshot('gpr_reconstruction.png')
