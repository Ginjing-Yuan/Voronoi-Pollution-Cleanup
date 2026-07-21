import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon as MplPolygon
from matplotlib.path import Path
import random

# ================================================================
# 1. 论文参数
# ================================================================
GAUSSIAN_NUM = 5
GAUSSIAN_AMPLITUDE = 0.5
GAUSSIAN_SIGMA = 60
NOISE_LEVEL = 0.0

ALPHA = 1.0
BETA = 0.0003
KAPPA_P = 1.0
OVERSPRAY_EPSILON = 0.02

KAPPA_V = 1.5
KAPPA_OMEGA = 2.5
LAMBDA_SINGULAR = 0.5
V_MAX = 2.0
OMEGA_MAX = 0.5
DT_MOVE = 0.2

WIDTH, HEIGHT = 600, 600
AGENTS_NUM = 10

# ★ 新增：高斯基重构参数
SIGMA_BASIS = 40.0  # 基函数宽度 σ_b
LAMBDA_RLS = 0.95  # RLS 遗忘因子
REGULARIZATION = 1e-3  # 正则化项

POLYGON_VERTICES = np.array([
    (80, 30), (270, 10), (390, 25), (500, 120), (520, 340),
    (510, 400), (490, 470), (200, 520), (80, 470), (25, 320), (35, 165)
])
POLYGON_PATH = Path(POLYGON_VERTICES)


# ================================================================
# 2. 区域掩码
# ================================================================
def create_region_mask(width, height):
    y, x = np.mgrid[0:height, 0:width]
    pts = np.column_stack([x.ravel(), y.ravel()])
    return POLYGON_PATH.contains_points(pts).reshape(height, width)


# ================================================================
# 3. ★ 分布式高斯基场重构器（替代GPR）
# ================================================================
class GaussianBasisFieldEstimator:
    """
    论文方法：分布式高斯基函数重构
    l_hat(q,t) = sum_k w_k(t) * phi_k(q)
    phi_k(q) = exp(-||q - c_k||^2 / (2*sigma_b^2))
    """

    def __init__(self, centers, sigma_basis=SIGMA_BASIS, lambda_rls=LAMBDA_RLS):
        """
        centers: (K, 2) 基函数中心点坐标
        """
        self.centers = np.asarray(centers)
        self.K = len(centers)  # 基函数数量
        self.sigma_b = sigma_basis
        self.lambda_rls = lambda_rls

        # 权重向量 w(t)
        self.w = np.zeros(self.K)

        # RLS 协方差矩阵 P(t)
        self.P = np.eye(self.K) / REGULARIZATION

        self.estimated_field = None

    def compute_basis_functions(self, query_points):
        """
        计算基函数值矩阵 Φ
        query_points: (N, 2)
        返回: (N, K) 矩阵，Φ[i,k] = phi_k(query_points[i])
        """
        # 计算每个查询点到每个中心的距离平方
        # query_points: (N, 2), centers: (K, 2)
        # 结果: (N, K)
        diff = query_points[:, None, :] - self.centers[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)

        # 高斯基函数
        Phi = np.exp(-dist_sq / (2 * self.sigma_b ** 2))
        return Phi

    def update_weights_rls(self, measurements, positions):
        """
        递推最小二乘（RLS）更新权重
        measurements: (N,) 传感器读数
        positions: (N, 2) 传感器位置
        """
        if len(measurements) == 0:
            return

        # ★ 修复：将 positions 转换为 numpy array
        positions = np.asarray(positions)

        # 计算基函数矩阵 Φ
        Phi = self.compute_basis_functions(positions)  # (N, K)
        y = np.asarray(measurements)  # (N,)

        # RLS 更新
        e = y - Phi @ self.w

        Phi_T = Phi.T  # (K, N)

        F = self.lambda_rls * np.eye(len(y)) + Phi @ self.P @ Phi_T
        K_gain = self.P @ Phi_T @ np.linalg.inv(F)

        self.w = self.w + K_gain @ e
        self.P = (1.0 / self.lambda_rls) * (self.P - K_gain @ Phi @ self.P)

    def update_field(self, positions, readings, width, height, region_mask, step=12):
        """
        更新全场估计
        """
        # 更新权重
        self.update_weights_rls(readings, positions)

        # 生成查询网格
        yg, xg = np.mgrid[0:height:step, 0:width:step]
        query_pts = np.column_stack([xg.ravel(), yg.ravel()])

        # 计算基函数值
        Phi = self.compute_basis_functions(query_pts)  # (N_query, K)

        # 估计场
        y_pred = Phi @ self.w  # (N_query,)

        # 重塑为网格
        low = y_pred.reshape(xg.shape)

        # 放大到原分辨率
        field = np.repeat(np.repeat(low, step, axis=0), step, axis=1)
        field = field[:height, :width]

        # 应用区域掩码
        field[~region_mask] = 0.0
        field = np.maximum(field, 0.0)  # 浓度非负

        self.estimated_field = field


# ================================================================
# 4. 污染环境
# ================================================================
class PollutionEnvironment:
    def __init__(self, width=WIDTH, height=HEIGHT):
        self.width = width
        self.height = height
        self.region_mask = create_region_mask(width, height)
        self.pollution_grid = np.zeros((height, width))
        self.initialized = False
        self.pollution_sources = []

    def generate_gmm_field(self):
        self.pollution_sources = []
        for _ in range(GAUSSIAN_NUM):
            while True:
                x = random.randint(60, self.width - 60)
                y = random.randint(60, self.height - 60)
                if self.region_mask[y, x]:
                    if all(np.hypot(x - s['x'], y - s['y']) >= 70 for s in self.pollution_sources):
                        break
            self.pollution_sources.append({'x': x, 'y': y})

        yy, xx = np.mgrid[0:self.height, 0:self.width]
        field = np.zeros((self.height, self.width))
        for src in self.pollution_sources:
            d2 = (xx - src['x']) ** 2 + (yy - src['y']) ** 2
            field += GAUSSIAN_AMPLITUDE * np.exp(-d2 / (2 * GAUSSIAN_SIGMA ** 2))

        field[~self.region_mask] = 0.0
        self.pollution_grid = np.maximum(field, 0.0)

    def init_steady(self):
        if not self.initialized:
            self.generate_gmm_field()
            self.initialized = True

    def get_concentration(self, x, y):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < self.width and 0 <= yi < self.height:
            return self.pollution_grid[yi, xi]
        return 0.0

    def apply_additive_spray(self, px, py, u_i, radius):
        cx, cy = int(round(px)), int(round(py))
        r = radius
        y0, y1 = max(0, cy - r), min(self.height, cy + r + 1)
        x0, x1 = max(0, cx - r), min(self.width, cx + r + 1)

        yy, xx = np.mgrid[y0:y1, x0:x1]
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2

        f_pq = ALPHA * np.exp(-BETA * d2)
        valid = (f_pq >= 0.01) & self.region_mask[y0:y1, x0:x1]

        delta = np.where(valid, f_pq * u_i, 0.0)
        self.pollution_grid[y0:y1, x0:x1] = np.maximum(0.0, self.pollution_grid[y0:y1, x0:x1] + delta)


# ================================================================
# 5. 固定传感器
# ================================================================
class Sensor:
    def __init__(self, x, y, sid):
        self.x, self.y, self.id = x, y, sid
        self.readings = []

    def measure(self, env):
        c = env.get_concentration(self.x, self.y)
        self.readings.append(c)
        return c


# ================================================================
# 6. 清洁 Agent
# ================================================================
class CleaningAgent:
    def __init__(self, x, y, aid):
        self.x, self.y, self.id = float(x), float(y), aid
        self.heading = 0.0
        self.v_i = 0.0
        self.omega_i = 0.0
        self.sigma_i = 0.0
        self.u_i = 0.0
        self.bar_l_i = 0.0
        self.effective_radius = int(np.sqrt(np.log(100 * ALPHA) / BETA))
        self.path_x, self.path_y = [x], [y]

        self.prev_cx = None
        self.prev_cy = None
        self.c_dot_x = 0.0
        self.c_dot_y = 0.0

    def decide_spray(self, avg_conc):
        self.bar_l_i = avg_conc
        if avg_conc > OVERSPRAY_EPSILON:
            self.sigma_i = 1.0
            self.u_i = -KAPPA_P * self.sigma_i * avg_conc
        else:
            self.sigma_i = 0.0
            self.u_i = 0.0

    def move_towards(self, tx, ty, env):
        dx, dy = tx - self.x, ty - self.y
        dist = np.hypot(dx, dy)

        if dist < 1.0:
            self.v_i = self.omega_i = 0.0
            self.prev_cx = tx
            self.prev_cy = ty
            return

        if self.prev_cx is not None:
            self.c_dot_x = (tx - self.prev_cx) / DT_MOVE
            self.c_dot_y = (ty - self.prev_cy) / DT_MOVE
        else:
            self.c_dot_x = 0.0
            self.c_dot_y = 0.0

        self.prev_cx = tx
        self.prev_cy = ty

        ct, st = np.cos(self.heading), np.sin(self.heading)
        ex = dx * ct + dy * st
        ey = -dx * st + dy * ct

        self.omega_i = KAPPA_OMEGA * np.arctan(ey / (ex + LAMBDA_SINGULAR))
        self.v_i = (KAPPA_V * ex + self.c_dot_x * ct + self.c_dot_y * st)

        self.v_i = np.clip(self.v_i, 0, V_MAX)
        self.omega_i = np.clip(self.omega_i, -OMEGA_MAX, OMEGA_MAX)

        self.heading += self.omega_i * DT_MOVE
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi

        nx = self.x + self.v_i * np.cos(self.heading) * DT_MOVE
        ny = self.y + self.v_i * np.sin(self.heading) * DT_MOVE

        ix, iy = int(round(nx)), int(round(ny))
        if 0 <= ix < env.width and 0 <= iy < env.height and env.region_mask[iy, ix]:
            self.x, self.y = nx, ny
        else:
            self.v_i = 0.0

        self.path_x.append(self.x)
        self.path_y.append(self.y)

    def stop(self):
        self.v_i = self.omega_i = 0.0
        self.path_x.append(self.x)
        self.path_y.append(self.y)

    def calculate_voronoi_centroid(self, my_points, bar_l_i):
        if len(my_points) == 0 or bar_l_i <= 1e-8:
            return self.x, self.y

        d2 = np.sum((my_points - np.array([self.x, self.y])) ** 2, axis=1)
        l_tilde = 2 * ALPHA * BETA * np.exp(-BETA * d2) * bar_l_i

        W = np.sum(l_tilde)
        if W < 1e-8:
            return self.x, self.y

        cx = np.sum(l_tilde * my_points[:, 0]) / W
        cy = np.sum(l_tilde * my_points[:, 1]) / W
        return cx, cy

    @staticmethod
    def compute_voronoi_boundaries(agents, env, step=5):
        pos = np.array([(a.x, a.y) for a in agents])
        yc = np.arange(0, env.height, step)
        xc = np.arange(0, env.width, step)
        xx, yy = np.meshgrid(xc, yc)
        flat_mask = env.region_mask[yy, xx].ravel()

        pts = np.column_stack([xx.ravel(), yy.ravel()])[flat_mask]
        if len(pts) == 0:
            return []

        dists = np.linalg.norm(pts[:, None, :] - pos[None, :, :], axis=2)
        owners = np.argmin(dists, axis=1)

        lr = -np.ones((len(yc), len(xc)), dtype=int)
        lr[flat_mask.reshape(len(yc), len(xc))] = owners

        segs = []
        for j in range(len(yc) - 1):
            for i in range(len(xc) - 1):
                o = lr[j, i]
                if o < 0:
                    continue
                if lr[j, i + 1] >= 0 and lr[j, i + 1] != o:
                    segs.append((xc[i] + step / 2, yc[j], xc[i] + step / 2, yc[j] + step))
                if lr[j + 1, i] >= 0 and lr[j + 1, i] != o:
                    segs.append((xc[i], yc[j] + step / 2, xc[i] + step, yc[j] + step / 2))
        return segs


# ================================================================
# 7. 多 Agent 仿真主控
# ================================================================
class MultiAgentSimulation:
    def __init__(self, width=WIDTH, height=HEIGHT, n_agents=AGENTS_NUM):
        self.width, self.height = width, height
        self.env = PollutionEnvironment(width, height)
        self.env.init_steady()

        self.sensors = [Sensor(x, y, i) for i, (x, y) in enumerate(self._sensor_positions())]
        self.agents = [CleaningAgent(x, y, i) for i, (x, y) in enumerate(self._agent_positions(n_agents))]

        # ★ 使用高斯基重构器（替代GPR）
        # 基函数中心：使用传感器位置
        basis_centers = np.array([[s.x, s.y] for s in self.sensors])
        self.field_est = GaussianBasisFieldEstimator(
            centers=basis_centers,
            sigma_basis=SIGMA_BASIS,
            lambda_rls=LAMBDA_RLS
        )

        self.J_history = []
        self.E_history = []
        self.M_history = []

        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))
        self.fig.suptitle('Multi-Agent Pollution Neutralization\n(Distributed Gaussian-Basis Reconstruction)',
                          fontsize=13, fontweight='bold')

        self.ax4 = self.axes[1, 1]
        self.ax4_r = self.ax4.twinx()
        self.ax4_r2 = self.ax4.twinx()
        self.ax4_r2.spines["right"].set_position(("axes", 1.15))

    def _sensor_positions(self):
        pos = [(int(vx), int(vy)) for vx, vy in POLYGON_VERTICES]
        n = len(POLYGON_VERTICES)
        for i in range(n):
            x1, y1 = POLYGON_VERTICES[i]
            x2, y2 = POLYGON_VERTICES[(i + 1) % n]
            for t in (1 / 3, 2 / 3):
                pos.append((int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))))
        for y in range(30, self.height, 70):
            for x in range(30, self.width, 70):
                if POLYGON_PATH.contains_point((x, y)):
                    pos.append((x, y))
        return pos

    def _agent_positions(self, n):
        return [(90, 440), (120, 402), (150, 364), (180, 327),
                (210, 289), (190, 251), (170, 213), (150, 176),
                (130, 138), (110, 100)][:n]

    def update(self, frame):
        for s in self.sensors:
            s.measure(self.env)

        positions = [[s.x, s.y] for s in self.sensors]
        readings = [s.readings[-1] if s.readings else 0.0 for s in self.sensors]
        for a in self.agents:
            positions.append([a.x, a.y])
            readings.append(self.env.get_concentration(a.x, a.y))

        if frame % 3 == 0:
            self.field_est.update_field(positions, readings, self.width, self.height, self.env.region_mask, step=12)

        vs = 6
        yi = np.arange(0, self.height, vs)
        xi = np.arange(0, self.width, vs)
        xx, yy = np.meshgrid(xi, yi)
        vmask = self.env.region_mask[yy, xx]
        pts = np.column_stack([xx[vmask], yy[vmask]])

        if len(pts) == 0:
            return frame

        apos = np.array([(a.x, a.y) for a in self.agents])
        dists = np.linalg.norm(pts[:, None, :] - apos[None, :, :], axis=2)
        nearest = np.argmin(dists, axis=1)

        J_total = 0.0
        E_total_frame = 0.0

        est = self.field_est.estimated_field if self.field_est.estimated_field is not None else np.zeros(
            (self.height, self.width))

        for i, agent in enumerate(self.agents):
            my_pts = pts[nearest == i]

            if len(my_pts) > 0:
                ix = np.clip(my_pts[:, 0].astype(int), 0, self.width - 1)
                iy = np.clip(my_pts[:, 1].astype(int), 0, self.height - 1)
                cell_concs = est[iy, ix]
                avg_conc = float(np.mean(cell_concs))
            else:
                avg_conc = 0.0

            agent.decide_spray(avg_conc)

            if agent.sigma_i > 0:
                tx, ty = agent.calculate_voronoi_centroid(my_pts, avg_conc)
                agent.move_towards(tx, ty, self.env)
                self.env.apply_additive_spray(agent.x, agent.y, agent.u_i, agent.effective_radius)

                if len(my_pts) > 0:
                    d2 = np.sum((my_pts - np.array([tx, ty])) ** 2, axis=1)
                    l_tilde = 2 * ALPHA * BETA * np.exp(-BETA * d2) * avg_conc
                    weighted_dist = ((my_pts[:, 0] - tx) ** 2 + (my_pts[:, 1] - ty) ** 2) * l_tilde
                    J_total += float(np.sum(weighted_dist))
            else:
                agent.stop()

            E_total_frame += (agent.u_i ** 2) * DT_MOVE

        M_total = float(np.sum(self.env.pollution_grid[self.env.region_mask]))

        self.J_history.append(J_total)
        self.E_history.append(E_total_frame)
        self.M_history.append(M_total)
        return frame

    def visualize(self, frame):
        poly_closed = np.vstack([POLYGON_VERTICES, POLYGON_VERTICES[0]])
        mask = self.env.region_mask

        for ax in self.axes.ravel()[:3]:
            ax.clear()
            ax.set_xlim(0, self.width)
            ax.set_ylim(0, self.height)
            ax.set_aspect('equal')
            ax.set_facecolor('white')

        dg = self.env.pollution_grid.copy()
        dg[~mask] = np.nan

        ax1 = self.axes[0, 0]
        ax1.imshow(dg, cmap='Blues', origin='lower', extent=[0, self.width, 0, self.height],
                   vmin=0, vmax=1.2, alpha=0.9, interpolation='bilinear')
        ax1.plot(poly_closed[:, 0], poly_closed[:, 1], 'k-', lw=1.5)
        ax1.set_title('Pollution Field (Gaussian-Basis Reconstruction)')

        ax2 = self.axes[0, 1]
        ax2.imshow(dg, cmap='Blues', origin='lower', extent=[0, self.width, 0, self.height],
                   vmin=0, vmax=1.2, alpha=0.25, interpolation='bilinear')
        ax2.plot(poly_closed[:, 0], poly_closed[:, 1], 'k-', lw=1.5)

        for x1, y1, x2, y2 in CleaningAgent.compute_voronoi_boundaries(self.agents, self.env):
            ax2.plot([x1, x2], [y1, y2], color='darkgray', alpha=0.7, lw=1.0)

        colors = plt.cm.tab10(np.linspace(0, 1, max(len(self.agents), 1)))
        for i, a in enumerate(self.agents):
            c = colors[i]
            ax2.plot(a.path_x[-40:], a.path_y[-40:], color=c, alpha=0.6, lw=1.5)
            sz = 8
            tip = (a.x + sz * np.cos(a.heading), a.y + sz * np.sin(a.heading))
            lf = (a.x + sz * 0.55 * np.cos(a.heading + 2.5), a.y + sz * 0.55 * np.sin(a.heading + 2.5))
            rt = (a.x + sz * 0.55 * np.cos(a.heading - 2.5), a.y + sz * 0.55 * np.sin(a.heading - 2.5))
            ax2.add_patch(MplPolygon([tip, lf, rt], closed=True, fc=c, ec='black', lw=1, zorder=5))

            if a.sigma_i > 0:
                ax2.add_patch(Circle((a.x, a.y), a.effective_radius, fill=False, color=c, ls='--', alpha=0.5, lw=1))
            else:
                ax2.plot(a.x, a.y, 'x', color='gray', ms=8, mew=2, zorder=6)
        ax2.set_title('Agents & Voronoi (× = stopped)')

        ax3 = self.axes[1, 0]
        ax3.imshow(dg, cmap='Blues', origin='lower', extent=[0, self.width, 0, self.height],
                   vmin=0, vmax=1.2, alpha=0.25, interpolation='bilinear')
        ax3.plot(poly_closed[:, 0], poly_closed[:, 1], 'k-', lw=1.5)
        sx = [s.x for s in self.sensors]
        sy = [s.y for s in self.sensors]
        sc = [s.readings[-1] if s.readings else 0 for s in self.sensors]
        ax3.scatter(sx, sy, c=sc, cmap='Reds', s=30, marker='^', edgecolors='black', vmin=0, vmax=0.8, zorder=4)
        ax3.set_title('Sensor Readings (fixed + onboard)')

        self.ax4.clear()
        self.ax4_r.clear()
        self.ax4_r2.clear()

        fx = list(range(len(self.J_history)))
        if fx:
            line1, = self.ax4.plot(fx, self.J_history, color='steelblue', lw=1.5, label='J_total (cost)')
            self.ax4.set_ylabel('J_total', color='steelblue', fontsize=9)
            self.ax4.tick_params(axis='y', labelcolor='steelblue', labelsize=8)

            line2, = self.ax4_r.plot(fx, self.E_history, color='orangered', lw=1.5, label='E_frame (energy)')
            self.ax4_r.set_ylabel('E_frame', color='orangered', fontsize=9)
            self.ax4_r.tick_params(axis='y', labelcolor='orangered', labelsize=8)

            line3, = self.ax4_r2.plot(fx, self.M_history, color='forestgreen', lw=1.5, label='M_total (mass)')
            self.ax4_r2.set_ylabel('M_total', color='forestgreen', fontsize=9)
            self.ax4_r2.tick_params(axis='y', labelcolor='forestgreen', labelsize=8)

            lines = [line1, line2, line3]
            labels = [l.get_label() for l in lines]
            self.ax4.legend(lines, labels, loc='upper right', fontsize=8)

        self.ax4.set_title('Evaluation Metrics (Paper Eq.18, 23)', fontsize=10)
        self.ax4.set_xlabel('Frame', fontsize=9)
        self.ax4.grid(True, alpha=0.2)

    def run(self, frames=200, interval=80, save_gif=False, gif_filename='pollution_neutralization.gif'):
        ani = FuncAnimation(self.fig, lambda f: self.visualize(self.update(f)),
                            frames=frames, interval=interval, blit=False, repeat=False)
        plt.tight_layout()

        if save_gif:
            print(f"正在保存 GIF → {gif_filename} ...")
            try:
                ani.save(gif_filename, writer='pillow', fps=10)
                print(f"✓ GIF 已保存: {gif_filename}")
            except Exception as e:
                print(f"✗ 保存失败: {e}\n  请运行 pip install pillow")
        else:
            plt.show()
        return ani


if __name__ == "__main__":
    sim = MultiAgentSimulation(WIDTH, HEIGHT, AGENTS_NUM)
    ani = sim.run(frames=200, interval=60, save_gif=True, gif_filename='pollution_neutralization.gif')
