import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon as MplPolygon
from matplotlib.path import Path
import random

# ================================================================
# 1. 论文参数 (严格对应 Section 4)
# ================================================================
GAUSSIAN_NUM = 5
GAUSSIAN_AMPLITUDE = 0.5
GAUSSIAN_SIGMA = 60

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

# ★ 论文 Section 4: Dx = Dy = 2m
DX = 2.0
DY = 2.0
AREA_ELEMENT = DX * DY  # 4.0 m^2

# ★ 有限喷洒半径 Ra (论文 Fig.2 示意)
RA = 120.0

WIDTH, HEIGHT = 600, 600
AGENTS_NUM = 10

SIGMA_BASIS = 40.0
LAMBDA_RLS = 0.95
REGULARIZATION = 1e-3

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
# 3. 分布式高斯基场重构器
# ================================================================
class GaussianBasisFieldEstimator:
    def __init__(self, centers, sigma_basis=SIGMA_BASIS, lambda_rls=LAMBDA_RLS):
        self.centers = np.asarray(centers)
        self.K = len(centers)
        self.sigma_b = sigma_basis
        self.lambda_rls = lambda_rls
        self.w = np.zeros(self.K)
        self.P = np.eye(self.K) / REGULARIZATION
        self.estimated_field = None

    def compute_basis_functions(self, query_points):
        diff = query_points[:, None, :] - self.centers[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        return np.exp(-dist_sq / (2 * self.sigma_b ** 2))

    def update_weights_rls(self, measurements, positions):
        if len(measurements) == 0: return
        positions = np.asarray(positions)
        Phi = self.compute_basis_functions(positions)
        y = np.asarray(measurements)
        e = y - Phi @ self.w
        Phi_T = Phi.T
        F = self.lambda_rls * np.eye(len(y)) + Phi @ self.P @ Phi_T
        K_gain = self.P @ Phi_T @ np.linalg.inv(F)
        self.w = self.w + K_gain @ e
        self.P = (1.0 / self.lambda_rls) * (self.P - K_gain @ Phi @ self.P)

    def update_field(self, positions, readings, width, height, region_mask, step=12):
        self.update_weights_rls(readings, positions)
        yg, xg = np.mgrid[0:height:step, 0:width:step]
        query_pts = np.column_stack([xg.ravel(), yg.ravel()])
        Phi = self.compute_basis_functions(query_pts)
        y_pred = Phi @ self.w
        low = y_pred.reshape(xg.shape)
        field = np.repeat(np.repeat(low, step, axis=0), step, axis=1)[:height, :width]
        field[~region_mask] = 0.0
        self.estimated_field = np.maximum(field, 0.0)


# ================================================================
# 4. 污染环境
# ================================================================
class PollutionEnvironment:
    def __init__(self, width=WIDTH, height=HEIGHT):
        self.width, self.height = width, height
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
                if self.region_mask[y, x] and all(
                        np.hypot(x - s['x'], y - s['y']) >= 70 for s in self.pollution_sources):
                    break
            self.pollution_sources.append({'x': x, 'y': y})

        yy, xx = np.mgrid[0:self.height, 0:self.width]
        field = np.zeros((self.height, self.width))
        for src in self.pollution_sources:
            d2 = (xx - src['x']) ** 2 + (yy - src['y']) ** 2
            field += GAUSSIAN_AMPLITUDE * np.exp(-d2 / (2 * GAUSSIAN_SIGMA ** 2))
        self.pollution_grid = np.maximum(field * self.region_mask, 0.0)
        self.initialized = True

    def get_concentration(self, x, y):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < self.width and 0 <= yi < self.height:
            return self.pollution_grid[yi, xi]
        return 0.0

    def apply_additive_spray(self, px, py, u_i, Ra=RA):
        """★ 严格实现论文公式(3)：有限范围喷洒核"""
        cx, cy = int(round(px)), int(round(py))
        r = int(np.ceil(Ra))
        y0, y1 = max(0, cy - r), min(self.height, cy + r + 1)
        x0, x1 = max(0, cx - r), min(self.width, cx + r + 1)

        yy, xx = np.mgrid[y0:y1, x0:x1]
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        d = np.sqrt(d2)

        f_pq = np.zeros_like(d)
        mask = d <= Ra
        f_pq[mask] = ALPHA * (np.exp(-BETA * d2[mask]) - np.exp(-BETA * Ra ** 2))

        valid = (f_pq > 0) & self.region_mask[y0:y1, x0:x1]
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

        self.prev_bar_l_i = None
        self.dJ_dt = 0.0
        self.m_Vi = 0.0
        self.e_ix = 0.0
        self.e_iy = 0.0  # ★ 新增：y方向跟踪误差

        self.path_x, self.path_y = [x], [y]

    def decide_spray(self, avg_conc):
        self.bar_l_i = avg_conc
        if avg_conc > OVERSPRAY_EPSILON:
            self.sigma_i = 1.0
            self.u_i = -KAPPA_P * self.sigma_i * avg_conc
        else:
            self.sigma_i = 0.0
            self.u_i = 0.0

    def calculate_voronoi_stats(self, my_points, bar_l_i, Ra=RA):
        """★ 严格实现论文公式(16)和(3)，包含面积元"""
        if len(my_points) == 0 or bar_l_i <= 1e-8:
            return self.x, self.y, 0.0, 0.0

        d2 = np.sum((my_points - np.array([self.x, self.y])) ** 2, axis=1)
        r = np.sqrt(d2)

        # ★ 公式(16): 用平滑高斯计算 m_Vi 和 c_Vi (含面积元)
        l_tilde = 2 * ALPHA * BETA * np.exp(-BETA * d2) * bar_l_i
        m_Vi = np.sum(l_tilde) * AREA_ELEMENT

        if m_Vi < 1e-8:
            return self.x, self.y, 0.0, 0.0

        cx = np.sum(l_tilde * my_points[:, 0]) * AREA_ELEMENT / m_Vi
        cy = np.sum(l_tilde * my_points[:, 1]) * AREA_ELEMENT / m_Vi

        # ★ 公式(3): 有限喷洒核，用于计算 J_i 和 dJ_i/dt (含面积元)
        f_pq = np.zeros_like(r)
        mask = r <= Ra
        f_pq[mask] = ALPHA * (np.exp(-BETA * d2[mask]) - np.exp(-BETA * Ra ** 2))
        I_fi = np.sum(f_pq) * AREA_ELEMENT

        return cx, cy, m_Vi, I_fi

    def move_towards(self, tx, ty, env):
        """★ 严格实现论文公式(20)和(21)"""
        dx, dy = tx - self.x, ty - self.y

        ct, st = np.cos(self.heading), np.sin(self.heading)
        ex = dx * ct + dy * st  # e_{i,x}
        ey = -dx * st + dy * ct  # e_{i,y}
        self.e_ix = ex
        self.e_iy = ey  # ★ 保存 y 方向误差

        if self.sigma_i > 0:
            self.omega_i = KAPPA_OMEGA * np.arctan2(ey, ex)
        else:
            self.omega_i = 0.0

        if self.sigma_i == 0:
            self.v_i = 0.0
        else:
            if abs(ex) >= LAMBDA_SINGULAR:
                if self.m_Vi > 1e-8:
                    compensation = self.dJ_dt / (self.m_Vi * ex)
                    self.v_i = KAPPA_V * ex - compensation
                else:
                    self.v_i = KAPPA_V * ex
            else:
                self.v_i = KAPPA_V * ex

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

    @staticmethod
    def compute_voronoi_boundaries(agents, env, step=5):
        """绘图用，步长可以大一些以加速"""
        pos = np.array([(a.x, a.y) for a in agents])
        yc = np.arange(0, env.height, step)
        xc = np.arange(0, env.width, step)
        xx, yy = np.meshgrid(xc, yc)
        flat_mask = env.region_mask[yy, xx].ravel()
        pts = np.column_stack([xx.ravel(), yy.ravel()])[flat_mask]
        if len(pts) == 0: return []
        dists = np.linalg.norm(pts[:, None, :] - pos[None, :, :], axis=2)
        owners = np.argmin(dists, axis=1)
        lr = -np.ones((len(yc), len(xc)), dtype=int)
        lr[flat_mask.reshape(len(yc), len(xc))] = owners
        segs = []
        for j in range(len(yc) - 1):
            for i in range(len(xc) - 1):
                o = lr[j, i]
                if o < 0: continue
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
        self.env.generate_gmm_field()

        self.sensors = [Sensor(x, y, i) for i, (x, y) in enumerate(self._sensor_positions())]
        self.agents = [CleaningAgent(x, y, i) for i, (x, y) in enumerate(self._agent_positions(n_agents))]

        basis_centers = np.array([[s.x, s.y] for s in self.sensors])
        self.field_est = GaussianBasisFieldEstimator(centers=basis_centers)

        # ★ 论文 Section 4 指标：时间积分值
        self.J_total_integral = 0.0
        self.E_total_integral = 0.0
        self.M_history = []

        # ★ 用于绘制瞬时曲线 (对应论文 Fig.5)
        self.J_instant_history = []
        self.E_instant_history = []

        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))
        self.fig.suptitle('Multi-Agent Pollution Neutralization\n(Strict Dx=Dy=2m, Formula 3/13/16/20/23)',
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
        for s in self.sensors: s.measure(self.env)

        positions = [[s.x, s.y] for s in self.sensors]
        readings = [s.readings[-1] if s.readings else 0.0 for s in self.sensors]
        for a in self.agents:
            positions.append([a.x, a.y])
            readings.append(self.env.get_concentration(a.x, a.y))

        if frame % 3 == 0:
            self.field_est.update_field(positions, readings, self.width, self.height, self.env.region_mask, step=12)

        # ★ 论文 Section 4: Dx = Dy = 2m
        vs_calc = 2
        yi = np.arange(0, self.height, vs_calc)
        xi = np.arange(0, self.width, vs_calc)
        xx, yy = np.meshgrid(xi, yi)
        vmask = self.env.region_mask[yy, xx]
        pts = np.column_stack([xx[vmask], yy[vmask]])
        if len(pts) == 0: return frame

        apos = np.array([(a.x, a.y) for a in self.agents])
        dists = np.linalg.norm(pts[:, None, :] - apos[None, :, :], axis=2)
        nearest = np.argmin(dists, axis=1)

        # ★ 瞬时指标 (对应论文 Fig.5 的纵坐标)
        J_instant = 0.0
        E_instant = 0.0

        est = self.field_est.estimated_field if self.field_est.estimated_field is not None else np.zeros(
            (self.height, self.width))

        for i, agent in enumerate(self.agents):
            my_pts = pts[nearest == i]
            if len(my_pts) > 0:
                ix = np.clip(my_pts[:, 0].astype(int), 0, self.width - 1)
                iy = np.clip(my_pts[:, 1].astype(int), 0, self.height - 1)
                avg_conc = float(np.mean(est[iy, ix]))
            else:
                avg_conc = 0.0

            agent.decide_spray(avg_conc)

            # ★ 公式(23): ∂Ji/∂t = \dot{\bar{l}}_i * ∫ f dq
            if agent.prev_bar_l_i is not None:
                dot_bar_l_i = (avg_conc - agent.prev_bar_l_i) / DT_MOVE
            else:
                dot_bar_l_i = 0.0
            agent.prev_bar_l_i = avg_conc

            if agent.sigma_i > 0:
                cx, cy, m_Vi, I_fi = agent.calculate_voronoi_stats(my_pts, avg_conc)
                agent.m_Vi = m_Vi
                agent.dJ_dt = dot_bar_l_i * I_fi

                agent.move_towards(cx, cy, self.env)
                self.env.apply_additive_spray(agent.x, agent.y, agent.u_i, RA)

                # ★ 论文公式(13): J_i = ∫_{V_i} f(p_i,q) \bar{l}_i dq ≈ Σ f * \bar{l}_i * Dx*Dy
                J_i = I_fi * agent.bar_l_i
                J_instant += J_i

                # ★ 论文公式: ||e_i|| = sqrt(e_{i,x}^2 + e_{i,y}^2)
                e_norm = np.hypot(agent.e_ix, agent.e_iy)
                E_instant += e_norm
            else:
                agent.stop()

        # ★ 时间积分 (对应论文 Table 1)
        self.J_total_integral += J_instant * DT_MOVE
        self.E_total_integral += E_instant * DT_MOVE

        # ★ 保存瞬时值用于绘图
        self.J_instant_history.append(J_instant)
        self.E_instant_history.append(E_instant)

        M_total = float(np.sum(self.env.pollution_grid[self.env.region_mask]))
        self.M_history.append(M_total)

        # 打印最终积分值 (对应论文 Table 1)
        if frame % 50 == 0:
            print(f"Frame {frame}: J_total={self.J_total_integral:.3f}, E_total={self.E_total_integral:.3f}")

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

        for x1, y1, x2, y2 in CleaningAgent.compute_voronoi_boundaries(self.agents, self.env, step=5):
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
                ax2.add_patch(Circle((a.x, a.y), RA, fill=False, color=c, ls='--', alpha=0.5, lw=1))
            else:
                ax2.plot(a.x, a.y, 'x', color='gray', ms=8, mew=2, zorder=6)
        ax2.set_title(f'Agents & Voronoi (Dx=Dy=2m, Ra={RA})')

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

        fx = list(range(len(self.J_instant_history)))
        if fx:
            # ★ 绘制瞬时值 J(P,t) 和 Σ||e_i|| (对应论文 Fig.5)
            line1, = self.ax4.plot(fx, self.J_instant_history, color='steelblue', lw=1.5, label='J(P,t) instant')
            self.ax4.set_ylabel('J(P,t)', color='steelblue', fontsize=9)
            self.ax4.tick_params(axis='y', labelcolor='steelblue', labelsize=8)

            line2, = self.ax4_r.plot(fx, self.E_instant_history, color='orangered', lw=1.5, label='Σ||e_i|| instant')
            self.ax4_r.set_ylabel('Σ||e_i||', color='orangered', fontsize=9)
            self.ax4_r.tick_params(axis='y', labelcolor='orangered', labelsize=8)

            line3, = self.ax4_r2.plot(fx, self.M_history, color='forestgreen', lw=1.5, label='M_total (mass)')
            self.ax4_r2.set_ylabel('M_total', color='forestgreen', fontsize=9)
            self.ax4_r2.tick_params(axis='y', labelcolor='forestgreen', labelsize=8)

            lines = [line1, line2, line3]
            labels = [l.get_label() for l in lines]
            self.ax4.legend(lines, labels, loc='upper right', fontsize=8)

        self.ax4.set_title(
            f'Instant Metrics (Fig.5) | J_total={self.J_total_integral:.1f}, E_total={self.E_total_integral:.1f}',
            fontsize=10)
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

        print(f"\n=== 最终积分结果 (对应论文 Table 1) ===")
        print(f"J_total = {self.J_total_integral:.3f}")
        print(f"E_total = {self.E_total_integral:.3f}")

        return ani


if __name__ == "__main__":
    sim = MultiAgentSimulation(WIDTH, HEIGHT, AGENTS_NUM)
    ani = sim.run(frames=150, interval=60, save_gif=True, gif_filename='pollution_neutralization.gif')
