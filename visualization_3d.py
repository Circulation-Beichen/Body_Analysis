import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import mediapipe as mp
import threading
import queue
import time
import json
import os

# 获取 Mediapipe Pose 的连接关系
mp_pose = mp.solutions.pose
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

# --- 新增：视角配置文件名 ---
VIEW_ANGLE_FILE = "view_angle.json"
# --- 新增：默认视角 --- (这个可以调整以获得最佳效果)
DEFAULT_ELEV = 10
DEFAULT_AZIM = -75 # 稍微倾斜的侧视图可能比 -90 更好

class VisualizationThread(threading.Thread):
    def __init__(self, init_elev=DEFAULT_ELEV, init_azim=DEFAULT_AZIM, pause_event=None):
        super().__init__(daemon=True) # 设置为守护线程，主线程退出时它也会退出
        self.queue = queue.Queue(maxsize=1) # 数据队列，只保留最新数据
        self.stop_event = threading.Event() # 停止信号
        self.pause_event = pause_event if pause_event else threading.Event() # 新增：暂停事件
        self.fig = None
        self.ax = None
        self.initialized = False
        self.init_elev = init_elev # 新增：存储初始仰角
        self.init_azim = init_azim # 新增：存储初始方位角

    def _init_plot(self):
        """初始化 Matplotlib 3D 绘图窗口 (在线程内执行)。"""
        if self.initialized:
            return
        print("[Vis Thread] 初始化 3D 可视化窗口...")
        try:
            # 在某些后端下，直接在非主线程创建 Figure 可能有问题
            # 但通常 plt.figure() 是可以的
            self.fig = plt.figure(figsize=(8, 6))
            self.ax = self.fig.add_subplot(111, projection='3d')
            self._setup_axes()
            plt.show(block=False)
            self.initialized = True
            print("[Vis Thread] 3D 可视化窗口已初始化。")
        except Exception as e:
            print(f"[Vis Thread] 初始化 Matplotlib 失败: {e}")
            self.initialized = False # 标记失败

    def _setup_axes(self):
        """设置坐标轴属性。"""
        if not self.ax:
             return
        # 调整坐标轴范围以适应 Mediapipe 世界坐标 (米)
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_zlim(-1.5, 1.5) # Z轴范围也可能是负值
        # 修改坐标轴标签单位
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Real-time 3D Pose Visualization (World Coords)')
        # Mediapipe 世界坐标 Y 轴向下，需要反转 Y 轴
        self.ax.invert_yaxis()
        # Mediapipe 世界坐标 Z 轴指向屏幕内，通常不需要反转 Z 轴
        # self.ax.invert_zaxis()
        # --- 修改: 设置新的视图角度，尝试面向 XZ 平面 ---
        self.ax.view_init(elev=self.init_elev, azim=0) # 仰角不变，方位角改为 0

    def _update_plot(self, points_3d):
        """
        更新 3D 图中的骨架 (在线程内执行)。
        Args:
            points_3d (dict): 3D 坐标字典。
        """
        if not self.initialized or not self.fig or not plt.fignum_exists(self.fig.number):
            # print("[Vis Thread] 警告: 绘图窗口未初始化或已关闭，尝试重新初始化。")
            # self._init_plot() # 尝试重新初始化可能导致问题，暂时禁用
            # if not self.initialized: # 如果重新初始化仍失败则返回
            #     return
            return # 如果窗口不在，则不更新

        if not points_3d:
            # 如果没有点，仍然需要清除旧图并保持窗口响应
            try:
                if plt.fignum_exists(self.fig.number): # 检查窗口是否存在
                    self.ax.cla() # 只清除绘图内容
                    # 注意：标签和标题可能也会被清除，这是此方法的副作用
                    # 但优先保证视角不被重置
                    # self._setup_axes() # 不再调用
                    plt.pause(0.05)
            except Exception as e:
                 print(f"[Vis Thread] 清除无数据图时出错: {e}")
                 self.initialized = False
                 self.stop_event.set()
            return

        try:
            # --- 修改：清除数据，重新设置范围/标签/标题，但不重置视角 --- 
            self.ax.cla() # 清除绘图数据
            
            # 重新设置坐标轴范围、标签和标题 (因为 cla() 会清除)
            self.ax.set_xlim(-1.5, 1.5)
            self.ax.set_ylim(-1.5, 1.5)
            self.ax.set_zlim(-1.5, 1.5)
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_zlabel('Z (m)')
            self.ax.set_title('Real-time 3D Pose Visualization (World Coords)')
            # 注意：invert_yaxis() 和 view_init() 不在此处调用
            # invert_yaxis() 在 _setup_axes 中调用一次即可
            # view_init() 由用户交互或初始设置决定

            valid_indices = list(points_3d.keys())
            xs = [points_3d[lm][0] for lm in valid_indices]
            ys = [points_3d[lm][1] for lm in valid_indices]
            zs = [points_3d[lm][2] for lm in valid_indices]

            self.ax.scatter(xs, ys, zs, c='r', marker='o', label='Joints')

            if POSE_CONNECTIONS:
                for connection in POSE_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    if start_idx in valid_indices and end_idx in valid_indices:
                        start_point = points_3d[start_idx]
                        end_point = points_3d[end_idx]
                        self.ax.plot([start_point[0], end_point[0]],
                                     [start_point[1], end_point[1]],
                                     [start_point[2], end_point[2]], 'k-')

            # 使用 pause 处理 GUI 事件并允许绘图更新
            plt.pause(0.01) # 暂停时间可以适当调整

        except Exception as e:
            print(f"[Vis Thread] 更新 3D 图时出错: {e}")
            # 可能窗口已关闭
            self.initialized = False # 标记为未初始化
            self.stop_event.set() # 如果绘图出错，也通知停止

    def run(self):
        """线程的主执行逻辑。"""
        self._init_plot()
        while not self.stop_event.is_set():
            try:
                # --- 新增: 检查暂停状态 ---
                if self.pause_event.is_set():
                    # 如果暂停，只处理窗口事件以保持交互性
                    if self.initialized and self.fig and plt.fignum_exists(self.fig.number):
                        try:
                            plt.pause(0.1) # 短暂暂停处理事件
                        except Exception:
                            self.initialized = False
                            self.stop_event.set()
                    else:
                        time.sleep(0.1) # 窗口不存在则等待
                    continue # 跳过后续的数据处理和绘图更新
                # --- 暂停检查结束 ---

                # 尝试从队列获取数据，设置超时以避免阻塞并允许检查 stop_event
                points_3d = self.queue.get(timeout=0.1)
                if points_3d is None: # 收到停止信号
                    break
                self._update_plot(points_3d)
            except queue.Empty:
                # 队列为空，表示主线程没有放入新数据
                # 如果没有暂停，为了保持窗口响应也要调用pause
                if not self.pause_event.is_set() and self.initialized and self.fig and plt.fignum_exists(self.fig.number):
                    try:
                        plt.pause(0.05) # 稍长暂停，避免CPU空转
                    except Exception:
                        self.initialized = False
                        self.stop_event.set()
                else:
                    # 如果窗口未初始化或已关闭，可以稍微等待一下
                    time.sleep(0.1)
            except Exception as e:
                 print(f"[Vis Thread] 运行循环出错: {e}")
                 self.stop_event.set() # 出现意外错误时停止线程

        self._close_plot()
        print("[Vis Thread] 可视化线程已停止。")

    def update_data(self, points_3d):
        """主线程调用此方法将新数据放入队列。"""
        if not self.stop_event.is_set():
            try:
                # 清空队列以丢弃旧数据，然后放入新数据
                # 使用 get_nowait 避免阻塞
                try:
                    while True:
                        self.queue.get_nowait()
                except queue.Empty:
                    pass # 队列已空
                self.queue.put_nowait(points_3d) # 放入最新数据
            except queue.Full:
                # 理论上 maxsize=1 时，清空后 put 不会满，但以防万一
                pass
            except Exception as e:
                 print(f"[Vis Thread] 更新数据到队列时出错: {e}")

    def stop(self):
        """主线程调用此方法来停止可视化线程。"""
        print("[Vis Thread] 收到停止信号...")
        self.stop_event.set()
        # 尝试向队列放入 None 信号以唤醒可能阻塞的 get
        try:
             self.queue.put_nowait(None)
        except queue.Full:
             # 如果队列满了（理论上不应该），可能需要先清空
             try:
                 self.queue.get_nowait()
             except queue.Empty:
                 pass # 队列本来就是空的
             try:
                 self.queue.put_nowait(None) # 再次尝试放入 None
             except queue.Full:
                 print("[Vis Thread] Warning: Queue still full after trying to clear. Stop signal might be delayed.")
                 pass # 如果还是满，暂时没办法
        except Exception as e:
            print(f"[Vis Thread] 发送停止信号到队列时出错: {e}")

    def get_current_view_and_save(self):
        """获取当前视角并保存到文件。"""
        print("[Vis Thread] get_current_view_and_save 开始执行。") # 调试打印
        if self.initialized and self.fig and self.ax and plt.fignum_exists(self.fig.number):
            print("[Vis Thread] 绘图窗口状态检查通过。") # 调试打印
            try:
                current_elev = self.ax.elev
                current_azim = self.ax.azim
                print(f"[Vis Thread] 获取到当前视角: elev={current_elev}, azim={current_azim}") # 调试打印
                print(f"[Vis Thread] 准备调用 save_view_angle...") # 调试打印
                save_view_angle(current_elev, current_azim)
                # 注意：save_view_angle 内部已有打印
                # 更新线程内部的初始视角，以便下次 setup_axes 能用上
                self.init_elev = current_elev
                self.init_azim = current_azim
                print("[Vis Thread] get_current_view_and_save 成功返回 True。") # 调试打印
                return True
            except Exception as e:
                print(f"[Vis Thread] 获取或保存视角时发生异常: {e}") # 调试打印
                return False
        else:
            # 详细打印检查失败的原因
            fig_exists = plt.fignum_exists(self.fig.number) if self.fig else False
            print(f"[Vis Thread] 绘图窗口状态检查失败: Initialized={self.initialized}, Fig exists={self.fig is not None}, Ax exists={self.ax is not None}, Fig Num Exists={fig_exists}") # 调试打印
            return False

    def _close_plot(self):
        """关闭 Matplotlib 窗口 (在线程内执行)。不再在此处保存视角。"""
        if self.initialized and self.fig and plt.fignum_exists(self.fig.number):
            try:
                plt.close(self.fig)
                print("[Vis Thread] Matplotlib 窗口已关闭。")
            except Exception as e:
                 print(f"[Vis Thread] 关闭 Matplotlib 窗口时出错: {e}")
        self.initialized = False

# --- 新增：加载视角函数 ---
def load_view_angle():
    """尝试从文件加载视角角度。如果文件不存在或无效，返回 None。"""
    if os.path.exists(VIEW_ANGLE_FILE):
        try:
            with open(VIEW_ANGLE_FILE, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'elev' in data and 'azim' in data:
                    print(f"从 {VIEW_ANGLE_FILE} 加载视角: elev={data['elev']}, azim={data['azim']}")
                    return data['elev'], data['azim']
                else:
                    print(f"警告: {VIEW_ANGLE_FILE} 文件格式无效。")
                    return None
        except Exception as e:
            print(f"加载视角文件 {VIEW_ANGLE_FILE} 时出错: {e}")
            return None
    else:
        return None

# --- 新增：保存视角函数 ---
def save_view_angle(elev, azim):
    """将视角角度保存到文件。"""
    print(f"[save_view_angle] 尝试保存 elev={elev}, azim={azim} 到 {VIEW_ANGLE_FILE}") # 调试打印
    try:
        data = {'elev': elev, 'azim': azim}
        with open(VIEW_ANGLE_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"[save_view_angle] 成功保存到 {VIEW_ANGLE_FILE}") # 调试打印
    except Exception as e:
        print(f"[save_view_angle] 保存文件时出错: {e}") # 调试打印

# --- 以下函数不再直接使用，逻辑移入类中 ---
# def init_plot(): ...
# def update_plot(points_3d): ...
# def close_plot(): ... 