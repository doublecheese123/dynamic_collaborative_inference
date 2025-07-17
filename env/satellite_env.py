from collections import deque
import math
import random

SLOT = 1  # 时隙长度  单位：s
R = 6371e3  # 地球半径  单位：m
H = 784e3  # 低轨卫星的高度  单位：m
ELEVATION = 70  # 卫星初始位置和地心的夹角
LIGHT_SPEED = 3e8  # 光速  单位：m/s
ANGULAR_VL = 0.063  # 卫星的角速度  单位：角度/s
G_ST = 35  # 卫星的发射天线增益  单位：dBi
G_GR = 35  # 地面服务器的接收天线增益
FRE = 12e9  # 载波频率  GHz
K = 1.38e-23  # 玻尔兹曼常数
T = 300  # 噪声温度 单位：K
POWER = 20  # 卫星发射功率  单位：W
GR_F = 1e12  # 地面总计算资源
SAT_COMP_POWER = 20  # 卫星实际推理时动态功耗  单位：W
SAT_COMP1 = [1.23e8 * 100, 2.02e8 * 100, 4.9e8 * 100]  # 模型划分点不同导致每个任务的计算量不同  (计算量 * 任务图片数量）
SAT_OFF1 = [3211264 * 100, 1605632 * 100, 802816 * 100]  # 模型1不同划分点的卫星计算量、中间数据传输量、地面计算量
GR_COMP1 = [4.45e8 * 100, 3.66e8 * 100, 7.85e8 * 100]
SAT_COMP2 = [1.9e9 * 3, 4.58e9 * 3, 5.74e9 * 3]  # 模型2不同划分点的卫星计算量、中间数据传输量、地面计算量
SAT_OFF2 = [13107200 * 3, 6553600 * 3, 13107200 * 3]
GR_COMP2 = [6.4e9 * 3, 3.72e9 * 3, 2.56e9 * 3]
TASK_TYPES = {
    0: {"name": "ic", "max_delay": 15},
    1: {"name": "od", "max_delay": 5}
}


class Task:
    def __init__(self, task_type):
        self.type = task_type
        self.max_delay = TASK_TYPES[task_type]["max_delay"]  # 时延要求
        self.sat_comp = None  # 卫星需要完成的计算量
        self.sat_off = None  # 卫星需要传输的中间数据量
        self.gr_comp = None  # 地面需要完成的计算量
        self.time = 0  # 生成任务的时间，每个时隙加1，若超过最大时延要求，则从队列中丢弃


class StateNormalizer:
    def __init__(self):
        # 添加动态归一化统计量
        self.queue1_max = 1
        self.queue2_max = 1
        self.data1_max = 1
        self.data2_max = 1
        self.data3_max = 1
        self.data4_max = 1
        self.distance_max = 1e6

    def step(self, state):
        # 更新最大值
        self.queue1_max = max(self.queue1_max, state[0])
        self.queue2_max = max(self.queue2_max, state[1])
        self.data1_max = max(self.data1_max, state[2])
        self.data2_max = max(self.data2_max, state[3])
        self.data3_max = max(self.data3_max, state[4])
        self.data4_max = max(self.data4_max, state[5])
        self.distance_max = max(self.distance_max, state[6])

        # 归一化
        normalized_state = [
            state[0] / self.queue1_max,
            state[1] / self.queue2_max,
            state[2] / self.data1_max,
            state[3] / self.data2_max,
            state[4] / self.data3_max,
            state[5] / self.data4_max,
            state[6] / self.distance_max
        ]
        return normalized_state


class SatEnv:
    def __init__(self, bandwidth, sat_comp_capability, max_task1_num, max_task2_num, coefficient):
        self.state_normalizer = StateNormalizer()
        self.bandwidth = bandwidth
        self.sat_comp_capability = sat_comp_capability
        self.max_task1_num = max_task1_num
        self.max_task2_num = max_task2_num
        self.coefficient = coefficient
        self.sat_comp_q1 = deque()
        self.sat_comp_q2 = deque()
        self.sat_off_q1 = deque()
        self.sat_off_q2 = deque()
        self.ground_comp_q1 = deque()
        self.ground_comp_q2 = deque()
        self.angle = ELEVATION
        self.angle_in_radians = math.radians(ELEVATION)
        self.position_x = math.cos(self.angle_in_radians) * (R + H)
        self.position_y = math.sin(self.angle_in_radians) * (R + H)
        self.distance = math.sqrt(self.position_x ** 2 + (self.position_y - R) ** 2)
        self.drop_task_num = 0
        self.drop_task1 = 0
        self.drop_task2 = 0
        self.total_energy = 0
        self.total_task = 0
        self.total_task1 = 0
        self.total_task2 = 0
        self.suc_task1 = 0
        self.suc_task2 = 0
        self.total_latency1 = 0
        self.total_latency2 = 0

    def reset(self):
        self.drop_task_num = 0
        self.drop_task1 = 0
        self.drop_task2 = 0
        self.total_energy = 0
        self.total_task = 0
        self.total_task1 = 0
        self.total_task2 = 0
        self.suc_task1 = 0
        self.suc_task2 = 0
        self.total_latency1 = 0
        self.total_latency2 = 0
        self.sat_comp_q1.clear()
        self.sat_comp_q2.clear()
        self.sat_off_q1.clear()
        self.sat_off_q2.clear()
        self.ground_comp_q1.clear()
        self.ground_comp_q2.clear()
        self.angle = ELEVATION
        self.angle_in_radians = math.radians(ELEVATION)
        self.position_x = math.cos(self.angle_in_radians) * (R + H)
        self.position_y = math.sin(self.angle_in_radians) * (R + H)
        self.distance = math.sqrt(self.position_x ** 2 + (self.position_y - R) ** 2)
        obs = [0, 0, 0, 0, 0, 0, self.distance]
        normalized_obs = self.state_normalizer.step(obs)
        return normalized_obs

    def generate_tasks(self):
        """生成当前时隙任务"""
        random1 = random.randint(1, self.max_task1_num)
        random2 = random.randint(1, self.max_task2_num)
        tasks1 = []
        tasks2 = []
        for _ in range(random1):
            tasks1.append(Task(0))
        for _ in range(random2):
            tasks2.append(Task(1))
        return tasks1, tasks2, random1, random2

    def sat_move(self):
        self.angle += (ANGULAR_VL * SLOT)
        self.angle_in_radians = math.radians(self.angle)
        if self.angle < 90:
            self.position_x = math.cos(self.angle_in_radians) * (R + H)
            self.position_y = math.sin(self.angle_in_radians) * (R + H)
        else:
            self.position_x = - math.cos(self.angle_in_radians) * (R + H)
            self.position_y = math.sin(self.angle_in_radians) * (R + H)
        self.distance = math.sqrt(self.position_x ** 2 + (self.position_y - R) ** 2)

    def generate_new_tasks(self):
        # 放入队列
        task1, task2, num_task1, num_task2 = self.generate_tasks()
        self.total_task += (num_task1 + num_task2)
        self.total_task1 += num_task1
        self.total_task2 += num_task2
        self.sat_comp_q1.extend(task1)
        self.sat_comp_q2.extend(task2)

    def calculate_snr(self, p, b):
        power = 10 * math.log(p, 10)
        l_db = 20 * math.log((4 * math.pi * self.distance * FRE) / LIGHT_SPEED, 10)
        n_linear = K * T * b
        n_dbw = 10 * math.log(n_linear, 10)
        pr_dbw = power + G_ST + G_GR - l_db
        snr_db = pr_dbw - n_dbw
        snr = 10 ** (snr_db / 10)
        return snr

    def step(self, action):
        task_drop_num1 = 0
        task_drop_num2 = 0
        sat_comp_drop_q1 = 0
        sat_comp_drop_q2 = 0
        sat_off_drop_q1 = 0
        sat_off_drop_q2 = 0
        gr_comp_drop_q1 = 0
        gr_comp_drop_q2 = 0
        suc_task_num1 = 0
        suc_task_num2 = 0
        total_latency1 = 0
        total_latency2 = 0

        # 更新任务时间，超时丢弃
        new_comp_queue1 = deque()
        for task in self.sat_comp_q1:
            task.time += 1
            if task.time > task.max_delay:
                self.drop_task_num += 1
                task_drop_num1 += 1
                sat_comp_drop_q1 += 1
                total_latency1 += task.time
            else:
                new_comp_queue1.append(task)
        self.sat_comp_q1 = new_comp_queue1

        new_comp_queue2 = deque()
        for task in self.sat_comp_q2:
            task.time += 1
            if task.time > task.max_delay:
                self.drop_task_num += 1
                task_drop_num2 += 1
                sat_comp_drop_q2 += 1
                total_latency2 += task.time
            else:
                new_comp_queue2.append(task)
        self.sat_comp_q2 = new_comp_queue2

        new_off_queue1 = deque()
        for task in self.sat_off_q1:
            task.time += 1
            if task.time > task.max_delay:
                self.drop_task_num += 1
                task_drop_num1 += 1
                sat_off_drop_q1 += 1
                total_latency1 += task.time
            else:
                new_off_queue1.append(task)
        self.sat_off_q1 = new_off_queue1

        new_off_queue2 = deque()
        for task in self.sat_off_q2:
            task.time += 1
            if task.time > task.max_delay:
                self.drop_task_num += 1
                task_drop_num2 += 1
                sat_off_drop_q2 += 1
                total_latency2 += task.time
            else:
                new_off_queue2.append(task)
        self.sat_off_q2 = new_off_queue2

        new_gr_comp_queue1 = deque()
        for task in self.ground_comp_q1:
            task.time += 1
            if task.time > task.max_delay:
                self.drop_task_num += 1
                task_drop_num1 += 1
                gr_comp_drop_q1 += 1
                total_latency1 += task.time
            else:
                new_gr_comp_queue1.append(task)
        self.ground_comp_q1 = new_gr_comp_queue1

        new_gr_comp_queue2 = deque()
        for task in self.ground_comp_q2:
            task.time += 1
            if task.time > task.max_delay:
                self.drop_task_num += 1
                task_drop_num2 += 1
                gr_comp_drop_q2 += 1
                total_latency2 += task.time
            else:
                new_gr_comp_queue2.append(task)
        self.ground_comp_q2 = new_gr_comp_queue2

        self.drop_task1 += task_drop_num1
        self.drop_task2 += task_drop_num2

        #  卫星移动
        self.sat_move()
        # 生成新任务
        self.generate_new_tasks()

        #  本时隙卫星、地面服务器的资源分配
        sf1 = action["continuous_actions"][0] * self.sat_comp_capability
        sf2 = (1 - action["continuous_actions"][0]) * self.sat_comp_capability
        p1 = action["continuous_actions"][3] * POWER
        p2 = (1 - action["continuous_actions"][3]) * POWER  # 传输功率分配
        b1 = action["continuous_actions"][1] * self.bandwidth
        b2 = (1 - action["continuous_actions"][1]) * self.bandwidth
        if p1 == 0:
            r1 = 0
        else:
            snr1 = self.calculate_snr(p1, b1)
            r1 = math.log(1 + snr1, 2) * b1  # 计算传输数据的速率
        if p2 == 0:
            r2 = 0
        else:
            snr2 = self.calculate_snr(p2, b2)
            r2 = math.log(1 + snr2, 2) * b2
        gf1 = action["continuous_actions"][2] * GR_F
        gf2 = (1 - action["continuous_actions"][2]) * GR_F

        #  当选择第一个划分点划分模型1时
        if action["discrete_actions"][0] == 0:
            #  计算本时隙可以完成卫星计算的任务，从卫星计算队列中弹出加入卫星卸载队列
            cs1 = int((SLOT * sf1) // SAT_COMP1[0])
            task_comp_num1 = 0  # 计算卫星本时隙实际完成计算的第一种类型的任务个数，因为队列中可能没有那么多任务
            for _ in range(cs1):
                if len(self.sat_comp_q1) > 0:
                    task = self.sat_comp_q1.popleft()
                    task.sat_off = SAT_OFF1[0]
                    task.gr_comp = GR_COMP1[0]
                    task_comp_num1 += 1
                    self.sat_off_q1.append(task)
            if sf1 == 0:
                sat_comp_time1 = 0
            else:
                sat_comp_time1 = (task_comp_num1 * SAT_COMP1[0]) / sf1  # 如果有可以计算的任务，计算完成这些任务的时间
            #   计算本时隙可以完成卸载的任务，从卫星卸载队列中弹出加入地面队列
            data_off1 = r1 * SLOT  # 本时隙可以卸载的数据量
            actual_off1 = 0  # 本时隙实际传输的数据量
            while len(self.sat_off_q1) > 0:
                task = self.sat_off_q1[0]
                if data_off1 > task.sat_off:
                    data_off1 -= task.sat_off
                    actual_off1 += task.sat_off
                    self.ground_comp_q1.append(self.sat_off_q1.popleft())
                else:
                    break
            # 如果有可以卸载的任务，计算卸载这些任务的时间（因为对该类型任务分配的带宽可能在一个时隙内一个任务都无法卸载完成）
            if r1 == 0:
                sat_off_time1 = 0
            else:
                sat_off_time1 = actual_off1 / r1
            #   计算本时隙可以完成地面剩余计算的任务，从地面计算队列弹出，任务完成
            data_gr_comp1 = gf1 * SLOT  # 本时隙地面服务器可以完成计算的数据量
            while len(self.ground_comp_q1) > 0:
                task = self.ground_comp_q1[0]
                if data_gr_comp1 > task.gr_comp:
                    data_gr_comp1 -= task.gr_comp
                    self.ground_comp_q1.popleft()
                    suc_task_num1 += 1
                    total_latency1 += task.time
                else:
                    break

        #  当选择第二个划分点划分模型1时
        if action["discrete_actions"][0] == 1:
            #  计算本时隙可以完成卫星计算的任务，从卫星计算队列中弹出加入卫星卸载队列
            cs1 = int((SLOT * sf1) // SAT_COMP1[1])
            task_comp_num1 = 0
            for _ in range(cs1):
                if len(self.sat_comp_q1) > 0:
                    task = self.sat_comp_q1.popleft()
                    task.sat_off = SAT_OFF1[1]
                    task.gr_comp = GR_COMP1[1]
                    task_comp_num1 += 1
                    self.sat_off_q1.append(task)
            if sf1 == 0:
                sat_comp_time1 = 0
            else:
                sat_comp_time1 = (task_comp_num1 * SAT_COMP1[1]) / sf1  # 同上，计算任务处理时延，但此时每个任务的计算量需要改
            #   计算本时隙可以完成卸载的任务，从卫星卸载队列中弹出加入地面队列
            data_off1 = r1 * SLOT  # 本时隙可以卸载的数据量
            actual_off1 = 0  # 本时隙实际传输的数据量
            while len(self.sat_off_q1) > 0:
                task = self.sat_off_q1[0]
                if data_off1 > task.sat_off:
                    data_off1 -= task.sat_off
                    actual_off1 += task.sat_off
                    self.ground_comp_q1.append(self.sat_off_q1.popleft())
                else:
                    break
            if r1 == 0:
                sat_off_time1 = 0
            else:
                sat_off_time1 = actual_off1 / r1
            #   计算本时隙可以完成地面剩余计算的任务，从地面计算队列弹出，任务完成
            data_gr_comp1 = gf1 * SLOT  # 本时隙地面服务器可以完成计算的数据量
            while len(self.ground_comp_q1) > 0:
                task = self.ground_comp_q1[0]
                if data_gr_comp1 > task.gr_comp:
                    data_gr_comp1 -= task.gr_comp
                    self.ground_comp_q1.popleft()
                    suc_task_num1 += 1
                    total_latency1 += task.time
                else:
                    break

        #  当选择第三个划分点划分模型1时
        if action["discrete_actions"][0] == 2:
            #  计算本时隙可以完成卫星计算的任务，从卫星计算队列中弹出加入卫星卸载队列
            cs1 = int((SLOT * sf1) // SAT_COMP1[2])
            task_comp_num1 = 0
            for _ in range(cs1):
                if len(self.sat_comp_q1) > 0:
                    task = self.sat_comp_q1.popleft()
                    task.sat_off = SAT_OFF1[2]
                    task.gr_comp = GR_COMP1[2]
                    task_comp_num1 += 1
                    self.sat_off_q1.append(task)
            if sf1 == 0:
                sat_comp_time1 = 0
            else:
                sat_comp_time1 = (task_comp_num1 * SAT_COMP1[2]) / sf1
            #   计算本时隙可以完成卸载的任务，从卫星卸载队列中弹出加入地面队列
            data_off1 = r1 * SLOT  # 本时隙可以卸载的数据量
            actual_off1 = 0  # 本时隙实际传输的数据量
            while len(self.sat_off_q1) > 0:
                task = self.sat_off_q1[0]
                if data_off1 > task.sat_off:
                    data_off1 -= task.sat_off
                    actual_off1 += task.sat_off
                    self.ground_comp_q1.append(self.sat_off_q1.popleft())
                else:
                    break
            if r1 == 0:
                sat_off_time1 = 0
            else:
                sat_off_time1 = actual_off1 / r1
            #   计算本时隙可以完成地面剩余计算的任务，从地面计算队列弹出，任务完成
            data_gr_comp1 = gf1 * SLOT  # 本时隙地面服务器可以完成计算的数据量
            while len(self.ground_comp_q1) > 0:
                task = self.ground_comp_q1[0]
                if data_gr_comp1 > task.gr_comp:
                    data_gr_comp1 -= task.gr_comp
                    self.ground_comp_q1.popleft()
                    suc_task_num1 += 1
                    total_latency1 += task.time
                else:
                    break

        #  当选择第一个划分点划分模型2时
        if action["discrete_actions"][1] == 0:
            #  计算本时隙可以完成卫星计算的任务，从卫星计算队列中弹出加入卫星卸载队列
            cs2 = int((SLOT * sf2) // SAT_COMP2[0])
            task_comp_num2 = 0  # 计算卫星在本时隙实际完成的第二种任务的个数
            for _ in range(cs2):
                if len(self.sat_comp_q2) > 0:
                    task = self.sat_comp_q2.popleft()
                    task.sat_off = SAT_OFF2[0]
                    task.gr_comp = GR_COMP2[0]
                    task_comp_num2 += 1
                    self.sat_off_q2.append(task)
            if sf2 == 0:
                sat_comp_time2 = 0
            else:
                sat_comp_time2 = (task_comp_num2 * SAT_COMP2[0]) / sf2
            #   计算本时隙可以完成卸载的任务，从卫星卸载队列中弹出加入地面队列
            data_off2 = r2 * SLOT  # 本时隙可以卸载的数据量
            actual_off2 = 0  # 本时隙实际传输的数据量
            while len(self.sat_off_q2) > 0:
                task = self.sat_off_q2[0]
                if data_off2 > task.sat_off:
                    data_off2 -= task.sat_off
                    actual_off2 += task.sat_off
                    self.ground_comp_q2.append(self.sat_off_q2.popleft())
                else:
                    break
            if r2 == 0:
                sat_off_time2 = 0
            else:
                sat_off_time2 = actual_off2 / r2
            #   计算本时隙可以完成地面剩余计算的任务，从地面计算队列弹出，任务完成
            data_gr_comp2 = gf2 * SLOT  # 本时隙地面服务器可以完成计算的数据量
            while len(self.ground_comp_q2) > 0:
                task = self.ground_comp_q2[0]
                if data_gr_comp2 > task.gr_comp:
                    data_gr_comp2 -= task.gr_comp
                    self.ground_comp_q2.popleft()
                    suc_task_num2 += 1
                    total_latency2 += task.time
                else:
                    break

        #  当选择第二个划分点划分模型2时
        if action["discrete_actions"][1] == 1:
            #  计算本时隙可以完成卫星计算的任务，从卫星计算队列中弹出加入卫星卸载队列
            cs2 = int((SLOT * sf2) // SAT_COMP2[1])
            task_comp_num2 = 0
            for _ in range(cs2):
                if len(self.sat_comp_q2) > 0:
                    task = self.sat_comp_q2.popleft()
                    task.sat_off = SAT_OFF2[1]
                    task.gr_comp = GR_COMP2[1]
                    task_comp_num2 += 1
                    self.sat_off_q2.append(task)
            if sf2 == 0:
                sat_comp_time2 = 0
            else:
                sat_comp_time2 = (task_comp_num2 * SAT_COMP2[1]) / sf2
            #   计算本时隙可以完成卸载的任务，从卫星卸载队列中弹出加入地面队列
            data_off2 = r2 * SLOT  # 本时隙可以卸载的数据量
            actual_off2 = 0  # 本时隙实际传输的数据量
            while len(self.sat_off_q2) > 0:
                task = self.sat_off_q2[0]
                if data_off2 > task.sat_off:
                    data_off2 -= task.sat_off
                    actual_off2 += task.sat_off
                    self.ground_comp_q2.append(self.sat_off_q2.popleft())
                else:
                    break
            if r2 == 0:
                sat_off_time2 = 0
            else:
                sat_off_time2 = actual_off2 / r2
            #   计算本时隙可以完成地面剩余计算的任务，从地面计算队列弹出，任务完成
            data_gr_comp2 = gf2 * SLOT  # 本时隙地面服务器可以完成计算的数据量
            while len(self.ground_comp_q2) > 0:
                task = self.ground_comp_q2[0]
                if data_gr_comp2 > task.gr_comp:
                    data_gr_comp2 -= task.gr_comp
                    self.ground_comp_q2.popleft()
                    suc_task_num2 += 1
                    total_latency2 += task.time
                else:
                    break

        #  当选择第三个划分点划分模型2时
        if action["discrete_actions"][1] == 2:
            #  计算本时隙可以完成卫星计算的任务，从卫星计算队列中弹出加入卫星卸载队列
            cs2 = int((SLOT * sf2) // SAT_COMP2[2])
            task_comp_num2 = 0
            for _ in range(cs2):
                if len(self.sat_comp_q2) > 0:
                    task = self.sat_comp_q2.popleft()
                    task.sat_off = SAT_OFF2[2]
                    task.gr_comp = GR_COMP2[2]
                    task_comp_num2 += 1
                    self.sat_off_q2.append(task)
            if sf2 == 0:
                sat_comp_time2 = 0
            else:
                sat_comp_time2 = (task_comp_num2 * SAT_COMP2[2]) / sf2
            #   计算本时隙可以完成卸载的任务，从卫星卸载队列中弹出加入地面队列
            data_off2 = r2 * SLOT  # 本时隙可以卸载的数据量
            actual_off2 = 0  # 本时隙实际传输的数据量
            while len(self.sat_off_q2) > 0:
                task = self.sat_off_q2[0]
                if data_off2 > task.sat_off:
                    data_off2 -= task.sat_off
                    actual_off2 += task.sat_off
                    self.ground_comp_q2.append(self.sat_off_q2.popleft())
                else:
                    break
            if r2 == 0:
                sat_off_time2 = 0
            else:
                sat_off_time2 = actual_off2 / r2
            #   计算本时隙可以完成地面剩余计算的任务，从地面计算队列弹出，任务完成
            data_gr_comp2 = gf2 * SLOT  # 本时隙地面服务器可以完成计算的数据量
            while len(self.ground_comp_q2) > 0:
                task = self.ground_comp_q2[0]
                if data_gr_comp2 > task.gr_comp:
                    data_gr_comp2 -= task.gr_comp
                    self.ground_comp_q2.popleft()
                    suc_task_num2 += 1
                    total_latency2 += task.time
                else:
                    break

        sat_comp_energy = SAT_COMP_POWER * (sat_comp_time1 + sat_comp_time2)
        sat_off_energy = p1 * sat_off_time1 + p2 * sat_off_time2

        #  计算队长
        data_sat_off_q1 = 0
        data_sat_off_q2 = 0
        data_gr_comp_q1 = 0
        data_gr_comp_q2 = 0
        for t in self.sat_off_q1:
            data_sat_off_q1 += t.sat_off
        for t in self.sat_off_q2:
            data_sat_off_q2 += t.sat_off
        for t in self.ground_comp_q1:
            data_gr_comp_q1 += t.gr_comp
        for t in self.ground_comp_q2:
            data_gr_comp_q2 += t.gr_comp

        #  下一个状态，包括队长和信道条件
        raw_next_state = [len(self.sat_comp_q1), len(self.sat_comp_q2), data_sat_off_q1, data_sat_off_q2,
                          data_gr_comp_q1, data_gr_comp_q2, self.distance]
        normalized_state = self.state_normalizer.step(raw_next_state)

        total_energy = sat_comp_energy + sat_off_energy
        reward = 0.1 * (- self.coefficient * total_energy + suc_task_num1 + suc_task_num2)
        self.total_energy += total_energy
        self.suc_task1 += suc_task_num1
        self.suc_task2 += suc_task_num2
        self.total_latency1 += total_latency1
        self.total_latency2 += total_latency2

        done = False

        return normalized_state, reward, done
