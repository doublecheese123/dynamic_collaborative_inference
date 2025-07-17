from env.satellite_env import SatEnv
from agent.hy_agent import HybridAgent
from util.replaybuffer import HybridReplayBuffer
import csv
import os

EPISODES = 800
TRAJECTORY = 600
BATCH_SIZE = 128
EPOCH = 4
STATE_DIM = 7


def train_with_bandwidth(bandwidth_value, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    file_path_reward = os.path.join(output_dir, "rewards.csv")
    file_path_drop = os.path.join(output_dir, "drop.csv")
    file_path_energy = os.path.join(output_dir, "energy.csv")
    file_path_drop_rate = os.path.join(output_dir, "drop_rate.csv")
    file_path_energy_rate = os.path.join(output_dir, "energy_rate.csv")
    file_path_latency = os.path.join(output_dir, "latency.csv")
    file_path_latency1 = os.path.join(output_dir, "latency1.csv")
    file_path_latency2 = os.path.join(output_dir, "latency2.csv")

    def create_csv(file_path, header):
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    create_csv(file_path_reward, ['Episode', 'Reward'])
    create_csv(file_path_drop, ['Episode', 'Task_drop'])
    create_csv(file_path_energy, ['Episode', 'Total_energy'])
    create_csv(file_path_drop_rate, ['Episode', 'Drop_rate'])
    create_csv(file_path_energy_rate, ['Episode', 'Energy_rate'])
    create_csv(file_path_latency, ['Episode', 'Latency'])
    create_csv(file_path_latency1, ['Episode', 'Latency1'])
    create_csv(file_path_latency2, ['Episode', 'Latency2'])

    env = SatEnv(bandwidth=bandwidth_value, sat_comp_capability=2e11,
                 max_task1_num=5, max_task2_num=12, coefficient=0.001)
    replay_buffer = HybridReplayBuffer()
    ha = HybridAgent(STATE_DIM)

    for episode in range(EPISODES):

        state = env.reset()
        total_reward = 0

        for step in range(TRAJECTORY):
            actions = ha.select_action(state)
            next_state, reward, done = env.step(actions)
            replay_buffer.add(state, actions["discrete_actions"], actions["continuous_actions"],
                              actions["log_probs"], reward, next_state, done)
            state = next_state
            total_reward += reward
            if replay_buffer.__len__() > 256:
                for _ in range(EPOCH):
                    ha.update(replay_buffer.sample(BATCH_SIZE))
                replay_buffer.clear()

        def append_to_csv(file_path, data):
            with open(file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)

        append_to_csv(file_path_reward, [episode, total_reward])
        append_to_csv(file_path_drop, [episode, env.drop_task_num])
        append_to_csv(file_path_energy, [episode, env.total_energy])
        append_to_csv(file_path_drop_rate, [episode, env.drop_task_num / env.total_task])
        append_to_csv(file_path_energy_rate, [episode, (env.total_task - env.drop_task_num) / env.total_energy])
        append_to_csv(file_path_latency, [episode, (env.total_latency1 + env.total_latency2) / (env.suc_task1 + env.drop_task1 + env.suc_task2 + env.drop_task2)])
        append_to_csv(file_path_latency1, [episode, env.total_latency1 / (env.suc_task1 + env.drop_task1)])
        append_to_csv(file_path_latency2, [episode, env.total_latency2 / (env.suc_task2 + env.drop_task2)])


def train_with_capability(capability_value, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    file_path_reward = os.path.join(output_dir, "rewards.csv")
    file_path_drop = os.path.join(output_dir, "drop.csv")
    file_path_energy = os.path.join(output_dir, "energy.csv")
    file_path_drop_rate = os.path.join(output_dir, "drop_rate.csv")
    file_path_energy_rate = os.path.join(output_dir, "energy_rate.csv")
    file_path_latency = os.path.join(output_dir, "latency.csv")
    file_path_latency1 = os.path.join(output_dir, "latency1.csv")
    file_path_latency2 = os.path.join(output_dir, "latency2.csv")

    def create_csv(file_path, header):
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    create_csv(file_path_reward, ['Episode', 'Reward'])
    create_csv(file_path_drop, ['Episode', 'Task_drop'])
    create_csv(file_path_energy, ['Episode', 'Total_energy'])
    create_csv(file_path_drop_rate, ['Episode', 'Drop_rate'])
    create_csv(file_path_energy_rate, ['Episode', 'Energy_rate'])
    create_csv(file_path_latency, ['Episode', 'Latency'])
    create_csv(file_path_latency1, ['Episode', 'Latency1'])
    create_csv(file_path_latency2, ['Episode', 'Latency2'])

    env = SatEnv(bandwidth=100e6, sat_comp_capability=capability_value,
                 max_task1_num=5, max_task2_num=12, coefficient=0.001)
    replay_buffer = HybridReplayBuffer()
    ha = HybridAgent(STATE_DIM)

    for episode in range(EPISODES):

        state = env.reset()
        total_reward = 0

        for step in range(TRAJECTORY):
            actions = ha.select_action(state)
            next_state, reward, done = env.step(actions)
            replay_buffer.add(state, actions["discrete_actions"], actions["continuous_actions"],
                              actions["log_probs"], reward, next_state, done)
            state = next_state
            total_reward += reward
            if replay_buffer.__len__() > 256:
                for _ in range(EPOCH):
                    ha.update(replay_buffer.sample(BATCH_SIZE))
                replay_buffer.clear()

        def append_to_csv(file_path, data):
            with open(file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)

        append_to_csv(file_path_reward, [episode, total_reward])
        append_to_csv(file_path_drop, [episode, env.drop_task_num])
        append_to_csv(file_path_energy, [episode, env.total_energy])
        append_to_csv(file_path_drop_rate, [episode, env.drop_task_num / env.total_task])
        append_to_csv(file_path_energy_rate, [episode, (env.total_task - env.drop_task_num) / env.total_energy])
        append_to_csv(file_path_latency, [episode, (env.total_latency1 + env.total_latency2) / (env.suc_task1 + env.drop_task1 + env.suc_task2 + env.drop_task2)])
        append_to_csv(file_path_latency1, [episode, env.total_latency1 / (env.suc_task1 + env.drop_task1)])
        append_to_csv(file_path_latency2, [episode, env.total_latency2 / (env.suc_task2 + env.drop_task2)])


def train_with_task(task_value, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    file_path_reward = os.path.join(output_dir, "rewards.csv")
    file_path_drop = os.path.join(output_dir, "drop.csv")
    file_path_energy = os.path.join(output_dir, "energy.csv")
    file_path_drop_rate = os.path.join(output_dir, "drop_rate.csv")
    file_path_energy_rate = os.path.join(output_dir, "energy_rate.csv")
    file_path_latency = os.path.join(output_dir, "latency.csv")
    file_path_latency1 = os.path.join(output_dir, "latency1.csv")
    file_path_latency2 = os.path.join(output_dir, "latency2.csv")

    def create_csv(file_path, header):
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    create_csv(file_path_reward, ['Episode', 'Reward'])
    create_csv(file_path_drop, ['Episode', 'Task_drop'])
    create_csv(file_path_energy, ['Episode', 'Total_energy'])
    create_csv(file_path_drop_rate, ['Episode', 'Drop_rate'])
    create_csv(file_path_energy_rate, ['Episode', 'Energy_rate'])
    create_csv(file_path_latency, ['Episode', 'Latency'])
    create_csv(file_path_latency1, ['Episode', 'Latency1'])
    create_csv(file_path_latency2, ['Episode', 'Latency2'])

    env = SatEnv(bandwidth=100e6, sat_comp_capability=2e11,
                 max_task1_num=task_value[0], max_task2_num=task_value[1], coefficient=0.0001)
    replay_buffer = HybridReplayBuffer()
    ha = HybridAgent(STATE_DIM)

    for episode in range(EPISODES):

        state = env.reset()
        total_reward = 0

        for step in range(TRAJECTORY):
            actions = ha.select_action(state)
            next_state, reward, done = env.step(actions)
            replay_buffer.add(state, actions["discrete_actions"], actions["continuous_actions"],
                              actions["log_probs"], reward, next_state, done)
            state = next_state
            total_reward += reward
            if replay_buffer.__len__() > 256:
                for _ in range(EPOCH):
                    ha.update(replay_buffer.sample(BATCH_SIZE))
                replay_buffer.clear()

        def append_to_csv(file_path, data):
            with open(file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)

        append_to_csv(file_path_reward, [episode, total_reward])
        append_to_csv(file_path_drop, [episode, env.drop_task_num])
        append_to_csv(file_path_energy, [episode, env.total_energy])
        append_to_csv(file_path_drop_rate, [episode, env.drop_task_num / env.total_task])
        append_to_csv(file_path_energy_rate, [episode, (env.total_task - env.drop_task_num) / env.total_energy])
        append_to_csv(file_path_latency, [episode, (env.total_latency1 + env.total_latency2) / (
                    env.suc_task1 + env.drop_task1 + env.suc_task2 + env.drop_task2)])
        append_to_csv(file_path_latency1, [episode, env.total_latency1 / (env.suc_task1 + env.drop_task1)])
        append_to_csv(file_path_latency2, [episode, env.total_latency2 / (env.suc_task2 + env.drop_task2)])


def train_with_coefficient(coefficient_value, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    file_path_reward = os.path.join(output_dir, "rewards.csv")
    file_path_drop = os.path.join(output_dir, "drop.csv")
    file_path_energy = os.path.join(output_dir, "energy.csv")
    file_path_drop_rate = os.path.join(output_dir, "drop_rate.csv")
    file_path_energy_rate = os.path.join(output_dir, "energy_rate.csv")
    file_path_latency = os.path.join(output_dir, "latency.csv")
    file_path_latency1 = os.path.join(output_dir, "latency1.csv")
    file_path_latency2 = os.path.join(output_dir, "latency2.csv")

    def create_csv(file_path, header):
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    create_csv(file_path_reward, ['Episode', 'Reward'])
    create_csv(file_path_drop, ['Episode', 'Task_drop'])
    create_csv(file_path_energy, ['Episode', 'Total_energy'])
    create_csv(file_path_drop_rate, ['Episode', 'Drop_rate'])
    create_csv(file_path_energy_rate, ['Episode', 'Energy_rate'])
    create_csv(file_path_latency, ['Episode', 'Latency'])
    create_csv(file_path_latency1, ['Episode', 'Latency1'])
    create_csv(file_path_latency2, ['Episode', 'Latency2'])

    env = SatEnv(bandwidth=100e6, sat_comp_capability=2e11,
                 max_task1_num=7, max_task2_num=14, coefficient=coefficient_value)
    replay_buffer = HybridReplayBuffer()
    ha = HybridAgent(STATE_DIM)

    for episode in range(EPISODES):

        state = env.reset()
        total_reward = 0

        for step in range(TRAJECTORY):
            actions = ha.select_action(state)
            next_state, reward, done = env.step(actions)
            replay_buffer.add(state, actions["discrete_actions"], actions["continuous_actions"],
                              actions["log_probs"], reward, next_state, done)
            state = next_state
            total_reward += reward
            if replay_buffer.__len__() > 256:
                for _ in range(EPOCH):
                    ha.update(replay_buffer.sample(BATCH_SIZE))
                replay_buffer.clear()

        def append_to_csv(file_path, data):
            with open(file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)

        append_to_csv(file_path_reward, [episode, total_reward])
        append_to_csv(file_path_drop, [episode, env.drop_task_num])
        append_to_csv(file_path_energy, [episode, env.total_energy])
        append_to_csv(file_path_drop_rate, [episode, env.drop_task_num / env.total_task])
        append_to_csv(file_path_energy_rate, [episode, (env.total_task - env.drop_task_num) / env.total_energy])
        append_to_csv(file_path_latency, [episode, (env.total_latency1 + env.total_latency2) / (
                env.suc_task1 + env.drop_task1 + env.suc_task2 + env.drop_task2)])
        append_to_csv(file_path_latency1, [episode, env.total_latency1 / (env.suc_task1 + env.drop_task1)])
        append_to_csv(file_path_latency2, [episode, env.total_latency2 / (env.suc_task2 + env.drop_task2)])


if __name__ == '__main__':
    bandwidth_settings = [50e6, 100e6, 150e6, 200e6, 250e6, 300e6]
    capability_settings = [1e11, 1.5e11, 2e11, 2.5e11, 3e11, 3.5e11]
    task_settings = [[3, 10], [4, 11], [5, 12], [6, 13], [7, 14]]
    coefficient_settings = [0.001, 0.05, 0.1]

    for bandwidth in bandwidth_settings:
        output_dir = f"D:/ppo/bandwidth_{bandwidth}"
        print(f"\ntrain with bandwidth {bandwidth} ...")
        results = train_with_bandwidth(bandwidth, output_dir)

    for capability in capability_settings:
        output_dir = f"D:/ppo/capability_{capability}"
        print(f"\ntrain with capability {capability} ...")
        results = train_with_capability(capability, output_dir)

    for task in task_settings:
        output_dir = f"D:/ppo/task_arrival_{task[0]}"
        print(f"\ntrain with task_arrival {task[0]} ...")
        results = train_with_task(task, output_dir)

    for coefficient in coefficient_settings:
        output_dir = f"D:/ppo/coefficient_{coefficient}"
        print(f"\ntrain with coefficient {coefficient} ...")
        results = train_with_coefficient(coefficient, output_dir)
