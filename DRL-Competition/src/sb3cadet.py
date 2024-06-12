import yaml
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from agents.RiskyValley import RiskyValley
import argparse
from agents.GolKenari import GolKenari
from agents.CustomAgent import CustomAgent
from CustomPolicy import CustomPolicyWithAttention
from stable_baselines3 import A2C, PPO
import datetime
def read_hypers():
    with open(f"./src/hyper.yaml", "r") as f:
        hyperparams_dict = yaml.safe_load(f)
        return hyperparams_dict["agentsofglory"]


# parser = argparse.ArgumentParser(description='Cadet Agents')
# parser.add_argument('map', metavar='map', type=str,
#                     help='Select Map to Train')
# parser.add_argument('--mode', metavar='mode', type=str, default="Train",
#                     help='Select Mode[Train,Sim]')
# parser.add_argument('--agentBlue', metavar='agentBlue', type=str, default="RayEnv",
#                     help='Class name of Blue Agent')
# parser.add_argument('--agentRed', metavar='agentRed', type=str,
#                     help='Class name of Red Agent')
# parser.add_argument('--numOfMatch', metavar='numOfMatch', type=int, nargs='?', default=10,
#                     help='Number of matches to play between agents')
# parser.add_argument('--render', action='store_true',
#                     help='Render the game')
# parser.add_argument('--gif', action='store_true',
#                     help='Create a gif of the game, also sets render')
# parser.add_argument('--img', action='store_true',
#                     help='Save images of each turn, also sets render')

# args = parser.parse_args()

# Initialize args to be able to define in code instead of command line
args = argparse.Namespace()

# Prepare args manually
# args.map = "GolKenariVadisi"
# args.map = "RiskyValley"
args.map = "RiskyValley"
# args.map = "TrainDuoTruckSmall"
# args.mode = "Train"
args.agentBlue = "CustomAgent"
args.agentRed = "RandomAgent"
args.numOfMatch = 1
args.render = False
args.gif = False
args.img = False
args.mode = "Train"
agents = [None, args.agentRed]

# Get current time as day hour and minute
currrent_time = datetime.datetime.now().strftime("%d-%H-%M")
common_part = f"{currrent_time}-MAP-{args.map}"
# common_part = 
_dir = f"logs/{common_part}"
_dir_model = f"models/{common_part}"

class LoggerCallback(BaseCallback):

    def __init__(self, _format, log_on_start=None, suffix=""):
        super().__init__()
        self._format = _format
        self.suffix = suffix
        if log_on_start is not None and not isinstance(log_on_start, (list, tuple)):
            log_on_start = tuple(log_on_start)
        self.log_on_start = log_on_start

    def _on_training_start(self) -> None:

        _logger = self.globals["logger"].Logger.CURRENT
        
        log_format = logger.make_output_format(self._format, _dir, self.suffix)
        _logger.output_formats.append(log_format)
        if self.log_on_start is not None:
            for pair in self.log_on_start:
                _logger.record(*pair, ("tensorboard", "stdout"))

    def _on_step(self) -> bool:
        """
        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True


if __name__ == "__main__":

    hyperparams = read_hypers()

    for agentsofglory in hyperparams:
        gamename, hyperparam = list(agentsofglory.items())[0]

        loggcallback = LoggerCallback(
            "json",
            [("hypers", hyperparam)]
        )

        env = SubprocVecEnv([lambda: CustomAgent(args, agents) for i in range(hyperparam["env"]["n_envs"])])
        checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=_dir_model, name_prefix='tsts')
        
        # model = PPO('MlpPolicy', env, verbose=1)
        model = PPO('MlpPolicy', env, policy_kwargs={'features_extractor_class': CustomPolicyWithAttention}, verbose=1, tensorboard_log="logs", **hyperparam["agent"])
        if args.mode == "Train":
            model.learn(total_timesteps=30000,callback=[checkpoint_callback, loggcallback], **hyperparam["learn"])
        elif args.mode == "Sim":
            model = PPO.load("./models/YOUR-MODEL-NAME/tsts_100000_steps")
            obs = env.reset()
            for i in range(10000):  # Run for a fixed number of steps or until done
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, dones, info = env.step(action)
        ## A2C
        # model = A2C(env=env,
        #             verbose=1,
        #             tensorboard_log="logs",
        #             **hyperparam["agent"])