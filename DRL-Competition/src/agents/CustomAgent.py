from os import kill
from agents.BaseLearningGym import BaseLearningAgentGym
import gym
from gym import spaces
import numpy as np
import yaml
from game import Game
from utilities import multi_forced_anchor, necessary_obs, decode_location, multi_reward_shape, enemy_locs, ally_locs, getDistance, truck_locs, get_best_direction, a_star_pathfinding,bfs_pathfinding ,custom_state_representation, get_best_direction_bfs

class CustomAgent(BaseLearningAgentGym):

    tagToString = {
            1: "Truck",
            2: "LightTank",
            3: "HeavyTank",
            4: "Drone",
        }

    def __init__(self, args, agents):
        super().__init__() 
        print(args, agents, "args")
        self.game = Game(args, agents)
        self.team = 0
        self.enemy_team = 1
        
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.nec_obs = None
        # print("x", self.game.map_x)
        # print("y", self.game.map_y)
        # print(self.game.map_x* self.game.map_y*10+8)
        val = self.game.map_x* self.game.map_y * 11+4
        # val = 4756
        self.observation_space = spaces.Box(
            low=-2,
            high=401,
            shape=(val,),
            dtype=np.int16
        )
        self.action_space = spaces.MultiDiscrete([7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5])
        self.previous_enemy_count = 4
        self.previous_ally_count = 4

    def setup(self, obs_spec, action_spec):
        self.observation_space = obs_spec
        self.action_space = action_spec
        print("setup")

    def reset(self):
        self.previous_enemy_count = 4
        self.previous_ally_count = 4
        self.episodes += 1
        self.steps = 0
        state = self.game.reset()
        self.nec_obs = state
        return self.decode_state(state)

    @staticmethod
    def _decode_state(obs, team, enemy_team):
        turn = obs['turn'] # 1
        max_turn = obs['max_turn'] # 1
        units = obs['units'] 
        hps = obs['hps'] 
        bases = obs['bases'] 
        score = obs['score'] # 2
        res = obs['resources'] 
        load = obs['loads']
        terrain = obs["terrain"] 
        y_max, x_max = res.shape
        my_units = []
        enemy_units = []
        resources = []
        resource_loc = np.argwhere(res == 1)
        best_dir = np.zeros_like(res)
        
        for i in range(y_max):
            for j in range(x_max):
                if units[team][i][j]<6 and units[team][i][j] != 0:
                    my_units.append(
                    {   
                        'unit': units[team][i][j],
                        'tag': CustomAgent.tagToString[units[team][i][j]],
                        'hp': hps[team][i][j],
                        'location': (i,j),
                        'load': load[team][i][j]
                    }
                    )
                    
                    if CustomAgent.tagToString[units[team][i][j]] == "Truck":
                        base_loc = np.argwhere(bases[team] == 1).squeeze()    
                        # If truck is loaded, find the closest base
                        if load[team][i][j] >= 2:
                            # Find our closest base
                            # Find best path to base by using bfs
                            # path = bfs_pathfinding([i, j], base_loc, terrain)
                            # Find the best direction to move
                            best_dir[i, j] = get_best_direction_bfs([i,j], base_loc,terrain)
                        
                        elif load[team][i][j] == 1:
                            resource_distances = sorted(resource_loc, key=lambda res_: getDistance([i,j], res_))[:2]
                            closest_resource = min(resource_distances, key=lambda res_: len(bfs_pathfinding([i,j], res_, obs['terrain'])))
                            distance_to_base = len(bfs_pathfinding([i,j], base_loc, obs['terrain']))
                            distance_to_resource = len(bfs_pathfinding([i,j], closest_resource, obs['terrain']))
                            # If the distance to the base is less than the distance to the resource, move to the base
                            if distance_to_base < distance_to_resource:
                                best_dir[i, j] = get_best_direction_bfs([i,j], base_loc, obs['terrain'])
                            else:
                                best_dir[i, j] = get_best_direction_bfs([i,j], closest_resource, obs['terrain'])
                        elif load[team][i][j] == 0:
                            resource_distances = sorted(resource_loc, key=lambda res_: getDistance([i,j], res_))[:5]
                            closest_resource = min(resource_distances, key=lambda res_: len(bfs_pathfinding([i,j], res_, obs['terrain'])))
                            best_dir[i, j] = get_best_direction_bfs([i,j], closest_resource, obs['terrain'])
                        
                        # print("Best direction for truck at", i, j, "is", best_dir[i, j])
                            
                if units[enemy_team][i][j]<6 and units[enemy_team][i][j] != 0:
                    enemy_units.append(
                    {   
                        'unit': units[enemy_team][i][j],
                        'tag': CustomAgent.tagToString[units[enemy_team][i][j]],
                        'hp': hps[enemy_team][i][j],
                        'location': (i,j),
                        'load': load[enemy_team][i][j]
                    }
                    )
                if res[i][j]==1:
                    resources.append((i,j))
                if bases[team][i][j]:
                    my_base = (i,j)
                if bases[enemy_team][i][j]:
                    enemy_base = (i,j)
        
        unitss = [*units[0].reshape(-1).tolist(), *units[1].reshape(-1).tolist()]
        hpss = [*hps[0].reshape(-1).tolist(), *hps[1].reshape(-1).tolist()]
        basess = [*bases[0].reshape(-1).tolist(), *bases[1].reshape(-1).tolist()]
        ress = [*res.reshape(-1).tolist()]
        loads = [*load[0].reshape(-1).tolist(), *load[1].reshape(-1).tolist()]
        terr = [*terrain.reshape(-1).tolist()]
        best_dir_flat = best_dir.reshape(-1).tolist()
        
        state = (*score.tolist(), turn, max_turn, *unitss, *hpss, *basess, *ress, *loads, *terr, *best_dir_flat)
        
        return np.array(state, dtype=np.int16), (x_max, y_max, my_units, enemy_units, resources, my_base,enemy_base)
    
    @staticmethod
    def just_decode_state(obs, team, enemy_team):
        state, _ = CustomAgent._decode_state(obs, team, enemy_team)
        return state

    def decode_state(self, obs):
        state, info = self._decode_state(obs, self.team, self.enemy_team)
        self.x_max, self.y_max, self.my_units, self.enemy_units, self.resources, self.my_base, self.enemy_base = info
        return state

    def take_action(self, action):
        return self.just_take_action(action, self.nec_obs, self.team) 

    @staticmethod
    def just_take_action(action, raw_state, team):
        movement = action[0:7]
        movement = movement.tolist()
        target = action[7:14]
        train = action[14]
        enemy_order = []
        allies = ally_locs(raw_state, team)
        enemies = enemy_locs(raw_state, team)

        if 0 > len(allies):
            print("why do you have negative allies ?")
            raise ValueError
        elif 0 == len(allies):
            locations = []
            movement = []
            target = []
            return [locations, movement, target, train]
        elif 0 < len(allies) <= 7:
            ally_count = len(allies)
            locations = allies

            counter = 0
            for j in target:
                if len(enemies) == 0:
                    enemy_order = [[6, 0] for i in range(ally_count)]
                    continue
                k = j % len(enemies)
                if counter == ally_count:
                    break
                if len(enemies) <= 0:
                    break
                enemy_order.append(enemies[k].tolist())
                counter += 1

            while len(enemy_order) > ally_count:
                enemy_order.pop()
            while len(movement) > ally_count:
                movement.pop()

        elif len(allies) > 7:
            ally_count = 7
            locations = allies

            counter = 0
            for j in target:
                if len(enemies) == 0:
                    enemy_order = [[6, 0] for i in range(ally_count)]
                    continue
                k = j % len(enemies)
                if counter == ally_count:
                    break
                if len(enemies) <= 0:
                    break
                enemy_order.append(enemies[k].tolist())
                counter += 1

            while len(locations) > 7:
                locations.pop(-1)

        # print("movement1", movement2)
        movement = multi_forced_anchor(movement, raw_state, team)

        if len(locations) > 0:
            locations = list(map(list, locations))

        #locations'dan biri, bir düşmana 2 adımda veya daha yakınsa dur (movement=0) ve ona ateş et (target = arg.min(distances))
        for i in range(len(locations)):
            for k in range(len(enemy_order)):
                
                if getDistance(locations[i], enemy_order[k]) <= 3:
                    movement[i] = 0
                    enemy_order[i] = enemy_order[k]


        locations = list(map(tuple, locations))
        return [locations, movement, enemy_order, train]
    
    def calculate_custom_rewards(self, obs, team, movement):
        load_reward = 0
        unload_reward = 0
        proximity_reward = 0
        survival_reward = 0
        resource_control_reward = 0
        strategic_positioning_reward = 0
        correct_movement_reward = 0
        incorrect_movement_penalty = 0

        bases = obs['bases'][team]
        units = obs['units'][team]
        enemy_bases = obs['bases'][(team + 1) % 2]
        enemy_units = obs['units'][(team + 1) % 2]
        loads = obs['loads'][team]
        resources = obs['resources']
        base_loc = np.argwhere(bases == 1).squeeze()
        enemy_base_loc = np.argwhere(enemy_bases == 1).squeeze()
        resource_loc = np.argwhere(resources == 1)
        enemy = enemy_locs(obs, team)
        allies = ally_locs(obs, team)
        trucks = truck_locs(obs, team)

        # Calculate load and unload rewards
        for truck in trucks:
            for reso in resource_loc:
                if not isinstance(truck, np.int64):
                    if (reso == truck).all() and loads[truck[0], truck[1]].max() != 3:
                        load_reward += 10
                if not isinstance(truck, np.int64):
                    if loads[truck[0], truck[1]].max() != 0 and (truck == base_loc).all():
                        unload_reward += 20

        # Calculate proximity reward
        for ally in ally_locs(obs, team):
            for enemy_base in enemy_base_loc:
                proximity_reward += 0

        # Calculate survival reward
        survival_reward = len(ally_locs(obs, team)) * 5

        # Calculate resource control reward
        for ally in ally_locs(obs, team):
            for reso in resource_loc:
                if (ally == reso).all():
                    resource_control_reward += 1

        # Calculate strategic positioning reward (example: rewarding units on high ground)
        for ally in ally_locs(obs, team):
            if obs['terrain'][ally[0]][ally[1]] == 2:  # Assuming '2' represents high ground
                strategic_positioning_reward += 2

        # Calculate correct and incorrect movement reward/penalty
        for i, ally in enumerate(allies):
            if len(trucks) == 0 or i > 6:
                break
            if isinstance(trucks[0], np.int64):
                trucks = np.expand_dims(trucks, axis=0)
            for truck in trucks:
                if (ally == truck).all():
                    if loads[truck[0], truck[1]] >= 2:
                        best_direction = get_best_direction_bfs(truck, base_loc, obs['terrain'])
                        if movement[i] == best_direction:
                            correct_movement_reward += 5
                    elif loads[truck[0], truck[1]] == 1:
                        resource_distances = sorted(resource_loc, key=lambda res: getDistance(truck, res))[:2]
                        closest_resource = min(resource_distances, key=lambda res: len(bfs_pathfinding(truck, res, obs['terrain'])))
                        distance_to_base = len(bfs_pathfinding(truck, base_loc, obs['terrain']))
                        distance_to_resource = len(bfs_pathfinding(truck, closest_resource, obs['terrain']))
                        if distance_to_base < distance_to_resource:
                            best_direction = get_best_direction_bfs(truck, base_loc, obs['terrain'])
                        else:
                            best_direction = get_best_direction_bfs(truck, closest_resource, obs['terrain'])
                        if movement[i] == best_direction:
                            correct_movement_reward += 5
                    elif loads[truck[0], truck[1]] == 0:
                        resource_distances = sorted(resource_loc, key=lambda res: getDistance(truck, res))[:5]
                        closest_resource = min(resource_distances, key=lambda res: len(bfs_pathfinding(truck, res, obs['terrain'])))
                        best_direction = get_best_direction_bfs(truck, closest_resource, obs['terrain'])
                        if movement[i] == best_direction:
                            correct_movement_reward += 5
                    # print(best_direction)

        total_reward = (load_reward + unload_reward + proximity_reward +
                        survival_reward + resource_control_reward +
                        strategic_positioning_reward + correct_movement_reward +
                        incorrect_movement_penalty)

        return total_reward

    
    def step(self, action):
        action = self.take_action(action)
        next_state, _, done =  self.game.step(action)
        reward = self.calculate_custom_rewards(self.nec_obs, self.team,action[1])
        _, enemy_count, ally_count = multi_reward_shape(self.nec_obs, self.team)

        self.previous_enemy_count = enemy_count
        self.previous_ally_count = ally_count
        info = {}
        self.steps += 1
        self.reward += reward

        self.nec_obs = next_state
        return self.decode_state(next_state), reward, done, info
    def render(self,):
        return None

    def close(self,):
        return None
