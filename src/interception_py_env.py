import math
import numpy as np
import gym
import time
from gym import spaces
from gym.utils import seeding


class InterceptionEnv(gym.Env):
    '''
    Description:
        The agent needs to intercept a target that moves along a predictable
        (straight-line) trajectory, with a sudden acceleration after X ms.
        The new speed is selected from a distribution. For any given state
        the agent may choose a paddle position which affects the travel
        speed with a log.
    Source:
        Diaz, G. J., Phillips, F., & Fajen, B. R. (2009). Intercepting moving
        targets: a little foresight helps a lot. Experimental brain research,
        195(3), 345-360.
    Observation:
        Type: Box(2)
        Num    Observation                  Min         Max
        0      Target distance              0.0         45.0
        1      Target velocity              8.18        20.0
        2       Subject distance            0.0         30.0
        3        Subject velocity           0.0         14.0
        4       Whether the target has       0           1
               changed speed (0 or 1)
    Actions:
        Type: Discrete(6)
        Num    Action
        0      Change the paddle positon to be 1 (0 means no accerleration)
        1      Change the paddle positon to be 2
        2      Change the paddle positon to be 3
        3       Change the paddle positon to be 4
        4       Change the paddle positon to be 5
        5       Change the paddle positon to be 6
        Note: Paddle position at one of N positions and changes are instantaneous.
        Change in speed determined by the difference between the current traveling
        speed and the new pedal position  V_dot =  K * ( Vp - Vs)
    Reward:
         Reward of 0 is awarded if the agent intercepts the target (position = 0.5)
         Reward of -1 is awarded everywhere else.
    Starting State:
         The simulated target approached the unmarked interception point from an
         initial distance of 45 m.
         Initial approach angle of 135, 140, or 145 degree from the subject’s
         path of motion.
         The starting velocity of the target is one from 11.25, 9.47, 8.18 m/s,
         which corresponds to initial first-order time-to-contact values of 4,
         4.75, and 5.5 seconds.
         Subject's initial distance is sampled from a uniform distribution
         between 25 and 30 meters.
    Episode Termination:
         The target position is at 0 (along target trajectory).
         The subject position is at 0 (along subject trajectory).
         Episode length is greater than 6 seconds (180 steps @ 30FPS).
    '''

    def __init__(self, target_speed_idx=0, approach_angle_idx=0):
        self.subject_min_position = 0.0
        self.subject_max_position = 30.0
        self.approach_angle_list = [135, 140, 145]
        self.target_init_speed_list = [11.25, 9.47, 8.18]
        self.approach_angle = self.approach_angle_list[approach_angle_idx]
        self.target_init_speed = self.target_init_speed_list[target_speed_idx]
        self.intercept_threshold = 0.35 * 2
        self.subject_init_distance_min = 20.0
        self.subject_init_distance_max = 30.0
        self.subject_speed_max = 14.0
        self.target_init_distance = 45.0
        self.time_to_change_speed_min = 2.5
        self.time_to_change_speed_max = 3.25
        self.speed_change_duration = 0.5
        self.target_max_speed = 20.0
        self.target_fspeed_mean = 15.0
        self.target_fspeed_std = 5.0
        self.target_min_speed = 10.0
        self.lag_coefficient = 0.017
        # FPS is set to match the frame rate of human subject experiments
        self.FPS = 60
        self.viewer = None
        self.action_type = 'acceleration'  # or 'speed'

        if self.action_type == 'speed':
            # instantaneous speed change
            self.action_space = spaces.Discrete(6)
            self.action_speed_mappings = [2.0, 4.0, 8.0, 10.0, 12.0, 14.0]
            self.low = np.array([0.0, np.min(self.target_init_speed_list), 0.0, 0.0, 0.0], dtype=np.float32)
            self.high = np.array([self.target_init_distance, self.target_max_speed, 1.0,
                                  self.subject_max_position, self.subject_speed_max], dtype=np.float32)
        elif self.action_type == 'acceleration':
            # accelerate / decelerate / none
            self.action_space = spaces.Discrete(5)
            self.action_acceleration_mappings = [0.05, 0.2, -0.05, -0.2, 0.0]
            self.low = np.array([-self.subject_speed_max], dtype=np.float32)
            self.high = np.array([self.subject_speed_max], dtype=np.float32)
        else:
            raise Exception("Action type {} is not valid!".format(self.action_type))

        self.observation_space = spaces.Box(
            self.low, self.high, dtype=np.float32)

        self.seed()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step_old(self, action):
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))
        target_dis, target_speed, has_changed_speed, subject_dis, subject_speed = self.state

        self.time += 1.0 / self.FPS
        if self.time >= self.time_to_change_speed and not has_changed_speed:
            has_changed_speed = 1
        if has_changed_speed:
            speed_proportion = (
                self.time - self.time_to_change_speed) / self.speed_change_duration
            target_speed = min(self.target_final_speed, speed_proportion * (
                self.target_final_speed - self.target_init_speed) + self.target_init_speed)

        # TODO add the lagging effect to subject speed
        subject_speed = self.action_speed_mappings[action]
        # subject_speed += (subject_speed - self.action_speed_mappings[action]) * self.lag_coefficient
        subject_dis -= subject_speed / self.FPS
        target_dis -= target_speed / self.FPS
        target_subject_dis = np.sqrt(np.square(target_dis) + np.square(
            subject_dis) - 2 * target_dis * subject_dis * np.cos(self.approach_angle * np.pi / 180))

        done = bool(
            subject_dis <= 0 or target_dis <= 0 or target_subject_dis <= self.intercept_threshold
        )
        reward = 100 if target_subject_dis <= self.intercept_threshold else 0

        self.state = (target_dis, target_speed,
                      has_changed_speed, subject_dis, subject_speed)
        return np.array(self.state), reward, done, {}


    def step(self, action):
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))
        target_dis, target_speed, has_changed_speed, subject_dis, subject_speed = self.info

        self.time += 1.0 / self.FPS
        if self.time >= self.time_to_change_speed and not has_changed_speed:
            has_changed_speed = 1
        if has_changed_speed:
            speed_proportion = (
                self.time - self.time_to_change_speed) / self.speed_change_duration
            target_speed = min(self.target_final_speed, speed_proportion * (
                self.target_final_speed - self.target_init_speed) + self.target_init_speed)

        # TODO add the lagging effect to subject speed
        subject_speed += self.action_acceleration_mappings[action]
        subject_dis -= subject_speed / self.FPS
        target_dis -= target_speed / self.FPS
        target_subject_dis = np.sqrt(np.square(target_dis) + np.square(
            subject_dis) - 2 * target_dis * subject_dis * np.cos(self.approach_angle * np.pi / 180))

        done = bool(
            subject_dis <= 0 or target_dis <= 0 or target_subject_dis <= self.intercept_threshold
        )
        reward = 100 if target_subject_dis <= self.intercept_threshold else 0

        self.info = (target_dis, target_speed,
                      has_changed_speed, subject_dis, subject_speed)

        estimated_speed = subject_dis / (target_dis / target_speed)
        self.state = subject_speed - estimated_speed

        return np.asarray([self.state], dtype=np.float32), reward, done, self.info


    def reset_old(self, target_speed_idx=0, approach_angle_idx=0):
        self.time_to_change_speed = self.np_random.uniform(
            low=self.time_to_change_speed_min, high=self.time_to_change_speed_max)
        self.approach_angle = self.approach_angle_list[approach_angle_idx]
        self.target_init_speed = self.target_init_speed_list[target_speed_idx]
        self.target_final_speed = np.clip(
            self.np_random.normal(
                loc=self.target_fspeed_mean, scale=self.target_fspeed_std),
            self.target_min_speed,
            self.target_max_speed
        )
        self.time = 0.0
        subject_init_distance = self.np_random.uniform(
            low=self.subject_init_distance_min, high=self.subject_init_distance_max)
        subject_init_speed = 0.0
        has_changed_speed = 0
        self.state = np.asarray([self.target_init_distance, self.target_init_speed,
                                 has_changed_speed, subject_init_distance, subject_init_speed], dtype=np.float32)
        return np.array(self.state)


    def reset(self):
        self.time_to_change_speed = self.np_random.uniform(
            low=self.time_to_change_speed_min, high=self.time_to_change_speed_max)
        self.target_final_speed = np.clip(
            self.np_random.normal(
                loc=self.target_fspeed_mean, scale=self.target_fspeed_std),
            self.target_min_speed,
            self.target_max_speed
        )
        self.time = 0.0
        subject_init_distance = self.np_random.uniform(
            low=self.subject_init_distance_min, high=self.subject_init_distance_max)
        subject_init_speed = 0.0
        has_changed_speed = 0
        self.info = np.asarray([self.target_init_distance, self.target_init_speed,
                                 has_changed_speed, subject_init_distance, subject_init_speed], dtype=np.float32)
        estimated_speed = subject_init_distance / (self.target_init_distance / self.target_init_speed)
        self.state = subject_init_speed - estimated_speed
        return np.asarray([self.state], dtype=np.float32)


    def render(self, mode='human'):
        if self.action_type == 'speed':
            target_dis, target_speed, has_changed_speed, subject_dis, subject_speed = self.state
        elif self.action_type == 'acceleration':
            target_dis, target_speed, has_changed_speed, subject_dis, subject_speed = self.info
        # scale = (1 / (self.intercept_threshold / 2)) * 4

        screen_width = 1000

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            import text_rendering

            actual_width = int((-math.cos(max(self.approach_angle_list) * math.pi / 180)
                                * self.subject_max_position + self.target_init_distance))
            self.scale = screen_width / actual_width
            screen_height = int((math.sin(min(self.approach_angle_list) *
                                          math.pi / 180) * self.subject_max_position) * self.scale + 100)
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # Set the world origin to somewhere that makes sense, keeping everything on screen at all times
            world_origin = rendering.Transform(translation=(
                self.target_init_distance * self.scale + 10, 90))

            # Set the background to black to make the subject and target stand out more
            background = rendering.make_polygon(
                [(0, 0), (screen_width, 0), (screen_width, screen_height), (0, screen_height)])
            background.set_color(0, 0, 0)
            self.viewer.add_geom(background)

            # Create the subject and target in red and green, and dashed lines as the projected path they travel on
            subject = rendering.make_circle(
                self.intercept_threshold / 2 * self.scale)
            subject.set_color(0, 1, 0)
            self.subject_trans = rendering.Transform()
            subject.add_attr(self.subject_trans)
            self.subject_rot = rendering.Transform()
            subject.add_attr(self.subject_rot)
            subject.add_attr(world_origin)

            subject_path = rendering.Line(start=(-10000, 0), end=(10000, 0))
            subject_path.set_color(0.5, 0.5, 0.5)
            subject_path.add_attr(rendering.LineStyle(0xFF80))
            subject_path.add_attr(self.subject_rot)
            subject_path.add_attr(world_origin)

            target = rendering.make_circle(
                self.intercept_threshold / 2 * self.scale)
            target.set_color(1, 0, 0)
            self.target_trans = rendering.Transform()
            target.add_attr(self.target_trans)
            target.add_attr(world_origin)

            target_path = rendering.Line(start=(-10000, 0), end=(10000, 0))
            target_path.set_color(0.5, 0.5, 0.5)
            target_path.add_attr(rendering.LineStyle(0xFF80))
            target_path.add_attr(world_origin)

            self.viewer.add_geom(subject_path)
            self.viewer.add_geom(target_path)
            self.viewer.add_geom(subject)
            self.viewer.add_geom(target)

            # Create the state display
            info_top = screen_height
            self.target_distance_label = text_rendering.Text(
                'Target Distance: ' + str(target_dis))
            info_top -= self.target_distance_label.text.content_height
            self.target_distance_label.add_attr(
                rendering.Transform(translation=(5, info_top)))
            self.viewer.add_geom(self.target_distance_label)

            self.target_speed_label = text_rendering.Text(
                'Target Speed: ' + str(target_speed))
            info_top -= self.target_speed_label.text.content_height
            self.target_speed_label.add_attr(
                rendering.Transform(translation=(5, info_top)))
            self.viewer.add_geom(self.target_speed_label)

            self.has_changed_speed_label = text_rendering.Text(
                'Has Changed Speed: ' + ('Yes' if has_changed_speed else 'No'))
            info_top -= self.has_changed_speed_label.text.content_height
            self.has_changed_speed_label.add_attr(
                rendering.Transform(translation=(5, info_top)))
            self.viewer.add_geom(self.has_changed_speed_label)

            self.subject_dis_label = text_rendering.Text(
                'Subject Distance: ' + str(subject_dis))
            info_top -= self.subject_dis_label.text.content_height
            self.subject_dis_label.add_attr(
                rendering.Transform(translation=(5, info_top)))
            self.viewer.add_geom(self.subject_dis_label)

            self.subject_speed_label = text_rendering.Text(
                'Subject Speed: ' + str(subject_speed))
            info_top -= self.subject_speed_label.text.content_height
            self.subject_speed_label.add_attr(
                rendering.Transform(translation=(5, info_top)))
            self.viewer.add_geom(self.subject_speed_label)

        # Update the state of the frame
        self.subject_trans.set_translation(-subject_dis * self.scale, 0)
        self.subject_rot.set_rotation(-self.approach_angle / 180 * math.pi)
        self.target_trans.set_translation(-target_dis * self.scale, 0)
        self.target_distance_label.set_text(
            'Target Distance: ' + str(target_dis))
        self.target_speed_label.set_text('Target Speed: ' + str(target_speed))
        self.has_changed_speed_label.set_text(
            'Has Changed Speed: ' + ('Yes' if has_changed_speed else 'No'))
        self.subject_dis_label.set_text(
            'Subject Distance: ' + str(subject_dis))
        self.subject_speed_label.set_text(
            'Subject Speed: ' + str(subject_speed))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


    def get_keys_to_action(self):
        # Control with letter keys on keyboards
        return {(): 0, (ord('a'),): 0, (ord('s'),): 1, (ord('d'),): 2, (ord('f'),): 3, (ord('g'),): 4, (ord('h'),): 5}


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    test = InterceptionEnv(target_speed_idx=2, approach_angle_idx=0)
    test.reset()
    frame_duration = 1 / test.FPS

    test.render()
    prev_time = time.time()
    while not test.step(0)[2]:
        time.sleep(max(frame_duration - (time.time() - prev_time), 0))
        prev_time = time.time()
        test.render()

    time.sleep(frame_duration - (time.time() - prev_time))
    test.render()

    input('press enter to close')
