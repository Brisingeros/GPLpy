import gym
from gym.wrappers import Monitor
env = Monitor(gym.make('LunarLanderContinuous-v2'), './video', force=True, video_callable=lambda episode_id: True)
env.seed(9756745635)
fitness = 0

for _ in range(10):
        observation = env.reset()
        done = False
        while not done:

                #################################

                x = observation[0]
                y = observation[1]
                vel_x = observation[2]
                vel_y = observation[3]
                ang = observation[4]
                vel_ang = observation[5]
                l_left = observation[6]
                l_right = observation[7]
                input = [0., 0.]

                #################################

                input = [max(min(i,1.0),-1.0) for i in [ ( ( ( observation[3] * -78.0698466187944 ) - ( ( -12.186430617542129 ) + ( 16.4559542711429 ) ) ) + ( ( -0.7513340629864018 ) * ( observation[5] - -16.145319203744336 ) ) ) , ( ( ( ( observation[4] ) + ( observation[4] ) ) + ( observation[4] + observation[4] ) ) - ( ( observation[4] + observation[3] ) - ( ( ( observation[6] - 15.332154602015805 ) + ( observation[3] ) ) * ( observation[2] - observation[5] ) ) ) ) ]]


                #################################
                #################################
                #################################
                observation, reward, done, info = env.step(input)
                fitness += reward

print(fitness)