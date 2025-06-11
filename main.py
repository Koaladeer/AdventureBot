from jericho import *

from JerichoWorld.build_dataset import build_dataset_walkthrough

# Create the environment, optionally specifying a random seed
env = FrotzEnv('games/z-machine-games-master/jericho-game-suite/zork1.z5')
initial_observation, info = env.reset()
done = False
# while not done:
#     # Take an action in the environment using the step fuction.
#     # The resulting text-observation, reward, and game-over indicator is returned.
#     observation, reward, done, info = env.step('open mailbox')
#     # Total score and move-count are returned in the info dictionary
#     print('observation:', observation)
#     print('reward:', reward)
#     print('info:', info)
#
#     print('Total Score', info['score'], 'Moves', info['moves'])
# print('Scored', info['score'], 'out of', env.get_max_score())


##walktrhough
walkthrough = env.get_walkthrough()
for act in walkthrough:
    print('-----------------')
    print('action taken:',act)
    #print('possible actions:',env.get_valid_actions())
    observation, reward, done, info = env.step(act)
    print('observation:', observation)
    print('reward:', reward)
    print('info:', info)
    print('Total Score', info['score'], 'Moves', info['moves'])
