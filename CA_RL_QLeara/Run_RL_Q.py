from Game import Game
from RL_Q import QLearningTable

def update():
    for episode in range(100):
        observation=env.reset()
        while True:
            env.render()

            action=RL.choose_action(str(observation))

            observation_,reward,done=env.step(action)

            RL.learn(str(observation),action,reward,str(observation_))
            print("action:",action,"---reward:",reward)

            observation=observation_

            if done:
                break
    print("game over")



if __name__ == '__main__':
    env=Game()
    RL=QLearningTable(action=list(range(env.n_actions)))
    update()