import gym

TEST_ENV_CARTPOLE = "CartPole-v0"
TEST_ENV_PONG_NO_FRAMESKIP = "PongNoFrameskip-v4"

env = gym.make(TEST_ENV_PONG_NO_FRAMESKIP)
env.reset()

for _ in range(1000):
    env.render()
    new_state, reward, is_done, _ = env.step(env.action_space.sample())

    print("just for fun reward: %s and new_state: %s " % (reward, new_state))

    if(is_done):
        print("Is done : %s" % is_done)
        break

env.close()