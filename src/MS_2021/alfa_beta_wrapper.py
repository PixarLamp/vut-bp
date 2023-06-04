# author: Michal Sova
# year: 2021
# file: alfa_beta_wrapper.py
# description: wrapper for alfa-beta method, setting up environment translates coordinates

from AT_2020 import alfa_beta, alfa_beta_mrx, environment

a_b = alfa_beta.AlfaBeta()
a_b.root.reset()


def env_set(agents, mrx, field):
    env = environment.Environment(field_size=field)
    env.reset()
    env.mrx = mrx[0] * field + mrx[1]
    env.agents.clear()
    for agent in agents:
        env.agents.append(agent[0] * field + agent[1])
    return env


def mrx_move(agents, mrx, field):
    env = env_set(agents, mrx, field)

    ab = alfa_beta_mrx.AlfaBeta(field_size=field)
    ab.root.reset()
    mrx_pos = ab.move_mrx(env.agents, env.mrx)
    mrx = [mrx_pos // field, mrx_pos % field]

    return mrx


def agents_move(agents, mrx, field, last_seen):
    env = env_set(agents, mrx, field)

    a_b.field_size = field
    # print(last_seen)
    if last_seen == 0 or a_b.root is None or not a_b.root.best_way:
        a_b.explore_state_space(env.agents, env.mrx)

    agent_moves = a_b.choose_new_move_agents()

    agents_ret = []
    for agent in agent_moves:
        agents_ret.append([agent // field, agent % field])
    # print(agents_ret)
    return agents_ret
