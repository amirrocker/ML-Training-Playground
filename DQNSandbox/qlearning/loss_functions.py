import torch
import torch.cuda
import torch.nn as nn

GAMMA = 0.99

def calc_loss(batch, net, tgt_net, device="cpu"):
    # get states, actions, rewards, dones and next_states from the received batch
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states, device=device).to(device=device)
    next_states_v = torch.tensor(next_states, device=device).to(device=device)
    actions_v = torch.tensor(actions, device=device).to(device=device)
    rewards_v = torch.tensor(rewards, device=device).to(device=device)
    done_mask = torch.tensor(dones, device=device).to(device=device)

    assert done_mask.size() == rewards_v.size()

    neural_net_result = net(states_v)
    #assert isinstance(neural_net_result, torch.tensor)
#    print(neural_net_result)

    result_actions_v = actions_v.unsqueeze(-1)
#    print(result_actions_v)

    unsqueeze_result = neural_net_result.gather(1, result_actions_v.long())
#    print(unsqueeze_result)

    squeeze_result = unsqueeze_result.squeeze(-1)
#    print(squeeze_result)

    # calculating the state_action values -> basically Q(s,a)
    #state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1).long()).squeeze(-1)

    tgt_net_result = tgt_net(next_states_v)

    next_state_values = tgt_net_result.max(1)[0]

    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()
    expected_state_action_values = next_state_values * GAMMA + rewards_v
    # return nn.MSELoss()(state_action_values, expected_state_action_values.float())
    return nn.MSELoss()(squeeze_result, expected_state_action_values.float())