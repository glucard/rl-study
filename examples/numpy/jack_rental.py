import math
import jax
import jax.random as jrd
import jax.numpy as jnp
import numpy as np

from tqdm.auto import tqdm

def poisson_prob(n, lam):
    return math.e**(-lam) * (lam**n) / jax.scipy.special.factorial(n)

def get_poisson_probs(max_cars_to_rent: int, lam: int):
    print(type(max_cars_to_rent))
    probs = [poisson_prob(i, lam) for i in range(max_cars_to_rent)]
    probs.append(1 - sum(probs))
    return jnp.array(probs)


def generate_states(
    max_cars:int = 20
):
    

    # V(s) = \summation_(s',r){ p(s',r | s,a) * [r + \gamma * V(s')] }

    #  V(s) = \summation_(s',r){ p(r|s,a)*r + p(r|s,a) * \gamma * V(s')] }
    #  V(s) = \summation_(r) { p(r|s,a)*r } + \summation_(s'){p(s'|s,a) * \gamma * V(s')] }


    prob_requests_1 = get_poisson_probs(max_cars, 3)
    prob_requests_2 = get_poisson_probs(max_cars, 4)

    prob_returns_1 = get_poisson_probs(max_cars, 3)
    prob_returns_2 = get_poisson_probs(max_cars, 2)    

    # expected_return


    # expected reward of starting with i cars
    expected_reward_1 = np.zeros((max_cars+1), dtype=np.float32)
    expected_reward_2 = np.zeros((max_cars+1), dtype=np.float32)
    
    # prob of ending with j cars while starting with i:  prob[i,j]
    probs_1 = np.zeros((max_cars+1, max_cars+1), dtype=np.float32)
    probs_2 = np.zeros((max_cars+1, max_cars+1), dtype=np.float32)
    for i in range(max_cars+1):
        for rented in range(max_cars+1):
            real_rented = min(i, rented)
            
            for returned in range(max_cars+1):

                j = i - real_rented + returned
                j = min(j, max_cars)
                
                probs_1[i][j] += prob_requests_1[rented] * prob_returns_1[returned]
                probs_2[i][j] += prob_requests_2[rented] * prob_returns_2[returned]
        
            expected_reward_1[i] += 10 * real_rented * prob_requests_1[rented]
            expected_reward_2[i] += 10 * real_rented * prob_requests_2[rented]


    possible_actions = jnp.array(range(max_cars * 2 + 1), dtype=jnp.int16) - max_cars
    print(possible_actions)

    # initialization
    state_values = jnp.zeros((max_cars+1, max_cars+1), dtype=jnp.float32)
    policy = jnp.full((max_cars+1, max_cars+1), fill_value=20, dtype=jnp.int16)
    print(policy)

    
    I, J = jnp.meshgrid(jnp.arange(max_cars + 1), jnp.arange(max_cars + 1), indexing='ij')

    while True:
        delta = 0
        old_state_values = state_values.copy()

        
        
        for cars_1 in tqdm(range(max_cars+1)):
            for cars_2 in range(max_cars+1):
                action = possible_actions[policy[cars_1, cars_2]]

                morning_1 = min(cars_1 - action, max_cars)
                morning_2 = min(cars_2 + action, max_cars)
                
                expected_reward = expected_reward_1[morning_1] + expected_reward_2[morning_2]


                # because loc 1 and 2 are independent
                # p(s'|s, a) = p((j_1,j_2)|(i_1, i_2), a) = p(j_1|i_1,a) \times p(j_2|i_2,a)
                
                # 2. The "Gamma * Expected V" part (Dot product sum of p * v)
                expected_val_given_loc1 = jnp.dot(old_state_values, probs_2[morning_2])
                expected_future_val = jnp.dot(probs_1[morning_1], expected_val_given_loc1)
                
                action_reward = -2 * abs(action)
                new_state_value = expected_reward + action_reward + 0.9 * expected_future_val

                state_values = state_values.at[cars_1, cars_2].set(new_state_value)

                delta = max(delta, abs(old_state_values[cars_1, cars_2] - new_state_value))

        print(delta)
        if delta < 0.1:
            break







master_key = jrd.PRNGKey(42)

generate_states()

# def generate_states(
#         key: jnp.ndarray,
#         max_cars_to_rent = 20,
#         rent_value: float = 10,
#         move_cost: float = 2,
#         teta = 0.1,
#         gamma = 0.9,

# ):
    
#     possible_actions = jnp.array(range(max_cars_to_rent * 2 + 1), dtype=jnp.int16) - max_cars_to_rent
#     print(possible_actions)

#     # initialization
#     state_values = jnp.zeros((max_cars_to_rent+1, max_cars_to_rent+1), dtype=jnp.float32)
#     policy = jnp.full((max_cars_to_rent+1, max_cars_to_rent+1), fill_value=20, dtype=jnp.int16)
#     print(policy)

#     prob_requests_first_loc = get_poisson_probs(max_cars_to_rent, 3)
#     prob_requests_second_loc = get_poisson_probs(max_cars_to_rent, 4)


#     prob_return_first_loc = get_poisson_probs(max_cars_to_rent, 3)
#     prob_return_second_loc = get_poisson_probs(max_cars_to_rent, 2)
    
    
#     # policy evaluation
#     while True:
#         delta = 0
#         old_state_values = state_values.copy()
        
#         # for each $s \in \mathbb{S}
#         for cars_in_first_location in tqdm(range(max_cars_to_rent+1)):
#             for cars_in_second_location in range(max_cars_to_rent+1):
#                 v = state_values[cars_in_first_location, cars_in_second_location]

#                 keys = jrd.split(key, 2)
#                 key = keys[0]
                
#                 # for each s', r
#                 # action = possible_actions[jrd.categorical(key=keys[1], logits=policy[cars_in_first_location, cars_in_second_location])]
#                 action = possible_actions[policy[cars_in_first_location, cars_in_second_location]]


#                 expected_requests = 0
#                 expected_next_state_value = 0


#                 if cars_in_first_location - action < 0 and cars_in_second_location + action < 0:
#                     raise ValueError("erro")

#                 morning_cars_1 = min(cars_in_first_location - action, max_cars_to_rent)
#                 morning_cars_2 = min(cars_in_second_location + action, max_cars_to_rent)

#                 for request1 in range(max_cars_to_rent + 1):
#                     for request2 in range(max_cars_to_rent + 1):
                        
#                         # probs
#                         prob_request = prob_requests_first_loc[request1] * prob_requests_second_loc[request2]
                        
#                         # rents
#                         cars_rented_1 = min(request1, morning_cars_1)
#                         cars_rented_2 = min(request2, morning_cars_2)
                        
#                         # expected request reward
#                         expected_requests += prob_request * (cars_rented_1+cars_rented_2)

#                         for return1 in range(max_cars_to_rent+1):
#                             for return2 in range(max_cars_to_rent+1):
                                
#                                 # probs
#                                 prob_return = prob_return_first_loc[return1] * prob_return_second_loc[return2]
#                                 prob_next_state = prob_return * prob_request

#                                 # cars at night
#                                 cars_night_1 = min(morning_cars_1 + return1 - cars_rented_1, max_cars_to_rent)
#                                 cars_night_2 = min(morning_cars_2 + return2 - cars_rented_2, max_cars_to_rent)
                                
#                                 expected_next_state_value += prob_next_state * old_state_values[cars_night_1, cars_night_2]        
                
#                 expected_requests_reward = -2 * abs(action) + expected_requests * 10

#                 new_state_value = expected_requests_reward + gamma * expected_next_state_value

#                 state_values = state_values.at[(cars_in_first_location, cars_in_second_location)].set(new_state_value)

#                 delta = max(delta, abs(v - state_values[cars_in_first_location, cars_in_second_location]))

#         print(delta)
#         if delta < teta:
#             break
#     print(state_values)





# master_key = jrd.PRNGKey(42)

# generate_states(master_key)