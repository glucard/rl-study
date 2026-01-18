# import jax
# import jax.numpy as jnp

# from typing import Callable, NamedTuple

# class JackRentalCarState(NamedTuple):

#     key_timestep: jnp.ndarray

#     rent_price: jnp.ndarray
#     move_cost: jnp.ndarray
    
#     available_cars: jnp.ndarray
#     avaiblade_cars_limit: jnp.ndarray
#     move_limit: int

#     expected_requests: jnp.ndarray
#     expected_returns: jnp.ndarray

#     state_values: jnp.ndarray
#     policy_probs: jnp.ndarray

# def create_jack_rental_car(
#         inital_key: jnp.ndarray,
#         rent_price: float | jnp.ndarray = 10,
#         move_cost: float | jnp.ndarray = 2,
#         available_cars: int | jnp.ndarray = 20,
#         avaiblade_cars_limit: int | jnp.ndarray = 20,
#         move_limit: int = 5,
#         expected_requests: int | jnp.ndarray = jnp.array([3, 4], dtype=jnp.int32),
#         expected_returns: int | jnp.ndarray = jnp.array([3, 2], dtype=jnp.int32),
# ) -> JackRentalCarState:

#     if isinstance(rent_price, (float, int)):
#         rent_price = jnp.full(2, rent_price, jnp.float32)

#     if isinstance(move_cost, (float, int)):
#         move_cost = jnp.full(2, move_cost, jnp.float32)

#     if isinstance(available_cars, int):
#         available_cars = jnp.full(2, available_cars, jnp.int32)

#     if isinstance(avaiblade_cars_limit, int):
#         avaiblade_cars_limit = jnp.full(2, avaiblade_cars_limit, jnp.int32)

#     if isinstance(expected_requests, int):
#         expected_requests = jnp.full(2, expected_requests, jnp.int32)

#     if isinstance(expected_returns, int):
#         expected_returns = jnp.full(2, expected_returns, jnp.int32)

#     assert isinstance(rent_price, jnp.ndarray)
#     assert isinstance(move_cost, jnp.ndarray)
#     assert isinstance(available_cars, jnp.ndarray)
#     assert isinstance(avaiblade_cars_limit, jnp.ndarray)
#     assert isinstance(move_limit, int)
#     assert isinstance(expected_requests, jnp.ndarray)
#     assert isinstance(expected_returns, jnp.ndarray)

#     state_values = jnp.full(
#         shape=(avaiblade_cars_limit[0],avaiblade_cars_limit[1]),
#         fill_value=0,
#         dtype=jnp.float32,
#     )

#     policy_actions = jnp.full(
#         shape=(
#             avaiblade_cars_limit[0],
#             avaiblade_cars_limit[1],
#             move_limit*2+1
#         ),
#         fill_value=1,
#         dtype=jnp.float32,
#     )

#     return JackRentalCarState(
#         key_timestep=inital_key,
#         rent_price=rent_price,
#         move_cost=move_cost,
#         available_cars=available_cars,
#         avaiblade_cars_limit=avaiblade_cars_limit,
#         move_limit=move_limit,
#         expected_requests=expected_requests,
#         expected_returns=expected_returns,
#         state_values=state_values,
#         policy_probs=policy_actions,
#     )


# def step_jack_rental_car(
#         state: JackRentalCarState,
#         _,
# ):
#     (
#         next_key_timestep,
#         key_rental,
#         key_returns
#     ) = jnp.split(state.key_timestep, 3)

#     requests = jax.random.poisson(key_rental, state.expected_requests, shape=(2,))
#     rented = jnp.minimum(requests, state.available_cars)

    


#     new_available_cars = jnp.minimum(
#         state.available_cars - rented + moved_cars,
#         state.avaiblade_cars_limit,
#     )
    
#     return state._replace(
#         available_cars=available_cars
#     )


    
# master_key = jax.random.PRNGKey(42)

# initial_state = create_jack_rental_car(master_key)

# print(initial_state)