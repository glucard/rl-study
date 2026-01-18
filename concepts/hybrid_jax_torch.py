import time
import jax
import jax.numpy as jnp
import flax.linen as nn
import gymnax

# ==========================================
# 1. Define the Network
# ==========================================
class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        logits = nn.Dense(self.action_dim)(x)
        return logits

# ==========================================
# 2. Define the Single Step (Compiled)
# ==========================================
def make_step_fn(env, network):
    
    # We create a closure that captures the STATIC configuration 
    # (env and network structure), but takes DYNAMIC state as arguments.
    
    def step_fn(runner_state, _):
        # Unpack the state that changes every step
        rng, obs, env_state, env_params, net_params = runner_state

        # --- A. AGENT STEP ---
        rng, rng_act = jax.random.split(rng)
        
        # FIX: We pass 'net_params' (the weights), NOT the class.
        logits = network.apply(net_params, obs)
        
        # Simple argmax policy for high-speed benchmark
        action = jnp.argmax(logits, axis=-1)

        # --- B. ENV STEP ---
        rng, rng_step = jax.random.split(rng)
        rng_step_split = jax.random.split(rng_step, obs.shape[0])
        
        # Vectorized step
        step_vmap = jax.vmap(env.step, in_axes=(0, 0, 0, None))
        next_obs, next_env_state, reward, done, info = step_vmap(
            rng_step_split, env_state, action, env_params
        )

        # Repack the state for the next iteration
        new_runner_state = (rng, next_obs, next_env_state, env_params, net_params)
        
        return new_runner_state, reward

    return step_fn

# ==========================================
# 3. Main Execution
# ==========================================
def main():
    # --- CONFIG ---
    NUM_ENVS = 4096
    # 10 Million steps to properly test the Ferrari engine
    TOTAL_STEPS = 10_000_000 
    BATCHES = TOTAL_STEPS // NUM_ENVS

    print(f"\U0001f680 Initializing Pure JAX (Target: >2M SPS)...")

    # 1. Setup Environment
    env, env_params = gymnax.make("CartPole-v1")
    
    # 2. Setup Network & Weights
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    network = ActorCritic(action_dim=2)
    # Initialize real weights
    network_params = network.init(init_rng, jnp.zeros((1, 4)))

    # 3. Setup Initial State
    print("\u23f3 Resetting Env...")
    rng, rng_reset = jax.random.split(rng)
    reset_keys = jax.random.split(rng_reset, NUM_ENVS)
    # Vectorized Reset
    reset_fn = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    obs, env_state = reset_fn(reset_keys, env_params)

    # 4. Pack everything into one big tuple
    # This tuple allows JAX to carry everything on the GPU loop
    init_runner_state = (rng, obs, env_state, env_params, network_params)

    # 5. Compile the Loop
    print("\u23f3 Compiling the 10-Million-Step Loop...")
    step_fn = make_step_fn(env, network)
    
    # jax.lax.scan is the "For Loop" that runs entirely on GPU
    # It takes the function, the initial state, and the length
    scan_fn = jax.jit(lambda s: jax.lax.scan(step_fn, s, None, length=BATCHES))
    
    # Warmup compilation (Run 1 step to trigger JIT)
    # We use a tiny length just to compile
    warmup_fn = jax.jit(lambda s: jax.lax.scan(step_fn, s, None, length=1))
    _ = warmup_fn(init_runner_state)
    print("\u2705 Compilation Done.")

    # 6. RUN
    print(f"\U0001f7e2 Running {TOTAL_STEPS:,} steps...")
    start = time.time()
    
    # This line sends the command to GPU and waits for the final result
    final_state, rewards = scan_fn(init_runner_state)
    
    # Block until GPU actually finishes calculation
    rewards.block_until_ready()
    
    end = time.time()
    duration = end - start
    sps = TOTAL_STEPS / duration

    print(f"\n\U0001f3c1 Finished!")
    print(f"Time: {duration:.4f}s")
    print(f"\U0001f680 SPS: {sps:,.0f}")
    
    # Sanity check to prove we actually did something
    print(f"Mean Reward: {rewards.mean():.2f}")

if __name__ == "__main__":
    main()