"""
DiLoCo: Distributed Low-Communication Training of Language Models

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - init_model_params
def init_model_params(input_dim, hidden_dim, output_dim, seed=0):
    # TODO: return dict with W1 (in,hid), b1 (hid,), W2 (hid,out), b2 (out,) as float64
    rng = np.random.RandomState(seed)
    
    # Initialize weights from a seeded numpy RNG
    W1 = rng.randn(input_dim, hidden_dim).astype(np.float64)
    W2 = rng.randn(hidden_dim, output_dim).astype(np.float64)
    
    # Biases start at zero
    b1 = np.zeros(hidden_dim, dtype=np.float64)
    b2 = np.zeros(output_dim, dtype=np.float64)
    
    return {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

# Step 2 - relu
import numpy as np

def relu(x):
    # TODO: return an array of the same shape as x with negatives clipped to 0.
    return np.maximum(0, x)

# Step 3 - model_forward
import numpy as np

def model_forward(params, x):
    """Run the 2-layer MLP forward pass and stash intermediates for backprop."""
    # TODO: compute z1, h1 = relu(z1), logits, and return (logits, cache).
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    
    # First layer: x @ W1 + b1
    z1 = x @ W1 + b1
    
    # ReLU activation
    h1 = relu(z1)
    
    # Second layer: h1 @ W2 + b2 (raw logits)
    logits = h1 @ W2 + b2
    
    # Cache intermediates for backprop
    cache = {
        'x': x,
        'z1': z1,
        'h1': h1,
        'logits': logits
    }
    
    return logits, cache

# Step 4 - softmax
import numpy as np

def softmax(logits):
    # TODO: return a row-wise, numerically-stable softmax of logits with shape (N, C).
    # Subtract the maximum value from each row for numerical stability
    # This prevents overflow when exponentiating large values
    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
    
    # Compute exponentials of shifted logits
    exp_logits = np.exp(logits_shifted)
    
    # Compute sum of exponentials for each row (keepdims=True for broadcasting)
    sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
    
    # Compute probabilities by dividing each row by its sum
    probs = exp_logits / sum_exp
    
    return probs

# Step 5 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(logits, labels):
    # TODO: Compute the mean cross-entropy loss between logits (N,C) and integer labels (N,).
    # Numerically stable implementation that works with large logits
    # Subtract max for stability, but preserve the loss magnitude
    
    # Shift logits for numerical stability
    max_logits = np.max(logits, axis=1, keepdims=True)
    logits_shifted = logits - max_logits
    
    # Compute softmax numerator and denominator in log space
    exp_logits = np.exp(logits_shifted)
    sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
    
    # Compute log probabilities directly (more stable)
    # log(p_i) = log(exp(logits_i - max)) - log(sum(exp(logits - max)))
    # = (logits_i - max) - log(sum_exp)
    N = logits.shape[0]
    log_probs = logits_shifted - np.log(sum_exp)
    
    # Get log probability of true class for each sample
    true_log_probs = log_probs[np.arange(N), labels]
    
    # Cross-entropy loss = -mean(log probability of true class)
    # Note: Don't add epsilon here since we're working in log space
    return -np.mean(true_log_probs)

# Step 6 - model_backward
import numpy as np

def model_backward(params, cache, labels):
    # TODO: return a dict of gradients {'W1','b1','W2','b2'} from cache and labels.
    # Extract from cache
    x = cache['x']
    z1 = cache['z1']
    h1 = cache['h1']
    logits = cache['logits']
    
    N = x.shape[0]
    
    # Extract parameters for shapes
    W2 = params['W2']
    
    # === Gradient of loss w.r.t. logits ===
    # For softmax + cross-entropy, dL/dlogits = softmax(logits) - one_hot(labels)
    probs = softmax(logits)
    # Create one-hot encoding of labels
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(N), labels] = 1.0
    dlogits = (probs - one_hot) / N  # Average over batch
    
    # === Backprop through output linear layer: logits = h1 @ W2 + b2 ===
    # dL/dW2 = h1.T @ dlogits
    # dL/db2 = sum(dlogits, axis=0)
    # dL/dh1 = dlogits @ W2.T
    dW2 = h1.T @ dlogits
    db2 = np.sum(dlogits, axis=0)
    dh1 = dlogits @ W2.T
    
    # === Backprop through ReLU: h1 = relu(z1) ===
    # dL/dz1 = dh1 * (z1 > 0)
    dz1 = dh1 * (z1 > 0)
    
    # === Backprop through input linear layer: z1 = x @ W1 + b1 ===
    # dL/dW1 = x.T @ dz1
    # dL/db1 = sum(dz1, axis=0)
    # (No need for dx as we don't propagate to input)
    dW1 = x.T @ dz1
    db1 = np.sum(dz1, axis=0)
    
    return {
        'W1': dW1,
        'b1': db1,
        'W2': dW2,
        'b2': db2
    }

# Step 7 - init_adamw_state
import numpy as np

def init_adamw_state(params):
    # TODO: Build the AdamW state dict with zeroed first/second moments and t=0.
    m = {}
    v = {}
    
    # For each parameter, create zero-filled arrays with the same shape and dtype
    for key, param in params.items():
        m[key] = np.zeros_like(param)  # First moment (mean)
        v[key] = np.zeros_like(param)  # Second moment (uncentered variance)
    
    return {
        'm': m,
        'v': v,
        't': 0  # Step counter starting at 0
    }

# Step 8 - update_adam_moments
import numpy as np

def update_adam_moments(state, grads, beta1, beta2):
    # TODO: increment state['t'] and update first/second moment EMAs for each param key.
    # Increment timestep
    state['t'] += 1
    
    # Update moments for each parameter
    for key in grads.keys():
        # First moment (mean): m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        state['m'][key] = beta1 * state['m'][key] + (1 - beta1) * grads[key]
        
        # Second moment (uncentered variance): v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        state['v'][key] = beta2 * state['v'][key] + (1 - beta2) * (grads[key] ** 2)
    
    return state

# Step 9 - bias_correct_moments
import numpy as np

def bias_correct_moments(state, beta1, beta2):
    # TODO: return (m_hat, v_hat) dicts with Adam bias-corrected moments at step state['t'].
    t = state['t']
    
    # Compute bias correction factors
    # For first moment: 1 - beta1^t
    # For second moment: 1 - beta2^t
    beta1_t = beta1 ** t
    beta2_t = beta2 ** t
    
    # Create new dictionaries for corrected moments
    m_hat = {}
    v_hat = {}
    
    # Apply bias correction to each parameter
    for key in state['m'].keys():
        # Bias-corrected first moment: m_hat = m / (1 - beta1^t)
        m_hat[key] = state['m'][key] / (1 - beta1_t)
        
        # Bias-corrected second moment: v_hat = v / (1 - beta2^t)
        v_hat[key] = state['v'][key] / (1 - beta2_t)
    
    return m_hat, v_hat

# Step 10 - adam_param_step
import numpy as np

def adam_param_step(params, m_hat, v_hat, lr, eps):
    # TODO: return new params updated by p - lr * m_hat / (sqrt(v_hat) + eps) elementwise.
    new_params = {}
    
    for key in params.keys():
        # Adam update: p_new = p - lr * m_hat / (sqrt(v_hat) + eps)
        # This is the adaptive step of AdamW
        new_params[key] = params[key] - lr * m_hat[key] / (np.sqrt(v_hat[key]) + eps)
    
    return new_params

# Step 11 - decoupled_weight_decay
import numpy as np

def decoupled_weight_decay(params, lr, weight_decay):
    # TODO: return a new params dict where each tensor is shrunk by AdamW's decoupled weight decay factor.
    new_params = {}
    
    for key in params.keys():
        # AdamW decoupled weight decay: p_new = p * (1 - lr * weight_decay)
        # This directly shrinks the parameters, independent of gradients
        decay_factor = 1 - lr * weight_decay
        new_params[key] = params[key] * decay_factor
    
    return new_params

# Step 12 - clone_params
import numpy as np

def clone_params(params):
    # TODO: return a new dict whose values are independent copies of the input arrays.
    cloned = {}
    
    for key, value in params.items():
        # Create a deep copy of each array using np.copy()
        cloned[key] = np.copy(value)
    
    return cloned

# Step 13 - scale_params
import numpy as np

def scale_params(params, scalar):
    # TODO: return a new dict where every array in params is multiplied by scalar.
    scaled = {}
    
    for key, value in params.items():
        # Multiply each array by the scalar and create an independent copy
        scaled[key] = value * scalar
    
    return scaled

# Step 14 - subtract_params
import numpy as np

def subtract_params(params_a, params_b):
    # TODO: return a new dict with params_a[k] - params_b[k] for each key.
    diff = {}
    
    for key in params_a.keys():
        # Element-wise subtraction: a[k] - b[k]
        # Creates a new independent array
        diff[key] = params_a[key] - params_b[key]
    
    return diff

# Step 15 - average_params
import numpy as np

def average_params(params_list):
    # TODO: return a new dict whose value at each key is the element-wise mean across the input dicts.
    # Get the keys from the first dict (assuming all dicts have the same keys)
    keys = params_list[0].keys()
    
    # Initialize the result dictionary
    avg = {}
    
    # For each key, compute the element-wise mean across all dicts
    for key in keys:
        # Stack all arrays for this key along a new axis (axis=0)
        # This creates a 3D array: (num_dicts, *array_shape)
        stacked = np.stack([params[key] for params in params_list], axis=0)
        
        # Compute the mean along the first axis (across dicts)
        avg[key] = np.mean(stacked, axis=0)
    
    return avg

# Step 16 - iid_shard_dataset
import numpy as np

def iid_shard_dataset(x, y, num_workers, seed=0):
    # TODO: partition (x, y) uniformly at random into num_workers disjoint IID shards.
    N = x.shape[0]
    
    # Set seed for reproducibility
    rng = np.random.RandomState(seed)
    
    # Create shuffled indices
    indices = np.arange(N)
    rng.shuffle(indices)
    
    # Shuffle the data using the shuffled indices
    x_shuffled = x[indices]
    y_shuffled = y[indices]
    
    # Determine shard sizes
    base_size = N // num_workers
    remainder = N % num_workers
    
    # Create shards
    shards = []
    start_idx = 0
    
    for worker_id in range(num_workers):
        # Current shard gets base_size + 1 if it's among the first 'remainder' shards
        shard_size = base_size + (1 if worker_id < remainder else 0)
        
        # Extract shard
        x_shard = x_shuffled[start_idx:start_idx + shard_size]
        y_shard = y_shuffled[start_idx:start_idx + shard_size]
        
        shards.append((x_shard, y_shard))
        start_idx += shard_size
    
    return shards

# Step 17 - noniid_shard_dataset
import numpy as np

def noniid_shard_dataset(x, y, num_workers, num_classes, seed=0):
    # TODO: partition (x, y) across workers so each worker owns a distinct subset of classes.
    # Initialize empty lists for each worker's data
    worker_x = [[] for _ in range(num_workers)]
    worker_y = [[] for _ in range(num_workers)]
    
    # Assign classes to workers by round-robin: class c goes to worker c % num_workers
    # Create a mapping from class to worker
    class_to_worker = {}
    for c in range(num_classes):
        worker_id = c % num_workers
        class_to_worker[c] = worker_id
    
    # Group each example by its assigned worker based on its label
    for i in range(len(y)):
        label = y[i]
        worker_id = class_to_worker[label]
        worker_x[worker_id].append(x[i])
        worker_y[worker_id].append(label)
    
    # Convert lists to numpy arrays and shuffle within each worker using seeded RNG
    shards = []
    rng = np.random.RandomState(seed)
    
    for worker_id in range(num_workers):
        # Convert to numpy arrays
        x_arr = np.array(worker_x[worker_id])
        y_arr = np.array(worker_y[worker_id])
        
        # Shuffle within the worker to randomize order (preserving row correspondence)
        if len(x_arr) > 0:
            indices = np.arange(len(x_arr))
            rng.shuffle(indices)
            x_arr = x_arr[indices]
            y_arr = y_arr[indices]
        
        shards.append((x_arr, y_arr))
    
    return shards

# Step 18 - sample_worker_batch
import numpy as np

def sample_worker_batch(x_shard, y_shard, batch_size, rng):
    # TODO: Sample batch_size examples from (x_shard, y_shard) using rng and return (x_batch, y_batch).
    n = x_shard.shape[0]
    
    # Determine if sampling with or without replacement
    replace = batch_size > n
    
    # Sample indices using the provided rng
    indices = rng.choice(n, size=batch_size, replace=replace)
    
    # Select the corresponding examples
    x_batch = x_shard[indices]
    y_batch = y_shard[indices]
    
    return x_batch, y_batch

# Step 19 - local_train_step
import numpy as np

def local_train_step(params, adam_state, x_batch, y_batch, lr, beta1, beta2, eps, weight_decay):
    # TODO: one AdamW update: forward, loss, backward, moment update, param step, weight decay.
    # 1. Forward pass
    logits, cache = model_forward(params, x_batch)
    
    # 2. Compute loss
    loss = cross_entropy_loss(logits, y_batch)
    
    # 3. Backward pass to get gradients
    grads = model_backward(params, cache, y_batch)
    
    # 4. Update AdamW moments (first and second moments)
    adam_state = update_adam_moments(adam_state, grads, beta1, beta2)
    
    # 5. Bias-correct the moments
    m_hat, v_hat = bias_correct_moments(adam_state, beta1, beta2)
    
    # 6. Apply Adam parameter update (adaptive step)
    new_params = adam_param_step(params, m_hat, v_hat, lr, eps)
    
    # 7. Apply decoupled weight decay
    new_params = decoupled_weight_decay(new_params, lr, weight_decay)
    
    return new_params, adam_state, loss

# Step 20 - inner_train_worker
import numpy as np

def inner_train_worker(params, x_shard, y_shard, num_inner_steps, batch_size, lr, beta1, beta2, eps, weight_decay, seed):
    # TODO: run num_inner_steps AdamW updates on this worker's shard from a copy of params
    # Create a private copy of the parameters (do not mutate the original)
    local_params = clone_params(params)
    
    # Initialize AdamW state
    adam_state = init_adamw_state(local_params)
    
    # Create a seeded RNG for this worker
    rng = np.random.default_rng(seed)
    
    total_loss = 0.0
    
    # Run num_inner_steps local updates
    for step in range(num_inner_steps):
        # Sample a mini-batch from the worker's shard
        x_batch, y_batch = sample_worker_batch(x_shard, y_shard, batch_size, rng)
        
        # Take one local training step
        local_params, adam_state, loss = local_train_step(
            local_params, adam_state, x_batch, y_batch,
            lr, beta1, beta2, eps, weight_decay
        )
        
        total_loss += loss
    
    # Compute mean loss across all inner steps
    mean_loss = total_loss / num_inner_steps
    
    return local_params, mean_loss

# Step 21 - init_outer_optimizer
import numpy as np

def init_outer_optimizer(params):
    # TODO: Build outer optimizer state with a zero momentum buffer matching params shapes.
    momentum = {}
    
    # Create zero-filled momentum buffers matching each parameter's shape and dtype
    for key, value in params.items():
        momentum[key] = np.zeros_like(value)
    
    return {'momentum': momentum}

# Step 22 - update_outer_momentum
import numpy as np

def update_outer_momentum(outer_state, outer_grad, momentum_coef):
    """Update Nesterov momentum buffer: m <- momentum_coef * m + outer_grad."""
    # TODO: for each key in outer_state['momentum'], set m[k] = momentum_coef * m[k] + outer_grad[k]
    # Get the momentum buffer from the state
    momentum = outer_state['momentum']
    
    # Update each momentum tensor
    for key in momentum.keys():
        momentum[key] = momentum_coef * momentum[key] + outer_grad[key]
    
    return outer_state

# Step 23 - nesterov_param_update
import numpy as np

def nesterov_param_update(params, outer_state, outer_grad, outer_lr, momentum_coef):
    # TODO: apply the Nesterov look-ahead step using the (already-updated) momentum buffer and outer_grad.
    new_params = {}
    
    for key in params.keys():
        # outer_state is the momentum buffer dict (already updated)
        m = outer_state[key]
        g = outer_grad[key]
        
        # Nesterov look-ahead update:
        # p_new = p - lr * (g + momentum_coef * m)
        # where m is the updated momentum buffer (m_new = momentum_coef * m_old + grad)
        new_params[key] = params[key] - outer_lr * (g + momentum_coef * m)
    
    return new_params

# Step 24 - compute_outer_gradient
import numpy as np

def compute_outer_gradient(global_params, worker_params_list):
    # TODO: return global_params minus the elementwise mean of worker_params_list, key by key.
    # Average the worker parameters
    avg_worker_params = average_params(worker_params_list)
    
    # Compute pseudo-gradient: global_params - average_worker_params
    outer_grad = subtract_params(global_params, avg_worker_params)
    
    return outer_grad

# Step 25 - run_diloco_round (not yet solved)
# TODO: implement

# Step 26 - train_diloco (not yet solved)
# TODO: implement

# Step 27 - train_synchronous_baseline (not yet solved)
# TODO: implement

# Step 28 - evaluate_loss (not yet solved)
# TODO: implement

# Step 29 - classification_accuracy (not yet solved)
# TODO: implement

# Step 30 - communication_savings (not yet solved)
# TODO: implement

