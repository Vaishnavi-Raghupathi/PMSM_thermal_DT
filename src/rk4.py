import numpy as np
from numba import njit
from src.pmsm_ode import pmsm_ode_numba


def rk4_step(ode_func, t, states, inputs, params, dt):
    """
    Single RK4 integration step.

    ode_func : function that returns derivatives [d(id)/dt, d(iq)/dt, d(wm)/dt]
    t        : current time
    states   : current state vector [id, iq, wm]
    inputs   : input vector [Vd, Vq, Tl] at current timestep
    params   : PMSMParameters dataclass
    dt       : timestep size in seconds
    """
    k1 = ode_func(t, states, inputs, params)
    k2 = ode_func(t + 0.5*dt, states + 0.5*dt*k1, inputs, params)
    k3 = ode_func(t + 0.5*dt, states + 0.5*dt*k2, inputs, params)
    k4 = ode_func(t + dt,     states + dt*k3,      inputs, params)

    return states + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)


@njit(cache=True)
def rk4_step_numba(t, states, inputs, params, dt):
    k1 = pmsm_ode_numba(t, states, inputs, params)
    k2 = pmsm_ode_numba(t + 0.5 * dt, states + 0.5 * dt * k1, inputs, params)
    k3 = pmsm_ode_numba(t + 0.5 * dt, states + 0.5 * dt * k2, inputs, params)
    k4 = pmsm_ode_numba(t + dt, states + dt * k3, inputs, params)

    return states + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_simulate(ode_func, initial_states, input_sequence, params, dt):
    """
    Run a full simulation over a sequence of inputs using RK4.

    initial_states  : [id0, iq0, wm0] starting conditions
    input_sequence  : array of shape (N, 3) - [Vd, Vq, Tl] at each timestep
    params          : PMSMParameters dataclass
    dt              : timestep size in seconds

    returns         : array of shape (N, 3) - [id, iq, wm] at each timestep
    """
    N = len(input_sequence)
    states_history = np.zeros((N, 3))
    states = np.array(initial_states, dtype=float)

    for i in range(N):
        states_history[i] = states
        inputs = input_sequence[i]
        states = rk4_step(ode_func, i*dt, states, inputs, params, dt)

    return states_history