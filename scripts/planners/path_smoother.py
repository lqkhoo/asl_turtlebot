import numpy as np
import scipy.interpolate

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    tups = zip(*path)
    x = np.array(tups[0], dtype=np.float)
    y = np.array(tups[1], dtype=np.float)
    N = x.shape[0]

    dx = np.diff(x, n=1)
    dy = np.diff(y, n=1)

    l2 = np.sqrt(dx*dx + dy*dy) # l2 distance between each point.
    dt_ = l2 / V_des            # Time to travel between each segment at nominal speed.
    t = np.zeros(N)
    t[1:] = np.cumsum(dt_)      # Time at each point if travelling at nominal velocity
    t_f = t[-1]                 # Final time
    
    tck, u = scipy.interpolate.splprep([x, y], k=3, s=alpha) # Fit the spline

    subsample = True
    if subsample:
        N_SAMPLES = 512
        u = np.linspace(0, 1, N_SAMPLES)

    ts = u * t_f
    # Hence
    du_dt = 1 / t_f
    d2u_dt2 = 0

    xi, yi = scipy.interpolate.splev(u, tck, der=0)
    thetai = np.arctan2(yi, xi)
    xi_dot, yi_dot = scipy.interpolate.splev(u, tck, der=1)
    xi_dot = xi_dot * du_dt
    yi_dot = yi_dot * du_dt

    xi_ddot, yi_ddot = scipy.interpolate.splev(u, tck, der=2)
    xi_ddot = xi_ddot * d2u_dt2
    yi_ddot = yi_ddot * d2u_dt2
    
    traj_smoothed = np.array([xi, yi, thetai, xi_dot, yi_dot, xi_ddot, yi_ddot]).T
    
    t_smoothed = ts

    ########## Code ends here ##########

    return traj_smoothed, t_smoothed