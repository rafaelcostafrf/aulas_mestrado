import numpy as np
from quadrotor_env import quad, plotter, sensor




a = quad(0.01, 1000, 1, 0)
plot = plotter(a, True)
sens = sensor(a, 0, 0, 0, 0, 0, 0, 0, 0)
a.reset()
done = False
sens.reset()
while not done:
    state, done = a.step(np.array([0, 0, 0, 0]))
    print('Debug')
    q, R = sens.triad()
    print(R)
    print('')
    print(a.mat_rot)
    print('')
    print(sens.accel_int(), a.state[0:6])
    plot.add()
plot.plot()