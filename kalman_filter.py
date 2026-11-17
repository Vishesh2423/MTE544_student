

import numpy as np



# TODO Part 3: Comment the code explaining each part
# Extended Kalman Filter (EKF) for the TurtleBot4.
#
# State vector (6x1):       x = [x, y, th, w, v, vdot]
#   x, y  : position of the robot in the odom frame [m]
#   th    : heading (yaw) [rad]
#   w     : angular velocity (yaw rate) [rad/s]
#   v     : forward linear velocity in the body frame [m/s]
#   vdot  : forward linear acceleration in the body frame [m/s^2]
#
# Measurement vector (4x1): z = [v, w, ax, ay]
#   v, w   : linear and angular velocity from the wheel odometry (/odom)
#   ax, ay : body-frame linear accelerations from the IMU (/imu)
#
# The motion model f(x) and measurement model h(x) are nonlinear, so the EKF
# linearizes them at every step with their Jacobians A = df/dx and C = dh/dx.
class kalman_filter:

    # TODO Part 3: Initialize the covariances and the states
    def __init__(self, P,Q,R, x, dt):

        self.P = P      # (6x6) covariance of the state estimate, i.e. how uncertain we are about x
        self.Q = Q      # (6x6) process noise covariance: how much we trust the motion model
        self.R = R      # (4x4) measurement noise covariance: how much we trust the sensors
        self.x = x      # (6,)  current state estimate [x, y, th, w, v, vdot]
        self.dt = dt    # time step between filter iterations [s]

    # TODO Part 3: Replace the matrices with Jacobians where needed
    def predict(self):
        # Prediction step: propagate the state and its uncertainty forward one dt
        # using only the motion model (no sensor data yet).

        # Linearize both models around the current (prior) estimate. A and C must be
        # evaluated BEFORE motion_model() overwrites self.x. C is stored here and
        # reused in update().
        self.A = self.jacobian_A()      # (6x6) Jacobian of the motion model, df/dx
        self.C = self.jacobian_H()      # (4x6) Jacobian of the measurement model, dh/dx

        # x = f(x): nonlinear state propagation
        self.motion_model()

        # P = A P A^T + Q: propagate the covariance through the linearized dynamics
        # and add the process noise, so the uncertainty grows during prediction.
        self.P= np.dot( np.dot(self.A, self.P), self.A.T) + self.Q

    # TODO Part 3: Calculate the Kalman gain and correct your estimates
    def update(self, z):
        # Update (correction) step: blend the predicted state with the measurement z.

        # S = C P C^T + R: innovation covariance, the expected uncertainty of the
        # measurement residual (prediction uncertainty mapped to measurement space + sensor noise).
        S = np.dot(np.dot(self.C, self.P), self.C.T) + self.R

        # K = P C^T S^-1: Kalman gain (6x4). It weighs the model against the sensors:
        # a large R gives a small K (trust the model); a large Q, which grows P, gives a large K (trust the sensors).
        kalman_gain= np.dot(np.dot(self.P, self.C.T), np.linalg.inv(S))

        # x = x + K (z - h(x)): correct the state with the innovation, the difference
        # between the actual measurement z and the measurement predicted from the state, h(x).
        self.x= self.x + np.dot(kalman_gain, z - self.measurement_model())

        # P = (I - K C) P: the uncertainty shrinks after including the measurement.
        self.P= np.dot(np.eye(self.A.shape[0]) - np.dot(kalman_gain, self.C), self.P)


    # TODO Part 3: Implement here the measurement model
    def measurement_model(self):
        # h(x): predicts what the sensors should read given the current state.
        x, y, th, w, v, vdot = self.x
        return np.array([
            v,# v    : odometry measures the forward velocity directly
            w,# w    : odometry measures the yaw rate directly
            vdot, # ax : IMU forward (tangential) acceleration = dv/dt
            v * w, # ay : IMU lateral acceleration = centripetal acceleration v*w
        ])

    # TODO Part 3: Impelment the motion model (state-transition matrice)
    def motion_model(self):
        # f(x): discrete-time unicycle kinematics (forward Euler integration over dt).
        # w and vdot are modeled as constant over one step; any change in them
        # is treated as process noise (Q).

        x, y, th, w, v, vdot = self.x
        dt = self.dt

        self.x = np.array([
            x + v * np.cos(th) * dt,    # move forward along the current heading (x component)
            y + v * np.sin(th) * dt,    # move forward along the current heading (y component)
            th + w * dt,                # integrate the yaw rate to get the heading
            w,                          # constant angular velocity
            v  + vdot*dt,               # integrate the acceleration to get the velocity
            vdot,                       # constant linear acceleration
        ])


    def jacobian_A(self):
        # A = df/dx: partial derivatives of the motion model with respect to each state,
        # evaluated at the current estimate. Used to propagate the covariance P.
        x, y, th, w, v, vdot = self.x
        dt = self.dt

        return np.array([
            #x, y,               th, w,             v, vdot
            [1, 0, -v*np.sin(th)*dt, 0, np.cos(th)*dt,  0],    # d(x_next)/d(.)   : depends on th and v (nonlinear)
            [0, 1,  v*np.cos(th)*dt, 0, np.sin(th)*dt,  0],    # d(y_next)/d(.)   : depends on th and v (nonlinear)
            [0, 0,                1, dt,           0,  0],    # d(th_next)/d(.)  = th + w*dt
            [0, 0,                0, 1,            0,  0],    # d(w_next)/d(.)   = w
            [0, 0,                0, 0,            1,  dt],   # d(v_next)/d(.)   = v + vdot*dt
            [0, 0,                0, 0,            0,  1 ]    # d(vdot_next)/d(.) = vdot
        ])

    # TODO Part 3: Implement here the jacobian of the H matrix (measurements)
    def jacobian_H(self):
        # C = dh/dx: partial derivatives of the measurement model with respect to each state.
        # Maps the state uncertainty into measurement space for S and K.
        x, y, th, w, v, vdot=self.x
        return np.array([
            #x,y,th,  w, v,vdot
            [0,0,0  , 0, 1, 0], # v  : d(v)/dv = 1
            [0,0,0  , 1, 0, 0], # w  : d(w)/dw = 1
            [0,0,0  , 0, 0, 1], # ax : d(vdot)/dvdot = 1
            [0,0,0  , v, w, 0], # ay : d(v*w)/dw = v, d(v*w)/dv = w (the only nonlinear measurement)
        ])

    # TODO Part 3: return the states here
    def get_states(self):
        # Returns the current state estimate [x, y, th, w, v, vdot].
        return self.x
