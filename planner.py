from math import exp

# Type of planner
POINT_PLANNER=0; TRAJECTORY_PLANNER=1



class planner:
    def __init__(self, type_, shape="parabola"):

        self.type=type_
        self.shape=shape

    
    def plan(self, goalPoint=[-1.0, -1.0]):
        
        if self.type==POINT_PLANNER:
            return self.point_planner(goalPoint)
        
        elif self.type==TRAJECTORY_PLANNER:
            return self.trajectory_planner(self.shape)


    def point_planner(self, goalPoint):
        x = goalPoint[0]
        y = goalPoint[1]
        return x, y

    # TODO Part 6: Implement the trajectories here
    def trajectory_planner(self, shape):

        if shape == "parabola":
            # y = x^2 for x in [0.0, 1.5]
            xs = [i*0.1 for i in range(0, 16)]
            return [[x, x**2] for x in xs]

        elif shape == "sigmoid":
            # y = 2 / (1 + e^(-2x)) - 1 for x in [0.0, 2.5]
            xs = [i*0.1 for i in range(0, 26)]
            return [[x, 2.0 / (1.0 + exp(-2.0*x)) - 1.0] for x in xs]

        else:
            raise ValueError(f"unknown trajectory: {shape}")

