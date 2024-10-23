import random
import math

simulations = 10000
 
circle_points = 0
square_points = 0

for i in range(simulations):
 
    # Randomly generate (x,y) values from a uniform distribution falling in the "unit square"
    rand_x = random.uniform(-1, 1)
    rand_y = random.uniform(-1, 1)
 
    # Compute distance between (x, y) and the origin with the pythagorean theorem: x^2 + y^2 = r^2
    origin_dist = math.sqrt(rand_x**2 + rand_y**2)
 
    # Checking if (x, y) lies inside the circle
    if origin_dist <= 1:
        circle_points += 1
 
    square_points += 1
 
    # Estimating value of pi,
    # pi = 4*(number of points generated inside the circle)/(number of points generated inside the square)
    pi = 4 * circle_points/ square_points

print(f"Estimation of Pi: {pi}, True Value: {math.pi}, Absolute Error: {abs(pi - math.pi)}")
