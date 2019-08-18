import numpy as np


def compute_error_for_line_given_points(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(points))


def step_gradient(b_currten, w_current, points, learingRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / N) * (y - (w_current * x) + b_currten)
        w_gradient += -(2 / N) * x * (y - ((w_current * x) + b_currten))
    new_b = b_currten - (learingRate * b_gradient)
    new_w = w_current - (learingRate * w_gradient)
    return [new_b, new_w]


def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iteratoins):
    b = starting_b
    w = starting_w
    for i in range(num_iteratoins):
        b, m = step_gradient(b, w, np.array(points), learning_rate)
    return [b, w]


def run():
    points = np.genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0
    initial_w = 0
    num_iteratoins = 100
    print("Starting gradient descent at b={0}, w={1}, error={2}"
          .format(initial_b, initial_w,
                  compute_error_for_line_given_points(initial_b, initial_w, points))
          )
    print("Runnint")
    [b,w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iteratoins)
    print("After{0}, w={2}, error={3}".
          format(num_iteratoins, b, w, compute_error_for_line_given_points(b, w, points))
          )

if __name__=='__main__'
    run()