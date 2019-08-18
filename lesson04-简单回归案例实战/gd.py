import numpy as np
'''
如果x的变化很小，并且变化值与导数值反号，则函数值下降。对于一元函数，x的变化只有两个方向，要么朝左，要么朝右。
'''

# y = wx + b
def compute_error_for_line_given_points(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # print("x:",x, "y:",y)
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(points))


def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / N) * (y - ((w_current * x) + b_current))  # 因为是所有的点的梯度累加，除以N后就不需要average了
        w_gradient += -(2 / N) * x * (y - ((w_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)  # 怎么理解这一步很重要！！
    new_w = w_current - (learningRate * w_gradient)
    print('new_b:', new_b, 'w:', new_w)
    return [new_b, new_w]


def gradient_descent_runner(points, starting_b, starting_w,
                            learning_rate, num_iterations):  # num_iterations迭代次数
    b = starting_b
    w = starting_w
    for i in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), learning_rate)
    return [b, w]


def run():
    points = np.genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0  # initial y-intercept guess
    initial_w = 0  # initial slope guess
    num_iterations = 100
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}"
          .format(initial_b, initial_w,
                  compute_error_for_line_given_points(initial_b, initial_w, points))
          )
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".
          format(num_iterations, b, m,
                 compute_error_for_line_given_points(b, m, points))
          )


if __name__ == '__main__':
    run()
