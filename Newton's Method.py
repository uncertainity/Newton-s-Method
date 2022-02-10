import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class newton_rhapson:

    def __init__(self,func_,x_old,iterations):
        self.x_old = x_old
        self.iterations = iterations
        self.func_ = func_

   # def func_(self,x):
   #     return x**2 + 5*x + 6

    def newton(self,x_old,threshold):
        with tf.GradientTape() as tape:
            fx = self.func_(x_old)
        f_x = tape.gradient(fx,x_old)
        if f_x == 0:
            x_new = x_old + np.random.uniform(-1,1)
        else:
            x_new = x_old - fx/f_x
        #print("x old:",x_old)ssss
        #print("fx:",fx)
        #print("f_x:",f_x)
        #print("thr f_x:",2*x_old + 5)
        #print("x new:",x_new)
        return x_new

    def newton_iter(self,x_old,iterations,threshold):
        x_old = tf.Variable(x_old,dtype = float)
        converge = 0
        for i in range(iterations):
            print("Iteration:",i+1)
            x_new = tf.Variable(self.newton(x_old,threshold))
            x_old = x_new
            if abs(self.func_(x_old).numpy()) < threshold:
                print("Convergence Achieved")
                converge = 1
                break
        if converge == 0:
            print("Current fx:",self.func_(x_old))
        print("Current Estimate of root:",x_old.numpy())
        return x_old

    def newton_main(self):
        x_old = self.x_old
        threshold = 1e-4
        iterations = self.iterations
        print("Initial Estimate:",x_old)
        print("No of iterations:",iterations)
        x_old = self.newton_iter(x_old,iterations,threshold)

initial_x = -4.0
iterations = 20

def poly(x):
     return x**2 + 5*x + 6

#obj_newton = newton_rhapson(poly,initial_x,iterations)
#obj_newton.newton_main()



class newton_multi:
    def __init__(self,func_,x1,x2,gamma,iterations):
        self.x1 = x1
        self.x2 = x2
        self.iterations = iterations
        self.func_ = func_
        self.gamma = gamma
        print("Initial Estimate:\n x1:",x1,"  x2:",x2)

    def modified_hessians(self,hessian_):
        ei_hessian_ = np.linalg.eig(hessian_)
        eigval_hessian_ = ei_hessian_[0]
        eigvec_hessian_ = ei_hessian_[1]

        #print("Eigenvalue before modifications:", eigval_hessian_)
        min_eig = np.min(eigval_hessian_)
        #print("minimum eigenvalue:", min_eig)
        if min_eig <= 0:
            eigval_hessian_ -= (min_eig - 1)

        diagonal_inverse = np.diag(1 / eigval_hessian_)
        #print("Eigen value after modifications:", eigval_hessian_)

        inverse_hessian_ = eigvec_hessian_ @ diagonal_inverse @ np.linalg.inv(eigvec_hessian_)
        #print("test check:\n", inverse_hessian_ @ hessian_)
        return inverse_hessian_

    def newton(self,x1,x2):
        x1 = tf.Variable(x1, dtype=float)
        x2 = tf.Variable(x2, dtype=float)
        with tf.GradientTape(persistent=True) as tape:
            with tf.GradientTape(persistent=True) as tape1:
                fx = self.func_(x1, x2)
            delta_fx1 = tape1.gradient(fx, x1)
            delta_fx2 = tape1.gradient(fx, x2)
            #print("Jacobian values")
            #print(delta_fx1)
            #print(delta_fx2)
        delta2_fx1 = tape.gradient(delta_fx1, x1)
        delta2_fx2 = tape.gradient(delta_fx2, x2)
        delta2_fx1fx2 = tape.gradient(delta_fx2, x1)
        if delta2_fx1fx2 is None:
            delta2_fx1fx2 = tf.constant(0.0)
        #print("Hessian Values")
        #print(delta2_fx1)
        #print(delta2_fx2)
        #print(delta2_fx1fx2)
        del tape
        del tape1
        jacobian_ = tf.constant([delta_fx1.numpy(), delta_fx2.numpy()], shape=(2, 1)).numpy()
        hessian_ = tf.constant(
            [[delta2_fx1.numpy(), delta2_fx1fx2.numpy()], [delta2_fx1fx2.numpy(), delta2_fx2.numpy()]]).numpy()
        inverse_hessian_ = self.modified_hessians(hessian_)
        x1_new = x1 - self.gamma * (inverse_hessian_ @ jacobian_)[0, 0]
        x2_new = x2 - self.gamma * (inverse_hessian_ @ jacobian_)[1, 0]
        return x1_new, x2_new, jacobian_

    def newton_iter(self):
        x1 = self.x1
        x2 = self.x2
        iterations = self.iterations
        converge = 0
        threshold = 1e-4
        jacobian_ = -1
        for i in range(iterations):
            #print("Iteration:", i + 1)
            x1_new, x2_new, jacobian_ = self.newton(x1,x2)
            #print("X1:", x1)
            #print("X2:", x2)
            #print("Current Jacobian:\n", jacobian_)
            x1 = x1_new
            x2 = x2_new

            if np.all(abs(jacobian_) < threshold):
                converge = 1
                print("Covergence achieved")
                break

        if converge == 0:
            print("Value of Jacobian:\n", jacobian_)

        print("Minimum value of the func:", self.func_(x1,x2))
        print("x1:", x1.numpy(), "\nx2:", x2.numpy())
        return x1.numpy(),x2.numpy()


def twodim_func(x1,x2):
    return (1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*x2**2)**2 + (2.625 - x1 + x1*x2**3)**2

iterations = 30
x1 = 4
x2 = 1
obj = newton_multi(twodim_func,x1,x2,0.4,iterations)
x1,x2 = obj.newton_iter()


