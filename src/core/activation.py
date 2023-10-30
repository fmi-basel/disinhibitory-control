# Parameterized activation functions
# Julian Rossbroich
# 2023

import jax 
import jax.numpy as jnp

from abc import ABC, abstractmethod

EPSILON = 1e-8

class ActivationFunction(ABC):
    """
    Abstract base class for parametrized activation functions.
    """
    
    def __init__(self):
        self._params = []
        self._param_names = []
        
    def add_param(self, value: float, name: str = None):
        self._params.append(value)
        self._param_names.append(str(name))
        return value
    
    @property
    def params(self):
        return dict(zip(self._param_names, self._params))
    
    @abstractmethod
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def deriv(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Derivative of the activation function
        """
        raise NotImplementedError
    
    @abstractmethod
    def inv(self, y: jnp.ndarray) -> jnp.ndarray:
        """
        Inverse of the activation function
        """
        raise NotImplementedError
    
    def inv_lin_taylor(self, y: jnp.ndarray, y0: jnp.ndarray) -> jnp.ndarray:
        """
        Linear approximation of the inverse function using taylor expansion around
        the point y0.
        
        :param y: The output of the function.
        :param y0: The output of the function at the point of linearization.
        :return: Linear approximation of the inverse function.
        """
        intercept, slope = self._get_taylorexp_model(y0)
        return intercept + slope * y
    
    def _get_taylorexp_model(self, y0):
        """
        Calculate a linear model approximating the inverse function using taylor expansion.
        
        :param y0: The output of the function at the point of linearization.
        :return: A tuple (intercept, slope) representing the linear model of the inverse function.
        """
        inv_y0 = self.inv(y0)
        deriv_y0 = self.deriv(inv_y0)
        intercept = inv_y0 - y0 / deriv_y0
        slope = 1 / deriv_y0
        
        return intercept, slope
    
    
class Linear(ActivationFunction):
    """
    Linear activation function.
    """
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x
    
    def deriv(self, x: jnp.ndarray) -> jnp.ndarray:
        return 1
    
    def inv(self, y: jnp.ndarray) -> jnp.ndarray:
        return y
    

class Tanh(ActivationFunction):

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.tanh(x)
    
    def deriv(self, x: jnp.ndarray) -> jnp.ndarray:
        return 1 - jax.nn.tanh(x)**2
    
    def inv(self, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.arctanh(y)


class ReLU(ActivationFunction):
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.relu(x)
    
    def deriv(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(x > 0, 1, 0)
    
    def inv(self, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(y > 0, y, 0)


class Sigmoid(ActivationFunction):
    
    def __init__(self):
        super().__init__()
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.sigmoid(x)
    
    def deriv(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.sigmoid(x) * (1 - jax.nn.sigmoid(x))
    
    def inv(self, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.log(y / (1 - y))


class SoftReLu(ActivationFunction):
    
    def __init__(self, scale: float, sharpness: float, shift: float):
        """
        Soft ReLU function with tunable parameters.
        
        :param scale: Controls the overall scaling of the function.
        :param sharpness: Controls the sharpness or smoothness of the function.
        :param shift: Controls the horizontal shift of the function.
        """
        super().__init__()
        self.scale = self.add_param(scale, 'scale')
        self.sharpness = self.add_param(sharpness, 'sharpness')
        self.shift = self.add_param(shift, 'shift')
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the value of the soft ReLU function for the given input.
        
        :param x: Input to the soft ReLU function.
        :return: Output of the soft ReLU function.
        """
        return self.scale * jnp.logaddexp(self.sharpness * (x - self.shift), 0)
        
    def deriv(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the derivative of the soft ReLU function for the given input.
        
        :param x: Input to the soft ReLU function.
        :return: Derivative of the soft ReLU function.
        """
        gradunfc = lambda x: jnp.sum(self.__call__(x))
        return jax.grad(gradunfc)(x)
        
    def inv(self, y: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the inverse of the soft ReLU function for the given output.
        
        :param y: Output of the soft ReLU function.
        :return: Input that produced the given output.
        """
        
        return (self.shift * self.sharpness + jnp.log(jnp.exp(y / self.scale) - 1)) / self.sharpness
        
    def inv(self, y: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the inverse of the soft ReLU function for the given output.
        
        :param y: Output of the soft ReLU function.
        :return: Input that produced the given output.
        """
        large_y = y / self.scale > 50   # Threshold to switch to linear approximation
                                        # Necessary for numerical stability
        stable_inv = jnp.log(jnp.expm1(y / self.scale))
        return jnp.where(large_y, y / self.scale, self.shift + (1 / self.sharpness) * stable_inv)
