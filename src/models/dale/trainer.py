import logging

import jax
from jax import vmap
import jax.numpy as jnp

from abc import abstractmethod
from typing import Iterable

# local
from src.core import FeedbackControlTrainer

# Logger
import logging
logger = logging.getLogger(__name__)

EPSILON = 1e-6
    
    
# EXACT TRAINER
#
# This trainer uses the exact inverse of the inhibitory activation function
# to calculate the gradients.
# # # # # # # # # # # # # # # # # # # #


class ExactTrainer(FeedbackControlTrainer):

    def _get_gradients(self, train_state, x, y, OL_y_pred, CL_y_pred, OL_state, CL_state):
    
        actE = self.model.vf.actE
        actI = self.model.vf.actI
        nb_hidden_layers = len(CL_state['vf']) - 1
        ff_inputs = self.model.vf.compute_ff_inputs(train_state.params, x, CL_state['vf'])
        
        errors = []
        
        # HIDDEN LAYERS
        # # # # # # # # # # #
        errors = []
        for l in range(nb_hidden_layers):
            
            # Unpack vectorfield state
            u_exc = CL_state['vf'][l]['exc']
            u_inh = CL_state['vf'][l]['inh']

            # Calculate error
            error = u_inh - actE(u_exc)
            error *= actE.deriv(u_exc)
            errors.append(error)
            
        # READOUT LAYER        
        if self.model.vf.fb_to_readout:
            error = ff_inputs[-1] - CL_state['vf'][-1]
        else:
            error = jax.grad(self.loss, argnums=0)(OL_y_pred, y)
        errors.append(error)
        
        grads = self.model.vf.calculate_gradients(train_state.params, 
                                                  x,
                                                  CL_state['vf'],
                                                  errors)
            
        return grads    


# LINEAR APPROX TRAINER
# # # # # # # # # # # # # # # # # # # #

class LinearApproxTrainer(FeedbackControlTrainer):
    """ Abstract Class for Linear Approximation trainers of all kind """
    
    # Override this method in subclasses
    @abstractmethod
    def approx_inv(self, u_inh, train_state, layer):   
        raise NotImplementedError

    def _get_gradients(self, train_state, x, y, OL_y_pred, CL_y_pred, OL_state, CL_state):
    
        actE = self.model.vf.actE
        actI = self.model.vf.actI
        nb_hidden_layers = len(CL_state['vf']) - 1
        ff_inputs = self.model.vf.compute_ff_inputs(train_state.params, x, CL_state['vf'])

        errors = []
        
        # HIDDEN LAYERS
        # # # # # # # # # # #
        errors = []
        for l in range(nb_hidden_layers):
            
            # Unpack vectorfield state
            u_exc = CL_state['vf'][l]['exc']
            u_inh = CL_state['vf'][l]['inh']

            # Calculate error
            error = (self.approx_inv(u_inh, train_state, layer=l) - actE(u_exc))
            error *= actE.deriv(u_exc)
            errors.append(error)
            
        # READOUT LAYER        
        if self.model.vf.fb_to_readout:
            error = ff_inputs[-1] - CL_state['vf'][-1]
        else:
            error = jax.grad(self.loss, argnums=0)(OL_y_pred, y)
        errors.append(error)
        
        grads = self.model.vf.calculate_gradients(train_state.params, 
                                                  x,
                                                  CL_state['vf'],
                                                  errors)
            
        return grads   
    
    
class TaylorApproxTrainer(LinearApproxTrainer):
    """
    Approximates the inverse function using first-order Taylor expansion
    around a fixed value y0.
    """
    y0: float = 1.0
    
    def approx_inv(self, u_inh, *args, **kwargs):   
        actI = self.model.vf.actI
        return actI.inv_lin_taylor(actI(u_inh), self.y0)
    

class AvgInhTaylorApproxTrainer(LinearApproxTrainer):
    """
    Uses the average inhibition across the minibatch as a linearization point.
    Average inhibition is calculated for each neuron separately.
    """
    avg_inh_lambda: float = 1.0
    avg_inh_init: float = 0.8
        
    def approx_inv(self, hidden_inh, train_state, layer, *args, **kwargs):   
        actI = self.model.vf.actI
        y0 = train_state.avg_inh[layer]
        return actI.inv_lin_taylor(actI(hidden_inh), y0)
    
    class ExtTrainState(FeedbackControlTrainer.ExtTrainState):
        avg_inh: Iterable[jnp.ndarray]

    def init_trainstate_params(self, params):
        """ Initializes the extra parameters of the train state. """
        
        if len(self.model.vf.sizes_hidden) == 1:
            sizes = [self.model.vf.sizes_hidden[0]] * self.model.vf.nb_hidden
        else:
            assert len(self.model.vf.sizes_hidden) == self.model.vf.nb_hidden, "Number of hidden layers does not match number of hidden sizes"
            sizes = self.model.vf.sizes_hidden
            
        avg_inh = [jnp.ones((size,)) * self.avg_inh_init for size in sizes]
        
        return {'avg_inh': avg_inh}
        
    def update_trainstate_params(self, trainstate, ol_sol):
        """ Updates the extra parameters of the train state. """
        
        avg_inh = trainstate.avg_inh
        
        # unpack ol_sol
        for l in range(self.model.vf.nb_hidden):
            actI = self.model.vf.actI
            u_inh = ol_sol.ys['vf'][l]['inh']            
            r_inh = actI(u_inh)        
            avg_inh[l] = avg_inh[l] * (1 - self.avg_inh_lambda) + self.avg_inh_lambda * jnp.mean(r_inh, axis=0)
            
        return {'avg_inh': avg_inh}