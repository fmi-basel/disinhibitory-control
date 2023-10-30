# Simple Dalian Fully Connected / Conv Vector fields
# One-to-one connectivity between exc and inh populations

import jax
import jax.numpy as jnp
from jax import jit

import flax
import flax.linen as nn

#local
from src.core.activation import ActivationFunction
from src.core.vectorfield import VectorField
from src.core.controller import Controller
from typing import Iterable


class FullyConnectedDalianVectorField(VectorField):
    
    # Architecture
    nb_hidden: int
    sizes_hidden: Iterable[int]
    use_bias: bool

    # Dynamics
    actE: ActivationFunction
    actI: ActivationFunction
    controller: Controller
    tauE: float
    tauI: float
    fb_to_readout: bool

    def _get_hidden_sizes(self):
        # Check sizes of hidden layers. 
        # If only one size is given, repeat it nb_hidden times
        if len(self.sizes_hidden) == 1:
            sizes = [self.sizes_hidden[0]] * self.nb_hidden
        else:
            assert len(self.sizes_hidden) == self.nb_hidden, "Number of hidden layers does not match number of hidden sizes"
            sizes = self.sizes_hidden    

        return sizes        
    
    
    def setup(self):
        sizes = self._get_hidden_sizes()
        self.hidden = [nn.Dense(sizes[i], use_bias=self.use_bias) 
                       for i in range(self.nb_hidden)]
        self.readout = nn.Dense(self.dim_output, use_bias=False)

    def _compute_ff_inputs(self, x, state_vf):
        """ Get feedforward inputs to the vector field """
        
        ff_inputs = []
        h = x
        for l in range(self.nb_hidden):
            ff_inputs.append(self.hidden[l](h))
            h = self.actE(state_vf[l]['exc'])
        ff_inputs.append(self.readout(h))
        
        return ff_inputs
    
    def compute_ff_inputs(self, params, x, state_vf):
        return self.apply(params, x, state_vf, method=self._compute_ff_inputs)

    def __call__(self, state, x, y, fb_weights, closedloop):
        """ ODE step for closed-loop dynamics """

        state_vf = state['vf']
        state_ctrl = state['ctrl']

        # Compute FF Inputs
        ff_inputs = self._compute_ff_inputs(x, state_vf)

        # Compute Feedback control
        if closedloop:
            y_pred = self.out(state)
            ctrl, delta_state_ctrl = self.controller(y_pred, y, state_ctrl)
        else:
            ctrl = jnp.zeros(self.dim_output)
            delta_state_ctrl = self.controller.get_initial_state()

        # Update hidden dynamics     
        delta_state_vf = []
        for l in range(self.nb_hidden):
            
            u_exc = state_vf[l]['exc']
            u_inh = state_vf[l]['inh']
            
            if closedloop:
                fb_input = jnp.dot(ctrl, fb_weights[l])
            else:
                fb_input = jnp.zeros_like(u_inh)
            
            delta_exc = 1 / self.tauE * (- u_exc + ff_inputs[l] - self.actI(u_inh))
            delta_inh = 1 / self.tauI * (- u_inh + self.actE(u_exc) - fb_input)
            
            delta_state_vf.append({'exc': delta_exc,
                                   'inh': delta_inh})
        
        # Update readout dynamics
        if self.fb_to_readout:
            delta_readout = 1 / self.tauE * (- state_vf[-1] + ff_inputs[-1] + ctrl)
        else:
            delta_readout = 1 / self.tauE * (- state_vf[-1] + ff_inputs[-1])

        delta_state_vf.append(delta_readout)

        return {'vf': delta_state_vf,
                'ctrl': delta_state_ctrl}
        
        
    def get_initial_state(self, x):
        """ Get initial vf state """
                
        # Hidden layers
        sizes = self._get_hidden_sizes()
        state_vf = []
        for i in range(self.nb_hidden):
            state_vf.append({'exc': jnp.zeros(sizes[i]),
                             'inh': jnp.zeros(sizes[i])})
        
        # Readout
        state_vf.append(jnp.zeros(self.dim_output))

        # Controller
        state_ctrl = self.controller.get_initial_state()

        return {'vf': state_vf,
                'ctrl': state_ctrl}
    
    
    def out(self, state):
        return state['vf'][-1]
    
    
    def calculate_jacobian(self, vf_state):
        """ Calculates the Jacobian at the current VF state """

        hidden_layer_state = vf_state[:-1]
        exc_layer_state = [l['exc'] for l in hidden_layer_state]
        inh_layer_state = [l['inh'] for l in hidden_layer_state]
        exc_derivs = jax.tree_map(self.actE.deriv, exc_layer_state)
        inh_derivs = jax.tree_map(self.actI.deriv, inh_layer_state)

        def surrogate_func(exc_state, exc_derivs):
            y = 0

            # hidden layers
            for l in range(1, self.nb_hidden):
                y = self.hidden[l]((y + exc_state[l-1]) * exc_derivs[l-1])
            
            # readout layer
            y = self.readout((y + exc_state[-1]) * exc_derivs[-1])

            return y
        
        jac = jax.jacrev(surrogate_func)(exc_layer_state, exc_derivs)

        # multiply each element of jac by the corresponding deriv of inh activation
        for l in range(self.nb_hidden):
            jac[l] *= jnp.expand_dims(inh_derivs[l], axis=0)
        
        return jac
    
    def calculate_gradients(self, params, x, vf_state, errors):
        """ 
        Given the parameter dict `params`,
        an input x, a VF state (sol.ys['vf']) 
        and a list of neuron-specific error signals,
        calculate the gradients wrt the weights
        """
        
        surrfunc = lambda p: self.apply(p, x, vf_state, errors, method=self._grads_surrfunc)
        return jax.grad(surrfunc)(params)
    
    def _grads_surrfunc(self, x, vf_state, errors):
        """ 
        Surrogate function for calculating gradients wrt the weights
        """
        
        y = 0
        h = x
        
        # Hidden
        for l in range(self.nb_hidden):
            y += jnp.sum(self.hidden[l](h) * errors[l])
            h = self.actE(vf_state[l]['exc'])
        
        # Readout
        y += jnp.sum(self.readout(h) * errors[-1])    
        
        return y
