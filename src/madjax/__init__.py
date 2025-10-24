"""The madjax package."""
import jax
import importlib
import itertools
import hashlib
from madjax.phasespace.flat_phase_space_generator import FlatInvertiblePhasespace


class MadJax(object):
    def __init__(self, config_name):
        self.config_name = config_name

        all_processes = importlib.import_module(
            '{}.processes.all_processes'.format(config_name)
        )
        self.parameters = importlib.import_module(
            '{}.model.parameters'.format(config_name)
        )
        self.processes = {
            k: v for k, v in all_processes.__dict__.items() if 'Matrix_' in k
        }

        self.permuted_processes = dict()
        for k, v in self.processes.items():
            PDG_IDs = v.pdg_order
            # Assume that we are dealing with 2 -> N scattering, not 1 -> N decay
            # If that is not the case, then this won't work correctly!
            for initial in itertools.permutations(PDG_IDs[:2]):
                for final in itertools.permutations(PDG_IDs[2:]):
                    if (initial+final) not in self.permuted_processes:
                        self.permuted_processes[initial+final] = dict()
                    self.permuted_processes[initial+final][v.process_id] = v
                    #self.permuted_processes[(initial+final, v.process_id)] = V

    def __hash__(self):
        return int.from_bytes(hashlib.md5(self.config_name.encode()).digest(), 'big')

    def __eq__(self, other):
        return isinstance(other, MadJax) and self.config_name == other.config_name

    def phasespace_generator(self, E_cm, process_name):
        def func(external_parameters):
            parameters = self.parameters.calculate_full_parameters(external_parameters)
            process = self.processes[process_name]()
            external_masses = process.get_external_masses(parameters)
            ps_generator = FlatInvertiblePhasespace(
                external_masses[0],
                external_masses[1],
                beam_Es=(E_cm / 2.0, E_cm / 2.0),
                beam_types=(0, 0),
            )
            # Ensure that E_cm offers enough twice as much energy as necessary
            # to produce the final states
            assert E_cm > sum(external_masses[1]) * 2.0

            return ps_generator
        return func

    
    def jacobian(self, E_cm, process_name, do_jit=True):
        ps = self.phasespace_generator(E_cm,process_name)
        def func(external_parameters, random_variables):
            ps_generator = ps(external_parameters)
            PS_point, jacobian = ps_generator.generateKinematics(E_cm, random_variables)
            return jacobian
        return jax.jit(func) if do_jit else func


    def phasespace_vectors(self, E_cm, process_name):
        ps = self.phasespace_generator(E_cm,process_name)
        def func(external_parameters, random_variables):
            ps_generator = ps(external_parameters)
            ps_point, jacobian = ps_generator.generateKinematics(E_cm, random_variables)
            return jax.numpy.array([v.vector for v in ps_point])
        return func

    def matrix_element(self, E_cm, process_name, return_grad=True, do_jit=True):
        ps = self.phasespace_generator(E_cm,process_name)
        def func(external_parameters, random_variables):
            parameters = self.parameters.calculate_full_parameters(external_parameters)
            ps_generator = ps(external_parameters)
            ps_point, jacobian = ps_generator.generateKinematics(E_cm, random_variables)
            process = self.processes[process_name]()
            return process.smatrix(ps_point, parameters)

        if return_grad:
            return jax.jit(jax.value_and_grad(func)) if do_jit else jax.value_and_grad(func)
        else:
            return jax.jit(func) if do_jit else func


    def matrix_element_and_jacobian(self, E_cm, process_name):
        ps = self.phasespace_generator(E_cm,process_name)
        def func(external_parameters, random_variables):
            parameters = self.parameters.calculate_full_parameters(external_parameters)
            ps_generator = ps(external_parameters)
            ps_point, jacobian = ps_generator.generateKinematics(E_cm, random_variables)
            process = self.processes[process_name]()
            return process.smatrix(ps_point, parameters), jacobian

        return func
