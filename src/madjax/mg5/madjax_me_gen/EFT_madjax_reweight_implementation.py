import math
import jax
import jax.numpy
import madjax
import madgraph.interface.reweight_interface as rwgt_interface
import madgraph.various.misc as misc
import madgraph.various.banner as banner
import madgraph.core.diagram_generation as diagram_generation
import madgraph.interface.common_run_interface as common_run_interface
import models.check_param_card as check_param_card
import re
import logging
import time
import shutil
import os
import sys
import itertools
from functools import partial

# Eliminate unnecessary warnings from JAX
logging.getLogger('jax._src.lib.xla_bridge').addFilter(lambda _: False)

pjoin = os.path.join

logger = logging.getLogger('decay.stdout') # -> stdout
#formatter = logging.Formatter('                                                                  %(filename)s:%(lineno)d:    %(message)s')
logger.setLevel(logging.DEBUG)
#handler = logging.StreamHandler()
#handler.setFormatter(formatter)
#logger.addHandler(handler)

jaxlogger = logging.getLogger("jax")
jaxlogger.setLevel(logging.DEBUG)

jax.config.update("jax_compilation_cache_dir", "jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
jax.config.update("jax_compilation_cache_include_metadata_in_key", False)
jax.config.update("jax_explain_cache_misses", True)

import pickle, hashlib

@partial(jax.jit, static_argnames=("other_param_names", "WC_names", "PDG_IDs", "numer"))
@jax.jacrev
@jax.jacfwd
def hess(WCs_plus_zero, fourvectors, helicities, other_params, other_param_names, WC_names, PDG_IDs, numer):
    #my_numerJMs = self.code_numerJMs[PDG_IDs]

    params = {WC_name : WC for WC_name, WC in zip(WC_names, WCs_plus_zero[1:])}
    params.update({other_param_name : other_param for other_param_name, other_param in zip(other_param_names, other_params)})
    #params.update(other_params)
    mod = numer.parameters.calculate_full_parameters(params)
    #mod = self.numer.parameters.calculate_full_parameters(params)
    madjax_vectors = [madjax.phasespace.vectors.LorentzVector(v) for v in fourvectors]
    M = 0
    for procID, JM in numer.permuted_processes[PDG_IDs].items():
        M += JM.s_smatrix(madjax_vectors, mod, [helicities])
    #for k, JMs in numer.permuted_processes.items():
    #    for procID, JM in JMs.items():
    #        M += JM.smatrix(madjax_vectors, mod, [helicities])
    #    #M += jax.lax.cond(jax.numpy.all(k[0] == PDG_IDs), lambda args: JM.smatrix([madjax.phasespace.vectors.LorentzVector(v) for v in args[0]], args[1], [args[2]]), lambda args: 0., (fourvectors, mod, helicities))
    #    #M += JM.smatrix(madjax_vectors, mod, [helicities])
    return jax.numpy.exp(WCs_plus_zero[0]) * M

@partial(jax.jit, static_argnames=("other_param_names", "WC_names", "PDG_IDs", "denom"))
#@jax.jit
def denom(WCs_sampling, fourvectors, helicities, other_params, other_param_names, WC_names, PDG_IDs, denom):
    #my_denomJMs = self.code_denomJMs[PDG_IDs]

    params = {WC_name : WC for WC_name, WC in zip(WC_names, WCs_sampling)}
    params.update({other_param_name : other_param for other_param_name, other_param in zip(other_param_names, other_params)})
    #params.update(other_params)
    mod = denom.parameters.calculate_full_parameters(params)
    #mod = self.denom.parameters.calculate_full_parameters(params)
    madjax_vectors = [madjax.phasespace.vectors.LorentzVector(v) for v in fourvectors]
    M = 0
    for procID, JM in denom.permuted_processes[PDG_IDs].items():
        M += JM.s_smatrix(madjax_vectors, mod, [helicities])
    #for k, JMs in denom.permuted_processes.items():
    #    for procID, JM in JMs.items():
    #        M += JM.smatrix(madjax_vectors, mod, [helicities])
    #    #M += jax.lax.cond(jax.numpy.all(k[0] == PDG_IDs), lambda args: JM.smatrix([madjax.phasespace.vectors.LorentzVector(v) for v in args[0]], args[1], [args[2]]), lambda args: 0., (fourvectors, mod, helicities))
    #    #M += jax.lax.cond(jax.numpy.all(k[0] == PDG_IDs), lambda args: 1, lambda args: 0, (jax.numpy.array(madjax_vectors), mod, helicities))
    #for JM in denomJMs:
    #    M += JM.smatrix(madjax_vectors, mod, [helicities])
    return M

@partial(jax.jit, static_argnames=("other_param_names", "WC_names", "PDG_IDs", "numerMJ", "denomMJ"))
#@jax.jit
def rewgt(WCs_plus_zero, WCs_sampling, fourvectors, helicities, other_params, other_param_names, WC_names, PDG_IDs, numerMJ, denomMJ):
    H = (hess(WCs_plus_zero, fourvectors, helicities, other_params, other_param_names, WC_names, PDG_IDs, numerMJ) /
         denom(WCs_sampling, fourvectors, helicities, other_params, other_param_names, WC_names, PDG_IDs, denomMJ))
    # Average the hessian matrix with its transpose, to even out any
    # differences between the forward and reverse derivatives, and
    # divide the main diagonal, except for the [0,0] element, by 2.
    # Then just return the lower triangular part of the matrix.  This
    # procedure allows us to reproduce the Taylor series correctly
    # without doing anything special.
    H2 = ((H + H.T - jax.numpy.diag(jax.numpy.diag(H)))/2).at[0,0].set(H[0,0])
    return H2[jax.numpy.tril_indices_from(H2)]

#sys.path.append('./rwgt')
##print(sys.path)
##sys.exit()
#from rw_mj_me.model.aloha_methods import *
#from madjax.wavefunctions import *
#
#import collections

#@partial(jax.jit, static_argnames=("other_param_names", "WC_names", "PDG_IDs", "numerMJ", "denomMJ"))
#def rewgt(WCs_plus_zero, WCs_sampling, fourvectors, helicities, other_params, other_param_names, WC_names, PDG_IDs, numerMJ, denomMJ):
#    params = collections.OrderedDict()
#    for WC_name, WC in zip(WC_names, WCs_sampling):
#        params[WC_name] = WC
#    for other_param_name, other_param in zip(other_param_names, other_params):
#        params[other_param_name] = other_param
#    #params = {WC_name : WC for WC_name, WC in zip(WC_names, WCs_sampling)}
#    #params.update({other_param_name : other_param for other_param_name, other_param in zip(other_param_names, other_params)})
#    mod = denomMJ.parameters.calculate_full_parameters(params)
#    madjax_vectors = [madjax.phasespace.vectors.LorentzVector(v) for v in fourvectors]
#    #print(list(mod.keys()))
#    #return WCs_plus_zero + mod['mdl_cuWRe']
#    #print(denomMJ.permuted_processes[PDG_IDs][1])
#    #model = mod
#    #p = madjax_vectors
#    #hel = helicities
#
#    #ngraphs = 20
#    #nexternal = 4
#    #nwavefuncs = 5
#    #ncolor = 3
#    #ZERO = 0.
#    ##  
#    ## Color matrix
#    ##  
#    #denom = [3,3,1.];
#    #cf = [[16,-2,6.],
#    #[-2,16,6],
#    #[2,2,6]];
#    ##
#    ## Model parameters
#    ##
#    #mdl_MH = model["mdl_MH"]
#    #mdl_MT = model["mdl_MT"]
#    #mdl_WH = model["mdl_WH"]
#    #mdl_WT = model["mdl_WT"]
#    #GC_1025 = model["GC_1025"]
#    #GC_31 = model["GC_31"]
#    #GC_32 = model["GC_32"]
#    #GC_347 = model["GC_347"]
#    #GC_348 = model["GC_348"]
#    #GC_351 = model["GC_351"]
#    #GC_352 = model["GC_352"]
#    #GC_385 = model["GC_385"]
#    #GC_386 = model["GC_386"]
#    #GC_478 = model["GC_478"]
#    #GC_6 = model["GC_6"]
#    #GC_7 = model["GC_7"]
#    ## ----------
#    ## Begin code
#    ## ----------
#    #amp = [0] * ngraphs
#    #w = [0] * 16
#    #w[0] = vxxxxx(p[0],ZERO,hel[0],-1)
#    #w[1] = vxxxxx(p[1],ZERO,hel[1],-1)
#    #w[2] = oxxxxx(p[2],mdl_MT,hel[2],+1)
#    #w[3] = ixxxxx(p[3],mdl_MT,hel[3],-1)
#    ## Amplitude(s) for diagram number 1
#    #amp[0]= FFVV3_0(w[3],w[2],w[0],w[1],GC_386)
#    ## Amplitude(s) for diagram number 2
#    #amp[1]= FFVV1_0(w[3],w[2],w[0],w[1],GC_385)
#    #w[4]= VVV3P0_1(w[0],w[1],GC_478,ZERO,ZERO)
#    ## Amplitude(s) for diagram number 3
#    #amp[2]= FFV1_0(w[3],w[2],w[4],GC_6)
#    #w[5]= VVV8P0_1(w[0],w[1],GC_32,ZERO,ZERO)
#    ## Amplitude(s) for diagram number 4
#    #amp[3]= FFV1_0(w[3],w[2],w[5],GC_6)
#    #w[6]= VVV7P0_1(w[0],w[1],GC_31,ZERO,ZERO)
#    ## Amplitude(s) for diagram number 5
#    #amp[4]= FFV1_0(w[3],w[2],w[6],GC_6)
#    #w[7]= VVV5P0_1(w[0],w[1],GC_7,ZERO,ZERO)
#    ## Amplitude(s) for diagram number 6
#    #amp[5]= FFV1_0(w[3],w[2],w[7],GC_6)
#    ## Amplitude(s) for diagram number 7
#    #amp[6]= FFV9_0(w[3],w[2],w[7],GC_352)
#    ## Amplitude(s) for diagram number 8
#    #amp[7]= FFV2_0(w[3],w[2],w[7],GC_351)
#    #w[8]= VVS2_3(w[0],w[1],GC_348,mdl_MH,mdl_WH)
#    ## Amplitude(s) for diagram number 9
#    #amp[8]= FFS2_0(w[3],w[2],w[8],GC_1025)
#    #w[9]= VVS4_3(w[0],w[1],GC_347,mdl_MH,mdl_WH)
#    ## Amplitude(s) for diagram number 10
#    #amp[9]= FFS2_0(w[3],w[2],w[9],GC_1025)
#    #w[10]= FFV1_1(w[2],w[0],GC_6,mdl_MT,mdl_WT)
#    ## Amplitude(s) for diagram number 11
#    #amp[10]= FFV1_0(w[3],w[10],w[1],GC_6)
#    ## Amplitude(s) for diagram number 12
#    #amp[11]= FFV9_0(w[3],w[10],w[1],GC_352)
#    ## Amplitude(s) for diagram number 13
#    #amp[12]= FFV2_0(w[3],w[10],w[1],GC_351)
#    #w[11]= FFV9_1(w[2],w[0],GC_352,mdl_MT,mdl_WT)
#    ## Amplitude(s) for diagram number 14
#    #amp[13]= FFV1_0(w[3],w[11],w[1],GC_6)
#    #w[12]= FFV2_1(w[2],w[0],GC_351,mdl_MT,mdl_WT)
#    ## Amplitude(s) for diagram number 15
#    #amp[14]= FFV1_0(w[3],w[12],w[1],GC_6)
#    #w[13]= FFV1_2(w[3],w[0],GC_6,mdl_MT,mdl_WT)
#    ## Amplitude(s) for diagram number 16
#    #amp[15]= FFV1_0(w[13],w[2],w[1],GC_6)
#    ## Amplitude(s) for diagram number 17
#    #amp[16]= FFV9_0(w[13],w[2],w[1],GC_352)
#    ## Amplitude(s) for diagram number 18
#    #amp[17]= FFV2_0(w[13],w[2],w[1],GC_351)
#    #w[14]= FFV9_2(w[3],w[0],GC_352,mdl_MT,mdl_WT)
#    ## Amplitude(s) for diagram number 19
#    #amp[18]= FFV1_0(w[14],w[2],w[1],GC_6)
#    #w[15]= FFV2_2(w[3],w[0],GC_351,mdl_MT,mdl_WT)
#    ## Amplitude(s) for diagram number 20
#    #amp[19]= FFV1_0(w[15],w[2],w[1],GC_6)
#    #return jax.numpy.array(amp)
#    #return denomMJ.permuted_processes[PDG_IDs][1].s_matrix(madjax_vectors, helicities, mod)
#    #M = 0
#    #for procID, JM in denomMJ.permuted_processes[PDG_IDs].items():
#    #    M += JM.s_smatrix(madjax_vectors, mod, [helicities])
#    #return M

class madjax_EFT:
    def __init__(self, madjax_instance_numerator, madjax_instance_denominator, WC_names=None):
        self.numer = madjax_instance_numerator
        self.denom = madjax_instance_denominator

        self.numerJMs = dict()
        for k, v in self.numer.processes.items():
            V = v()
            # Assume that we are dealing with 2 -> N scattering, not 1 -> N decay
            # If that is not the case, then this won't work correctly!
            PDG_IDs = v.pdg_order
            for initial in itertools.permutations(PDG_IDs[:2]):
                for final in itertools.permutations(PDG_IDs[2:]):
                    self.numerJMs[(initial+final, v.process_id)] = V

        self.denomJMs = dict()
        for k, v in self.denom.processes.items():
            V = v()
            # Assume that we are dealing with 2 -> N scattering, not 1 -> N decay
            # If that is not the case, then this won't work correctly!
            PDG_IDs = v.pdg_order
            for initial in itertools.permutations(PDG_IDs[:2]):
                for final in itertools.permutations(PDG_IDs[2:]):
                    self.denomJMs[(initial+final, v.process_id)] = V

        self.tag_map = dict()
        for k, V in self.numerJMs.items():
            self.tag_map[k[0]] = V.pdg_order

        self.proc_map = dict()
        self.WC_names = WC_names
        self.codes = { 1: 'd',
                      -1: 'dx',
                       2: 'u',
                      -2: 'ux',
                       3: 's',
                      -3: 'sx',
                       4: 'c',
                      -4: 'cx',
                       5: 'b',
                      -5: 'bx',
                       6: 't',
                      -6: 'tx',
                      21: 'g',
                      23: 'z',
                      25: 'h'
                     }

        self.code_numerJMs = dict()
        self.code_denomJMs = dict()

    def set_WC_names(self, WC_names):
        self.WC_names = WC_names
        self.WC_names.sort()

    #def _new_hess(self, PDG_IDs):
        #incoming = ''.join([self.codes[ID] for ID in PDG_IDs[:2]])
        #outgoing = ''.join([self.codes[ID] for ID in PDG_IDs[2:]])
        #code = f'{incoming}_{outgoing}'

        #logger.info(f'Compiling {code}')

        #self.code_numerJMs[tuple(PDG_IDs)] = [v for k, v in self.numerJMs.items()  if k[0] == PDG_IDs]
        #self.code_denomJMs[tuple(PDG_IDs)] = [v for k, v in self.denomJMs.items()  if k[0] == PDG_IDs]
        #my_numerJMs = [v for k, v in self.numerJMs.items() if k[0] == PDG_IDs]
        #my_denomJMs = [v for k, v in self.denomJMs.items() if k[0] == PDG_IDs]



        #self.proc_map[tuple(PDG_IDs)] = (hess, denom, rewgt)

    def __call__(self, WCs, WCs_sampling, event, other_params=dict()):
        flat_PDG_IDs = self.tag_map[tuple(sum(event.get_tag_and_order()[1], start=[]))]
        PDG_IDs = event.get_tag_and_order()[1]
        fourvectors = event.get_momenta([flat_PDG_IDs[:2], flat_PDG_IDs[2:]])
        helicities = event.get_helicity([flat_PDG_IDs[:2], flat_PDG_IDs[2:]])

        boost_pz = sum([p[3] for p in fourvectors[:2]])
        boost_e  = sum([p[0] for p in fourvectors[:2]])

        boost_v = boost_pz / boost_e
        boost_gamma = 1 / math.sqrt(1 - boost_v**2)
        boosted_fourvectors = [[(p[0] - boost_v * p[3]) * boost_gamma,
                                p[1],
                                p[2],
                                (p[3] - boost_v * p[0]) * boost_gamma]
                               for p in fourvectors]

        j_fourvectors = jax.numpy.array(boosted_fourvectors)
        j_helicities = jax.numpy.array(helicities)

        other_params[('sminputs', 3)] = event.aqcd

        other_param_names_list = list(other_params.keys())
        other_param_names_list.sort()
        other_param_values_list = [other_params[name] for name in other_param_names_list]

        print("JSW JSW JSW")
        #print("hash of closure vars:", hashlib.sha256(pickle.dumps(hess.__closure__)).hexdigest())
        #print("hash of closure vars:", hashlib.sha256(pickle.dumps(denom.__closure__)).hexdigest())
        #print("hash of closure vars:", hashlib.sha256(pickle.dumps(rewgt.__closure__)).hexdigest())
        print(hash(tuple(self.WC_names)))
        print(hash(tuple(flat_PDG_IDs)))
        print(hash(self.numer))
        print(hash(self.denom))
        print("jsw jsw jsw")

        #if flat_PDG_IDs not in self.proc_map:
        #    self._new_hess(flat_PDG_IDs)

        #(hess, denom, rewgt) = self.proc_map[flat_PDG_IDs]

        # return hess(jax.numpy.insert(WCs, 0, 0.0), j_fourvectors, j_helicities, other_params)
        # return denom(WCs_sampling, j_fourvectors, j_helicities, other_params)
        #logger.debug(("MJ", WCs_sampling, jax.numpy.array([0.0] + WCs)))
        #logger.debug(("MJ", j_fourvectors, j_helicities, flat_PDG_IDs, PDG_IDs))
        #logger.debug(("MJ numerator", hess(jax.numpy.array([0.0] + WCs), j_fourvectors, j_helicities, other_params, PDG_IDs)[0][0]))
        #logger.debug(("MJ denominator", denom(WCs_sampling, j_fourvectors, j_helicities, other_params, PDG_IDs)))
        return rewgt(
                jax.numpy.array([0.0] + WCs),
                WCs_sampling,
                j_fourvectors,
                j_helicities,
                jax.numpy.array(other_param_values_list),
                tuple(other_param_names_list),
                tuple(self.WC_names),
                tuple(flat_PDG_IDs),
                self.numer,
                self.denom,
                ##tuple(numerJMs),
                ##tuple(denomJMs),
                ##numer_calculate_full_parameters,
                ##denom_calculate_full_parameters,
                )
        #return rewgt(
        #        jax.numpy.array([0.0] + WCs),
        #        WCs_sampling,
        #        j_fourvectors,
        #        j_helicities,
        #        jax.numpy.array(other_param_values_list),
        #        tuple(other_param_names_list),
        #        tuple(self.WC_names),
        #        tuple(flat_PDG_IDs),
        #        self.numer,
        #        self.denom,
        #        #tuple(numerJMs),
        #        #tuple(denomJMs),
        #        #numer_calculate_full_parameters,
        #        #denom_calculate_full_parameters,
        #        )

class EFT_madjax_reweight(rwgt_interface.ReweightInterface):
    @misc.mute_logger()
    def create_standalone_tree_directory(self, data ,second=False):
        """generate the various directory for the weight evaluation"""

        mgcmd = self.mg5cmd
        path_me = data['path']
        # 2. compute the production matrix element -----------------------------
        has_nlo = False
        mgcmd.exec_cmd("set group_subprocesses False")

        if not second:
            logger.info('generating the square matrix element for reweighting')
        else:
            logger.info('generating the square matrix element for reweighting (second model and/or processes)')
        start = time.time()
        commandline=''
        for i,proc in enumerate(data['processes']):
            if '[' not in proc:
                commandline += "add process %s ;" % proc
            else:
                has_nlo = True
                if self.banner.get('run_card','ickkw') == 3:
                    if len(proc) == min([len(p.strip()) for p in data['processes']]):
                        commandline += self.get_LO_definition_from_NLO(proc, self.model)
                    else:
                        commandline += self.get_LO_definition_from_NLO(proc,
                                                     self.model, real_only=True)
                else:
                    commandline += self.get_LO_definition_from_NLO(proc, self.model)

        commandline = commandline.replace('add process', 'generate',1)
        logger.info(commandline)
        try:
            mgcmd.exec_cmd(commandline, precmd=True, errorhandling=False)
        except diagram_generation.NoDiagramException:
            commandline=''
            for proc in data['processes']:
                if '[' not in proc:
                    raise
                # pass to virtsq=
                base, post = proc.split('[',1)
                nlo_order, post = post.split(']',1)
                if '=' not in nlo_order:
                    nlo_order = 'virt=%s' % nlo_order
                elif 'noborn' in nlo_order:
                    nlo_order = nlo_order.replace('noborn', 'virt')
                commandline += "add process %s [%s] %s;" % (base,nlo_order,post)
            commandline = commandline.replace('add process', 'generate',1)
            if commandline:
                logger.info("RETRY with %s", commandline)
                mgcmd.exec_cmd(commandline, precmd=True)
                has_nlo = False
        except Exception as error:
            misc.sprint(type(error))
            raise

        commandline = 'output madjax %s --prefix=int' % pjoin(path_me,data['paths'][0])
        mgcmd.exec_cmd(commandline, precmd=True)
        logger.info('Done %.4g' % (time.time()-start))
        self.has_standalone_dir = True

        # 4. Check MadLoopParam for Loop induced
        if os.path.exists(pjoin(path_me, data['paths'][0], 'Cards', 'MadLoopParams.dat')):
            MLCard = banner.MadLoopParam(pjoin(path_me, data['paths'][0], 'Cards', 'MadLoopParams.dat'))
            MLCard.set('WriteOutFilters', False)
            MLCard.set('UseLoopFilter', False)
            MLCard.set("DoubleCheckHelicityFilter", False)
            MLCard.set("HelicityFilterLevel", 0)
            MLCard.write(pjoin(path_me, data['paths'][0], 'SubProcesses', 'MadLoopParams.dat'),
                         pjoin(path_me, data['paths'][0], 'Cards', 'MadLoopParams.dat'),
                         commentdefault=False)

        if os.path.exists(pjoin(path_me, data['paths'][1], 'Cards', 'MadLoopParams.dat')):
            if self.multicore == 'create':
                print("compile OLP", data['paths'][1])
                # It is potentially unsafe to use several cores, We limit ourself to one for now
                # n_cores = self.mother.options['nb_core']
                n_cores = 1
                misc.compile(['OLP_static'], cwd=pjoin(path_me, data['paths'][1],'SubProcesses'),
                             nb_core=self.mother.options['nb_core'])

        return has_nlo


    @misc.mute_logger()
    def create_standalone_virt_directory(self, data ,second=False):
        """generate the various directory for the weight evaluation"""

        mgcmd = self.mg5cmd
        path_me = data['path']
        # Do not pass here for LO/NLO_tree
        start = time.time()
        commandline=''
        for proc in data['processes']:
            if '[' not in proc:
                pass
            else:
                proc = proc.replace('[', '[ virt=')
                commandline += "add process %s ;" % proc
        commandline = re.sub('@\s*\d+', '', commandline)
        # deactivate golem since it creates troubles
        old_options = dict(mgcmd.options)
        if mgcmd.options['golem']:
            logger.info(" When doing NLO reweighting, MG5aMC cannot use the loop reduction algorithms Golem")
        mgcmd.options['golem'] = None
        commandline = commandline.replace('add process', 'generate',1)
        logger.info(commandline)
        mgcmd.exec_cmd(commandline, precmd=True)
        commandline = 'output madjax %s --prefix=int -f' % pjoin(path_me, data['paths'][1])
        mgcmd.exec_cmd(commandline, precmd=True)

        #put back golem to original value
        mgcmd.options['golem'] = old_options['golem']
        # update make_opts

        if not mgcmd.options['lhapdf']:
            raise Exception("NLO reweighting requires LHAPDF to work correctly")

        # Download LHAPDF SET
        common_run_interface.CommonRunCmd.install_lhapdf_pdfset_static(\
            mgcmd.options['lhapdf'], None, self.banner.run_card.get_lhapdf_id())

    @misc.mute_logger()
    def create_standalone_directory(self, second=False):
        """generate the various directory for the weight evaluation"""

        data={}
        if not second:
            data['paths'] = ['rw_mj_me', 'rw_mj_mevirt']
            # model
            info = self.banner.get('proc_card', 'full_model_line')
            if '-modelname' in info:
                data['mg_names'] = False
            else:
                data['mg_names'] = True
            data['model_name'] = self.banner.get('proc_card', 'model')
            #processes
            data['processes'] = [line[9:].strip() for line in self.banner.proc_card
                     if line.startswith('generate')]
            data['processes'] += [' '.join(line.split()[2:]) for line in self.banner.proc_card
                      if re.search('^\s*add\s+process', line)]
            #object_collector
            #self.id_to_path = {}
            #data['id2path'] = self.id_to_path
        else:
            for key in list(self.f2pylib.keys()):
                if 'rw_mj_me_%s' % self.nb_library in key[0]:
                    del self.f2pylib[key]

            self.nb_library += 1
            data['paths'] = ['rw_mj_me_%s' % self.nb_library, 'rw_mj_mevirt_%s' % self.nb_library]


            # model
            if self.second_model:
                data['mg_names'] = True
                if ' ' in self.second_model:
                    args = self.second_model.split()
                    if '--modelname' in args:
                        data['mg_names'] = False
                    data['model_name'] = args[0]
                else:
                    data['model_name'] = self.second_model
            else:
                data['model_name'] = None
            #processes
            if self.second_process:
                data['processes'] = self.second_process
            else:
                data['processes'] = [line[9:].strip() for line in self.banner.proc_card
                                 if line.startswith('generate')]
                data['processes'] += [' '.join(line.split()[2:])
                                      for line in self.banner.proc_card
                                      if re.search('^\s*add\s+process', line)]
            #object_collector
            #self.id_to_path_second = {}
            #data['id2path'] = self.id_to_path_second

        # 0. clean previous run ------------------------------------------------
        if not self.rwgt_dir:
            path_me = self.me_dir
        else:
            path_me = self.rwgt_dir
        data['path'] = path_me

        for i in range(2):
            pdir = pjoin(path_me,data['paths'][i])
            if os.path.exists(pdir):
                try:
                    shutil.rmtree(pdir)
                except Exception as error:
                    misc.sprint('fail to rm rwgt dir:', error)
                    pass

        # 1. prepare the interface----------------------------------------------
        mgcmd = self.mg5cmd
        complex_mass = False
        has_cms = re.compile(r'''set\s+complex_mass_scheme\s*(True|T|1|true|$|;)''')
        for line in self.banner.proc_card:
            if line.startswith('set'):
                mgcmd.exec_cmd(line, printcmd=False, precmd=False, postcmd=False)
                if has_cms.search(line):
                    complex_mass = True
            elif line.startswith('define'):
                try:
                    mgcmd.exec_cmd(line, printcmd=False, precmd=False, postcmd=False)
                except madgraph.InvalidCmd:
                    pass

        # 1. Load model---------------------------------------------------------
        if  not data['model_name'] and not second:
            raise self.InvalidCmd('Only UFO model can be loaded in this module.')
        elif data['model_name']:
            self.load_model(data['model_name'], data['mg_names'], complex_mass)
            modelpath = self.model.get('modelpath')
            if os.path.basename(modelpath) != mgcmd._curr_model['name']:
                name, restrict = mgcmd._curr_model['name'].rsplit('-',1)
                if os.path.exists(pjoin(os.path.dirname(modelpath),name, 'restrict_%s.dat' % restrict)):
                    modelpath = pjoin(os.path.dirname(modelpath), mgcmd._curr_model['name'])

            commandline="import model %s " % modelpath
            if not data['mg_names']:
                commandline += ' -modelname '
            mgcmd.exec_cmd(commandline)

            #multiparticles
            for name, content in self.banner.get('proc_card', 'multiparticles'):
                try:
                    mgcmd.exec_cmd("define %s = %s" % (name, content))
                except madgraph.InvalidCmd:
                    pass

        if  second and 'tree_path' in self.dedicated_path:
            files.ln(self.dedicated_path['tree_path'], path_me,name=data['paths'][0])
            if 'virtual_path' in self.dedicated_path:
                has_nlo=True
            else:
                has_nlo=False
        else:
            has_nlo = self.create_standalone_tree_directory(data, second)

        if has_nlo and not self.rwgt_mode:
            self.rwgt_mode = ['NLO']

        # 5. create the virtual for NLO reweighting  ---------------------------
        if second and 'virtual_path' in self.dedicated_path:
            files.ln(self.dedicated_path['virtual_path'], path_me, name=data['paths'][1])
        elif has_nlo and 'NLO' in self.rwgt_mode:
            self.create_standalone_virt_directory(data, second)

            if self.multicore == 'create':
                print("compile OLP", data['paths'][1])
                try:
                    misc.compile(['OLP_static'], cwd=pjoin(path_me, data['paths'][1],'SubProcesses'),
                             nb_core=self.mother.options['nb_core'])
                except:
                    misc.compile(['OLP_static'], cwd=pjoin(path_me, data['paths'][1],'SubProcesses'),
                             nb_core=1)
        elif has_nlo and not second and self.rwgt_mode == ['NLO_tree']:
            # We do not have any virtual reweighting to do but we still have to
            #combine the weights.
            #Idea:create a fake directory.
            start = time.time()
            commandline='import model loop_sm;generate g g > e+ ve [virt=QCD]'
            # deactivate golem since it creates troubles
            old_options = dict(mgcmd.options)
            mgcmd.options['golem'] = None
            commandline = commandline.replace('add process', 'generate',1)
            logger.info(commandline)
            mgcmd.exec_cmd(commandline, precmd=True)
            commandline = 'output madjax %s --prefix=int -f' % pjoin(path_me, data['paths'][1])
            mgcmd.exec_cmd(commandline, precmd=True)
            #put back golem to original value
            mgcmd.options['golem'] = old_options['golem']
            # update make_opts
            if not mgcmd.options['lhapdf']:
                raise Exception("NLO_tree reweighting requires LHAPDF to work correctly")

            # Download LHAPDF SET
            common_run_interface.CommonRunCmd.install_lhapdf_pdfset_static(\
                mgcmd.options['lhapdf'], None, self.banner.run_card.get_lhapdf_id())

        # 6. If we need a new model/process-------------------------------------
        if (self.second_model or self.second_process or self.dedicated_path) and not second :
            self.create_standalone_directory(second=True)

        if not second:
            self.has_nlo = has_nlo

    def compile(self):
        pass

    def load_module(self, metag=1):
        if not self.rwgt_dir:
            path_me = self.me_dir
        else:
            path_me = self.rwgt_dir

        self.madjax_objects = {}
        rwgt_dir_possibility =   ['rw_mj_me','rw_mj_me_%s' % self.nb_library,'rw_mj_mevirt','rw_mj_mevirt_%s' % self.nb_library]
        for onedir in rwgt_dir_possibility:
            if not os.path.exists(pjoin(path_me,onedir)):
                continue
            for tag in [2*metag, 2*metag+1]:
                with misc.TMP_variable(sys, 'path', [pjoin(path_me), pjoin(path_me, onedir)] + sys.path):
                    if (onedir,tag) not in self.madjax_objects:
                        self.madjax_objects[(onedir,tag)] = madjax.MadJax(onedir)
                if (self.second_model or self.second_process or self.dedicated_path):
                    break

        with misc.TMP_variable(sys, 'path', [pjoin(path_me)] + sys.path):
            self.madjax_denominator = madjax.MadJax('rw_mj_me')
        if self.second_process:
            with misc.TMP_variable(sys, 'path', [pjoin(path_me)] + sys.path):
                self.madjax_numerator = madjax.MadJax('rw_mj_me_2')
        else:
            self.madjax_numerator = self.madjax_denominator

        self.madjax_EFT = madjax_EFT(self.madjax_numerator, self.madjax_denominator)


    def save_to_pickle(self):
        pass

    def load_from_pickle(self, keep_name=False):
        self.create_standalone_directory()
        self.compile()

    def calculate_weight(self, event):
        if self.has_nlo and self.rwgt_mode != "LO":
            raise NotImplementedError

        event.parse_reweight()
        orig_wgt = event.wgt

        # I guess we don't really have the machinery to handle changing the event kinematics in this reweighting plugin

        hess_tril = self.madjax_EFT(
                self.WCs,
                self.WCs_sampling,
                event,
                self.other_params
                )

        #print(event.reweight_data)
        weights = {'orig': orig_wgt, '': hess_tril[0] * orig_wgt}
        event.reweight_order.extend(self.weight_names)
        event.reweight_data.update(dict(zip(self.weight_names, (hess_tril * orig_wgt).tolist())))
        #print(event.reweight_data)

        return weights












    def handle_param_card(self, model_line, args, type_rwgt):


        if self.rwgt_dir:
            path_me = self.rwgt_dir
        else:
            path_me = self.me_dir

        if self.second_model or self.second_process or self.dedicated_path:
            rw_dir = pjoin(path_me, 'rw_mj_me_%s' % self.nb_library)
        else:
            rw_dir = pjoin(path_me, 'rw_mj_me')

        if not '--keep_card' in args:
            if self.has_nlo and self.rwgt_mode != "LO":
                rwdir_virt = rw_dir.replace('rw_mj_me', 'rw_mj_mevirt')
            with open(pjoin(rw_dir, 'Cards', 'param_card.dat'), 'w') as fsock:
                fsock.write(self.banner['slha'])
            out, cmd = common_run_interface.CommonRunCmd.ask_edit_card_static(cards=['param_card.dat'],
                                   ask=self.ask, pwd=rw_dir, first_cmd=self.stored_line,
                                   write_file=False, return_instance=True
                                   )
            self.stored_line = None
            card = cmd.param_card
            new_card = card.write()
        elif self.new_param_card:
            new_card = self.new_param_card.write()
        else:
            new_card = open(pjoin(rw_dir, 'Cards', 'param_card.dat')).read()

        # check for potential scan in the new card
        pattern_scan = re.compile(r'''^(decay)?[\s\d]*scan''', re.I+re.M)
        param_card_iterator = []
        if pattern_scan.search(new_card):
            try:
                import internal.extended_cmd as extended_internal
                Shell_internal = extended_internal.CmdShell
            except:
                Shell_internal = None
            import madgraph.interface.extended_cmd as extended_cmd
            if not isinstance(self.mother, (extended_cmd.CmdShell, Shell_internal)):
                raise Exception("scan are not allowed on the Web")
            # at least one scan parameter found. create an iterator to go trough the cards
            main_card = check_param_card.ParamCardIterator(new_card)
            if self.options['rwgt_name']:
                self.options['rwgt_name'] = '%s_0' % self.options['rwgt_name']

            param_card_iterator = main_card
            first_card = param_card_iterator.next(autostart=True)
            new_card = first_card.write()
            self.new_param_card = first_card
            #first_card.write(pjoin(rw_dir, 'Cards', 'param_card.dat'))

        # check if "Auto" is present for a width parameter)
        if 'block' not in new_card.lower():
            raise Exception(str(new_card))
        tmp_card = new_card.lower().split('block',1)[1]
        if "auto" in tmp_card:
            if param_card_iterator:
                first_card.write(pjoin(rw_dir, 'Cards', 'param_card.dat'))
            else:
                ff = open(pjoin(rw_dir, 'Cards', 'param_card.dat'),'w')
                ff.write(new_card)
                ff.close()

            self.mother.check_param_card(pjoin(rw_dir, 'Cards', 'param_card.dat'))
            new_card = open(pjoin(rw_dir, 'Cards', 'param_card.dat')).read()


        # Find new tag in the banner and add information if needed
        if 'initrwgt' in self.banner and self.output_type == 'default':
            if 'name=\'mg_reweighting\'' in self.banner['initrwgt']:
                blockpat = re.compile(r'''<weightgroup name=\'mg_reweighting\'\s*weight_name_strategy=\'includeIdInWeightName\'>(?P<text>.*?)</weightgroup>''', re.I+re.M+re.S)
                before, content, after = blockpat.split(self.banner['initrwgt'])
                header_rwgt_other = before + after
                pattern = re.compile('<weight id=\'(?:rwgt_(?P<id>\d+)|(?P<id2>[_\w\-\.]+))(?P<rwgttype>\s*|_\w+)\'>(?P<info>.*?)</weight>', re.S+re.I+re.M)
                mg_rwgt_info = pattern.findall(content)
                maxid = 0
                for k,(i, fulltag, nlotype, diff) in enumerate(mg_rwgt_info):
                    if i:
                        if int(i) > maxid:
                            maxid = int(i)
                        mg_rwgt_info[k] = (i, nlotype, diff) # remove the pointless fulltag tag
                    else:
                        mg_rwgt_info[k] = (fulltag, nlotype, diff) # remove the pointless id tag

                maxid += 1
                rewgtid = maxid
                if self.options['rwgt_name']:
                    #ensure that the entry is not already define if so overwrites it
                    for (i, nlotype, diff) in mg_rwgt_info[:]:
                        for flag in type_rwgt:
                            if 'rwgt_%s' % i == '%s%s' %(self.options['rwgt_name'],flag) or \
                                i == '%s%s' % (self.options['rwgt_name'], flag):
                                    logger.warning("tag %s%s already defines, will replace it", self.options['rwgt_name'],flag)
                                    mg_rwgt_info.remove((i, nlotype, diff))

            else:
                header_rwgt_other = self.banner['initrwgt']
                mg_rwgt_info = []
                rewgtid = 1
        else:
            self.banner['initrwgt']  = ''
            header_rwgt_other = ''
            mg_rwgt_info = []
            rewgtid = 1

        # add the reweighting in the banner information:
        #starts by computing the difference in the cards.
        s_orig = self.banner['slha']
        self.orig_param_card_text = s_orig
        s_new = new_card
        self.new_param_card = check_param_card.ParamCard(s_new.splitlines())

        #define tag for the run
        if self.options['rwgt_name']:
            tag = self.options['rwgt_name']
        else:
            tag = str(rewgtid)

        if 'rwgt_info' in self.options and self.options['rwgt_info']:
            card_diff = self.options['rwgt_info']
            for name in type_rwgt:
                mg_rwgt_info.append((tag, name, self.options['rwgt_info']))
        elif not self.second_model and not self.dedicated_path:
            old_param = check_param_card.ParamCard(s_orig.splitlines())
            new_param =  self.new_param_card
            card_diff = old_param.create_diff(new_param)
            if card_diff == '' and not self.second_process:
                    logger.warning(' REWEIGHTING: original card and new card are identical.')
            try:
                if old_param['sminputs'].get(3).value - new_param['sminputs'].get(3).value > 1e-3 * new_param['sminputs'].get(3).value:
                    logger.warning("We found different value of alpha_s. Note that the value of alpha_s used is the one associate with the event and not the one from the cards.")
            except Exception as error:
                logger.debug("error in check of alphas: %s" % str(error))
                pass #this is a security
            if not self.second_process:
                for name in type_rwgt:
                    mg_rwgt_info.append((tag, name, card_diff))
            else:
                str_proc = "\n change process  ".join([""]+self.second_process)
                for name in type_rwgt:
                    mg_rwgt_info.append((tag, name, str_proc + '\n'+ card_diff))
        else:
            if self.second_model:
                str_info = "change model %s" % self.second_model
            else:
                str_info =''
            if self.second_process:
                str_info += "\n change process  ".join([""]+self.second_process)
            if self.dedicated_path:
                for k,v in self.dedicated_path.items():
                    str_info += "\n change %s %s" % (k,v)
            card_diff = str_info
            str_info += '\n' + s_new
            for name in type_rwgt:
                mg_rwgt_info.append((tag, name, str_info))
        # re-create the banner.
        self.banner['initrwgt'] = header_rwgt_other
        if self.output_type == 'default':
            self.banner['initrwgt'] += '\n<weightgroup name=\'mg_reweighting\' weight_name_strategy=\'includeIdInWeightName\'>\n'
        else:
            self.banner['initrwgt'] += '\n<weightgroup name=\'main\'>\n'
        for tag, rwgttype, diff in mg_rwgt_info:
            if tag.isdigit():
                self.banner['initrwgt'] += '<weight id=\'rwgt_%s%s\'>%s</weight>\n' % \
                                       (tag, rwgttype, diff)
            else:
                self.banner['initrwgt'] += '<weight id=\'%s%s\'>%s</weight>\n' % \
                                       (tag, rwgttype, diff)
        self.banner['initrwgt'] += '\n</weightgroup>\n'
        self.banner['initrwgt'] = self.banner['initrwgt'].replace('\n\n', '\n')


        logger.info('starts to compute weight for events with the following modification to the param_card:')
        logger.info(card_diff.replace('\n','\nKEEP:'))
        try:
            self.run_card = banner.Banner(self.banner).charge_card('run_card')
        except Exception:
            logger.debug('no run card found -- reweight interface')
            self.run_card = None

        if self.options['rwgt_name']:
            tag_name = self.options['rwgt_name']
        else:
            tag_name = 'rwgt_%s' % rewgtid

        # Essentially a copy of ParamCard.create_diff(self, new_card):
        self.diff_params = set()
        self.block_to_pname = dict()
        self.block_to_pname[None] = "SM"
        for blockname, block in old_param.items():
            for param in block:
                assert len(param.lhacode) == 1, "If lhacode has a length different from 1, then I'm not sure what to do.  Contact MadJax developers."
                lhacode = param.lhacode[0]
                value = param.value
                new_value = new_param[blockname].get(lhacode).value
                if not misc.equal(value, new_value, 6, zero_limit=False):
                    self.diff_params.add((blockname, lhacode))

                comment = param.comment
                if comment.strip().startswith('set of param :'):
                    all_var = list(re.findall(r'''[^-]1\*(\w*)\b''', comment))
                elif len(comment.split()) == 1:
                    all_var = [comment.strip()]
                else:
                    split = comment.split()
                    if len(split) == 2:
                        if re.search(r'''\[[A-Z]\]eV\^''', split[1]):
                            all_var = [comment.strip()]
                    elif len(split) >= 2 and split[1].startswith('('):
                        all_var = [split[0].strip()]
                    else:
                        if not blockname.startswith('qnumbers'):
                            logger.debug("Do not recognize information for %s %s : %s",
                                    blockname, lhacode, comment)
                        continue
                assert len(all_var) == 1, "If all_var has a length larger than 1, then I'm not sure what to do.  Contact MadJax developers."
                self.block_to_pname[(blockname, lhacode)] = all_var[0]

        self.diff_params = list(self.diff_params)
        self.diff_params.sort()
        self.old_param = old_param
        self.new_param = new_param

        self.weight_names = []
        self.weight_indices = []

        for indices in zip(*jax.numpy.tril_indices(len(self.diff_params)+1)):
            weight_name = '_'.join([tag_name] + [self.block_to_pname[([None] + self.diff_params)[ind]] for ind in indices])
            self.weight_names.append(weight_name)
            self.weight_indices.append(indices)


        self.madjax_EFT.set_WC_names(self.diff_params)

        self.other_params = {}
        for blockname, block in self.old_param.items():
            for param in block:
                lhacode = param.lhacode[0]
                value = param.value
                if (blockname, lhacode) not in self.diff_params:
                    self.other_params[(blockname, lhacode)] = value

        self.WCs_sampling = [self.old_param[blockname].get(lhacode).value for blockname, lhacode in self.diff_params]
        self.WCs = [self.new_param[blockname].get(lhacode).value for blockname, lhacode in self.diff_params]

        return param_card_iterator, tag_name

import madgraph.iolibs.files as files
class Double_reweight(rwgt_interface.ReweightInterface):
    def __init__(self, *args, **kwargs):
        self.MJ = EFT_madjax_reweight(*args, **kwargs)
        super().__init__(*args, **kwargs)

    #def create_standalone_tree_directory(self, data, second=False):
    #    self.MJ.create_standalone_tree_directory(data, second)
    #    return super().create_standalone_tree_directory(data, second)

    #def create_standalone_virt_directory(self, data, second=False):
    #    self.MJ.create_standalone_virt_directory(data, second)
    #    return super().create_standalone_virt_directory(data, second)

    #def create_standalone_directory(self, data, second=False):
    #    self.MJ.create_standalone_directory(data, second)
    #    return super().create_standalone_directory(data, second)

    #def compile(self):
    #    self.MJ.compile()
    #    return super().compile()

    #def load_module(self, metag=1):
    #    self.MJ.load_module(metag)
    #    return super().load_module(metag)

    #def save_to_pickle(self):
    #    self.MJ.save_to_pickle()
    #    return super().save_to_pickle()

    #def load_from_pickle(self, keep_name=False):
    #    self.MJ.load_from_pickle(keep_name)
    #    return super().load_from_pickle(keep_name)

    #def calculate_weight(self, event):
    #    conventional_weight = super().calculate_weight(event)
    #    MJ_weight = self.MJ.calculate_weight(event)
    #    print(conventional_weight, MJ_weight)
    #    conventional_weight.update(MJ_weight)
    #    return conventional_weight

    #def handle_param_card(self, model_line, args, type_rwgt):
    #    self.MJ.handle_param_card(model_line, args, type_rwgt)
    #    return super().handle_param_card(model_line, args, type_rwgt)

    @misc.mute_logger()
    def do_launch(self, line):
        """end of the configuration launched the code"""
        
        args = self.split_arg(line)
        opts = self.check_launch(args)
        if opts['rwgt_name']:
            self.options['rwgt_name'] = opts['rwgt_name']
            # MJ
            self.MJ.options['rwgt_name'] = opts['rwgt_name']
            # END MJ
        if opts['rwgt_info']:
            self.options['rwgt_info'] = opts['rwgt_info']
            # MJ
            self.MJ.options['rwgt_info'] = opts['rwgt_info']
            # END MJ
        model_line = self.banner.get('proc_card', 'full_model_line')

        if not self.has_standalone_dir:                           
            if self.rwgt_dir and os.path.exists(pjoin(self.rwgt_dir,'rw_me','rwgt.pkl')):
                self.load_from_pickle()
                if opts['rwgt_name']:
                    self.options['rwgt_name'] = opts['rwgt_name']
                if not self.rwgt_dir:
                    self.me_dir = self.rwgt_dir
                self.load_module()       # load the fortran information from the f2py module
            elif self.multicore == 'wait':
                i=0
                while not os.path.exists(pjoin(self.me_dir,'rw_me','rwgt.pkl')):
                    time.sleep(10+i)
                    i+=5
                    print('wait for pickle')                  
                print("loading from pickle")
                if not self.rwgt_dir:
                    self.rwgt_dir = self.me_dir
                self.load_from_pickle(keep_name=True)
                self.load_module()
            else:
                self.create_standalone_directory()
                self.compile()
                self.load_module()  
                if self.multicore == 'create':
                    self.load_module()
                    if not self.rwgt_dir:
                        self.rwgt_dir = self.me_dir
                    self.save_to_pickle()      

        # MJ
        if not self.MJ.has_standalone_dir:                           
            if self.MJ.rwgt_dir and os.path.exists(pjoin(self.MJ.rwgt_dir,'rw_me','rwgt.pkl')):
                self.MJ.load_from_pickle()
                if opts['rwgt_name']:
                    self.MJ.options['rwgt_name'] = opts['rwgt_name']
                if not self.MJ.rwgt_dir:
                    self.MJ.me_dir = self.MJ.rwgt_dir
                self.MJ.load_module()       # load the fortran information from the f2py module
            elif self.MJ.multicore == 'wait':
                i=0
                while not os.path.exists(pjoin(self.MJ.me_dir,'rw_me','rwgt.pkl')):
                    time.sleep(10+i)
                    i+=5
                    print('wait for pickle')                  
                print("loading from pickle")
                if not self.MJ.rwgt_dir:
                    self.MJ.rwgt_dir = self.MJ.me_dir
                self.MJ.load_from_pickle(keep_name=True)
                self.MJ.load_module()
            else:
                self.MJ.create_standalone_directory()
                self.MJ.compile()
                self.MJ.load_module()  
                if self.MJ.multicore == 'create':
                    self.MJ.load_module()
                    if not self.MJ.rwgt_dir:
                        self.MJ.rwgt_dir = self.MJ.me_dir
                    self.MJ.save_to_pickle()      
        # END MJ
        
        # get the mode of reweighting #LO/NLO/NLO_tree/...
        type_rwgt = self.get_weight_names()

        # get iterator over param_card and the name associated to the current reweighting.
        param_card_iterator, tag_name = self.handle_param_card(model_line, args, type_rwgt)
        
        if self.rwgt_dir:
            path_me =self.rwgt_dir
        else:
            path_me = self.me_dir 
            
        if self.second_model or self.second_process or self.dedicated_path:
            rw_dir = pjoin(path_me, 'rw_me_%s' % self.nb_library)
        else:
            rw_dir = pjoin(path_me, 'rw_me')
                
        start = time.time()
        # initialize the collector for the various re-weighting
        cross, ratio, ratio_square,error = {},{},{}, {}
        for name in type_rwgt + ['orig']:
            cross[name], error[name] = 0.,0.
            ratio[name],ratio_square[name] = 0., 0.# to compute the variance and associate error

        if self.output_type == "default":
            output = open( self.lhe_input.path +'rw', 'w')
            #write the banner to the output file
            self.banner.write(output, close_tag=False)
        else:
            output = {}
            if tag_name.isdigit():
                name_tag= 'rwgt_%s' % tag_name
            else:
                name_tag = tag_name
            base = os.path.dirname(self.lhe_input.name)
            for rwgttype in  type_rwgt:
                output[(name_tag,rwgttype)] = lhe_parser.EventFile(pjoin(base,'rwgt_events%s_%s.lhe.gz' %(rwgttype,tag_name)), 'w')
                #write the banner to the output file
                self.banner.write(output[(name_tag,rwgttype)], close_tag=False)
                
        if self.lhe_input.closed:
            self.lhe_input = lhe_parser.EventFile(self.lhe_input.name)

        self.lhe_input.seek(0)
        for event_nb,event in enumerate(self.lhe_input):
            #control logger
            if (event_nb % max(int(10**int(math.log10(float(event_nb)+1))),10)==0): 
                    running_time = misc.format_timer(time.time()-start)
                    logger.info('Event nb %s %s' % (event_nb, running_time))
            if (event_nb==10001): logger.info('reducing number of print status. Next status update in 10000 events')
            if (event_nb==100001): logger.info('reducing number of print status. Next status update in 100000 events')


                
            weight = self.calculate_weight(event)
            # MJ
            MJ_weight = self.MJ.calculate_weight(event)
            #weight.update(MJ_weight)
            # END MJ
            if not isinstance(weight, dict):
                weight = {'':weight}
            
            for name in weight:
                cross[name] += weight[name]
                ratio[name] += weight[name]/event.wgt
                ratio_square[name] += (weight[name]/event.wgt)**2

            # ensure to have a consistent order of the weights. new one are put 
            # at the back, remove old position if already defines
            for tag in type_rwgt:
                try:
                    event.reweight_order.remove('%s%s'  % (tag_name,tag))
                except ValueError:
                    continue
            
            event.reweight_order += ['%s%s' % (tag_name,name) for name in type_rwgt]  
            if self.output_type == "default":
                for name in weight:
                    if 'orig' in name:
                        continue             
                    event.reweight_data['%s%s' % (tag_name,name)] = weight[name]
                    #write this event with weight
                output.write(str(event))
            else:
                for i,name in enumerate(weight):
                    if 'orig' in name:
                        continue 
                    if weight[name] == 0:
                        continue
                    new_evt = lhe_parser.Event(str(event))
                    new_evt.wgt = weight[name]
                    new_evt.parse_reweight()
                    new_evt.reweight_data = {}  
                    output[(tag_name,name)].write(str(new_evt))
            #print(event.reweight_data)
            #logger.debug(f"{float(event.reweight_data['MJEFT_SM_SM'])} {float(event.reweight_data['mjeft'])} {float(event.reweight_data['MJEFT_SM_SM']) / float(event.reweight_data['mjeft'])}")

        # check normalisation of the events:
        if self.run_card and 'event_norm' in self.run_card:
            if self.run_card['event_norm'] in ['average','bias']:
                for key, value in cross.items():
                    cross[key] = value / (event_nb+1)
                
        running_time = misc.format_timer(time.time()-start)
        logger.info('All event done  (nb_event: %s) %s' % (event_nb+1, running_time))        
        
        
        if self.output_type == "default":
            output.write('</LesHouchesEvents>\n')
            output.close()
        else:
            for key in output:
                output[key].write('</LesHouchesEvents>\n')
                output[key].close()
                if self.systematics and len(output) ==1:
                    try:
                        logger.info('running systematics computation')
                        import madgraph.various.systematics as syst
                        
                        if not isinstance(self.systematics, bool):
                            args = [output[key].name, output[key].name] + self.systematics
                        else:
                            args = [output[key].name, output[key].name]
                        if self.mother and self.mother.options['lhapdf']:
                            args.append('--lhapdf_config=%s' % self.mother.options['lhapdf'])
                        syst.call_systematics(args, result=open('rwg_syst_%s.result' % key[0],'w'),
                                              log=logger.info)
                    except Exception:
                        logger.error('fail to add systematics')
                        raise
        # add output information        
        if self.mother and hasattr(self.mother, 'results'):
            run_name = self.mother.run_name
            results = self.mother.results
            results.add_run(run_name, self.run_card, current=True)
            results.add_detail('nb_event', event_nb+1)
            name = type_rwgt[0]
            results.add_detail('cross', cross[name])
            event_nb +=1
            for name in type_rwgt:
                variance = ratio_square[name]/event_nb - (ratio[name]/event_nb)**2
                orig_cross, orig_error = self.orig_cross
                error[name] = math.sqrt(max(0,variance/math.sqrt(event_nb))) * orig_cross + ratio[name]/event_nb * orig_error
            results.add_detail('error', error[type_rwgt[0]])
            import madgraph.interface.madevent_interface as ME_interface

        self.lhe_input.close()
        if not self.mother:
            name, ext = self.lhe_input.name.rsplit('.',1)
            target = '%s_out.%s' % (name, ext)            
        elif self.output_type != "default" :
            target = pjoin(self.mother.me_dir, 'Events', run_name, 'events.lhe')
        else:
            target = self.lhe_input.name
        
        if self.output_type == "default":
            files.mv(output.name, target)
            logger.info('Event %s have now the additional weight' % self.lhe_input.name)
        elif self.output_type == "unweight":
            for key in output:
                #output[key].write('</LesHouchesEvents>\n')
                #output.close()
                lhe = lhe_parser.EventFile(output[key].name)
                nb_event = lhe.unweight(target)
                if self.mother and  hasattr(self.mother, 'results'):
                    results = self.mother.results
                    results.add_detail('nb_event', nb_event)
                    results.current.parton.append('lhe')
                logger.info('Event %s is now unweighted under the new theory: %s(%s)' % (lhe.name, target, nb_event))                
        else:
            if self.mother and  hasattr(self.mother, 'results'):
                results = self.mother.results
                results.current.parton.append('lhe')       
            logger.info('Eventfiles is/are now created with new central weight')
        
        if self.multicore != 'create':
            for name in cross:
                if name == 'orig':
                    continue
                logger.info('new cross-section is %s: %g pb (indicative error: %g pb)' %\
                        ('(%s)' %name if name else '',cross[name], error[name]))
            
        self.terminate_fortran_executables(new_card_only=True)
        # MJ
        self.MJ.terminate_fortran_executables(new_card_only=True)
        # END MJ
        #store result
        for name in cross:
            if name == 'orig':
                self.all_cross_section[name] = (cross[name], error[name])
            else:
                self.all_cross_section[(tag_name,name)] = (cross[name], error[name])

        # perform the scanning
        if param_card_iterator:
            if self.options['rwgt_name']:
                reweight_name = self.options['rwgt_name'].rsplit('_',1)[0] # to avoid side effect during the scan
            else:
                reweight_name = None
            for i,card in enumerate(param_card_iterator):
                if reweight_name:
                    self.options['rwgt_name'] = '%s_%s' % (reweight_name, i+1)
                self.new_param_card = card
                #card.write(pjoin(rw_dir, 'Cards', 'param_card.dat'))
                self.exec_cmd("launch --keep_card", printcmd=False, precmd=True)
        
        self.options['rwgt_name'] = None

    def handle_param_card(self, model_line, args, type_rwgt):

        if self.rwgt_dir:
            path_me = self.rwgt_dir
        else:
            path_me = self.me_dir

        if self.second_model or self.second_process or self.dedicated_path:
            rw_dir    = pjoin(path_me, 'rw_me_%s'    % self.nb_library)
            MJ_rw_dir = pjoin(path_me, 'rw_mj_me_%s' % self.nb_library)
        else:
            rw_dir    = pjoin(path_me, 'rw_me')
            MJ_rw_dir = pjoin(path_me, 'rw_mj_me')

        if not '--keep_card' in args:
            if self.has_nlo and self.rwgt_mode != "LO":
                rwdir_virt = rw_dir.replace('rw_me', 'rw_mevirt')
                MJ_rwdir_virt = MJ_rw_dir.replace('rw_mj_me', 'rw_mj_mevirt')
            with open(pjoin(rw_dir, 'Cards', 'param_card.dat'), 'w') as fsock:
                fsock.write(self.banner['slha'])
            with open(pjoin(MJ_rw_dir, 'Cards', 'param_card.dat'), 'w') as fsock:
                fsock.write(self.banner['slha'])
            out, cmd = common_run_interface.CommonRunCmd.ask_edit_card_static(cards=['param_card.dat'],
                                   ask=self.ask, pwd=rw_dir, first_cmd=self.stored_line,
                                   write_file=False, return_instance=True
                                   )
            self.stored_line = None
            card = cmd.param_card
            new_card = card.write()
        elif self.new_param_card:
            new_card = self.new_param_card.write()
        else:
            new_card = open(pjoin(rw_dir, 'Cards', 'param_card.dat')).read()

        # check for potential scan in the new card
        pattern_scan = re.compile(r'''^(decay)?[\s\d]*scan''', re.I+re.M)
        param_card_iterator = []
        if pattern_scan.search(new_card):
            try:
                import internal.extended_cmd as extended_internal
                Shell_internal = extended_internal.CmdShell
            except:
                Shell_internal = None
            import madgraph.interface.extended_cmd as extended_cmd
            if not isinstance(self.mother, (extended_cmd.CmdShell, Shell_internal)):
                raise Exception("scan are not allowed on the Web")
            # at least one scan parameter found. create an iterator to go trough the cards
            main_card = check_param_card.ParamCardIterator(new_card)
            if self.options['rwgt_name']:
                self.options['rwgt_name'] = '%s_0' % self.options['rwgt_name']
                self.MJ.options['rwgt_name'] = '%s_0' % self.options['rwgt_name']

            param_card_iterator = main_card
            first_card = param_card_iterator.next(autostart=True)
            new_card = first_card.write()
            self.new_param_card = first_card
            self.MJ.new_param_card = first_card
            #first_card.write(pjoin(rw_dir, 'Cards', 'param_card.dat'))

        # check if "Auto" is present for a width parameter)
        if 'block' not in new_card.lower():
            raise Exception(str(new_card))
        tmp_card = new_card.lower().split('block',1)[1]
        if "auto" in tmp_card:
            if param_card_iterator:
                first_card.write(pjoin(rw_dir, 'Cards', 'param_card.dat'))
                first_card.write(pjoin(MJ_rw_dir, 'Cards', 'param_card.dat'))
            else:
                ff = open(pjoin(rw_dir, 'Cards', 'param_card.dat'),'w')
                ff.write(new_card)
                ff.close()
                ff = open(pjoin(MJ_rw_dir, 'Cards', 'param_card.dat'),'w')
                ff.write(new_card)
                ff.close()

            self.mother.check_param_card(pjoin(rw_dir, 'Cards', 'param_card.dat'))
            self.MJ.mother.check_param_card(pjoin(MJ_rw_dir, 'Cards', 'param_card.dat'))
            new_card = open(pjoin(rw_dir, 'Cards', 'param_card.dat')).read()


        # Find new tag in the banner and add information if needed
        if 'initrwgt' in self.banner and self.output_type == 'default':
            if 'name=\'mg_reweighting\'' in self.banner['initrwgt']:
                blockpat = re.compile(r'''<weightgroup name=\'mg_reweighting\'\s*weight_name_strategy=\'includeIdInWeightName\'>(?P<text>.*?)</weightgroup>''', re.I+re.M+re.S)
                before, content, after = blockpat.split(self.banner['initrwgt'])
                header_rwgt_other = before + after
                pattern = re.compile('<weight id=\'(?:rwgt_(?P<id>\d+)|(?P<id2>[_\w\-\.]+))(?P<rwgttype>\s*|_\w+)\'>(?P<info>.*?)</weight>', re.S+re.I+re.M)
                mg_rwgt_info = pattern.findall(content)
                maxid = 0
                for k,(i, fulltag, nlotype, diff) in enumerate(mg_rwgt_info):
                    if i:
                        if int(i) > maxid:
                            maxid = int(i)
                        mg_rwgt_info[k] = (i, nlotype, diff) # remove the pointless fulltag tag
                    else:
                        mg_rwgt_info[k] = (fulltag, nlotype, diff) # remove the pointless id tag

                maxid += 1
                rewgtid = maxid
                if self.options['rwgt_name']:
                    #ensure that the entry is not already define if so overwrites it
                    for (i, nlotype, diff) in mg_rwgt_info[:]:
                        for flag in type_rwgt:
                            if 'rwgt_%s' % i == '%s%s' %(self.options['rwgt_name'],flag) or \
                                i == '%s%s' % (self.options['rwgt_name'], flag):
                                    logger.warning("tag %s%s already defines, will replace it", self.options['rwgt_name'],flag)
                                    mg_rwgt_info.remove((i, nlotype, diff))

            else:
                header_rwgt_other = self.banner['initrwgt']
                mg_rwgt_info = []
                rewgtid = 1
        else:
            self.banner['initrwgt']  = ''
            self.MJ.banner['initrwgt']  = ''
            header_rwgt_other = ''
            mg_rwgt_info = []
            rewgtid = 1

        # add the reweighting in the banner information:
        #starts by computing the difference in the cards.
        s_orig = self.banner['slha']
        self.orig_param_card_text = s_orig
        self.MJ.orig_param_card_text = s_orig
        s_new = new_card
        self.new_param_card = check_param_card.ParamCard(s_new.splitlines())
        self.MJ.new_param_card = check_param_card.ParamCard(s_new.splitlines())

        #define tag for the run
        if self.options['rwgt_name']:
            tag = self.options['rwgt_name']
        else:
            tag = str(rewgtid)

        if 'rwgt_info' in self.options and self.options['rwgt_info']:
            card_diff = self.options['rwgt_info']
            for name in type_rwgt:
                mg_rwgt_info.append((tag, name, self.options['rwgt_info']))
        elif not self.second_model and not self.dedicated_path:
            old_param = check_param_card.ParamCard(s_orig.splitlines())
            new_param =  self.new_param_card
            card_diff = old_param.create_diff(new_param)
            if card_diff == '' and not self.second_process:
                    logger.warning(' REWEIGHTING: original card and new card are identical.')
            try:
                if old_param['sminputs'].get(3).value - new_param['sminputs'].get(3).value > 1e-3 * new_param['sminputs'].get(3).value:
                    logger.warning("We found different value of alpha_s. Note that the value of alpha_s used is the one associate with the event and not the one from the cards.")
            except Exception as error:
                logger.debug("error in check of alphas: %s" % str(error))
                pass #this is a security
            if not self.second_process:
                for name in type_rwgt:
                    mg_rwgt_info.append((tag, name, card_diff))
            else:
                str_proc = "\n change process  ".join([""]+self.second_process)
                for name in type_rwgt:
                    mg_rwgt_info.append((tag, name, str_proc + '\n'+ card_diff))
        else:
            if self.second_model:
                str_info = "change model %s" % self.second_model
            else:
                str_info =''
            if self.second_process:
                str_info += "\n change process  ".join([""]+self.second_process)
            if self.dedicated_path:
                for k,v in self.dedicated_path.items():
                    str_info += "\n change %s %s" % (k,v)
            card_diff = str_info
            str_info += '\n' + s_new
            for name in type_rwgt:
                mg_rwgt_info.append((tag, name, str_info))
        # re-create the banner.
        self.banner['initrwgt'] = header_rwgt_other
        self.MJ.banner['initrwgt'] = header_rwgt_other
        if self.output_type == 'default':
            self.banner['initrwgt'] += '\n<weightgroup name=\'mg_reweighting\' weight_name_strategy=\'includeIdInWeightName\'>\n'
            self.MJ.banner['initrwgt'] += '\n<weightgroup name=\'mg_reweighting\' weight_name_strategy=\'includeIdInWeightName\'>\n'
        else:
            self.banner['initrwgt'] += '\n<weightgroup name=\'main\'>\n'
            self.MJ.banner['initrwgt'] += '\n<weightgroup name=\'main\'>\n'
        for tag, rwgttype, diff in mg_rwgt_info:
            if tag.isdigit():
                self.banner['initrwgt'] += '<weight id=\'rwgt_%s%s\'>%s</weight>\n' % \
                                       (tag, rwgttype, diff)
                self.MJ.banner['initrwgt'] += '<weight id=\'rwgt_%s%s\'>%s</weight>\n' % \
                                       (tag, rwgttype, diff)
            else:
                self.banner['initrwgt'] += '<weight id=\'%s%s\'>%s</weight>\n' % \
                                       (tag, rwgttype, diff)
                self.MJ.banner['initrwgt'] += '<weight id=\'%s%s\'>%s</weight>\n' % \
                                       (tag, rwgttype, diff)
        self.banner['initrwgt'] += '\n</weightgroup>\n'
        self.MJ.banner['initrwgt'] += '\n</weightgroup>\n'
        self.banner['initrwgt'] = self.banner['initrwgt'].replace('\n\n', '\n')
        self.MJ.banner['initrwgt'] = self.MJ.banner['initrwgt'].replace('\n\n', '\n')


        logger.info('starts to compute weight for events with the following modification to the param_card:')
        logger.info(card_diff.replace('\n','\nKEEP:'))
        try:
            self.run_card = banner.Banner(self.banner).charge_card('run_card')
            self.MJ.run_card = banner.Banner(self.MJ.banner).charge_card('run_card')
        except Exception:
            logger.debug('no run card found -- reweight interface')
            self.run_card = None
            self.MJ.run_card = None

        if self.options['rwgt_name']:
            tag_name = self.options['rwgt_name'].lower()
        else:
            tag_name = 'rwgt_%s' % rewgtid

        # Essentially a copy of ParamCard.create_diff(self, new_card):
        self.MJ.diff_params = set()
        self.MJ.block_to_pname = dict()
        self.MJ.block_to_pname[None] = "SM"
        for blockname, block in old_param.items():
            for param in block:
                assert len(param.lhacode) == 1, "If lhacode has a length different from 1, then I'm not sure what to do.  Contact MadJax developers."
                lhacode = param.lhacode[0]
                value = param.value
                new_value = new_param[blockname].get(lhacode).value
                if not misc.equal(value, new_value, 6, zero_limit=False):
                    self.MJ.diff_params.add((blockname, lhacode))

                comment = param.comment
                if comment.strip().startswith('set of param :'):
                    all_var = list(re.findall(r'''[^-]1\*(\w*)\b''', comment))
                elif len(comment.split()) == 1:
                    all_var = [comment.strip()]
                else:
                    split = comment.split()
                    if len(split) == 2:
                        if re.search(r'''\[[A-Z]\]eV\^''', split[1]):
                            all_var = [comment.strip()]
                    elif len(split) >= 2 and split[1].startswith('('):
                        all_var = [split[0].strip()]
                    else:
                        if not blockname.startswith('qnumbers'):
                            logger.debug("Do not recognize information for %s %s : %s",
                                    blockname, lhacode, comment)
                        continue
                assert len(all_var) == 1, "If all_var has a length larger than 1, then I'm not sure what to do.  Contact MadJax developers."
                self.MJ.block_to_pname[(blockname, lhacode)] = all_var[0]

        self.MJ.diff_params = list(self.MJ.diff_params)
        self.MJ.old_param = old_param
        self.MJ.new_param = new_param

        self.MJ.weight_names = []
        self.MJ.weight_indices = []

        for indices in zip(*jax.numpy.tril_indices(len(self.MJ.diff_params)+1)):
            weight_name = '_'.join([tag_name.upper()] + [self.MJ.block_to_pname[([None] + self.MJ.diff_params)[ind]] for ind in indices])
            self.MJ.weight_names.append(weight_name)
            self.MJ.weight_indices.append(indices)


        self.MJ.madjax_EFT.set_WC_names(self.MJ.diff_params)

        self.MJ.other_params = {}
        for blockname, block in self.MJ.old_param.items():
            for param in block:
                lhacode = param.lhacode[0]
                value = param.value
                if (blockname, lhacode) not in self.MJ.diff_params:
                    self.MJ.other_params[(blockname, lhacode)] = value

        self.MJ.WCs_sampling = [self.MJ.old_param[blockname].get(lhacode).value for blockname, lhacode in self.MJ.diff_params]
        self.MJ.WCs = [self.MJ.new_param[blockname].get(lhacode).value for blockname, lhacode in self.MJ.diff_params]

        #initialise module.
        for (path,tag), module in self.f2pylib.items():
            with misc.chdir(pjoin(os.path.dirname(rw_dir), path)):
                with misc.stdchannel_redirected(sys.stdout, os.devnull):
                    if 'rw_me_' in path or tag == 3:
                        param_card = self.new_param_card
                    else:
                        param_card = check_param_card.ParamCard(self.orig_param_card_text)
                    module.initialise('../Cards/param_card.dat')
                    for block in param_card:
                        if block.lower() == 'qnumbers':
                            continue
                        for param   in param_card[block]:
                            lhacode = param.lhacode
                            value = param.value
                            name = '%s_%s' % (block.upper(), '_'.join([str(i) for i in lhacode]))
                            module.change_para(name, value)
#                    misc.sprint("recompute module")
                    module.update_all_coup()

        return param_card_iterator, tag_name

    def do_import(self, inputfile, allow_madspin=False):
        super().do_import(inputfile, allow_madspin)
        self.MJ.do_import(inputfile, allow_madspin)

    #def get_LO_definition_from_NLO(self, *args, **kwargs):
    #    super().get_LO_definition_from_NLO(*args, **kwargs)
    #    self.MJ.get_LO_definition_from_NLO(*args, **kwargs)

    #def check_events(self, *args, **kwargs):
    #    super().check_events(*args, **kwargs)
    #    self.MJ.check_events(*args, **kwargs)

    #def complete_import(self, *args, **kwargs):
    #    super().complete_import(*args, **kwargs)
    #    self.MJ.complete_import(*args, **kwargs)

    #def help_change(self, *args, **kwargs):
    #    super().help_change(*args, **kwargs)
    #    self.MJ.help_change(*args, **kwargs)

    def do_change(self, *args, **kwargs):
        super().do_change(*args, **kwargs)
        self.MJ.do_change(*args, **kwargs)

    #def check_launch(self, *args, **kwargs):
    #    super().check_launch(*args, **kwargs)
    #    self.MJ.check_launch(*args, **kwargs)

    #def help_launch(self, *args, **kwargs):
    #    super().help_launch(*args, **kwargs)
    #    self.MJ.help_launch(*args, **kwargs)

    #def get_weight_names(self, *args, **kwargs):
    #    super().get_weight_names(*args, **kwargs)
    #    self.MJ.get_weight_names(*args, **kwargs)

    #def do_set(self, *args, **kwargs):
    #    super().do_set(*args, **kwargs)
    #    return self.MJ.do_set(*args, **kwargs)

    #def default(self, *args, **kwargs):
    #    super().default(*args, **kwargs)
    #    self.MJ.default(*args, **kwargs)

    #def do_compute_widths(self, *args, **kwargs):
    #    super().do_compute_widths(*args, **kwargs)
    #    self.MJ.do_compute_widths(*args, **kwargs)

    #def change_kinematics(self, *args, **kwargs):
    #    super().change_kinematics(*args, **kwargs)
    #    self.MJ.change_kinematics(*args, **kwargs)

    #def calculate_nlo_weight(self, *args, **kwargs):
    #    super().calculate_nlo_weight(*args, **kwargs)
    #    self.MJ.calculate_nlo_weight(*args, **kwargs)

    #def combine_wgt_local(self, *args, **kwargs):
    #    super().combine_wgt_local(*args, **kwargs)
    #    self.MJ.combine_wgt_local(*args, **kwargs)

    #def __getattr__(self, name):
    #    # Check if the attribute exists on internal objects and is callable
    #    if hasattr(super(), name) and callable(getattr(super(), name)):
    #        print('Call both', name)
    #        def method_wrapper(*args, **kwargs):
    #            getattr(self.MJ, name)(*args, **kwargs)
    #            return getattr(super(), name)(*args, **kwargs)
    #        return method_wrapper
    #    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

#class EFT_madjax_reweight(rwgt_interface.ReweightInterface):
#    def create_standalone_tree_directory(self, data ,second=False):
#    def create_standalone_virt_directory(self, data ,second=False):
#    def create_standalone_directory(self, second=False):
#    def compile(self):
#    def load_module(self, metag=1):
#    def save_to_pickle(self):
#    def load_from_pickle(self, keep_name=False):
#    def calculate_weight(self, event):
#    def handle_param_card(self, model_line, args, type_rwgt):

#    def __init__(self, event_path=None, allow_madspin=False, mother=None, *completekey, **stdin):
#    def do_import(self, inputfile, allow_madspin=False):
#    def get_LO_definition_from_NLO(proc, model, real_only=False):
#    def check_events(self):
#    def complete_import(self, text, line, begidx, endidx):
#    def help_change(self):
#    def do_change(self, line):
#    def check_launch(self, args):
#    def help_launch(self):
#    def get_weight_names(self):
#    def do_launch(self, line):
#    def handle_param_card(self, model_line, args, type_rwgt):
#    def do_set(self, line):
#    def default(self, line, log=True):
#    def write_reweighted_event(self, event, tag_name, **opt):
#    def do_compute_widths(self, line):
#    def change_kinematics(self, event):
#    def calculate_weight(self, event):
#    def calculate_nlo_weight(self, event):
#    def combine_wgt_local(self, scale2s, pdgs, bjxs, base_wgts, gss, qcdpowers, pdf):
#    def invert_momenta(p):
#    def rename_f2py_lib(Pdir, tag):
#    def calculate_matrix_element(self, event, hypp_id, scale2=0):
#    def terminate_fortran_executables(self, new_card_only=False):
#    def do_quit(self, line):
#    def __del__(self):
#    def adding_me(self, matrix_elements, path):
#    def create_standalone_tree_directory(self, data ,second=False):
#    def create_standalone_virt_directory(self, data ,second=False):
#    def create_standalone_directory(self, second=False):
#    def compile(self):
#    def load_module(self, metag=1):
#    def load_model(self, name, use_mg_default, complex_mass=False):
#    def save_to_pickle(self):
#    def load_from_pickle(self, keep_name=False):



