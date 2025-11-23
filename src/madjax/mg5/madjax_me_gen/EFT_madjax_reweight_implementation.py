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
import madgraph.iolibs.files as files
import re
import logging
import time
import shutil
import os
import sys
import itertools
from functools import partial
from jax import checkpoint
from jax.experimental.serialize_executable import serialize as serialize_compiled
from jax.experimental.serialize_executable import deserialize_and_load as deserialize_compiled
import pickle

# Eliminate unnecessary warnings from JAX
logging.getLogger('jax._src.lib.xla_bridge').addFilter(lambda _: False)

pjoin = os.path.join

logger = logging.getLogger('decay.stdout') # -> stdout
logger.setLevel(logging.DEBUG)

jaxlogger = logging.getLogger("jax")
jaxlogger.setLevel(logging.DEBUG)

jax.config.update("jax_enable_x64", False)

rewgt_path = "rewgt_functions"
os.makedirs(rewgt_path, exist_ok=True)

@partial(jax.jit, static_argnames=("other_param_names", "WC_names", "PDG_IDs", "numer"))
@jax.jacrev
@jax.jacfwd
def hess(WCs_plus_zero, fourvectors, helicities, other_params, other_param_names, WC_names, PDG_IDs, numer):
    params = {WC_name : WC for WC_name, WC in zip(WC_names, WCs_plus_zero[1:])}
    params.update({other_param_name : other_param for other_param_name, other_param in zip(other_param_names, other_params)})
    mod = numer.parameters.calculate_full_parameters(params)
    madjax_vectors = [madjax.phasespace.vectors.LorentzVector(v) for v in fourvectors]
    M = 0
    for procID, JM in numer.permuted_processes[PDG_IDs].items():
        M += JM.static_smatrix(madjax_vectors, mod, [helicities])
    return jax.numpy.exp(WCs_plus_zero[0]) * M

@partial(jax.jit, static_argnames=("other_param_names", "WC_names", "PDG_IDs", "denom"))
def denom(WCs_sampling, fourvectors, helicities, other_params, other_param_names, WC_names, PDG_IDs, denom):
    params = {WC_name : WC for WC_name, WC in zip(WC_names, WCs_sampling)}
    params.update({other_param_name : other_param for other_param_name, other_param in zip(other_param_names, other_params)})
    mod = denom.parameters.calculate_full_parameters(params)
    madjax_vectors = [madjax.phasespace.vectors.LorentzVector(v) for v in fourvectors]
    M = 0
    for procID, JM in denom.permuted_processes[PDG_IDs].items():
        M += JM.static_smatrix(madjax_vectors, mod, [helicities])
    return M

@partial(jax.jit, static_argnames=("other_param_names", "WC_names", "PDG_IDs", "numerMJ", "denomMJ"))
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

class madjax_EFT:
    def __init__(self, madjax_instance_numerator, madjax_instance_denominator, WC_names=None):
        self.numer = madjax_instance_numerator
        self.denom = madjax_instance_denominator
        self._memory_cache = {}

        self.tag_map = dict()
        for k, v in self.numer.processes.items():
            # Assume that we are dealing with 2 -> N scattering, not 1 -> N decay
            # If that is not the case, then this won't work correctly!
            PDG_IDs = v.pdg_order
            for initial in itertools.permutations(PDG_IDs[:2]):
                for final in itertools.permutations(PDG_IDs[2:]):
                    self.tag_map[initial+final] = v.pdg_order

        self.WC_names = WC_names
        if self.WC_names is not None:
            self.WC_names.sort()

    def set_WC_names(self, WC_names):
        self.WC_names = WC_names
        self.WC_names.sort()

    def reset(self):
        self._memory_cache = {}

    def __call__(self, WCs, WCs_sampling, event, other_params=dict()):
        flat_PDG_IDs = self.tag_map[tuple(sum(event.get_tag_and_order()[1], start=[]))]
        event_id = event.get_tag_and_order()[1]
        flat_event_id = [x for sublist in event_id for x in sublist]
        string_event_id = [str(item) for item in flat_event_id]
        concat_event_id = ''.join(string_event_id)
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

        if concat_event_id in self._memory_cache:
            compiled_rewgt = self._memory_cache[concat_event_id]

        else:
            # compile here?
            if os.path.exists(f"rewgt_functions/compiled_{concat_event_id}"):
                print("Loading compiled function from disk")
                with open(f"rewgt_functions/compiled_{concat_event_id}", "rb") as f:
                    serialized, in_tree, out_tree = pickle.load(f)
                    compiled_rewgt = deserialize_compiled(serialized, in_tree, out_tree)
                self._memory_cache[concat_event_id] = compiled_rewgt
            else:
                print("Compiling from scratch")
                #traced_rewgt = rewgt.trace(
                lowered_rewgt = rewgt.lower(
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
                        )
                #lowered_rewgt = traced_rewgt.lower()
                compiled_rewgt = lowered_rewgt.compile()
                serialized_rewgt = serialize_compiled(compiled_rewgt)

                with open(f"rewgt_functions/compiled_{concat_event_id}", "wb") as f:
                    pickle.dump(serialized_rewgt, f)


        return compiled_rewgt(
                jax.numpy.array([0.0] + WCs),
                WCs_sampling,
                j_fourvectors,
                j_helicities,
                jax.numpy.array(other_param_values_list)#,
                #tuple(other_param_names_list),
                #tuple(self.WC_names),
                #tuple(flat_PDG_IDs),
                #self.numer,
                #self.denom,
                )

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
            data['paths'] = ['rw_me', 'rw_mevirt']
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
                if 'rw_me_%s' % self.nb_library in key[0]:
                    del self.f2pylib[key]

            self.nb_library += 1
            data['paths'] = ['rw_me_%s' % self.nb_library, 'rw_mevirt_%s' % self.nb_library]


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
        rwgt_dir_possibility =   ['rw_me','rw_me_%s' % self.nb_library,'rw_mevirt','rw_mevirt_%s' % self.nb_library]
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
            self.madjax_denominator = madjax.MadJax('rw_me')
        if self.second_process:
            with misc.TMP_variable(sys, 'path', [pjoin(path_me)] + sys.path):
                self.madjax_numerator = madjax.MadJax('rw_me_2')
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

        weights = {'orig': orig_wgt, '': hess_tril[0] * orig_wgt}
        event.reweight_order.extend(self.weight_names)
        event.reweight_data.update(dict(zip(self.weight_names, (hess_tril * orig_wgt).tolist())))

        return weights

    def handle_param_card(self, model_line, args, type_rwgt):

        if self.rwgt_dir:
            path_me = self.rwgt_dir
        else:
            path_me = self.me_dir

        if self.second_model or self.second_process or self.dedicated_path:
            rw_dir = pjoin(path_me, 'rw_me_%s' % self.nb_library)
        else:
            rw_dir = pjoin(path_me, 'rw_me')

        if not '--keep_card' in args:
            if self.has_nlo and self.rwgt_mode != "LO":
                rwdir_virt = rw_dir.replace('rw_me', 'rw_mevirt')
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

    @misc.mute_logger()
    def do_launch(self, line):
        """end of the configuration launched the code"""

        args = self.split_arg(line)
        opts = self.check_launch(args)
        if opts['rwgt_name']:
            self.options['rwgt_name'] = opts['rwgt_name']
        if opts['rwgt_info']:
            self.options['rwgt_info'] = opts['rwgt_info']
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

        pdgIds_list = []
        seen_pdgs=set()
        for event_nb, event in enumerate(self.lhe_input):
            nested_pdgs = event.get_tag_and_order()[1]

            # 2. Flatten the list (using the fast list comprehension method)
            flat_pdgs = [x for sublist in nested_pdgs for x in sublist]

            # 3. Convert to tuple to check for uniqueness
            pdg_tuple = tuple(flat_pdgs)

            # 4. Only append if we haven't seen this combination before
            if pdg_tuple not in seen_pdgs:
                seen_pdgs.add(pdg_tuple)
                pdgIds_list.append(flat_pdgs)
        print('===================================================')
        print(pdgIds_list)

        if self.lhe_input.closed:
            self.lhe_input = lhe_parser.EventFile(self.lhe_input.name)
        self.lhe_input.seek(0)

        for subproc in pdgIds_list:
            print("Reweighting for subprocess:", subproc)
            if self.lhe_input.closed:
                self.lhe_input = lhe_parser.EventFile(self.lhe_input.name)
            self.lhe_input.seek(0)
            for event_nb,event in enumerate(self.lhe_input):
                event_id = event.get_tag_and_order()[1]
                flat_event_id = [x for sublist in event_id for x in sublist]
                if flat_event_id != subproc:
                    continue
                #control logger
                if (event_nb % max(int(10**int(math.log10(float(event_nb)+1))),10)==0):
                        running_time = misc.format_timer(time.time()-start)
                        logger.info('Event nb %s %s' % (event_nb, running_time))
                if (event_nb==10001): logger.info('reducing number of print status. Next status update in 10000 events')
                if (event_nb==100001): logger.info('reducing number of print status. Next status update in 100000 events')


                weight = self.calculate_weight(event)
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
            self.madjax_EFT.reset()
            jax.clear_caches()
            #backend = jax.lib.xla_bridge.get_backend()
            #for buf in backend.live_buffers():
            #    buf.delete()

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
