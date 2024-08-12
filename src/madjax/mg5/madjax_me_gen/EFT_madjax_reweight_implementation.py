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
import os
import sys
import itertools

# Eliminate unnecessary warnings from JAX
logging.getLogger('jax._src.lib.xla_bridge').addFilter(lambda _: False)

pjoin = os.path.join

logger = logging.getLogger('decay.stdout') # -> stdout

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
                     }

    def set_WC_names(self, WC_names):
        self.WC_names = WC_names

    def _new_hess(self, PDG_IDs):
        incoming = ''.join([self.codes[ID] for ID in PDG_IDs[:2]])
        outgoing = ''.join([self.codes[ID] for ID in PDG_IDs[2:]])
        code = f'{incoming}_{outgoing}'

        print(f'Compiling {code}')

        my_numerJMs = [v for k, v in self.numerJMs.items() if k[0] == PDG_IDs]
        my_denomJMs = [v for k, v in self.denomJMs.items() if k[0] == PDG_IDs]

        @jax.jit
        @jax.jacfwd
        @jax.jacrev
        def hess(WCs_plus_zero, fourvectors, helicities, other_params):
            params = {WC_name : WC for WC_name, WC in zip(self.WC_names, WCs_plus_zero[1:])}
            params.update(other_params)
            mod = self.numer.parameters.calculate_full_parameters(params)
            madjax_vectors = [madjax.phasespace.vectors.LorentzVector(v) for v in fourvectors]
            M = 0
            for JM in my_numerJMs:
                M += JM.smatrix(madjax_vectors, mod, [helicities])
            return jax.numpy.exp(WCs_plus_zero[0]) * M

        @jax.jit
        def denom(WCs_sampling, fourvectors, helicities, other_params):
            params = {WC_name : WC for WC_name, WC in zip(self.WC_names, WCs_sampling)}
            params.update(other_params)
            mod = self.denom.parameters.calculate_full_parameters(params)
            madjax_vectors = [madjax.phasespace.vectors.LorentzVector(v) for v in fourvectors]
            M = 0
            for JM in my_denomJMs:
                M += JM.smatrix(madjax_vectors, mod, [helicities])
            return M

        @jax.jit
        def rewgt(WCs_plus_zero, WCs_sampling, fourvectors, helicities, other_params):
            H = (hess(WCs_plus_zero, fourvectors, helicities, other_params) /
                 denom(WCs_sampling, fourvectors, helicities, other_params))
            H2 = (H + H.T - jax.numpy.diag(jax.numpy.diag(H))).at[0,0].set(H[0,0])
            return H2[jax.numpy.tril_indices_from(H2)]


        self.proc_map[tuple(PDG_IDs)] = (hess, denom, rewgt)

    def __call__(self, WCs, WCs_sampling, event, other_params=dict()):
        flat_PDG_IDs = self.tag_map[tuple(sum(event.get_tag_and_order()[1], start=[]))]
        PDG_IDs = event.get_tag_and_order()[1]
        fourvectors = event.get_momenta(PDG_IDs)
        helicities = event.get_helicity(PDG_IDs)

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

        if flat_PDG_IDs not in self.proc_map:
            self._new_hess(flat_PDG_IDs)

        (hess, denom, rewgt) = self.proc_map[flat_PDG_IDs]

        # return hess(jax.numpy.insert(WCs, 0, 0.0), j_fourvectors, j_helicities, other_params)
        # return denom(WCs_sampling, j_fourvectors, j_helicities, other_params)
        return rewgt(
                jax.numpy.array([0.0] + WCs),
                WCs_sampling,
                j_fourvectors,
                j_helicities,
                other_params
                )

class EFT_madjax_reweight(rwgt_interface.ReweightInterface):
    def calculate_matrix_element(self, event, hypp_id, scale2=0):
        """routine to return the matrix element"""

        print("JSW")
        return super().calculate_matrix_element(event, hypp_id, scale2)

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

        weights = {'orig': orig_wgt}
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
