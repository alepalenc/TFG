import tensorflow as tf
from tensorflow.contrib import predictor


from matplotlib import pyplot as plt
import numpy as np
import random


import sys 
sys.path.insert(0,'..')
sys.path.insert(0,'../..')


from cnf_dataset import clauses_to_matrix
from gsat import GSAT
from walksat import WalkSAT
from walksat_sta import WalkSAT_sta
from walksat_dyn import WalkSAT_dyn
from walksat_cl_sta import WalkSATCL_sta
from walksat_cl_dyn import WalkSATCL_dyn
from cnf import get_random_kcnf, CNF, get_sats_SR, get_pos_SR
from tqdm import tqdm
from collections import Counter


import math
from collections import defaultdict





with tf.device('/device:GPU:0'):
    export_dir = "model_sr50"
    predict_fn = predictor.from_saved_model(export_dir)
    
    
    np.set_printoptions(precision=3, suppress=True)


    BATCH_SIZE = 1
    MAX_FLIPS_GSAT = {10:20, 20:50, 30:100, 40:100, 50:100, 60:100, 70:100}
    MAX_FLIPS_STA = 100
    MAX_TRIES_STA = 10
    MAX_FLIPS_DYN = 1000
    MEMORY_STEPS = 100
    VERBOSE = 0
    
    
    
    
    
    ################
    ###   GSAT   ###
    ################
    
    def compute_steps_gsat(sats, gsat_cls):
        flips = []
        tries = []
        flips_all = []
        tries_all = []
        solved = 0
        for sat in tqdm(sats):
            gsat = gsat_cls(max_flips=MAX_FLIPS_GSAT[M], max_tries=1000//MAX_FLIPS_GSAT[M], verbose=VERBOSE)
            res = gsat.run(sat)
            if res is not None:
                flips.append(gsat.number_of_flips+gsat.MAX_FLIPS*gsat.number_of_tries)
                tries.append(gsat.number_of_tries+1)
                flips_all.append(gsat.number_of_flips+gsat.MAX_FLIPS*gsat.number_of_tries)
                tries_all.append(gsat.number_of_tries+1)
                solved += 1
            else:
                flips_all.append(None)
                tries_all.append(None)
        print("\tWith {} MAX_FLIPS and {} MAX_TRIES solved {} problems out of {}".format(MAX_FLIPS_GSAT[M], 1000//MAX_FLIPS_GSAT[M], solved, len(sats)))
        print("\tFlips    : ",flips)
        print("\tTries    : ",tries)
        print("\tAll flips: ",flips_all)
        print("\tAll tries: ",tries_all)
        return flips, tries
    
    
    def compute_and_print_steps_gsat(sats, gsat_cls):
        flips, tries = compute_steps_gsat(sats, gsat_cls)
        print("\t#Sats: {}; avg flip: {:.2f}; stdev flip: {:.2f}; avg try: {:.2f}; stdev try: {:.2f}".format(
            len(flips), np.mean(flips), np.std(flips), np.mean(tries), np.std(tries)))
        plt.figure(figsize=(14, 4))
        plt.subplot(1, 2, 1)
        plt.title("Flips of {}".format(gsat_cls.__name__))
        plt.hist(flips, bins=40)
        plt.ylim((0, len(sats)))
        
        plt.subplot(1, 2, 2)
        plt.title("Tries of {}".format(gsat_cls.__name__))
        plt.hist(tries, bins=range(1,1000//MAX_FLIPS_GSAT[M]+1))
        plt.ylim((0, len(sats)))
        plt.show()
    
    
    
    
    
    ###################
    ###   WalkSAT   ###
    ###################
    
    def compute_steps_walksat(sats, walksat_cls):
        flips = []
        tries = []
        flips_all = []
        tries_all = []
        solved = 0
        for sat in tqdm(sats):
            walksat = walksat_cls(max_flips=MAX_FLIPS_STA, max_tries=MAX_TRIES_STA, verbose=VERBOSE)
            res = walksat.run(sat)
            if res is not None:
                flips.append(walksat.number_of_flips+walksat.MAX_FLIPS*walksat.number_of_tries)
                tries.append(walksat.number_of_tries+1)
                flips_all.append(walksat.number_of_flips+walksat.MAX_FLIPS*walksat.number_of_tries)
                tries_all.append(walksat.number_of_tries+1)
                solved += 1
            else:
                flips_all.append(None)
                tries_all.append(None)
        print("\tWith {} MAX_FLIPS and {} MAX_TRIES solved {} problems out of {}".format(MAX_FLIPS_STA, MAX_TRIES_STA, solved, len(sats)))
        print("\tFlips    : ",flips)
        print("\tTries    : ",tries)
        print("\tAll flips: ",flips_all)
        print("\tAll tries: ",tries_all)
        return flips, tries
    
    
    def compute_and_print_steps_walksat(sats, walksat_cls):
        flips, tries = compute_steps_walksat(sats, walksat_cls)
        print("\t#Sats: {}; avg flip: {:.2f}; stdev flip: {:.2f}; avg try: {:.2f}; stdev try: {:.2f}".format(
            len(flips), np.mean(flips), np.std(flips), np.mean(tries), np.std(tries)))
        plt.figure(figsize=(14, 4))
        plt.subplot(1, 2, 1)
        plt.title("Flips of {}".format(walksat_cls.__name__))
        plt.hist(flips, bins=40)
        plt.ylim((0, len(sats)))
        
        plt.subplot(1, 2, 2)
        plt.title("Tries of {}".format(walksat_cls.__name__))
        plt.hist(tries, bins=range(1,MAX_TRIES_STA+1))
        plt.ylim((0, len(sats)))
        plt.show()
    
    
    
    
    
    ############################
    ###   NeuroWalkSAT-sta   ###
    ############################
    
    class NeuroWalkSAT_sta_worst(WalkSAT_sta):
        def suggest(self, cnf: CNF, state, random_clause, break_counts, clauses_satisfiability):
            reduced_state = state.copy()
            unsat_clauses = [cnf.clauses[i] for i in np.argwhere(np.array(clauses_satisfiability)==False).squeeze(axis=1)]
            joined_unsat_clauses = set([l for c in unsat_clauses for l in c])
            
            if self.verbose > 0: print(f"Joined unsat clauses: {joined_unsat_clauses}")
            for l in joined_unsat_clauses:
                reduced_state.remove(-l)
            
            # Assign values to the variables in reduced_cnf with reduced_state
            new_clauses = []
            for c in cnf.clauses:
                clause = set(c)
                satisfied = False
                for v in reduced_state:
                    if v in clause:
                        satisfied = True
                        break
                    elif -v in clause:
                        clause.remove(-v)
                if not satisfied:
                    new_clauses.append(clause)
            reduced_cnf = CNF(new_clauses)
            
            if self.verbose > 0: print(f"Reduced CNF clauses: {reduced_cnf.clauses}")
            
            """
            # Delete empty clauses if needed
            reduced_cnf.clauses = tuple(filter(lambda a: a != tuple(), reduced_cnf.clauses))
            if self.verbose > 0: print(f"Reduced CNF: {reduced_cnf.clauses}")
            """
            
            clause_num = len(reduced_cnf.clauses)
            var_num = max(reduced_cnf.vars)
            inputs = np.asarray([clauses_to_matrix(reduced_cnf.clauses, clause_num, var_num)] * BATCH_SIZE)
            
            policy_probs = predict_fn({"input": inputs})['policy_probabilities']
            if self.verbose > 0: print(f"Policy probs: {policy_probs}")
            
            worst_prob = 2.0
            worst_lit = None
            for lit in joined_unsat_clauses:
                lit_prob = policy_probs[0][abs(lit)-1][0 if lit < 0 else 1]
                if lit_prob < worst_prob:
                    worst_prob = lit_prob
                    worst_lit = lit
            
            return worst_lit
    
    
    class NeuroWalkSAT_sta_best(WalkSAT_sta):
        def suggest(self, cnf: CNF, state, random_clause, break_counts, clauses_satisfiability):
            reduced_state = state.copy()
            unsat_clauses = [cnf.clauses[i] for i in np.argwhere(np.array(clauses_satisfiability)==False).squeeze(axis=1)]
            joined_unsat_clauses = set([l for c in unsat_clauses for l in c])
            
            if self.verbose > 0: print(f"Joined unsat clauses: {joined_unsat_clauses}")
            for l in joined_unsat_clauses:
                reduced_state.remove(-l)
            
            # Assign values to the variables in reduced_cnf with reduced_state
            new_clauses = []
            for c in cnf.clauses:
                clause = set(c)
                satisfied = False
                for v in reduced_state:
                    if v in clause:
                        satisfied = True
                        break
                    elif -v in clause:
                        clause.remove(-v)
                if not satisfied:
                    new_clauses.append(clause)
            reduced_cnf = CNF(new_clauses)
            
            if self.verbose > 0: print(f"Reduced CNF clauses: {reduced_cnf.clauses}")
            
            """
            # Delete empty clauses if needed
            reduced_cnf.clauses = tuple(filter(lambda a: a != tuple(), reduced_cnf.clauses))
            if self.verbose > 0: print(f"Reduced CNF: {reduced_cnf.clauses}")
            """
            
            clause_num = len(reduced_cnf.clauses)
            var_num = max(reduced_cnf.vars)
            inputs = np.asarray([clauses_to_matrix(reduced_cnf.clauses, clause_num, var_num)] * BATCH_SIZE)
            
            policy_probs = predict_fn({"input": inputs})['policy_probabilities']
            if self.verbose > 0: print(f"Policy probs: {policy_probs}")
            
            best_prob = -1.0
            best_lit = None
            for lit in joined_unsat_clauses:
                lit_prob = policy_probs[0][abs(lit)-1][0 if lit > 0 else 1]
                if lit_prob > best_prob:
                    best_prob = lit_prob
                    best_lit = lit
            
            return best_lit
    
    
    def compute_steps_sta(sats, walksat_cls):
        flips = []
        tries = []
        flips_all = []
        tries_all = []
        solved = 0
        for sat in tqdm(sats):
            walksat = walksat_cls(max_flips=MAX_FLIPS_STA, max_tries=MAX_TRIES_STA, verbose=VERBOSE)
            res = walksat.run(sat)
            if res is not None:
                flips.append(walksat.number_of_flips+walksat.MAX_FLIPS*walksat.number_of_tries)
                tries.append(walksat.number_of_tries+1)
                flips_all.append(walksat.number_of_flips+walksat.MAX_FLIPS*walksat.number_of_tries)
                tries_all.append(walksat.number_of_tries+1)
                solved += 1
            else:
                flips_all.append(None)
                tries_all.append(None)
        print("\tWith {} MAX_FLIPS and {} MAX_TRIES solved {} problems out of {}".format(MAX_FLIPS_STA, MAX_TRIES_STA, solved, len(sats)))
        print("\tFlips    : ",flips)
        print("\tTries    : ",tries)
        print("\tAll flips: ",flips_all)
        print("\tAll tries: ",tries_all)
        return flips, tries
    
    
    def compute_and_print_steps_sta(sats, walksat_cls):
        flips, tries = compute_steps_sta(sats, walksat_cls)
        print("\t#Sats: {}; avg flip: {:.2f}; stdev flip: {:.2f}; avg try: {:.2f}; stdev try: {:.2f}".format(
            len(flips), np.mean(flips), np.std(flips), np.mean(tries), np.std(tries)))
    
    
    
    
    
    ############################
    ###   NeuroWalkSAT-dyn   ###
    ############################
    
    class NeuroWalkSAT_dyn_worst(WalkSAT_dyn):
        def suggest(self, cnf: CNF, state, random_clause, break_counts, clauses_satisfiability):
            reduced_state = state.copy()
            unsat_clauses = [cnf.clauses[i] for i in np.argwhere(np.array(clauses_satisfiability)==False).squeeze(axis=1)]
            joined_unsat_clauses = set([l for c in unsat_clauses for l in c])
            
            if self.verbose > 0: print(f"Joined unsat clauses: {joined_unsat_clauses}")
            for l in joined_unsat_clauses:
                reduced_state.remove(-l)
            
            # Assign values to the variables in reduced_cnf with reduced_state
            new_clauses = []
            for c in cnf.clauses:
                clause = set(c)
                satisfied = False
                for v in reduced_state:
                    if v in clause:
                        satisfied = True
                        break
                    elif -v in clause:
                        clause.remove(-v)
                if not satisfied:
                    new_clauses.append(clause)
            reduced_cnf = CNF(new_clauses)
            
            if self.verbose > 0: print(f"Reduced CNF clauses: {reduced_cnf.clauses}")
            
            """
            # Delete empty clauses if needed
            reduced_cnf.clauses = tuple(filter(lambda a: a != tuple(), reduced_cnf.clauses))
            if self.verbose > 0: print(f"Reduced CNF: {reduced_cnf.clauses}")
            """
            
            clause_num = len(reduced_cnf.clauses)
            var_num = max(reduced_cnf.vars)
            inputs = np.asarray([clauses_to_matrix(reduced_cnf.clauses, clause_num, var_num)] * BATCH_SIZE)
            
            policy_probs = predict_fn({"input": inputs})['policy_probabilities']
            if self.verbose > 0: print(f"Policy probs: {policy_probs}")
            
            worst_prob = 2.0
            worst_lit = None
            for lit in joined_unsat_clauses:
                lit_prob = policy_probs[0][abs(lit)-1][0 if lit < 0 else 1]
                if lit_prob < worst_prob:
                    worst_prob = lit_prob
                    worst_lit = lit
            
            return worst_lit
    
    
    class NeuroWalkSAT_dyn_best(WalkSAT_dyn):
        def suggest(self, cnf: CNF, state, random_clause, break_counts, clauses_satisfiability):
            reduced_state = state.copy()
            unsat_clauses = [cnf.clauses[i] for i in np.argwhere(np.array(clauses_satisfiability)==False).squeeze(axis=1)]
            joined_unsat_clauses = set([l for c in unsat_clauses for l in c])
            
            if self.verbose > 0: print(f"Joined unsat clauses: {joined_unsat_clauses}")
            for l in joined_unsat_clauses:
                reduced_state.remove(-l)
            
            # Assign values to the variables in reduced_cnf with reduced_state
            new_clauses = []
            for c in cnf.clauses:
                clause = set(c)
                satisfied = False
                for v in reduced_state:
                    if v in clause:
                        satisfied = True
                        break
                    elif -v in clause:
                        clause.remove(-v)
                if not satisfied:
                    new_clauses.append(clause)
            reduced_cnf = CNF(new_clauses)
            
            if self.verbose > 0: print(f"Reduced CNF clauses: {reduced_cnf.clauses}")
            
            """
            # Delete empty clauses if needed
            reduced_cnf.clauses = tuple(filter(lambda a: a != tuple(), reduced_cnf.clauses))
            if self.verbose > 0: print(f"Reduced CNF: {reduced_cnf.clauses}")
            """
            
            clause_num = len(reduced_cnf.clauses)
            var_num = max(reduced_cnf.vars)
            inputs = np.asarray([clauses_to_matrix(reduced_cnf.clauses, clause_num, var_num)] * BATCH_SIZE)
            
            policy_probs = predict_fn({"input": inputs})['policy_probabilities']
            if self.verbose > 0: print(f"Policy probs: {policy_probs}")
            
            best_prob = -1.0
            best_lit = None
            for lit in joined_unsat_clauses:
                lit_prob = policy_probs[0][abs(lit)-1][0 if lit > 0 else 1]
                if lit_prob > best_prob:
                    best_prob = lit_prob
                    best_lit = lit
            
            return best_lit
    
    
    def compute_steps_dyn(sats, walksat_cls):
        flips = []
        tries = []
        flips_all = []
        tries_all = []
        solved = 0
        for sat in tqdm(sats):
            walksat = walksat_cls(max_flips=MAX_FLIPS_DYN, verbose=VERBOSE)
            res = walksat.run(sat)
            if res is not None:
                flips.append(walksat.number_of_flips)
                tries.append(walksat.number_of_tries+1)
                flips_all.append(walksat.number_of_flips)
                tries_all.append(walksat.number_of_tries+1)
                solved += 1
            else:
                flips_all.append(None)
                tries_all.append(None)
        print("\tWith {} MAX_FLIPS solved {} problems out of {}".format(MAX_FLIPS_DYN, solved, len(sats)))
        print("\tFlips    : ",flips)
        print("\tTries    : ",tries)
        print("\tAll flips: ",flips_all)
        print("\tAll tries: ",tries_all)
        return flips, tries
    
    
    def compute_and_print_steps_dyn(sats, walksat_cls):
        flips, tries = compute_steps_dyn(sats, walksat_cls)
        print("\t#Sats: {}; avg flip: {:.2f}; stdev flip: {:.2f}; avg try: {:.2f}; stdev try: {:.2f}".format(
            len(flips), np.mean(flips), np.std(flips), np.mean(tries), np.std(tries)))
    
    
    
    
    
    ##############################
    ###   NeuroWalkSATCL-sta   ###
    ##############################
    
    class NeuroWalkSATCL_sta_worst(WalkSATCL_sta):
        def suggest(self, cnf: CNF, state, random_clause, break_counts, clauses_satisfiability):
            reduced_state = state.copy()
            unsat_clauses = [cnf.clauses[i] for i in np.argwhere(np.array(clauses_satisfiability)==False).squeeze(axis=1)]
            joined_unsat_clauses = set([l for c in unsat_clauses for l in c])
            
            if self.verbose > 0: print(f"Joined unsat clauses: {joined_unsat_clauses}")
            for l in joined_unsat_clauses:
                reduced_state.remove(-l)
            
            # Assign values to the variables in reduced_cnf with reduced_state
            reduced_cnf_clauses = []
            for c in cnf.clauses:
                clause = set(c)
                satisfied = False
                for v in reduced_state:
                    if v in clause:
                        satisfied = True
                        break
                    elif -v in clause:
                        clause.remove(-v)
                if not satisfied:
                    reduced_cnf_clauses.append(clause)
            reduced_cnf = CNF(reduced_cnf_clauses)
            
            if self.verbose > 0: print(f"Reduced CNF clauses: {reduced_cnf.clauses}")
            
            # Apply unit propagation and pure literal elimination
            reduced_cnf, assigned_vars = self.shorten_cnf(reduced_cnf)
            
            if self.verbose > 0: print(f"Reduced CNF clauses (after shorten_cnf): {reduced_cnf.clauses}")
            if self.verbose > 0: print(f"Assigned variables (after shorten_cnf): {assigned_vars}")
            
            if reduced_cnf.is_true():
                # If the reduced_cnf is trivially true, return the literales which need to be flipped
                return assigned_vars
            elif reduced_cnf.is_false():
                # If the reduced_cnf is unsatisfiable, add a new clause with the negation of the literals in reduced_state
                #   and flip the variable in reduced_state with minimum break-count
                new_clause = tuple([-l for l in reduced_state])
                
                if self.verbose > 0: print(f"Reduced CNF is unsatisfiable. New clause: {new_clause}")
                
                break_counts = []
                literal = new_clause[len(break_counts)]
                bc = self.get_break_count(literal, cnf.clauses, clauses_satisfiability, state)
                break_counts.append(bc)
                while break_counts[-1] != 0 and len(break_counts) < len(new_clause):
                    literal = new_clause[len(break_counts)]
                    bc = self.get_break_count(literal, cnf.clauses, clauses_satisfiability, state)
                    break_counts.append(bc)
                
                if self.verbose > 0: print(f"    Break counts: {break_counts}")
                
                if break_counts[-1] == 0:
                    # If exists a variable with break_count=0, flip that variable
                    suggested_var = new_clause[len(break_counts)-1]
                    if self.verbose > 0: print(f"    Flip variable with break-count=0 ({suggested_var})")
                else:
                    # Flip the variable with minimum break_count
                    suggested_var = self.suggest_walksat(cnf, state, new_clause, break_counts, clauses_satisfiability)
                    if self.verbose > 0: print(f"    Flip suggested variable ({suggested_var})")
                
                # Add the new learnt clause
                self.learnt_clauses.append(new_clause)
                if self.max_num_lc < len(self.learnt_clauses):
                    self.max_num_lc = len(self.learnt_clauses)
                if self.max_length_lc < len(new_clause):
                    self.max_length_lc = len(new_clause)
                self.memory_matrix = np.vstack((self.memory_matrix, np.zeros(self.MEMORY_STEPS,dtype=bool)))
                
                # Delete the easier clause to satisfy (that with fewer ones) if the learnt clauses limit has been reached
                while len(self.learnt_clauses) > self.MAX_NUM_LEARNT_CLAUSES:
                    lc_index = np.argmin(np.sum(self.memory_matrix[len(self.original_clauses):,:], axis=1))
                    if self.verbose > 0: print(f"Deleted clause ({len(self.original_clauses)+lc_index}): {self.learnt_clauses[lc_index]}")
                    self.learnt_clauses.pop(lc_index)
                    self.memory_matrix = np.delete(self.memory_matrix, len(self.original_clauses)+lc_index, axis=0)
                
                # Add the new list of learnt clauses to the cnf
                cnf.clauses = self.original_clauses + tuple(self.learnt_clauses)
                
                return suggested_var
            else:
                clause_num = len(reduced_cnf.clauses)
                var_num = max(reduced_cnf.vars)
                inputs = np.asarray([clauses_to_matrix(reduced_cnf.clauses, clause_num, var_num)] * BATCH_SIZE)

                policy_probs = predict_fn({"input": inputs})['policy_probabilities']
                if self.verbose > 0: print(f"Policy probs: {policy_probs}")

                worst_prob = 2.0
                worst_lit = None
                for lit in joined_unsat_clauses:
                    if abs(lit) in reduced_cnf.vars:
                        lit_prob = policy_probs[0][abs(lit)-1][0 if lit < 0 else 1]
                        if lit_prob < worst_prob:
                            worst_prob = lit_prob
                            worst_lit = lit

                return worst_lit
    
    
    class NeuroWalkSATCL_sta_best(WalkSATCL_sta):
        def suggest(self, cnf: CNF, state, random_clause, break_counts, clauses_satisfiability):
            reduced_state = state.copy()
            unsat_clauses = [cnf.clauses[i] for i in np.argwhere(np.array(clauses_satisfiability)==False).squeeze(axis=1)]
            joined_unsat_clauses = set([l for c in unsat_clauses for l in c])
            
            if self.verbose > 0: print(f"Joined unsat clauses: {joined_unsat_clauses}")
            for l in joined_unsat_clauses:
                reduced_state.remove(-l)
            
            # Assign values to the variables in reduced_cnf with reduced_state
            reduced_cnf_clauses = []
            for c in cnf.clauses:
                clause = set(c)
                satisfied = False
                for v in reduced_state:
                    if v in clause:
                        satisfied = True
                        break
                    elif -v in clause:
                        clause.remove(-v)
                if not satisfied:
                    reduced_cnf_clauses.append(clause)
            reduced_cnf = CNF(reduced_cnf_clauses)
            
            if self.verbose > 0: print(f"Reduced CNF clauses: {reduced_cnf.clauses}")
            
            # Apply unit propagation and pure literal elimination
            reduced_cnf, assigned_vars = self.shorten_cnf(reduced_cnf)
            
            if self.verbose > 0: print(f"Reduced CNF clauses (after shorten_cnf): {reduced_cnf.clauses}")
            if self.verbose > 0: print(f"Assigned variables (after shorten_cnf): {assigned_vars}")
            
            if reduced_cnf.is_true():
                # If the reduced_cnf is trivially true, return the literales which need to be flipped
                return assigned_vars
            elif reduced_cnf.is_false():
                # If the reduced_cnf is unsatisfiable, add a new clause with the negation of the literals in reduced_state
                #   and flip the variable in reduced_state with minimum break-count
                new_clause = tuple([-l for l in reduced_state])
                
                if self.verbose > 0: print(f"Reduced CNF is unsatisfiable. New clause: {new_clause}")
                
                break_counts = []
                literal = new_clause[len(break_counts)]
                bc = self.get_break_count(literal, cnf.clauses, clauses_satisfiability, state)
                break_counts.append(bc)
                while break_counts[-1] != 0 and len(break_counts) < len(new_clause):
                    literal = new_clause[len(break_counts)]
                    bc = self.get_break_count(literal, cnf.clauses, clauses_satisfiability, state)
                    break_counts.append(bc)
                
                if self.verbose > 0: print(f"    Break counts: {break_counts}")
                
                if break_counts[-1] == 0:
                    # If exists a variable with break_count=0, flip that variable
                    suggested_var = new_clause[len(break_counts)-1]
                    if self.verbose > 0: print(f"    Flip variable with break-count=0 ({suggested_var})")
                else:
                    # Flip the variable with minimum break_count
                    suggested_var = self.suggest_walksat(cnf, state, new_clause, break_counts, clauses_satisfiability)
                    if self.verbose > 0: print(f"    Flip suggested variable ({suggested_var})")
                
                # Add the new learnt clause
                self.learnt_clauses.append(new_clause)
                if self.max_num_lc < len(self.learnt_clauses):
                    self.max_num_lc = len(self.learnt_clauses)
                if self.max_length_lc < len(new_clause):
                    self.max_length_lc = len(new_clause)
                self.memory_matrix = np.vstack((self.memory_matrix, np.zeros(self.MEMORY_STEPS,dtype=bool)))
                
                # Delete the easier clause to satisfy (that with fewer ones) if the learnt clauses limit has been reached
                while len(self.learnt_clauses) > self.MAX_NUM_LEARNT_CLAUSES:
                    lc_index = np.argmin(np.sum(self.memory_matrix[len(self.original_clauses):,:], axis=1))
                    if self.verbose > 0: print(f"Deleted clause ({len(self.original_clauses)+lc_index}): {self.learnt_clauses[lc_index]}")
                    self.learnt_clauses.pop(lc_index)
                    self.memory_matrix = np.delete(self.memory_matrix, len(self.original_clauses)+lc_index, axis=0)
                
                # Add the new list of learnt clauses to the cnf
                cnf.clauses = self.original_clauses + tuple(self.learnt_clauses)
                
                return suggested_var
            else:
                clause_num = len(reduced_cnf.clauses)
                var_num = max(reduced_cnf.vars)
                inputs = np.asarray([clauses_to_matrix(reduced_cnf.clauses, clause_num, var_num)] * BATCH_SIZE)

                policy_probs = predict_fn({"input": inputs})['policy_probabilities']
                if self.verbose > 0: print(f"Policy probs: {policy_probs}")
                
                best_prob = -1.0
                best_lit = None
                for lit in joined_unsat_clauses:
                    if abs(lit) in reduced_cnf.vars:
                        lit_prob = policy_probs[0][abs(lit)-1][0 if lit > 0 else 1]
                        if lit_prob > best_prob:
                            best_prob = lit_prob
                            best_lit = lit

                return best_lit
    
    
    def compute_steps_cl_sta(sats, walksat_cls):
        flips = []
        tries = []
        flips_all = []
        tries_all = []
        max_num_lc = []
        max_length_lc = []
        solved = 0
        for sat in tqdm(sats):
            walksat = walksat_cls(max_flips=MAX_FLIPS_STA, max_tries=MAX_TRIES_STA, max_num_learnt_clauses=10*len(sat.vars), memory_steps=MEMORY_STEPS, verbose=VERBOSE)
            res = walksat.run(sat)
            if res is not None:
                flips.append(walksat.number_of_flips+walksat.MAX_FLIPS*walksat.number_of_tries)
                tries.append(walksat.number_of_tries+1)
                flips_all.append(walksat.number_of_flips+walksat.MAX_FLIPS*walksat.number_of_tries)
                tries_all.append(walksat.number_of_tries+1)
                max_num_lc.append(walksat.max_num_lc)
                max_length_lc.append(walksat.max_length_lc)
                solved += 1
            else:
                flips_all.append(None)
                tries_all.append(None)
        print("\tWith {} MAX_FLIPS and {} MAX_TRIES solved {} problems out of {}".format(MAX_FLIPS_STA, MAX_TRIES_STA, solved, len(sats)))
        print("\tFlips    : ",flips)
        print("\tTries    : ",tries)
        print("\tAll flips: ",flips_all)
        print("\tAll tries: ",tries_all)
        return flips, tries, max_num_lc, max_length_lc
    
    
    def compute_and_print_steps_cl_sta(sats, walksat_cls):
        flips, tries, max_num_lc, max_length_lc = compute_steps_cl_sta(sats, walksat_cls)
        print("\t#Sats: {}; avg flip: {:.2f}; stdev flip: {:.2f}; avg try: {:.2f}; stdev try: {:.2f}".format(
            len(flips), np.mean(flips), np.std(flips), np.mean(tries), np.std(tries)))
        if max(max_num_lc)>0:
            print("\tavg max_num_lc: {:.2f}; stdev max_num_lc: {:.2f}; avg max_length_lc: {:.2f}; stdev max_length_lc: {:.2f}".format(
                np.mean(max_num_lc), np.std(max_num_lc), np.mean(max_length_lc), np.std(max_length_lc)))

    
    
    
    
    
    ##############################
    ###   NeuroWalkSATCL-dyn   ###
    ##############################
    
    class NeuroWalkSATCL_dyn_worst(WalkSATCL_dyn):
        def suggest(self, cnf: CNF, state, random_clause, break_counts, clauses_satisfiability):
            reduced_state = state.copy()
            unsat_clauses = [cnf.clauses[i] for i in np.argwhere(np.array(clauses_satisfiability)==False).squeeze(axis=1)]
            joined_unsat_clauses = set([l for c in unsat_clauses for l in c])
            
            if self.verbose > 0: print(f"Joined unsat clauses: {joined_unsat_clauses}")
            for l in joined_unsat_clauses:
                reduced_state.remove(-l)
            
            # Assign values to the variables in reduced_cnf with reduced_state
            reduced_cnf_clauses = []
            for c in cnf.clauses:
                clause = set(c)
                satisfied = False
                for v in reduced_state:
                    if v in clause:
                        satisfied = True
                        break
                    elif -v in clause:
                        clause.remove(-v)
                if not satisfied:
                    reduced_cnf_clauses.append(clause)
            reduced_cnf = CNF(reduced_cnf_clauses)
            
            if self.verbose > 0: print(f"Reduced CNF clauses: {reduced_cnf.clauses}")
            
            # Apply unit propagation and pure literal elimination
            reduced_cnf, assigned_vars = self.shorten_cnf(reduced_cnf)
            
            if self.verbose > 0: print(f"Reduced CNF clauses (after shorten_cnf): {reduced_cnf.clauses}")
            if self.verbose > 0: print(f"Assigned variables (after shorten_cnf): {assigned_vars}")
            
            if reduced_cnf.is_true():
                # If the reduced_cnf is trivially true, return the literales which need to be flipped
                return assigned_vars
            elif reduced_cnf.is_false():
                # If the reduced_cnf is unsatisfiable, add a new clause with the negation of the literals in reduced_state
                #   and flip the variable in reduced_state with minimum break-count
                new_clause = tuple([-l for l in reduced_state])
                
                if self.verbose > 0: print(f"Reduced CNF is unsatisfiable. New clause: {new_clause}")
                
                break_counts = []
                literal = new_clause[len(break_counts)]
                bc = self.get_break_count(literal, cnf.clauses, clauses_satisfiability, state)
                break_counts.append(bc)
                while break_counts[-1] != 0 and len(break_counts) < len(new_clause):
                    literal = new_clause[len(break_counts)]
                    bc = self.get_break_count(literal, cnf.clauses, clauses_satisfiability, state)
                    break_counts.append(bc)
                
                if self.verbose > 0: print(f"    Break counts: {break_counts}")
                
                if break_counts[-1] == 0:
                    # If exists a variable with break_count=0, flip that variable
                    suggested_var = new_clause[len(break_counts)-1]
                    if self.verbose > 0: print(f"    Flip variable with break-count=0 ({suggested_var})")
                else:
                    # Flip the variable with minimum break_count
                    suggested_var = self.suggest_walksat(cnf, state, new_clause, break_counts, clauses_satisfiability)
                    if self.verbose > 0: print(f"    Flip suggested variable ({suggested_var})")
                
                # Add the new learnt clause
                self.learnt_clauses.append(new_clause)
                if self.max_num_lc < len(self.learnt_clauses):
                    self.max_num_lc = len(self.learnt_clauses)
                if self.max_length_lc < len(new_clause):
                    self.max_length_lc = len(new_clause)
                self.memory_matrix = np.vstack((self.memory_matrix, np.zeros(self.MEMORY_STEPS,dtype=bool)))
                
                # Delete the easier clause to satisfy (that with fewer ones) if the learnt clauses limit has been reached
                while len(self.learnt_clauses) > self.MAX_NUM_LEARNT_CLAUSES:
                    lc_index = np.argmin(np.sum(self.memory_matrix[len(self.original_clauses):,:], axis=1))
                    if self.verbose > 0: print(f"Deleted clause ({len(self.original_clauses)+lc_index}): {self.learnt_clauses[lc_index]}")
                    self.learnt_clauses.pop(lc_index)
                    self.memory_matrix = np.delete(self.memory_matrix, len(self.original_clauses)+lc_index, axis=0)
                
                # Add the new list of learnt clauses to the cnf
                cnf.clauses = self.original_clauses + tuple(self.learnt_clauses)
                
                return suggested_var
            else:
                clause_num = len(reduced_cnf.clauses)
                var_num = max(reduced_cnf.vars)
                inputs = np.asarray([clauses_to_matrix(reduced_cnf.clauses, clause_num, var_num)] * BATCH_SIZE)

                policy_probs = predict_fn({"input": inputs})['policy_probabilities']
                if self.verbose > 0: print(f"Policy probs: {policy_probs}")

                worst_prob = 2.0
                worst_lit = None
                for lit in joined_unsat_clauses:
                    if abs(lit) in reduced_cnf.vars:
                        lit_prob = policy_probs[0][abs(lit)-1][0 if lit < 0 else 1]
                        if lit_prob < worst_prob:
                            worst_prob = lit_prob
                            worst_lit = lit

                return worst_lit
    
    
    class NeuroWalkSATCL_dyn_best(WalkSATCL_dyn):
        def suggest(self, cnf: CNF, state, random_clause, break_counts, clauses_satisfiability):
            reduced_state = state.copy()
            unsat_clauses = [cnf.clauses[i] for i in np.argwhere(np.array(clauses_satisfiability)==False).squeeze(axis=1)]
            joined_unsat_clauses = set([l for c in unsat_clauses for l in c])
            
            if self.verbose > 0: print(f"Joined unsat clauses: {joined_unsat_clauses}")
            for l in joined_unsat_clauses:
                reduced_state.remove(-l)
            
            # Assign values to the variables in reduced_cnf with reduced_state
            reduced_cnf_clauses = []
            for c in cnf.clauses:
                clause = set(c)
                satisfied = False
                for v in reduced_state:
                    if v in clause:
                        satisfied = True
                        break
                    elif -v in clause:
                        clause.remove(-v)
                if not satisfied:
                    reduced_cnf_clauses.append(clause)
            reduced_cnf = CNF(reduced_cnf_clauses)
            
            if self.verbose > 0: print(f"Reduced CNF clauses: {reduced_cnf.clauses}")
            
            # Apply unit propagation and pure literal elimination
            reduced_cnf, assigned_vars = self.shorten_cnf(reduced_cnf)
            
            if self.verbose > 0: print(f"Reduced CNF clauses (after shorten_cnf): {reduced_cnf.clauses}")
            if self.verbose > 0: print(f"Assigned variables (after shorten_cnf): {assigned_vars}")
            
            if reduced_cnf.is_true():
                # If the reduced_cnf is trivially true, return the literales which need to be flipped
                return assigned_vars
            elif reduced_cnf.is_false():
                # If the reduced_cnf is unsatisfiable, add a new clause with the negation of the literals in reduced_state
                #   and flip the variable in reduced_state with minimum break-count
                new_clause = tuple([-l for l in reduced_state])
                
                if self.verbose > 0: print(f"Reduced CNF is unsatisfiable. New clause: {new_clause}")
                
                break_counts = []
                literal = new_clause[len(break_counts)]
                bc = self.get_break_count(literal, cnf.clauses, clauses_satisfiability, state)
                break_counts.append(bc)
                while break_counts[-1] != 0 and len(break_counts) < len(new_clause):
                    literal = new_clause[len(break_counts)]
                    bc = self.get_break_count(literal, cnf.clauses, clauses_satisfiability, state)
                    break_counts.append(bc)
                
                if self.verbose > 0: print(f"    Break counts: {break_counts}")
                
                if break_counts[-1] == 0:
                    # If exists a variable with break_count=0, flip that variable
                    suggested_var = new_clause[len(break_counts)-1]
                    if self.verbose > 0: print(f"    Flip variable with break-count=0 ({suggested_var})")
                else:
                    # Flip the variable with minimum break_count
                    suggested_var = self.suggest_walksat(cnf, state, new_clause, break_counts, clauses_satisfiability)
                    if self.verbose > 0: print(f"    Flip suggested variable ({suggested_var})")
                
                # Add the new learnt clause
                self.learnt_clauses.append(new_clause)
                if self.max_num_lc < len(self.learnt_clauses):
                    self.max_num_lc = len(self.learnt_clauses)
                if self.max_length_lc < len(new_clause):
                    self.max_length_lc = len(new_clause)
                self.memory_matrix = np.vstack((self.memory_matrix, np.zeros(self.MEMORY_STEPS,dtype=bool)))
                
                # Delete the easier clause to satisfy (that with fewer ones) if the learnt clauses limit has been reached
                while len(self.learnt_clauses) > self.MAX_NUM_LEARNT_CLAUSES:
                    lc_index = np.argmin(np.sum(self.memory_matrix[len(self.original_clauses):,:], axis=1))
                    if self.verbose > 0: print(f"Deleted clause ({len(self.original_clauses)+lc_index}): {self.learnt_clauses[lc_index]}")
                    self.learnt_clauses.pop(lc_index)
                    self.memory_matrix = np.delete(self.memory_matrix, len(self.original_clauses)+lc_index, axis=0)
                
                # Add the new list of learnt clauses to the cnf
                cnf.clauses = self.original_clauses + tuple(self.learnt_clauses)
                
                return suggested_var
            else:
                clause_num = len(reduced_cnf.clauses)
                var_num = max(reduced_cnf.vars)
                inputs = np.asarray([clauses_to_matrix(reduced_cnf.clauses, clause_num, var_num)] * BATCH_SIZE)

                policy_probs = predict_fn({"input": inputs})['policy_probabilities']
                if self.verbose > 0: print(f"Policy probs: {policy_probs}")
                
                best_prob = -1.0
                best_lit = None
                for lit in joined_unsat_clauses:
                    if abs(lit) in reduced_cnf.vars:
                        lit_prob = policy_probs[0][abs(lit)-1][0 if lit > 0 else 1]
                        if lit_prob > best_prob:
                            best_prob = lit_prob
                            best_lit = lit

                return best_lit
    
    
    def compute_steps_cl_dyn(sats, walksat_cls):
        flips = []
        tries = []
        flips_all = []
        tries_all = []
        max_num_lc = []
        max_length_lc = []
        solved = 0
        for sat in tqdm(sats):
            walksat = walksat_cls(max_flips=MAX_FLIPS_DYN, max_num_learnt_clauses=10*len(sat.vars), memory_steps=MEMORY_STEPS, verbose=VERBOSE)
            res = walksat.run(sat)
            if res is not None:
                flips.append(walksat.number_of_flips)
                tries.append(walksat.number_of_tries+1)
                flips_all.append(walksat.number_of_flips)
                tries_all.append(walksat.number_of_tries+1)
                max_num_lc.append(walksat.max_num_lc)
                max_length_lc.append(walksat.max_length_lc)
                solved += 1
            else:
                flips_all.append(None)
                tries_all.append(None)
        print("\tWith {} MAX_FLIPS solved {} problems out of {}".format(MAX_FLIPS_DYN, solved, len(sats)))
        print("\tFlips    : ",flips)
        print("\tTries    : ",tries)
        print("\tAll flips: ",flips_all)
        print("\tAll tries: ",tries_all)
        return flips, tries, max_num_lc, max_length_lc
    
    
    def compute_and_print_steps_cl_dyn(sats, walksat_cls):
        flips, tries, max_num_lc, max_length_lc = compute_steps_cl_dyn(sats, walksat_cls)
        print("\t#Sats: {}; avg flip: {:.2f}; stdev flip: {:.2f}; avg try: {:.2f}; stdev try: {:.2f}".format(
            len(flips), np.mean(flips), np.std(flips), np.mean(tries), np.std(tries)))
        if max(max_num_lc)>0:
            print("\tavg max_num_lc: {:.2f}; stdev max_num_lc: {:.2f}; avg max_length_lc: {:.2f}; stdev max_length_lc: {:.2f}".format(
                np.mean(max_num_lc), np.std(max_num_lc), np.mean(max_length_lc), np.std(max_length_lc)))
    
    
    
    
    
    
    # s - number of samples
    # n - max number of clauses, use 100 * m
    # m - number of variables

    def print_all(s, n, m):
        global S, N, M
        S = s
        N = n # number of clauses
        M = m # number of variables
        
        MAX_ATTEMPTS = 100000
        sats = []
        
        random.seed(8)
        np.random.seed(8)
        
        for index in range(MAX_ATTEMPTS):
            if len(sats) >= S:
                break
            sat = get_pos_SR(M, M, N)
            sats.append(sat)
        assert len(sats) == S
        print("We have generated {} formulas".format(len(sats)))
        print()
        print("*****   GSAT   *****")
        compute_and_print_steps_gsat(sats, GSAT)
        print()
        print("*****   WalkSAT   *****")
        compute_and_print_steps_walksat(sats, WalkSAT)
        print()
        print("*****   NeuroWalkSAT-sta-worst   *****")
        compute_and_print_steps_sta(sats, NeuroWalkSAT_sta_worst)
        print()
        print("*****   NeuroWalkSAT-sta-best   *****")
        compute_and_print_steps_sta(sats, NeuroWalkSAT_sta_best)
        print()
        print("*****   NeuroWalkSAT-dyn-worst   *****")
        compute_and_print_steps_dyn(sats, NeuroWalkSAT_dyn_worst)
        print()
        print("*****   NeuroWalkSAT-dyn-best   *****")
        compute_and_print_steps_dyn(sats, NeuroWalkSAT_dyn_best)
        print()
        print("*****   NeuroWalkSATCL-sta-worst   *****")
        compute_and_print_steps_cl_sta(sats, NeuroWalkSATCL_sta_worst)
        print()
        print("*****   NeuroWalkSATCL-sta-best   *****")
        compute_and_print_steps_cl_sta(sats, NeuroWalkSATCL_sta_best)
        print()
        print("*****   NeuroWalkSATCL-dyn-worst   *****")
        compute_and_print_steps_cl_dyn(sats, NeuroWalkSATCL_dyn_worst)
        print()
        print("*****   NeuroWalkSATCL-dyn-best   *****")
        compute_and_print_steps_cl_dyn(sats, NeuroWalkSATCL_dyn_best)
        print()
        print()
        print()
    
    
    
    
    
    print("######################")
    print("#####   SR(10)   #####")
    print("######################")
    print()
    print_all(100, 200, 10)
    
    print("######################")
    print("#####   SR(20)   #####")
    print("######################")
    print()
    print_all(100, 200, 20)
    
    print("######################")
    print("#####   SR(30)   #####")
    print("######################")
    print()
    print_all(100, 500, 30)
    
    print("######################")
    print("#####   SR(40)   #####")
    print("######################")
    print()
    print_all(100, 1000, 40)
    
    print("######################")
    print("#####   SR(50)   #####")
    print("######################")
    print()
    print_all(100, 1000, 50)
    
    print("######################")
    print("#####   SR(60)   #####")
    print("######################")
    print()
    print_all(100, 1000, 60)
    
    print("######################")
    print("#####   SR(70)   #####")
    print("######################")
    print()
    print_all(100, 1000, 70)

