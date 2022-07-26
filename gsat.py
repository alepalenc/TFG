import random

import numpy as np

from cnf import get_random_kcnf, CNF



class GSAT(object):
    def __init__(self, max_flips=100, max_tries=10, verbose=0):
        self.MAX_FLIPS = max_flips
        self.MAX_TRIES = max_tries
        self.number_of_flips = 0
        self.number_of_tries = 0
        self.verbose = verbose
    
    
    def run(self, cnf: CNF, init_state=None):
        assert isinstance(cnf, CNF)
        
        if self.verbose > 0: print('\n\n', cnf, '\n', cnf.clauses)
        
        num_clauses = len(cnf.clauses)
        
        self.number_of_tries = 0
        
        while self.number_of_tries < self.MAX_TRIES:
            if self.verbose > 0: print(f"\n\n///// NUMBER OF TRIES: {self.number_of_tries} /////")
            
            self.number_of_flips = 0
            
            # Set a random state
            if init_state == None:
                state = set([v*random.choice([1,-1]) for v in cnf.vars])
            else:
                state = init_state
            
            if self.verbose > 0: print(f"* Initial state: {state}")
                
            # Get a boolean array with the satisfiability of each clause
            clauses_satisfiability = [any(l in state for l in c) for c in cnf.clauses]
            num_satisfied_clauses = sum(clauses_satisfiability)
            if self.verbose > 0: print(f"Satisfiability of each clause ({num_satisfied_clauses}/{num_clauses}): {clauses_satisfiability}")
            
            # If all clauses are satisfied, return the state
            if num_satisfied_clauses == num_clauses:
                return state
            
            while self.number_of_flips < self.MAX_FLIPS:
                if self.verbose > 0: print(f"\n/// NUMBER OF FLIPS: {self.number_of_flips} ///\n")
                
                # Flip the variable which minimises the number of unsatisfied clauses
                best_lit = None
                best_clauses_satisfiability = None
                best_num_satisfied_clauses = -1
                for lit in state:
                    new_state = state.copy()
                    new_state.remove(lit)
                    new_state.add(-lit)
                    current_clauses_satisfiability = [any(l in new_state for l in c) for c in cnf.clauses]
                    current_num_satisfied_clauses = sum(current_clauses_satisfiability)
                    if current_num_satisfied_clauses > best_num_satisfied_clauses:
                        best_lit = lit
                        best_clauses_satisfiability = current_clauses_satisfiability
                        best_num_satisfied_clauses = current_num_satisfied_clauses
                
                if self.verbose > 0: print(f"* Flip suggested variable ({-best_lit})")
                state.remove(best_lit)
                state.add(-best_lit)
                clauses_satisfiability = best_clauses_satisfiability
                num_satisfied_clauses = best_num_satisfied_clauses
                
                if self.verbose > 0: print(f"    --> New state: {state}")
                if self.verbose > 0: print(f"Satisfiability of each clause ({num_satisfied_clauses}/{num_clauses}): {clauses_satisfiability}")
                
                self.number_of_flips += 1
                
                if num_satisfied_clauses == num_clauses:
                    return state
            
            
            self.number_of_tries += 1
        
        # If no solution is found, return None
        return None

