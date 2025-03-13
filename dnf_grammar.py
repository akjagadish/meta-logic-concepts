import multiprocessing
import time

import math
import numpy as np
import random

import logging
import sys

sys.setrecursionlimit(200)


class Primitive:
    def __init__(self, output_type, input_types, function, name):
        self.output_type = output_type
        self.input_types = input_types
        self.name = name
        self.function = function

class Grammar:
    def __init__(self):
        self.rules = {}
        self.weights = {}

    def add(self, primitive, weight):
        if primitive.output_type not in self.rules:
            self.rules[primitive.output_type] = []
            self.weights[primitive.output_type] = []

        self.rules[primitive.output_type].append(primitive)
        self.weights[primitive.output_type].append(weight)

    def delete(self, output_type):
        if output_type in self.rules:
            del self.rules[output_type]
            del self.weights[output_type]
        else:
            print(f"No rules found for output type '{output_type}'.")

    def __iter__(self):
        return iter(self.rules.items())

    def sample(self, nonterminal):
        if nonterminal in self.rules:
            choices = self.rules[nonterminal]
            weights = self.weights[nonterminal]
        else:
            raise ValueError("No choices available for nonterminal: " + nonterminal)

        choice = random.choices(choices, weights=weights)[0]
        
        if len(choice.input_types) == 0:
            # All base cases now return a function that takes (x, Set)
            return choice.function(), choice.name

        arg_functions = []
        arg_names = []
        
        for input_type in choice.input_types:
            arg_function, arg_name = self.sample(input_type)
            arg_functions.append(arg_function)
            arg_names.append(arg_name)
        
        # Function composition with consistent signatures
        # new_function = lambda x, Set: choice.function(*arg_functions)(x, Set)
        new_function = lambda *args: choice.function(*arg_functions)(*args)
        
        new_name = choice.name % tuple(arg_names)
        
        return new_function, new_name

    def pretty_print(self):
        for lhs in self.rules:
            for primitive in self.rules[lhs]:
                print(primitive.name)
                print(primitive.input_types)
                # rhs = primitive.name % tuple(primitive.input_types)
                rhs = primitive.name + " (" + ", ".join(primitive.input_types) + ")"
                print(lhs + " -> " + rhs)


class DNFHypothesis:
    def __init__(self, n_features=4, no_true_false_top=True, b=1):

        # Used for determining probability of outlier
        self.b = b
        self.p_outlier = math.exp(-1 * self.b) / (1 + math.exp(-1 * self.b))

        self.grammar = Grammar()

        if no_true_false_top:
            s1 = Primitive("S", ["D_top"], lambda x: x, "∀x l(x) <=> %s")
            self.grammar.add(s1, 1.0)  # Don't have to worry about probability - only one option

            d_top = Primitive("D_top", ["C_top", "D"], lambda x, y: x or y, "(%s or %s)")
            self.grammar.add(d_top, 1.0)  # Don't have to worry about probability - only one option

            c_top = Primitive("C_top", ["P", "C"], lambda x, y: x and y, "(%s and %s)")
            self.grammar.add(c_top, 1.0)  # Don't have to worry about probability - only one option

            d1 = Primitive("D", ["C_top", "D"], lambda x, y: x or y, "(%s or %s)")
            d2 = Primitive("D", [], lambda f: False, "False")

            # Random probabilities for "D" rules
            d_probs = np.random.dirichlet((1, 1))
            p_d1 = d_probs[0]
            p_d2 = d_probs[1]
            self.grammar.add(d1, p_d1)
            self.grammar.add(d2, p_d2)

        else:
            s1 = Primitive("S", ["D"], lambda x: x, "∀x l(x) <=> %s")
            self.grammar.add(s1, 1.0)  # Don't have to worry about probability - only one option

            d1 = Primitive("D", ["C", "D"], lambda x, y: x or y, "(%s or %s)")
            d2 = Primitive("D", [], lambda f: False, "False")

            # Random probabilities for "D" rules
            d_probs = np.random.dirichlet((1, 1))
            p_d1 = d_probs[0]
            p_d2 = d_probs[1]
            self.grammar.add(d1, p_d1)
            self.grammar.add(d2, p_d2)

        c1 = Primitive("C", ["P", "C"], lambda x, y: x and y, "(%s and %s)")
        c2 = Primitive("C", [], lambda f: True, "True")

        # Random probabilities for "C" rules
        c_probs = np.random.dirichlet((1, 1))
        p_c1 = c_probs[0]
        p_c2 = c_probs[1]
        self.grammar.add(c1, p_c1)
        self.grammar.add(c2, p_c2)

        p_probs = np.random.dirichlet([1 for _ in range(n_features)])
        for i in range(n_features):
            p_primitive = Primitive("P", ["F" + str(i + 1)], lambda x: x, "%s")
            self.grammar.add(p_primitive, p_probs[i])

            f_probs = np.random.dirichlet((1, 1))
            f1_primitive = Primitive("F" + str(i + 1), [], lambda f, i=i: f[i] == 1, "f_" + str(i + 1) + "(x) = 1")
            self.grammar.add(f1_primitive, f_probs[0])

            f2_primitive = Primitive("F" + str(i + 1), [], lambda f, i=i: f[i] == 0, "f_" + str(i + 1) + "(x) = 0")
            self.grammar.add(f2_primitive, f_probs[1])

        dataset_created = False
        while not dataset_created:
            # try/except to catch cases with recursion that's too deep
            try:
                self.function, self.name = self.grammar.sample("S")
                # print("FCT", self.function, self.name)
                example_input = [0 for _ in range(n_features)]
                pred = self.function(example_input)
                dataset_created = True
            except:
                pass

    def function_with_outliers(self, inp):

        correct_output = self.function(inp)
        if random.random() < self.p_outlier:
            return not correct_output
        else:
            return correct_output


class SimpleBooleanHypothesis:
    def __init__(self):
        
        self.alpha = np.random.uniform()
        self.gamma = np.random.uniform()
        self.grammar = Grammar()

        S = Primitive("S", ["BOOL"], lambda x: x, "%s")
        self.grammar.add(S, 1.0)  # Don't have to worry about probability - only one option

        # BOOL1 = Primitive("BOOL", ["BOOL", "BOOL"], lambda x, y: x & y, "(%s and %s)")
        # BOOL2 = Primitive("BOOL", ["BOOL", "BOOL"], lambda x, y: x | y, "(%s or %s)")
        # BOOL3 = Primitive("BOOL", ["BOOL"], lambda x: not x, "not (%s)")
        # BOOL4 = Primitive("BOOL", ["F"], lambda x: x, "%s")

        # p = 0.3
        # p_BOOL = [0.5 * p, 0.5 * p, 0.5 * (1-p), 0.5 * (1-p)]
        
        # p_BOOL = [0.2, 0.2, 0.1, 0.5] #np.random.dirichlet((20, 20, 5, 5))
        # print(p_BOOL)
        # 2*p1 + 2*p2 + 1*p3 + 0*p4 + 0*p5 + 0*p6 < 1
        # while 2 * p_BOOL[0] + 2 * p_BOOL[1] + p_BOOL[2] >= 1 or np.sum(p_BOOL) == 0:
        #     p_BOOL = np.random.dirichlet((10, 10, 1, 1))
        #     print(p_BOOL)
        #     break
            # T = np.random.gamma(1,1)
            # p_BOOL = np.power(p_BOOL, 1/T)
            # if np.sum(p_BOOL) == 0:
            #     continue
            # p_BOOL = p_BOOL / np.sum(p_BOOL)
        # # print(p_BOOL)
        # assert(np.sum(p_BOOL) >= 0.999 and np.sum(p_BOOL) <= 1.001)

        # self.grammar.add(BOOL1, p_BOOL[0])
        # self.grammar.add(BOOL2, p_BOOL[1])
        # self.grammar.add(BOOL3, p_BOOL[2])
        # self.grammar.add(BOOL4, p_BOOL[3])

        ''' if we include true and false add BOOL_top'''
        # S = Primitive("S", ["BOOL_top"], lambda x: x, "%s")
        # self.grammar.add(S, 1.0)  # Don't have to worry about probability - only one option

        # BOOL_top1 = Primitive("BOOL_top", ["BOOL", "BOOL"], lambda x, y: x & y, "(%s and %s)")
        # BOOL_top2 = Primitive("BOOL_top", ["BOOL", "BOOL"], lambda x, y: x | y, "(%s or %s)")
        # BOOL_top3 = Primitive("BOOL_top", ["BOOL"], lambda x: not x, "not (%s)")
        # BOOL_top4 = Primitive("BOOL_top", ["F"], lambda x: x, "%s")

        # p_BOOL_top = np.random.dirichlet((np.random.gamma(1,1), np.random.gamma(1,1), np.random.gamma(1,1), np.random.gamma(1,1)))
        # self.grammar.add(BOOL_top1, p_BOOL_top[0])
        # self.grammar.add(BOOL_top2, p_BOOL_top[1])
        # self.grammar.add(BOOL_top3, p_BOOL_top[2])
        # self.grammar.add(BOOL_top4, p_BOOL_top[3])

        # OLD code without BOOL_top
        # S = Primitive("S", ["BOOL"], lambda x: x, "%s")
        # self.grammar.add(S, 1.0)  # Don't have to worry about probability - only one option

        BOOL1 = Primitive("BOOL", ["BOOL", "BOOL"], lambda x, y: x & y, "(%s and %s)")
        BOOL2 = Primitive("BOOL", ["BOOL", "BOOL"], lambda x, y: x | y, "(%s or %s)")
        BOOL3 = Primitive("BOOL", ["BOOL"], lambda x: not x, "not (%s)")
        BOOL4 = Primitive("BOOL", [], lambda t: True, "true")
        BOOL5 = Primitive("BOOL", [], lambda f: False, "false")
        BOOL6 = Primitive("BOOL", ["F"], lambda x: x, "%s")
        
        p_BOOL = np.array([0, 0, 0, 0, 0, 0])
        # 2*p1 + 2*p2 + 1*p3 + 0*p4 + 0*p5 + 0*p6 < 1
        while 2 * p_BOOL[0] + 2 * p_BOOL[1] + p_BOOL[2] >= 1 or np.sum(p_BOOL) == 0:
            p_BOOL = np.random.dirichlet((np.random.gamma(1,1), np.random.gamma(1,1), np.random.gamma(1,1),
                                          np.random.gamma(1,1), np.random.gamma(1,1), np.random.gamma(1,1)))
            T = np.random.gamma(1,1)
            p_BOOL = np.power(p_BOOL, 1/T)
            if np.sum(p_BOOL) == 0:
                continue
            p_BOOL = p_BOOL / np.sum(p_BOOL)
        # print(p_BOOL)
        assert(np.sum(p_BOOL) >= 0.999 and np.sum(p_BOOL) <= 1.001)

        self.grammar.add(BOOL1, p_BOOL[0])
        self.grammar.add(BOOL2, p_BOOL[1])
        self.grammar.add(BOOL3, p_BOOL[2])
        self.grammar.add(BOOL4, p_BOOL[3])
        self.grammar.add(BOOL5, p_BOOL[4])
        self.grammar.add(BOOL6, p_BOOL[5])

        F1 = Primitive("F", ["COLOR"], lambda x: x, "%s")
        F2 = Primitive("F", ["SHAPE"], lambda x: x, "%s")
        F3 = Primitive("F", ["SIZE"], lambda x: x, "%s")

        # p_F = np.array([1/3, 1/3, 1/3])
        p_F = np.array([0, 0, 0])
        while np.sum(p_F) == 0:
            p_F = np.random.dirichlet((np.random.gamma(1,1), np.random.gamma(1,1), np.random.gamma(1,1)))
            T = np.random.gamma(1,1)
            p_F = np.power(p_F, 1/T)
        p_F = p_F / np.sum(p_F)
        assert(np.sum(p_F) >= 0.999 and np.sum(p_F) <= 1.001)

        self.grammar.add(F1, p_F[0])
        self.grammar.add(F2, p_F[1])
        self.grammar.add(F3, p_F[2])

        # colors [blue, green, yellow] blue: [1, 0, 0], green = [0, 1, 0], yellow = [0, 0, 1]
        COLOR1 = Primitive("COLOR", [], lambda x: x[0] == 1, "blue")
        COLOR2 = Primitive("COLOR", [], lambda x: x[1] == 1, "green")
        COLOR3 = Primitive("COLOR", [], lambda x: x[2] == 1, "yellow")

        # p_COLOR = np.array([1/3, 1/3, 1/3])
        p_COLOR = np.array([0, 0, 0])
        while np.sum(p_COLOR) == 0:
            p_COLOR = np.random.dirichlet((np.random.gamma(1,1), np.random.gamma(1,1), np.random.gamma(1,1)))
            T = np.random.gamma(1,1)
            p_COLOR = np.power(p_COLOR, 1/T)
        p_COLOR = p_COLOR / np.sum(p_COLOR)
        assert(np.sum(p_COLOR) >= 0.999 and np.sum(p_COLOR) <= 1.001)

        self.grammar.add(COLOR1, p_COLOR[0])
        self.grammar.add(COLOR2, p_COLOR[1])
        self.grammar.add(COLOR3, p_COLOR[2])

        # shapes [circle, square, triangle] circle: [1, 0, 0], square= [0, 1, 0], triangle = [0, 0, 1]
        SHAPE1 = Primitive("SHAPE", [], lambda x: x[3] == 1, "circle")
        SHAPE2 = Primitive("SHAPE", [], lambda x: x[4] == 1, "rectangle")
        SHAPE3 = Primitive("SHAPE", [], lambda x: x[5] == 1, "triangle")

        # p_SHAPE = np.array([1/3, 1/3, 1/3])
        p_SHAPE = np.array([0, 0, 0])
        while np.sum(p_SHAPE) == 0:
            p_SHAPE = np.random.dirichlet((np.random.gamma(1,1), np.random.gamma(1,1), np.random.gamma(1,1)))
            T = np.random.gamma(1,1)
            p_SHAPE = np.power(p_SHAPE, 1/T)
        p_SHAPE = p_SHAPE / np.sum(p_SHAPE)
        assert(np.sum(p_SHAPE) >= 0.999 and np.sum(p_SHAPE) <= 1.001)

        self.grammar.add(SHAPE1, p_SHAPE[0])
        self.grammar.add(SHAPE2, p_SHAPE[1])
        self.grammar.add(SHAPE3, p_SHAPE[2])

        # sizes [small, medium, large]: size1: [1, 0, 0], size2 = [0, 1, 0], size3 = [0, 0, 1]
        SIZE1 = Primitive("SIZE", [], lambda x: x[6] == 1, "size1")
        SIZE2 = Primitive("SIZE", [], lambda x: x[7] == 1, "size2")
        SIZE3 = Primitive("SIZE", [], lambda x: x[8] == 1, "size3")

        # p_SIZE = np.array([1/3, 1/3, 1/3])
        p_SIZE = np.array([0, 0, 0])
        while np.sum(p_SIZE) == 0:
            p_SIZE = np.random.dirichlet((np.random.gamma(1,1), np.random.gamma(1,1), np.random.gamma(1,1)))
            T = np.random.gamma(1,1)
            p_SIZE = np.power(p_SIZE, 1/T)
        p_SIZE = p_SIZE / np.sum(p_SIZE)
        assert(np.sum(p_SIZE) >= 0.999 and np.sum(p_SIZE) <= 1.001)

        self.grammar.add(SIZE1, p_SIZE[0])
        self.grammar.add(SIZE2, p_SIZE[1])
        self.grammar.add(SIZE3, p_SIZE[2])

        dataset_created = False
        while not dataset_created:
            # try/except to catch cases with recursion that's too deep
            try:
                self.function, self.name = self.grammar.sample("S")
                # print("FCT", self.function, self.name)
                example_input = [0 for _ in range(9)]
                pred = self.function(example_input)
                dataset_created = True
            except:
                pass

    def function_with_outliers(self, inp):

        correct_output = self.function(inp)
        if random.random() < self.alpha: # label is random
            if random.random() < self.gamma:
                return correct_output
            else:
                 return not correct_output
        else:
             return correct_output


class FOLHypothesis_init:
    def __init__(self):
        self.counter = 0
        self.alpha = np.random.uniform()
        self.gamma = np.random.uniform()
        self.grammar = Grammar()

        START = Primitive("START", ["BOOL"], lambda x, Set: x, "%s")
        self.grammar.add(START, 1.0)  # Don't have to worry about probability - only one option

        SET1 = Primitive("SET", [], lambda x, Set: Set, " ")
        SET2 = Primitive("SET", [], lambda x, Set: [elem for elem in Set if elem != x], " ")

        p_SET = np.array([0.5, 0.5]) #example probabilities
        p_SET = p_SET / np.sum(p_SET)
        self.grammar.add(SET1, p_SET[0])
        self.grammar.add(SET2, p_SET[1])


        BOOL1 = Primitive("BOOL", ["BOOL", "BOOL"], lambda x, y: x & y, "(%s and %s)")
        BOOL2 = Primitive("BOOL", ["BOOL", "BOOL"], lambda x, y: x | y, "(%s or %s)")
        BOOL3 = Primitive("BOOL", ["BOOL"], lambda x: not x, "not (%s)")
        BOOL4 = Primitive("BOOL", [], lambda t: True, "true")
        BOOL5 = Primitive("BOOL", [], lambda f: False, "false")
        BOOL6 = Primitive("BOOL", ["F"], lambda x: x, "%s")

        BOOL7 = Primitive("BOOL", ["F", "SET"], lambda x, Set: all(y for y in Set), "For all x in Set, x is")
        # BOOL7 = Primitive("BOOL", ["F", "SET"], lambda F, Set: all(F(x) for x in Set), "For all x in Set, (F x)")
        # BOOL8 = Primitive("BOOL", ["F", "SET"], lambda F, Set: any(F(x) for x in Set), "There exists some x in Set such that (F x)")
        BOOL8 = Primitive("BOOL", ["F", "SET"], lambda x, Set: any(y for y in Set), "There exists some x in Set such that x is")

        BOOL9 = Primitive("BOOL", [], lambda x, y: x[6:9].index(1)  > y[6:9].index(1),  "(size > %s %s)")  # x is larger than y
        BOOL10 = Primitive("BOOL", [], lambda x, y: x[6:9].index(1) >= y[6:9].index(1), "(size >= %s %s)")  # x is larger than or equal to y

        BOOL11 = Primitive("BOOL", [], lambda x, y: x[3:6] == y[3:6], "(equal-shape? %s %s)")
        BOOL12 = Primitive("BOOL", [], lambda x, y: x[0:3] == y[0:3], "(equal-color? %s %s)")
        BOOL13 = Primitive("BOOL", [], lambda x, y: x[6:9] == y[6:9], "(equal-size? %s %s)")
        
        p_BOOL = np.array([0] * 13)
        # 2*p1 + 2*p2 + 1*p3 + 0*p4 + 0*p5 + 0*p6 < 1
        while 2 * p_BOOL[0] + 2 * p_BOOL[1] + p_BOOL[2] >= 1 or np.sum(p_BOOL) == 0:
            print(p_BOOL)
            p_BOOL = np.random.dirichlet((np.random.gamma(1,1), np.random.gamma(1,1), np.random.gamma(1,1),
                                          np.random.gamma(1,1), np.random.gamma(1,1), np.random.gamma(1,1),
                                          np.random.gamma(1,1), np.random.gamma(1,1), np.random.gamma(1,1),
                                          np.random.gamma(1,1), np.random.gamma(1,1), np.random.gamma(1,1),
                                          np.random.gamma(1,1)))
            T = np.random.gamma(1,1)
            p_BOOL = np.power(p_BOOL, 1/T)
            if np.sum(p_BOOL) == 0:
                continue
            p_BOOL = p_BOOL / np.sum(p_BOOL)
        print("BOOLS", p_BOOL)
        assert(np.sum(p_BOOL) >= 0.999 and np.sum(p_BOOL) <= 1.001)

        self.grammar.add(BOOL1, p_BOOL[0])
        self.grammar.add(BOOL2, p_BOOL[1])
        self.grammar.add(BOOL3, p_BOOL[2])
        self.grammar.add(BOOL4, p_BOOL[3])
        self.grammar.add(BOOL5, p_BOOL[4])
        self.grammar.add(BOOL6, p_BOOL[5])
        self.grammar.add(BOOL7, p_BOOL[6])
        self.grammar.add(BOOL8, p_BOOL[7])
        self.grammar.add(BOOL9, p_BOOL[8])
        self.grammar.add(BOOL10, p_BOOL[9])
        self.grammar.add(BOOL11, p_BOOL[10])
        self.grammar.add(BOOL12, p_BOOL[11])
        self.grammar.add(BOOL13, p_BOOL[12])

        F1 = Primitive("F", ["COLOR"], lambda x: x, "%s")
        F2 = Primitive("F", ["SHAPE"], lambda x: x, "%s")
        F3 = Primitive("F", ["SIZE"], lambda x: x, "%s")
        F3 = Primitive("F", ["BOOL"], lambda x: x, "%s")

        # p_F = np.array([1/3, 1/3, 1/3])
        p_F = np.array([0, 0, 0, 0])
        while np.sum(p_F) == 0:
            p_F = np.random.dirichlet((np.random.gamma(1,1), np.random.gamma(1,1), np.random.gamma(1,1), np.random.gamma(1,1)))
            T = np.random.gamma(1,1)
            p_F = np.power(p_F, 1/T)
        p_F = p_F / np.sum(p_F)
        assert(np.sum(p_F) >= 0.999 and np.sum(p_F) <= 1.001)
        print("Fs", p_F)

        self.grammar.add(F1, p_F[0])
        self.grammar.add(F2, p_F[1])
        self.grammar.add(F3, p_F[2])

        # colors [blue, green, yellow] blue: [1, 0, 0], green = [0, 1, 0], yellow = [0, 0, 1]
        COLOR1 = Primitive("COLOR", [], lambda x: x[0] == 1, "blue")
        COLOR2 = Primitive("COLOR", [], lambda x: x[1] == 1, "green")
        COLOR3 = Primitive("COLOR", [], lambda x: x[2] == 1, "yellow")

        # p_COLOR = np.array([1/3, 1/3, 1/3])
        p_COLOR = np.array([0, 0, 0])
        while np.sum(p_COLOR) == 0:
            p_COLOR = np.random.dirichlet((np.random.gamma(1,1), np.random.gamma(1,1), np.random.gamma(1,1)))
            T = np.random.gamma(1,1)
            p_COLOR = np.power(p_COLOR, 1/T)
        p_COLOR = p_COLOR / np.sum(p_COLOR)
        assert(np.sum(p_COLOR) >= 0.999 and np.sum(p_COLOR) <= 1.001)
        print("COLORs", p_COLOR)

        self.grammar.add(COLOR1, p_COLOR[0])
        self.grammar.add(COLOR2, p_COLOR[1])
        self.grammar.add(COLOR3, p_COLOR[2])

        # shapes [circle, square, triangle] circle: [1, 0, 0], square= [0, 1, 0], triangle = [0, 0, 1]
        SHAPE1 = Primitive("SHAPE", [], lambda x: x[3] == 1, "circle")
        SHAPE2 = Primitive("SHAPE", [], lambda x: x[4] == 1, "rectangle")
        SHAPE3 = Primitive("SHAPE", [], lambda x: x[5] == 1, "triangle")

        # p_SHAPE = np.array([1/3, 1/3, 1/3])
        p_SHAPE = np.array([0, 0, 0])
        while np.sum(p_SHAPE) == 0:
            p_SHAPE = np.random.dirichlet((np.random.gamma(1,1), np.random.gamma(1,1), np.random.gamma(1,1)))
            T = np.random.gamma(1,1)
            p_SHAPE = np.power(p_SHAPE, 1/T)
        p_SHAPE = p_SHAPE / np.sum(p_SHAPE)
        assert(np.sum(p_SHAPE) >= 0.999 and np.sum(p_SHAPE) <= 1.001)
        print("SHAPEs", p_SHAPE)

        self.grammar.add(SHAPE1, p_SHAPE[0])
        self.grammar.add(SHAPE2, p_SHAPE[1])
        self.grammar.add(SHAPE3, p_SHAPE[2])

        # sizes [small, medium, large]: size1: [1, 0, 0], size2 = [0, 1, 0], size3 = [0, 0, 1]
        SIZE1 = Primitive("SIZE", [], lambda x: x[6] == 1, "size1")
        SIZE2 = Primitive("SIZE", [], lambda x: x[7] == 1, "size2")
        SIZE3 = Primitive("SIZE", [], lambda x: x[8] == 1, "size3")

        # p_SIZE = np.array([1/3, 1/3, 1/3])
        p_SIZE = np.array([0, 0, 0])
        while np.sum(p_SIZE) == 0:
            p_SIZE = np.random.dirichlet((np.random.gamma(1,1), np.random.gamma(1,1), np.random.gamma(1,1)))
            T = np.random.gamma(1,1)
            p_SIZE = np.power(p_SIZE, 1/T)
        p_SIZE = p_SIZE / np.sum(p_SIZE)
        assert(np.sum(p_SIZE) >= 0.999 and np.sum(p_SIZE) <= 1.001)
        print("SIZEs", p_SIZE)

        self.grammar.add(SIZE1, p_SIZE[0])
        self.grammar.add(SIZE2, p_SIZE[1])
        self.grammar.add(SIZE3, p_SIZE[2])

        dataset_created = False
        print(self.grammar.pretty_print())
        self.function, self.name = self.grammar.sample("START")

        # while not dataset_created:
        #     # try/except to catch cases with recursion that's too deep
        #     try:
        #         self.function, self.name = self.grammar.sample("START")
        #         print("FCT", self.function, self.name)
        #         example_input = [0 for _ in range(9)]
        #         pred = self.function(example_input)
        #         dataset_created = True
        #     except:
        #         pass

    def function_with_outliers(self, inp):

        correct_output = self.function(inp)
        if random.random() < self.alpha: # label is random
            if random.random() < self.gamma:
                return correct_output
            else:
                 return not correct_output
        else:
             return correct_output

class FOLHypothesis_comp:
    def __init__(self):
        self.counter = 0
        self.alpha = np.random.uniform()
        self.gamma = np.random.uniform()
        self.grammar = Grammar()

        START = Primitive("START", ["BOOL"], lambda f: lambda x, Set: f(x, Set), "%s")
        self.grammar.add(START, 1.0)

        # Set primitives
        SET1 = Primitive("SET", [], lambda: lambda x, Set: Set, "Set")
        SET2 = Primitive("SET", [], lambda: lambda x, Set: [elem for elem in Set if not np.array_equal(elem, x)], "Set-without-x")
        
        p_SET = np.array([0.6, 0.4])
        self.grammar.add(SET1, p_SET[0])
        self.grammar.add(SET2, p_SET[1])

        # Boolean operations
        BOOL1 = Primitive("BOOL", ["BOOL", "BOOL"], 
                         lambda f1, f2: lambda x, Set: f1(x, Set) and f2(x, Set), 
                         "(%s and %s)")
        BOOL2 = Primitive("BOOL", ["BOOL", "BOOL"], 
                         lambda f1, f2: lambda x, Set: f1(x, Set) or f2(x, Set), 
                         "(%s or %s)")
        BOOL3 = Primitive("BOOL", ["BOOL"], 
                         lambda f: lambda x, Set: not f(x, Set), 
                         "not (%s)")

        # Quantifiers with comparison handling
        BOOL6 = Primitive("BOOL", ["SET", "COMP"], 
                         lambda s, comp: lambda x, Set: all(comp(x, y, Set) for y in s(x, Set)), 
                         "For all y in %s, %s")
        BOOL7 = Primitive("BOOL", ["SET", "COMP"], 
                         lambda s, comp: lambda x, Set: any(comp(x, y, Set) for y in s(x, Set)), 
                         "There exists y in %s such that %s")

        # Comparisons - now as a separate type COMP
        COMP1 = Primitive("COMP", [], 
                         lambda: lambda x, y, Set: x[6:9].index(1) > y[6:9].index(1), 
                         "x is larger than y")
        COMP2 = Primitive("COMP", [], 
                         lambda: lambda x, y, Set: x[6:9].index(1) >= y[6:9].index(1), 
                         "x is at least as large as y")
        COMP3 = Primitive("COMP", [], 
                         lambda: lambda x, y, Set: x[3:6] == y[3:6], 
                         "x has same shape as y")
        COMP4 = Primitive("COMP", [], 
                         lambda: lambda x, y, Set: x[0:3] == y[0:3], 
                         "x has same color as y")
        COMP5 = Primitive("COMP", [], 
                         lambda: lambda x, y, Set: x[6:9] == y[6:9], 
                         "x has same size as y")

        # Basic properties (for non-comparative statements)
        F1 = Primitive("F", ["COLOR"], 
                      lambda color: lambda x, Set: color(x, Set), 
                      "%s")
        F2 = Primitive("F", ["SHAPE"], 
                      lambda shape: lambda x, Set: shape(x, Set), 
                      "%s")
        F3 = Primitive("F", ["SIZE"], 
                      lambda size: lambda x, Set: size(x, Set), 
                      "%s")

        # Base types
        COLOR1 = Primitive("COLOR", [], lambda: lambda x, Set: x[0] == 1, "blue")
        COLOR2 = Primitive("COLOR", [], lambda: lambda x, Set: x[1] == 1, "green")
        COLOR3 = Primitive("COLOR", [], lambda: lambda x, Set: x[2] == 1, "yellow")

        SHAPE1 = Primitive("SHAPE", [], lambda: lambda x, Set: x[3] == 1, "circle")
        SHAPE2 = Primitive("SHAPE", [], lambda: lambda x, Set: x[4] == 1, "rectangle")
        SHAPE3 = Primitive("SHAPE", [], lambda: lambda x, Set: x[5] == 1, "triangle")

        SIZE1 = Primitive("SIZE", [], lambda: lambda x, Set: x[6] == 1, "small")
        SIZE2 = Primitive("SIZE", [], lambda: lambda x, Set: x[7] == 1, "medium")
        SIZE3 = Primitive("SIZE", [], lambda: lambda x, Set: x[8] == 1, "large")

        # Add primitives with probabilities
        primitives = {
            'BOOL': ([BOOL1, BOOL2, BOOL3, BOOL6, BOOL7], 5),
            'COMP': ([COMP1, COMP2, COMP3, COMP4, COMP5], 5),
            'F': ([F1, F2, F3], 3),
            'COLOR': ([COLOR1, COLOR2, COLOR3], 3),
            'SHAPE': ([SHAPE1, SHAPE2, SHAPE3], 3),
            'SIZE': ([SIZE1, SIZE2, SIZE3], 3)
        }

        for type_name, (prims, count) in primitives.items():
            p = np.random.dirichlet(np.ones(count))
            for prim, prob in zip(prims, p):
                self.grammar.add(prim, prob)

        self.function, self.name = self.grammar.sample("START")

class FOLHypothesis:
    def __init__(self):
        self.counter = 0
        self.alpha = np.random.uniform()
        self.gamma = np.random.uniform()
        self.grammar = Grammar()

        START = Primitive("START", ["BOOL"], lambda f: lambda x, Set: f(x, Set), "%s")
        self.grammar.add(START, 1.0)

        # Set primitives
        SET1 = Primitive("SET", [], lambda: lambda x, Set: Set, "Set")
        SET2 = Primitive("SET", [], lambda: lambda x, Set: [elem for elem in Set if not np.array_equal(elem, x)], "Set-without-x")
        
        p_SET = np.array([0.6, 0.4])
        self.grammar.add(SET1, p_SET[0])
        self.grammar.add(SET2, p_SET[1])

        # Boolean operations
        BOOL1 = Primitive("BOOL", ["BOOL", "BOOL"], 
                         lambda f1, f2: lambda x, Set: f1(x, Set) and f2(x, Set), 
                         "(%s and %s)")
        BOOL2 = Primitive("BOOL", ["BOOL", "BOOL"], 
                         lambda f1, f2: lambda x, Set: f1(x, Set) or f2(x, Set), 
                         "(%s or %s)")
        BOOL3 = Primitive("BOOL", ["BOOL"], 
                         lambda f: lambda x, Set: not f(x, Set), 
                         "not (%s)")

        # Quantifiers take SET and PRED
        BOOL6 = Primitive("BOOL", ["SET", "PRED"], 
                         lambda s, pred: lambda x, Set: all(pred(x, y, Set) for y in s(x, Set)), 
                         "For all y in %s, %s")
        BOOL7 = Primitive("BOOL", ["SET", "PRED"], 
                         lambda s, pred: lambda x, Set: any(pred(x, y, Set) for y in s(x, Set)), 
                         "There exists y in %s such that %s")

        # PRED wraps both COMP and F
        PRED1 = Primitive("PRED", ["COMP"], 
                         lambda comp: comp,  # Pass through the comparison function directly
                         "%s")
        PRED2 = Primitive("PRED", ["F"], 
                         lambda f: lambda x, y, Set: f(y, Set),  # Adapt F to take 3 args
                         "%s")

        # Comparisons - now with explicit 3-argument functions
        COMP1 = Primitive("COMP", [], 
                         lambda: lambda x, y, Set: x[6:9].index(1) > y[6:9].index(1), 
                         "x is larger than y")
        COMP2 = Primitive("COMP", [], 
                         lambda: lambda x, y, Set: x[6:9].index(1) >= y[6:9].index(1), 
                         "x is at least as large as y")
        COMP3 = Primitive("COMP", [], 
                         lambda: lambda x, y, Set: x[3:6] == y[3:6], 
                         "x has same shape as y")
        COMP4 = Primitive("COMP", [], 
                         lambda: lambda x, y, Set: x[0:3] == y[0:3], 
                         "x has same color as y")
        COMP5 = Primitive("COMP", [], 
                         lambda: lambda x, y, Set: x[6:9] == y[6:9], 
                         "x has same size as y")

        # Basic properties
        F1 = Primitive("F", ["COLOR"], 
                      lambda color: lambda x, Set: color(x, Set), 
                      "%s")
        F2 = Primitive("F", ["SHAPE"], 
                      lambda shape: lambda x, Set: shape(x, Set), 
                      "%s")
        F3 = Primitive("F", ["SIZE"], 
                      lambda size: lambda x, Set: size(x, Set), 
                      "%s")

        # Direct property check for non-quantified statements
        BOOL8 = Primitive("BOOL", ["F"],
                         lambda f: lambda x, Set: f(x, Set),
                         "%s")

        # Base types
        COLOR1 = Primitive("COLOR", [], lambda: lambda x, Set: x[0] == 1, "blue")
        COLOR2 = Primitive("COLOR", [], lambda: lambda x, Set: x[1] == 1, "green")
        COLOR3 = Primitive("COLOR", [], lambda: lambda x, Set: x[2] == 1, "yellow")

        SHAPE1 = Primitive("SHAPE", [], lambda: lambda x, Set: x[3] == 1, "circle")
        SHAPE2 = Primitive("SHAPE", [], lambda: lambda x, Set: x[4] == 1, "rectangle")
        SHAPE3 = Primitive("SHAPE", [], lambda: lambda x, Set: x[5] == 1, "triangle")

        SIZE1 = Primitive("SIZE", [], lambda: lambda x, Set: x[6] == 1, "small")
        SIZE2 = Primitive("SIZE", [], lambda: lambda x, Set: x[7] == 1, "medium")
        SIZE3 = Primitive("SIZE", [], lambda: lambda x, Set: x[8] == 1, "large")

        # Add primitives with probabilities
        primitives = {
            'BOOL': ([BOOL1, BOOL2, BOOL3, BOOL6, BOOL7, BOOL8], 6),
            'PRED': ([PRED1, PRED2], 2),
            'COMP': ([COMP1, COMP2, COMP3, COMP4, COMP5], 5),
            'F': ([F1, F2, F3], 3),
            'COLOR': ([COLOR1, COLOR2, COLOR3], 3),
            'SHAPE': ([SHAPE1, SHAPE2, SHAPE3], 3),
            'SIZE': ([SIZE1, SIZE2, SIZE3], 3)
        }

        for type_name, (prims, count) in primitives.items():
            p = np.random.dirichlet(np.ones(count))
            for prim, prob in zip(prims, p):
                self.grammar.add(prim, prob)

        # dataset_created = False
        # while not dataset_created:
        #     # try/except to catch cases with recursion that's too deep
        #     try:
        #         self.function, self.name = self.grammar.sample("START")
        #         # print("FCT", self.function, self.name)
        #         # example_input = [0 for _ in range(9)]
        #         # pred = self.function(example_input)
        #         dataset_created = True
        #     except RecursionError:
        #         print("RecursionError occurred")
        attempts = 0
        dataset_created = False
        max_attempts = 200
        
        while not dataset_created and attempts < max_attempts:
            try:
                # Reset recursion limit for each attempt
                sys.setrecursionlimit(200)
                
                # Your sampling logic here
                self.function, self.name = self.grammar.sample("START")
                dataset_created = True
                
            except RecursionError:
                attempts += 1
                # Optional: Add a small delay or print debugging info
                print(f"Recursion error on attempt {attempts}")
                # Clear any stored state that might affect next attempt
                # self.grammar.reset()  # If applicable
                
        if not dataset_created:
            raise RuntimeError(f"Failed to create dataset after {self.max_attempts} attempts")

        # self.function, self.name = self.grammar.sample("START")

        # dataset_created = False
        # while not dataset_created:
        #     # try/except to catch cases with recursion that's too deep
        #     try:
        #         self.function, self.name = self.grammar.sample("START")
        #         # print(self.name)
        #         # example_input = [0 for _ in range(9)]
        #         # pred = self.function(example_input)
        #         # print(pred)
        #         dataset_created = True
        #         # print(dataset_created)
        #         break
        #     except:
        #         pass

    def function_with_outliers(self, inp):
        correct_output = self.function(inp)
        if random.random() < self.alpha: # label is random
            if random.random() < self.gamma:
                return correct_output
            else:
                 return not correct_output
        else:
             return correct_output

if __name__ == "__main__":
    # my_hyp = DNFHypothesis(n_features=4, no_true_false_top=True, b=1)
    # print(my_hyp.name)

    # feature_values = [[0, 0, 0, 0],
    #                   [0, 0, 0, 1],
    #                   [0, 0, 1, 0],
    #                   [0, 0, 1, 1],
    #                   [0, 1, 0, 0],
    #                   [0, 1, 0, 1],
    #                   [0, 1, 1, 0],
    #                   [0, 1, 1, 1],
    #                   [1, 0, 0, 0],
    #                   [1, 0, 0, 1],
    #                   [1, 0, 1, 0],
    #                   [1, 0, 1, 1],
    #                   [1, 1, 0, 0],
    #                   [1, 1, 0, 1],
    #                   [1, 1, 1, 0],
    #                   [1, 1, 1, 1]]

    # for features in feature_values:
    #     print(features, my_hyp.function(features), my_hyp.function_with_outliers(features))

    # print("")
    # print("Grammar:")
    # my_hyp.grammar.pretty_print()

    # varied = 0
    # total = 0
    # for _ in range(1):
    #     my_hyp = DNFHypothesis()
    #     all_true = True
    #     all_false = True
    #     for features in feature_values:
    #         output = my_hyp.function(features)
    #         if output:
    #             all_false = False
    #         else:
    #             all_true = False
    #     if not all_false and not all_true:
    #         varied += 1
    #     total += 1

    # print("Varied grammars:", varied, total)

    # # feature_values = np.random.randint(2, size=(5, 9))

    # # for features in feature_values:
    # #     print(features, my_hyp.function(features), my_hyp.function_with_outliers(features))
    my_hyp = FOLHypothesis()
    print(my_hyp.name)

    feature_values = [[0, 0, 1, 0, 1, 0, 1, 0, 0],
                      [0, 1, 0, 1, 0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0, 1, 0, 1, 0],
                      [0, 0, 1, 1, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0]]

    for features in feature_values[:4]:
        print(features)
        print(my_hyp.function(features, feature_values[:4]))
        # print(my_hyp.function_with_outliers(features, feature_values))

    # print("")
    # print("Grammar:")
    # my_hyp.grammar.pretty_print()

    varied = 0
    total = 0
    for _ in range(100000):
        my_hyp = FOLHypothesis()
        print(my_hyp.name)
        all_true = True
        all_false = True
        for features in feature_values[:4]:
            output = my_hyp.function(features, feature_values[:4])
            if output:
                all_false = False
            else:
                all_true = False
        if not all_false and not all_true:
            varied += 1
        total += 1

    print("Varied grammars:", varied, total)

    # feature_values = np.random.randint(2, size=(5, 9))

    # for features in feature_values:
    #     print(features, my_hyp.function(features), my_hyp.function_with_outliers(features))

