# Copyright 2019 snkas
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ortools.linear_solver import pywraplp
import re


_REGEXP_SINGLE_VAR_NAME_START = r"^[A-Za-z]+[A-Za-z0-9_]*"
_REGEXP_SINGLE_VAR_NAME_ALL = r"^[A-Za-z]+[A-Za-z0-9_]*$"


def _is_valid_constant_float(value) -> bool:
    if re.search(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$", value.strip()) is None:
        return False
    else:
        return True


def _retrieve_core_line(line: str, line_nr: int) -> str:

    # Remove whitespace
    line = line.strip()

    # Remove any comments
    if line.find("//") != -1:
        line = line.split("//")[0]

    # Remove again any whitespace
    line = line.strip()

    # Empty lines are allowed
    if len(line) == 0:
        return line

    # Check for the semicolon at the end
    if line[-1] != ";":
        raise ValueError("Line %d does not end with a semi-colon." % line_nr)

    # Remove the semicolon
    line = line[0:len(line) - 1]

    # Remove again any whitespace
    line = line.strip()

    # Semicolon-only lines are not allowed
    if len(line) == 0:
        raise ValueError("Line %d ends with semi-colon but is empty otherwise." % line_nr)

    return line


def _parse_declaration(solver: pywraplp.Solver, core_line: str, line_nr: int, var_names: set):
    spl_whitespace = core_line.split(maxsplit=1)

    if spl_whitespace[0] != "int":
        raise ValueError("Declaration on line %d should start with \"int \"." % line_nr)

    if len(spl_whitespace) != 2:
        raise ValueError("Declaration on line %d has no variables." % line_nr)

    spl_variables = spl_whitespace[1].split(",")
    for raw_var in spl_variables:
        clean_var = raw_var.strip()
        if not re.match(_REGEXP_SINGLE_VAR_NAME_ALL, clean_var):
            raise ValueError("Non-permitted variable name (\"%s\") on line %d." % (clean_var, line_nr))
        if clean_var in var_names:
            raise ValueError("Variable \"%s\" declared again on line %d." % (clean_var, line_nr))
        var_names.add(clean_var)
        solver.IntVar(-solver.infinity(), solver.infinity(), clean_var)


def _set_coefficients(solver: pywraplp.Solver, objective_or_constraint, coefficient_part: str, line_nr: int,
                      var_names: set):

    # Strip the coefficient whitespace
    remainder = coefficient_part.strip()
    if len(remainder) == 0:
        raise ValueError("No variables present in equation on line %d." % line_nr)

    # All variables found
    var_names_found = set()

    running_constant_sum = 0.0
    had_at_least_one_variable = False
    while len(remainder) != 0:

        # Combination sign
        coefficient = 1.0
        combination_sign_match = re.search(r"^[-+]", remainder)
        if combination_sign_match is not None:
            if combination_sign_match.group() == "-":
                coefficient = -1.0
            remainder = remainder[1:].strip()

        # Real sign
        sign_match = re.search(r"^[-+]", remainder)
        if sign_match is not None:
            if sign_match.group() == "-":
                coefficient = coefficient * -1.0
            remainder = remainder[1:]  # There is no strip() here, as it must be directly in front of the mantissa

        # Mantissa and exponent
        mantissa_exp_match = re.search(r"^(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", remainder)
        whitespace_after = False
        if mantissa_exp_match is not None:
            coefficient = coefficient * float(mantissa_exp_match.group())
            remainder = remainder[mantissa_exp_match.span()[1]:]
            stripped_remainder = remainder.strip()
            if len(remainder) != len(stripped_remainder):
                whitespace_after = True
            remainder = stripped_remainder

        # Variable name
        var_name_match = re.search(_REGEXP_SINGLE_VAR_NAME_START, remainder)
        if var_name_match is not None:

            # It must have had at least one variable
            had_at_least_one_variable = True

            # Retrieve clean variable name
            clean_var = var_name_match.group()
            var_names.add(clean_var)
            if clean_var in var_names_found:
                raise ValueError("Variable \"%s\" found more than once on line %d." % (clean_var, line_nr))
            var_names_found.add(clean_var)
            solver_var = solver.LookupVariable(clean_var)
            if solver_var is None:
                solver_var = solver.NumVar(-solver.infinity(), solver.infinity(), clean_var)

            # Set coefficient
            objective_or_constraint.SetCoefficient(solver_var, coefficient)

            # Strip what we matched
            remainder = remainder[var_name_match.span()[1]:]
            stripped_remainder = remainder.strip()
            whitespace_after = False
            if len(remainder) != len(stripped_remainder):
                whitespace_after = True
            remainder = stripped_remainder

        elif mantissa_exp_match is None:
            raise ValueError("Cannot process remainder coefficients of \"%s\" on line %d." % (remainder, line_nr))

        else:
            running_constant_sum += coefficient

        # At the end of each element there either:
        # (a) Must be whitespace (e.g., x1 x2 <= 10)
        # (b) The next combination sign (e.g., x1+x2 <= 10)
        # (c) Or it was the last one, as such remainder is empty (e.g., x1 <= 10)
        if len(remainder) != 0 and not whitespace_after and remainder[0:1] != "-" and remainder[0:1] != "+":
            raise ValueError(
                "Unexpected next character \"%s\" on line %d (expected whitespace or "
                "combination sign character)." % (remainder[0:1], line_nr)
            )

    # There must have been at least one variable
    if not had_at_least_one_variable:
        raise ValueError("Not a single variable present in the coefficients on line %d." % line_nr)

    return running_constant_sum


def _attempt_to_improve_var_bounds_one_hs(solver: pywraplp.Solver, coefficient_part: str, is_leq: bool,
                                          right_hand_side: str):
    coefficient_part = coefficient_part.strip()
    if re.match(_REGEXP_SINGLE_VAR_NAME_ALL, coefficient_part):
        if is_leq:
            solver.LookupVariable(coefficient_part).SetUb(float(right_hand_side))
        else:
            solver.LookupVariable(coefficient_part).SetLb(float(right_hand_side))


def _attempt_to_improve_var_bounds_two_hs(solver: pywraplp.Solver, coefficient_part: str, is_leq: bool,
                                          left_hand_side: str, right_hand_side: str):
    coefficient_part = coefficient_part.strip()
    if re.match(_REGEXP_SINGLE_VAR_NAME_ALL, coefficient_part):
        if is_leq:
            solver.LookupVariable(coefficient_part).SetLb(float(left_hand_side))
            solver.LookupVariable(coefficient_part).SetUb(float(right_hand_side))
        else:
            solver.LookupVariable(coefficient_part).SetLb(float(right_hand_side))
            solver.LookupVariable(coefficient_part).SetUb(float(left_hand_side))


def _parse_objective_function(solver: pywraplp.Solver, core_line: str, line_nr: int, var_names: set):
    spl_colon = core_line.split(":", maxsplit=1)

    objective = solver.Objective()

    # Set maximization / minimization if specified
    if len(spl_colon) == 1:
        raise ValueError("Objective function on line %d must start with \"max:\" or \"min:\"." % line_nr)
    else:
        if spl_colon[0] != "max" and spl_colon[0] != "min":
            raise ValueError("Objective function on line %d must start with \"max:\" or \"min:\"." % line_nr)
        elif spl_colon[0] == "max":
            objective.SetMaximization()
        else:
            objective.SetMinimization()

        # Set the remainder
        coefficient_part = spl_colon[1].strip()

        # Finally set the coefficients of the objective
        constant = _set_coefficients(solver, objective, coefficient_part, line_nr, var_names)
        objective.SetOffset(constant)


def _parse_constraint(solver: pywraplp.Solver, core_line: str, line_nr: int, var_names: set):

    # We don't care about the coefficient name before the colon
    constraint_part = core_line
    spl_colon = core_line.split(":", maxsplit=1)
    if len(spl_colon) > 1:
        constraint_part = spl_colon[1].strip()

    # Equality constraint
    if constraint_part.find("=") >= 0 and constraint_part.find("<=") == -1 and constraint_part.find(">=") == -1:
        equality_spl = constraint_part.split("=")
        if len(equality_spl) > 2:
            raise ValueError("Equality constraint on line %d has multiple equal signs." % line_nr)
        if not _is_valid_constant_float(equality_spl[1]):
            raise ValueError("Right hand side (\"%s\") of equality constraint on line %d is not a float "
                             "(e.g., variables are not allowed there!)."
                             % (equality_spl[1], line_nr))
        equal_value = float(equality_spl[1])
        constraint = solver.Constraint(equal_value, equal_value)
        constant = _set_coefficients(solver, constraint, equality_spl[0], line_nr, var_names)
        constraint.SetLb(constraint.Lb() - constant)
        constraint.SetUb(constraint.Ub() - constant)
        _attempt_to_improve_var_bounds_two_hs(solver, equality_spl[0], True, equality_spl[1], equality_spl[1])

    # Inequality constraints
    else:

        # Replace all of these inequality signs, because they are equivalent
        constraint_part = constraint_part.replace("<=", "<").replace(">=", ">")

        # lower bound < ... < upper bound
        if constraint_part.count("<") == 2:
            spl = constraint_part.split("<")
            if not _is_valid_constant_float(spl[0]):
                raise ValueError("Left hand side (\"%s\") of inequality constraint on line %d is not a float "
                                 "(e.g., variables are not allowed there!)."
                                 % (spl[0], line_nr))
            if not _is_valid_constant_float(spl[2]):
                raise ValueError("Right hand side (\"%s\") of inequality constraint on line %d is not a float "
                                 "(e.g., variables are not allowed there!)."
                                 % (spl[2], line_nr))
            constraint = solver.Constraint(float(spl[0]), float(spl[2]))
            constant = _set_coefficients(solver, constraint, spl[1], line_nr, var_names)
            constraint.SetLb(constraint.Lb() - constant)
            constraint.SetUb(constraint.Ub() - constant)
            _attempt_to_improve_var_bounds_two_hs(solver, spl[1], True, spl[0], spl[2])

        # upper bound > ... > lower bound
        elif constraint_part.count(">") == 2:
            spl = constraint_part.split(">")
            if not _is_valid_constant_float(spl[0]):
                raise ValueError("Left hand side (\"%s\") of inequality constraint on line %d is not a float "
                                 "(e.g., variables are not allowed there!)."
                                 % (spl[0], line_nr))
            if not _is_valid_constant_float(spl[2]):
                raise ValueError("Right hand side (\"%s\") of inequality constraint on line %d is not a float "
                                 "(e.g., variables are not allowed there!)."
                                 % (spl[2], line_nr))
            constraint = solver.Constraint(float(spl[2]), float(spl[0]))
            constant = _set_coefficients(solver, constraint, spl[1], line_nr, var_names)
            constraint.SetLb(constraint.Lb() - constant)
            constraint.SetUb(constraint.Ub() - constant)
            _attempt_to_improve_var_bounds_two_hs(solver, spl[1], False, spl[0], spl[2])

        # ... < upper bound
        elif constraint_part.count("<") == 1:
            spl = constraint_part.split("<")
            if not _is_valid_constant_float(spl[1]):
                raise ValueError("Right hand side (\"%s\") of inequality constraint on line %d is not a float "
                                 "(e.g., variables are not allowed there!)."
                                 % (spl[1], line_nr))
            constraint = solver.Constraint(-solver.infinity(), float(spl[1]))
            constant = _set_coefficients(solver, constraint, spl[0], line_nr, var_names)
            constraint.SetUb(constraint.Ub() - constant)
            _attempt_to_improve_var_bounds_one_hs(solver, spl[0], True, spl[1])

        # ... > lower bound
        elif constraint_part.count(">") == 1:
            spl = constraint_part.split(">")
            if not _is_valid_constant_float(spl[1]):
                raise ValueError("Right hand side (\"%s\") of inequality constraint on line %d is not a float "
                                 "(e.g., variables are not allowed there!)."
                                 % (spl[1], line_nr))
            constraint = solver.Constraint(float(spl[1]), solver.infinity())
            constant = _set_coefficients(solver, constraint, spl[0], line_nr, var_names)
            constraint.SetLb(constraint.Lb() - constant)
            _attempt_to_improve_var_bounds_one_hs(solver, spl[0], False, spl[1])

        # ...
        elif constraint_part.count(">") == 0 and constraint_part.count("<") == 0:
            raise ValueError("No (in)equality sign present for constraint on line %d." % line_nr)

        # Some strange combination
        else:
            raise ValueError("Too many (in)equality signs present for constraint on line %d." % line_nr)


def _set_declarations(solver: pywraplp.Solver, lp_filename: str, var_names: set):

    with open(lp_filename, "r") as lp_file:

        line_nr = 0
        in_objective_function = True
        in_constraints = False
        for line in lp_file:
            line_nr += 1

            # Retrieve the core of the line without heading or trailing whitespace and without comments
            core_line = _retrieve_core_line(line, line_nr)

            # Skip over empty lines
            if len(core_line) == 0:
                continue

            # The first non-empty core line must be the objective function
            if in_objective_function:
                in_objective_function = False
                in_constraints = True

            # Go over until we hit a line without colon and starting with "int<whitespace>"
            elif in_constraints:
                if core_line.find(":") == -1 and core_line.split()[0] == "int":
                    in_constraints = False
                    _parse_declaration(solver, core_line, line_nr, var_names)

            # Any core line from here on must be a declaration
            else:
                _parse_declaration(solver, core_line, line_nr, var_names)


def _set_objective_function_and_constraints(solver: pywraplp.Solver, lp_filename: str, var_names: set):

    with open(lp_filename, "r") as lp_file:

        line_nr = 0
        in_objective_function = True
        for line in lp_file:
            line_nr += 1

            # Retrieve the core of the line without heading or trailing whitespace and without comments
            core_line = _retrieve_core_line(line, line_nr)

            # Skip over empty lines
            if len(core_line) == 0:
                continue

            # The first non-empty core line must be the objective function
            if in_objective_function:
                _parse_objective_function(solver, core_line, line_nr, var_names)
                in_objective_function = False  # As such, next line we are in the constraints

            # If we are not in objective function, we are in the constraints section
            # Go over until we hit a line without colon and starting with "int<whitespace>"
            # (which would indicate the end of the constraint section)
            else:
                if core_line.find(":") == -1 and core_line.split()[0] == "int":
                    break  # The declaration section is not used in this function, so exit
                else:
                    _parse_constraint(solver, core_line, line_nr, var_names)


def parse_lp_file(lp_filename: str, use_solver=pywraplp.Solver.GLOP_LINEAR_PROGRAMMING):
    """
    Read in a linear program defined in a LP file.

    The LP specification format is similar to LPSolve 5.1.
    For the accepted format, consult the README of the python-ortools-lp-parser repository.

    :param lp_filename:  .lp file name (i.e. "/path/to/file.lp")
    :param use_solver:   Solver instance to use (optional)
                         (1) pywraplp.Solver.CLP_LINEAR_PROGRAMMING)
                         (2) pywraplp.Solver.GLOP_LINEAR_PROGRAMMING) -> Default
                         (3) pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
                         (4) pywraplp.Solver.BOP_INTEGER_PROGRAMMING)

    :return: Dictionary {
                "solver": pywraplp.Solver instance with the linear program in it
                "var_names": set of all variable names in the solver
            }
    """

    # Set of the names of all variables
    var_names = set()

    # Solver instantiation
    solver = pywraplp.Solver("LpFromFile", use_solver)

    # Set declarations (first pass over file)
    _set_declarations(solver, lp_filename, var_names)

    # Set objective function and constraints (second pass over file)
    _set_objective_function_and_constraints(solver, lp_filename, var_names)

    # Finally return
    return {
        "solver": solver,
        "var_names": var_names
    }
