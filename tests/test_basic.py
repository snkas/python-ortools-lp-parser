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

import ortoolslpparser
import unittest
import os
from ortools.linear_solver import pywraplp

TEMP_FILE = "program.lp"


def write_lp_program(lines):
    with open(TEMP_FILE, "w+") as program_lp_file:
        for line in lines:
            program_lp_file.write("%s\r\n" % line)


class TestValid(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        if os.path.isfile(TEMP_FILE):
            os.remove(TEMP_FILE)

    def test_readme(self):

        write_lp_program([
            "max: x1 - x2;",
            "x1 >= 0.3;",
            "x1 <= 30.6;",
            "x2 >= 24.9;",
            "x2 <= 50.1;",
        ])

        solver = ortoolslpparser.parse_lp_file(TEMP_FILE)["solver"]
        status = solver.Solve()
        self.assertEqual(status, solver.OPTIMAL)

        self.assertAlmostEqual(solver.LookupVariable("x1").solution_value(), 30.6)
        self.assertAlmostEqual(solver.LookupVariable("x2").solution_value(), 24.9)
        self.assertAlmostEqual(solver.Objective().Value(), 5.7)

    def test_simple_max(self):

        write_lp_program([
            "max: x1 + x2;",
            "x1 >= 0;",
            "x1 <= 30;",
            "x2 >= 24;",
            "x2 <= 50;"
        ])

        solver = ortoolslpparser.parse_lp_file(TEMP_FILE)["solver"]
        status = solver.Solve()
        self.assertEqual(status, solver.OPTIMAL)

        self.assertAlmostEqual(solver.LookupVariable("x1").solution_value(), 30.0)
        self.assertAlmostEqual(solver.LookupVariable("x2").solution_value(), 50.0)
        self.assertAlmostEqual(solver.Objective().Value(), 80.0)

    def test_simple_max_coefficient(self):

        write_lp_program([
            "max: +1.0x1 + x2;",
            "x1 >= 0;",
            "x1 <= 30;",
            "x2 >= 24;",
            "x2 <= 50;"
        ])

        solver = ortoolslpparser.parse_lp_file(TEMP_FILE)["solver"]
        status = solver.Solve()
        self.assertEqual(status, solver.OPTIMAL)

        self.assertAlmostEqual(solver.LookupVariable("x1").solution_value(), 30.0)
        self.assertAlmostEqual(solver.LookupVariable("x2").solution_value(), 50.0)
        self.assertAlmostEqual(solver.Objective().Value(), 80.0)

    def test_underscore_var_names(self):

        write_lp_program([
            "max: x_1 + x_2;",
            "x_1 >= 0;",
            "x_1 <= 30;",
            "x_2 >= 24;",
            "x_2 <= 50;",
            "x3_ <= 200;"
        ])

        solver = ortoolslpparser.parse_lp_file(TEMP_FILE)["solver"]
        status = solver.Solve()
        self.assertEqual(status, solver.OPTIMAL)

        self.assertAlmostEqual(solver.LookupVariable("x_1").solution_value(), 30.0)
        self.assertAlmostEqual(solver.LookupVariable("x_2").solution_value(), 50.0)
        self.assertAlmostEqual(solver.Objective().Value(), 80.0)

    def test_fail_constraint_constant(self):

        write_lp_program([
            "min: x1;",
            "x1 + 5 >= 0;",
            "x1 <= 30;",
        ])

        solver = ortoolslpparser.parse_lp_file(TEMP_FILE)["solver"]
        status = solver.Solve()
        self.assertEqual(status, solver.OPTIMAL)

        self.assertAlmostEqual(solver.LookupVariable("x1").solution_value(), -5.0)
        self.assertAlmostEqual(solver.Objective().Value(), -5.0)

    def test_simple_equality(self):

        write_lp_program([
            "max: x1 + x2;",
            "x1 = 30;",
            "x2 <= 99.2;"
        ])

        solver = ortoolslpparser.parse_lp_file(TEMP_FILE)["solver"]
        status = solver.Solve()
        self.assertEqual(status, solver.OPTIMAL)

        self.assertAlmostEqual(solver.LookupVariable("x1").solution_value(), 30.0)
        self.assertAlmostEqual(solver.LookupVariable("x2").solution_value(), 99.2)
        self.assertAlmostEqual(solver.Objective().Value(), 129.2)

    def test_underscore_in_name(self):

        write_lp_program([
            "max: x_1 + x_2;",
            "x_1 = 30;",
            "x_2 <= 99.2;"
        ])

        solver = ortoolslpparser.parse_lp_file(TEMP_FILE)["solver"]
        status = solver.Solve()
        self.assertEqual(status, solver.OPTIMAL)

        self.assertAlmostEqual(solver.LookupVariable("x_1").solution_value(), 30.0)
        self.assertAlmostEqual(solver.LookupVariable("x_2").solution_value(), 99.2)
        self.assertAlmostEqual(solver.Objective().Value(), 129.2)

    def test_simple_min(self):

        write_lp_program([
            "min: x1 + x2;",
            "x1 >= 0;",
            "x1 <= 30;",
            "x2 >= 24;",
            "x2 <= 50;"
        ])

        solver = ortoolslpparser.parse_lp_file(TEMP_FILE)["solver"]
        status = solver.Solve()
        self.assertEqual(status, solver.OPTIMAL)

        self.assertAlmostEqual(solver.LookupVariable("x1").solution_value(), 0.0)
        self.assertAlmostEqual(solver.LookupVariable("x2").solution_value(), 24.0)
        self.assertAlmostEqual(solver.Objective().Value(), 24.0)

    def test_simple_coefficients(self):

        write_lp_program([
            "max: +-0.1x1 + 0.3 x2;",
            "x1 >= 0;",
            "x1 <= 30;",
            "x2 >= 24;",
            "x2 <= 50;"
        ])

        solver = ortoolslpparser.parse_lp_file(TEMP_FILE)["solver"]
        status = solver.Solve()
        self.assertEqual(status, solver.OPTIMAL)

        self.assertAlmostEqual(solver.LookupVariable("x1").solution_value(), 0.0)
        self.assertAlmostEqual(solver.LookupVariable("x2").solution_value(), 50.0)
        self.assertAlmostEqual(solver.Objective().Value(), 15.0)

    def test_simple_whitespace_after_coefficient(self):

        write_lp_program([
            "max: --x1 + +x2+1--1;",
            "x1 >= 0;",
            "x1 <= 30;",
            "x2 >= 24;",
            "x2 <= 50;"
        ])

        solver = ortoolslpparser.parse_lp_file(TEMP_FILE)["solver"]
        status = solver.Solve()
        self.assertEqual(status, solver.OPTIMAL)

        self.assertAlmostEqual(solver.LookupVariable("x1").solution_value(), 30.0)
        self.assertAlmostEqual(solver.LookupVariable("x2").solution_value(), 50.0)
        self.assertAlmostEqual(solver.Objective().Value(), 82.0)

    def test_simple_naming(self):

        write_lp_program([
            "max: x1 + x2;",
            "my_constraint: x1 >= 0;",
            "x1 <= 30;",
            "x2 >= 24;",
            "x2 <= 50;"
        ])

        solver = ortoolslpparser.parse_lp_file(TEMP_FILE)["solver"]
        status = solver.Solve()
        self.assertEqual(status, solver.OPTIMAL)

        self.assertAlmostEqual(solver.LookupVariable("x1").solution_value(), 30.0)
        self.assertAlmostEqual(solver.LookupVariable("x2").solution_value(), 50.0)
        self.assertAlmostEqual(solver.Objective().Value(), 80.0)

    def test_simple_max_condensed(self):

        write_lp_program([
            "max: -x1 + x2;",
            "0 <= x1 <= 30.6;",
            "50.5 >= x2 >= 24;",
        ])

        solver = ortoolslpparser.parse_lp_file(TEMP_FILE)["solver"]
        status = solver.Solve()
        self.assertEqual(status, solver.OPTIMAL)

        self.assertAlmostEqual(solver.LookupVariable("x1").solution_value(), 0.0)
        self.assertAlmostEqual(solver.LookupVariable("x2").solution_value(), 50.5)
        self.assertAlmostEqual(solver.Objective().Value(), 50.5)

    def test_simple_integers(self):

        write_lp_program([
            "max: x1 + x2;",
            "0 <= x1 <= 30.9;",
            "50.4 >= x2 >= 24.8;",
            "int x1, x2;",
        ])

        solver = ortoolslpparser.parse_lp_file(
            TEMP_FILE,
            use_solver=pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
        )["solver"]
        status = solver.Solve()
        self.assertEqual(status, solver.OPTIMAL)

        self.assertEqual(solver.LookupVariable("x1").solution_value(), 30)
        self.assertEqual(solver.LookupVariable("x2").solution_value(), 50)
        self.assertAlmostEqual(solver.Objective().Value(), 80.0)

    def test_simple_integers_two_line(self):

        write_lp_program([
            "max: x1 + x2 + x3;",
            "0 <= x1 <= 30.9;",
            "50.4 >= x2 >= 24.8;",
            "x3 <= 99.2;",
            "int x1, x2;",
            "int x3;",
        ])

        parse_result = ortoolslpparser.parse_lp_file(
            TEMP_FILE,
            use_solver=pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
        )
        solver = parse_result["solver"]
        status = solver.Solve()
        self.assertEqual(status, solver.OPTIMAL)

        self.assertEqual(solver.LookupVariable("x1").solution_value(), 30)
        self.assertEqual(solver.LookupVariable("x2").solution_value(), 50)
        self.assertEqual(solver.LookupVariable("x3").solution_value(), 99)
        self.assertAlmostEqual(solver.Objective().Value(), 179.0)

        self.assertEqual(len(parse_result["var_names"]), 3)
        self.assertTrue("x1" in parse_result["var_names"])
        self.assertTrue("x2" in parse_result["var_names"])
        self.assertTrue("x3" in parse_result["var_names"])

    def test_crazy_signs(self):

        write_lp_program([
            "max: -x1 - -x2;",
            "x1 >= 0;",
            "x1 <= 30;",
            "x2 >= 24;",
            "x2 <= 50;"
        ])

        solver = ortoolslpparser.parse_lp_file(TEMP_FILE)["solver"]
        status = solver.Solve()
        self.assertEqual(status, solver.OPTIMAL)

        self.assertAlmostEqual(solver.LookupVariable("x1").solution_value(), 0.0)
        self.assertAlmostEqual(solver.LookupVariable("x2").solution_value(), 50.0)
        self.assertAlmostEqual(solver.Objective().Value(), 50.0)

    def test_objective_constant(self):

        write_lp_program([
            "max: x1 + 80;",
            "x1 >= 0;",
            "x1 <= 30;",
        ])

        solver = ortoolslpparser.parse_lp_file(TEMP_FILE)["solver"]
        status = solver.Solve()
        self.assertEqual(status, solver.OPTIMAL)

        self.assertAlmostEqual(solver.LookupVariable("x1").solution_value(), 30.0)
        self.assertAlmostEqual(solver.Objective().Value(), 110.0)

    def test_inequalities(self):

        write_lp_program([
            "max: x1 + x2;",
            "x1 >= 0;",
            "x1 < 30;",
            "x2 > 24;",
            "x2 < 50;"
        ])

        solver = ortoolslpparser.parse_lp_file(TEMP_FILE)["solver"]
        status = solver.Solve()
        self.assertEqual(status, solver.OPTIMAL)

        self.assertAlmostEqual(solver.LookupVariable("x1").solution_value(), 30.0)
        self.assertAlmostEqual(solver.LookupVariable("x2").solution_value(), 50.0)
        self.assertAlmostEqual(solver.Objective().Value(), 80.0)

    def test_unbound(self):

        write_lp_program([
            "max: x1;",
        ])

        solver = ortoolslpparser.parse_lp_file(TEMP_FILE)["solver"]
        status = solver.Solve()
        self.assertNotEqual(status, solver.OPTIMAL)

        # print(status)
        # print("%d-%d-%d-%d-%d-%d" % (
        #     solver.OPTIMAL,     # 0
        #     solver.FEASIBLE,    # 1
        #     solver.INFEASIBLE,  # 2
        #     solver.UNBOUNDED,   # 3
        #     solver.ABNORMAL,    # 4
        #     solver.NOT_SOLVED)) # 6

    def test_infeasible(self):

        write_lp_program([
            "max: x1;",
            "x1 >= 10;",
            "x1 <= 5;"
        ])

        solver = ortoolslpparser.parse_lp_file(TEMP_FILE)["solver"]
        status = solver.Solve()
        self.assertNotEqual(status, solver.OPTIMAL)

    def test_comments(self):

        write_lp_program([
            "max: x1 - x2;",
            "",
            "x1 >= 0;",
            "// Test",
            "     //    x1 >= 45;",
            "x1 <= 30;",
            "x2 >= 24;",
            "",
            "x2 <= 50;    // Some text"
        ])

        solver = ortoolslpparser.parse_lp_file(TEMP_FILE)["solver"]
        status = solver.Solve()
        self.assertEqual(status, solver.OPTIMAL)

        self.assertAlmostEqual(solver.LookupVariable("x1").solution_value(), 30.0)
        self.assertAlmostEqual(solver.LookupVariable("x2").solution_value(), 24.0)
        self.assertAlmostEqual(solver.Objective().Value(), 6.0)

    def test_exponents(self):

        write_lp_program([
            "max: 3e8x1 + 2e8 x2;",  # 3e8 * x1 + 2e8 * x2
            "e8 x1 >= 0;",
            "e8 x1 <= 30e2;",
            "e8 >= 24;",
            "e8 <= 50;",
            "x2 >= 9;",
            "x2 <= 5e1;",
        ])

        solver = ortoolslpparser.parse_lp_file(TEMP_FILE)["solver"]
        status = solver.Solve()
        self.assertEqual(status, solver.OPTIMAL)

        self.assertAlmostEqual(solver.LookupVariable("x1").solution_value(), 2976.0)
        self.assertAlmostEqual(solver.LookupVariable("x2").solution_value(), 50.0)
        self.assertAlmostEqual(solver.LookupVariable("e8").solution_value(), 24.0)
        self.assertAlmostEqual(solver.Objective().Value(), 902800000000.0)

    def test_illegal_sign(self):

        write_lp_program([
            "max: x-1 + x2 - x1;",  # 320 - 1 + 50 - 2 = 367
            "x + 80 <= 400;",
            "x1 >= 2;",
            "x1 <= 30;",
            "x2 >= 24;",
            "x2 <= 50;"
        ])

        solver = ortoolslpparser.parse_lp_file(TEMP_FILE)["solver"]
        status = solver.Solve()
        self.assertEqual(status, solver.OPTIMAL)

        self.assertAlmostEqual(solver.LookupVariable("x").solution_value(), 320.0)
        self.assertAlmostEqual(solver.LookupVariable("x1").solution_value(), 2.0)
        self.assertAlmostEqual(solver.LookupVariable("x2").solution_value(), 50.0)
        self.assertAlmostEqual(solver.Objective().Value(), 367.0)

    def test_standalone_coefficient(self):

        write_lp_program([
            "max: 33.333333333333 3x1 + x2;",
            "x1 >= 3;",
            "x1 <= 30.44;",
            "x2 >= 24;",
            "x2 <= 50;",
        ])

        solver = ortoolslpparser.parse_lp_file(TEMP_FILE)["solver"]
        status = solver.Solve()
        self.assertEqual(status, solver.OPTIMAL)

        self.assertAlmostEqual(solver.LookupVariable("x1").solution_value(), 30.44)
        self.assertAlmostEqual(solver.LookupVariable("x2").solution_value(), 50.0)
        self.assertAlmostEqual(solver.Objective().Value(), 174.6533333333)


class TestInvalid(unittest.TestCase):

    def test_illegal_var(self):

        write_lp_program([
            "max: x|1 + x2;",
            "x1 >= 0;",
            "x1 <= 30;",
            "x2 >= 24;",
            "x2 <= 50;"
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_forgot_semicolon_objective(self):

        write_lp_program([
            "max: x1 + x2",
            "x1 >= 0;",
            "x1 <= 30;",
            "x2 >= 24;",
            "x2 <= 50;"
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_no_var_in_coefficients(self):

        write_lp_program([
            "max: x1 + x2;",
            "x1 >= 0;",
            "30 <= 30;",
            "x2 >= 24;",
            "x2 <= 50;"
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_forgot_semicolon_constraint(self):

        write_lp_program([
            "max: x1 + x2;",
            "x1 >= 0;",
            "x1 <= 30;",
            "x2 >= 24;",
            "x2 <= 50"
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_multi_equal(self):

        write_lp_program([
            "max: x1 + x2;",
            "9 >= x1 >= 0 >= x2;",
            "x1 <= 30;",
            "x2 >= 24;",
            "x2 <= 50;"
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_bad_right_hand_side_leq(self):

        write_lp_program([
            "max: x1 + x2;",
            "x1 >= 3;",
            "x1 <= 30.44.44;",
            "x2 >= 24;",
            "x2 <= 50;"
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_bad_right_hand_side_geq(self):

        write_lp_program([
            "max: x1 + x2;",
            "x1 >= 3;",
            "x1 >= 30.44.44;",
            "x2 >= 24;",
            "x2 <= 50;"
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_bad_right_hand_side_leq_leq(self):

        write_lp_program([
            "max: x1 + x2;",
            "3 <= x1 <= 30.44.44;",
            "x2 >= 24;",
            "x2 <= 50;"
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_bad_left_hand_side_leq_leq(self):

        write_lp_program([
            "max: x1 + x2;",
            "2.7.7 <= x1 <= 30;",
            "x2 >= 24;",
            "x2 <= 50;"
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_bad_right_hand_side_geq_geq(self):

        write_lp_program([
            "max: x1 + x2;",
            "30 >= x1 >= 2.68.5;",
            "x2 >= 24;",
            "x2 <= 50;"
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_bad_left_hand_side_geq_geq(self):

        write_lp_program([
            "max: x1 + x2;",
            "30.4.4a >= x1 >= 2.685;",
            "x2 >= 24;",
            "x2 <= 50;"
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_semicolon_only(self):

        write_lp_program([
            "max: x1 + x2;",
            "x1 >= 3;",
            "x1 <= 30.44;",
            "x2 >= 24;",
            "x2 <= 50;",
            ";"
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_double_semicolon(self):

        write_lp_program([
            "max: x1 + x2;",
            "x1 >= 3;",
            "x1 <= 30.44;",
            "x2 >= 24;",
            "x2 <= 50;;",
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_invalid_goal(self):

        write_lp_program([
            "abc: x1 + x2;",
            "x1 >= 3;",
            "x1 <= 30.44;",
            "x2 >= 24;",
            "x2 <= 50;",
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_no_variables(self):

        write_lp_program([
            "max: x1 + x2;",
            "x1 >= 3;",
            " <= 30.44;",
            "x2 >= 24;",
            "x2 <= 50;",
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_no_inequality(self):

        write_lp_program([
            "max: x1 + x2;",
            "x1 >= 3;",
            "x1 <= 30.44;",
            "x2 >= 24;",
            "x2;",
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_strange_declaration(self):

        write_lp_program([
            "max: x1 + x2;",
            "0 <= x1 <= 30.9;",
            "50.4 >= x2 >= 24.8;",
            "sos x1, x2;",
        ])

        try:
            ortoolslpparser.parse_lp_file(
                TEMP_FILE,
                use_solver=pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
            )
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_strange_declaration_second(self):

        write_lp_program([
            "max: x1 + x2;",
            "0 <= x1 <= 30.9;",
            "50.4 >= x2 >= 24.8;",
            "int x1;",
            "sos x2;",
        ])

        try:
            ortoolslpparser.parse_lp_file(
                TEMP_FILE,
                use_solver=pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
            )
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_empty_int_vars(self):

        write_lp_program([
            "max: x1 + x2;",
            "0 <= x1 <= 30.9;",
            "50.4 >= x2 >= 24.8;",
            "int ;",
        ])

        try:
            ortoolslpparser.parse_lp_file(
                TEMP_FILE,
                use_solver=pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
            )
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_bad_coefficient(self):

        write_lp_program([
            "max: 33.33.33x1 + x2;",
            "x1 >= 3;",
            "x1 <= 30.44;",
            "x2 >= 24;",
            "x2 <= 50;",
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_trailing_stuff(self):

        write_lp_program([
            "max: 33.33x1 + x2   +  ;",
            "x1 >= 3;",
            "x1 <= 30.44;",
            "x2 >= 24;",
            "x2 <= 50;",
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_multiple_eq(self):

        write_lp_program([
            "max: 33.33x1 + x2;",
            "x1 >= 3;",
            "x1 <= 30.44;",
            "x1 = 29.3 = 29.3;",
            "x2 >= 24;",
            "x2 <= 50;",
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_bad_right_hand_side_eq(self):

        write_lp_program([
            "max: 33.33x1 + x2;",
            "x1 >= 3;",
            "x1 <= 30.44;",
            "x1 = 29.3agd;",
            "x2 >= 24;",
            "x2 <= 50;",
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_invalid_coefficient_char(self):

        write_lp_program([
            "max: 33.33|x1 + x2;",
            "x1 >= 3;",
            "x1 <= 30.44;",
            "x2 >= 24;",
            "x2 <= 50;",
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_int_invalid_var_name(self):

        write_lp_program([
            "max: x1 + x2;",
            "0 <= x1 <= 30.9;",
            "50.4 >= x2 >= 24.8;",
            "int ajd|bqx1;",
        ])

        try:
            ortoolslpparser.parse_lp_file(
                TEMP_FILE,
                use_solver=pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
            )
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_int_invalid_var_name_start(self):

        write_lp_program([
            "max: x1 + x2;",
            "0 <= x1 <= 30.9;",
            "50.4 >= x2 >= 24.8;",
            "int 4inv1;",
        ])

        try:
            ortoolslpparser.parse_lp_file(
                TEMP_FILE,
                use_solver=pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
            )
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_int_none_only_comma(self):

        write_lp_program([
            "max: x1 + x2;",
            "0 <= x1 <= 30.9;",
            "50.4 >= x2 >= 24.8;",
            "int , ;",
        ])

        try:
            ortoolslpparser.parse_lp_file(
                TEMP_FILE,
                use_solver=pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
            )
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_fail_multi_line_comment(self):

        write_lp_program([
            "max: x1 + x2;",
            "/*x1 >= 0.0;*/;",
            "x1 <= 30.9;",
            "50.4 >= x2 >= 24.8;",
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_sign_detached(self):

        write_lp_program([
            "max: x1 + + x2;",
            "x1 >= 0.0;",
            "x1 <= 30.9;",
            "50.4 >= x2 >= 24.8;",
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_triple_sign(self):

        write_lp_program([
            "max: x1 +++x2;",
            "x1 >= 0.0;",
            "x1 <= 30.9;",
            "50.4 >= x2 >= 24.8;",
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_no_objective(self):

        write_lp_program([
            "x1 + x2;",
            "x1 >= 0;",
            "x1 <= 30;",
            "x2 >= 24;",
            "x2 <= 50;"
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_example_mixed_integer(self):

        write_lp_program([
            "max: x1 + x2;",
            "x1 >= 0.3;",
            "x1 <= 30.6;",
            "x2 >= 24.9;",
            "x2 <= 50.1;",
            "int x1;"
        ])

        result = ortoolslpparser.parse_lp_file(
            TEMP_FILE,
            use_solver=pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
        )
        solver = result["solver"]
        status = solver.Solve()
        self.assertEqual(status, solver.OPTIMAL)
        self.assertEqual(solver.LookupVariable("x1").solution_value(), 30)
        self.assertEqual(solver.LookupVariable("x2").solution_value(), 50.1)

    def test_twice_same_var_objective(self):

        write_lp_program([
            "max: x1 + x2 + x1;",
            "x1 >= 0.0;",
            "x1 <= 30.9;",
            "24.4 >= x2 >= 29.8;",
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_twice_same_var_constraint(self):

        write_lp_program([
            "max: x1 + x2;",
            "x1 + 2x1 >= 0.0;",
            "x1 <= 30.9;",
            "24.4 >= x2 >= 29.8;",
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_twice_declaration(self):

        write_lp_program([
            "max: x1 + x2;",
            "x1 >= 0.0;",
            "x1 <= 30.9;",
            "24.4 >= x2 >= 29.8;",
            "int x1;",
            "int x1;"
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_twice_declaration_same_line(self):

        write_lp_program([
            "max: x1 + x2;",
            "x1 >= 0.0;",
            "x1 <= 30.9;",
            "24.4 >= x2 >= 29.8;",
            "int x2;",
            "int x1, x1;"
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_invalid_var_name_objective_function(self):

        write_lp_program([
            "max: x1^ + x2;",
            "x1^ >= 0;",
            "x1^ <= 30;",
            "x2 >= 24;",
            "x2 <= 50;"
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_invalid_var_name_constraint(self):

        write_lp_program([
            "max: x1 + x2;",
            "x1 >= 0;",
            "x1 <= 30;",
            "x2 >= 24;",
            "x2 <= 50;",
            "x$3 <= 200;"
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_invalid_var_name_start_underscore(self):

        write_lp_program([
            "max: x1 + x2;",
            "x1 >= 0;",
            "x1 <= 30;",
            "x2 >= 24;",
            "x2 <= 50;",
            "_x3 <= 200;"
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_invalid_var_name_declarations(self):

        write_lp_program([
            "max: x1 + x2;",
            "x1 >= 0;",
            "x1 <= 30;",
            "x2 >= 24;",
            "x2 <= 50;",
            "int x1, x3^;"
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_no_whitespace_after_variable(self):

        write_lp_program([
            "max: x1 + x2;",
            "x1 = 30;",
            "x1|x2 <= 99.2;"
        ])

        try:
            ortoolslpparser.parse_lp_file(TEMP_FILE)
            self.assertTrue(False)
        except ValueError as e:
            self.assertEqual(e.args[0], "Whitespace or combination sign is missing on line 3.")
            self.assertTrue(True)

    def test_improve_var_bounds_no_single_var_name_match(self):

        write_lp_program([
            "max: x1 + x2;",
            "20 <= x1 + x2 <= 30;",
            "10 <= x1 <= 10;",
        ])

        solver = ortoolslpparser.parse_lp_file(TEMP_FILE)["solver"]
        status = solver.Solve()
        self.assertEqual(status, solver.OPTIMAL)

        self.assertAlmostEqual(solver.LookupVariable("x1").solution_value(), 10.0)
        self.assertAlmostEqual(solver.LookupVariable("x2").solution_value(), 20.0)
        self.assertAlmostEqual(solver.Objective().Value(), 30.0)
