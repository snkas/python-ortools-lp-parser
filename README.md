# Python OR-Tools LP Parser

[![Build Status](https://travis-ci.org/snkas/python-ortools-lp-parser.svg?branch=master)](https://travis-ci.org/snkas/python-ortools-lp-parser) [![codecov](https://codecov.io/gh/snkas/python-ortools-lp-parser/branch/master/graph/badge.svg)](https://codecov.io/gh/snkas/python-ortools-lp-parser)

It is remarkable that the or-tools open source project (https://developers.google.com/optimization/) enables a decent linear solver to be included in Python via `pip` without manual compilation. The Python wrapper of the or-tools optimization API (i.e. `pip install ortools`) unfortunately does not support the reading of linear program files. This small module is to enable Python developers to read in LP formatted linear program files.

**This fan-written LP parser is NOT IN ANY WAY affiliated with or endorsed by the or-tools developers. This parser is distributed on an "as is" basis, without warranties or conditions of any kind (see also the Apache 2.0 License in ./LICENSE).**

It aims to follow the LPSolve 5.1 LP format (http://lpsolve.sourceforge.net/5.1/lp-format.htm). The format accepted by this parser is different in the following ways:

1. An objective function / constraint / declaration must be on a single line terminated by a semicolon (;)
2. Multi-line comments are not allowed
3. The objective function must have "max:" or "min:" at the start of it
4. Keywords (i.e., "max", "min", "int") must be lowercase
5. Semi-continuous variables are not allowed
6. Special ordered sets (SOS) are not allowed
7. Constraint names are completely ignored (e.g., "my_constraint_row_1: x1 >= x2" is just parsed to "x1 >= x2")
8. At most 1 detached (whitespaced) sign and 1 directly in front of a coefficient/variable is permitted (e.g., "max: --x1 + +3.0" is allowed, "max: --x1 + + +3.0" is not, "max: --x1 + + 3.0" is also not).
9. Variable names are more restrictive. A variable name is case sensitive. It must start with an alphabetic character. It must consist solely of alphabetic characters, numeric characters, and underscore characters. This corresponds to the following regular expression: `[A-Za-z]+[A-Za-z0-9_]*`.

## Installation

**Requirements**
* Python 3.5+
* `pip install ortools`

**Option 1**

```bash
$ pip install git+https://github.com/snkas/python-ortools-lp-parser
```

You can now include it using: `import ortoolslpparser`

**Option 2**

Clone/download this Git repository. Then, execute the following to install the package locally:

```bash
$ bash install_local.sh
```

You can now include it using: `import ortoolslpparser`

## Getting started

Create a LP formatted linear program called `program.lp`:

```
max: x1 - x2;
x1 >= 0.3;
x1 <= 30.6;
x2 >= 24.9;
x2 <= 50.1;
```

Create a Python file called `example.py`:

```python
import ortoolslpparser

parse_result = ortoolslpparser.parse_lp_file("program.lp")
solver = parse_result["solver"]
solver.Solve()
print("Objective value: %f" % solver.Objective().Value())
for var_name in parse_result["var_names"]:
    print("Variable %s: %f" % (var_name, solver.LookupVariable(var_name).solution_value()))
```

Then run `python3 example.py`. It should output:

```
Objective value: 5.700000
Variable x1: 30.600000
Variable x2: 24.900000
```

## Testing

Run all tests (local version):
```bash
$ python -m pytest
```

Run all tests (global pip-installed version):
```bash
$ pytest
```

Calculate coverage locally (output in `htmlcov/`):
```bash
$ bash calculate_coverage.sh
```

## General advice

* **Declare the tightest bounds possible.** The default type of a variable is a floating point number in the range (-inf, inf). Make sure that you declare for each variable as tight bounds as possible: this can help the solver. In particular, if you know a certain variable `x` is a non-negative number, add a `x >= 0;` constraint to limit the range to [0, inf). The Glop solver tends to not be as good at solving (-inf, inf)-bounded variables: in some instances it declares the program as ABNORMAL.
