from ortools.linear_solver import pywraplp
import ortoolslpparser

parse_result = ortoolslpparser.parse_lp_file(
    "program_mixed_integer.lp",
    use_solver=pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
)
solver = parse_result["solver"]
solver.Solve()
print("Objective value: %f" % solver.Objective().Value())
for var_name in parse_result["var_names"]:
    print("Variable %s: %f" % (var_name, solver.LookupVariable(var_name).solution_value()))
