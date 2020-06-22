import ortoolslpparser

parse_result = ortoolslpparser.parse_lp_file("program.lp")
solver = parse_result["solver"]
solver.Solve()
print("Objective value: %f" % solver.Objective().Value())
for var_name in parse_result["var_names"]:
    print("Variable %s: %f" % (var_name, solver.LookupVariable(var_name).solution_value()))
