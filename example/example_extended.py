import ortoolslpparser

parse_result = ortoolslpparser.parse_lp_file("program.lp")
solver = parse_result["solver"]
result = solver.Solve()

if result == solver.OPTIMAL:
    print("Value of objective function: %f" % solver.Objective().Value())
    print("Actual values of the variables:")
    for var_name in parse_result["var_names"]:
        print("%s %f" % (var_name, solver.LookupVariable(var_name).solution_value()))

else:
    print("Linear program was not solved.")
    error_msg = "UNKNOWN"
    if result == solver.OPTIMAL:
        error_msg = "OPTIMAL"
    elif result == solver.FEASIBLE:
        error_msg = "FEASIBLE"
    elif result == solver.INFEASIBLE:
        error_msg = "INFEASIBLE"
    elif result == solver.UNBOUNDED:
        error_msg = "UNBOUNDED"
    elif result == solver.ABNORMAL:
        error_msg = "ABNORMAL"
    elif result == solver.NOT_SOLVED:
        error_msg = "NOT_SOLVED"
    print("Error returned by OR-tools: %s (code: %d)"  % (error_msg, result))
