from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pickle
import numpy as np


def create_data_model(dt):
    """Stores the data for the problem."""
    # data = pickle.load(open("data/vrp/vrp20_datasets_seed1234.pkl", "rb"))
    # dt = data[0]
    loc = np.array(dt[1]) * 100
    dist_mat = list((np.sum((loc - loc[:,None])**2, axis=2))**.5)
    data = {}
    data['distance_matrix'] = dist_mat
    data['demands'] = dt[2]
    total_demand = 0
    for demand in data['demands']:
        total_demand += demand
    vehicle_capacity = 50
    if total_demand % vehicle_capacity > 0:
        data['num_vehicles'] = int(total_demand/vehicle_capacity) + 1
    else:
        data['num_vehicles'] = int(total_demand / vehicle_capacity)
    # data['num_vehicles'] = 5
    data['vehicle_capacities'] = []
    for i in range(data['num_vehicles']):
        data['vehicle_capacities'].append(vehicle_capacity)
    data['depot'] = 0
    return data


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    total_distance = 0
    total_load = 0
    nodes_visited = []
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            nodes_visited.append(node_index)
            route_load += data['demands'][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index),
                                                 route_load)
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        plan_output += 'Load of the route: {}\n'.format(route_load)
        print(plan_output)
        total_distance += route_distance
        total_load += route_load

    data = {
        "route": nodes_visited
    }
    # pickle.dump(data, open("cvrp_100_rep.pkl", "wb"))
    # print('Total distance of all routes: {}m'.format(total_distance))
    # print('Total load of all routes: {}'.format(total_load))

    return total_distance
    #


def main():
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data_sets = pickle.load(open("data/vrp_100.pkl", "rb"))
    solutions = []

    for i in range(1):
        data = create_data_model(data_sets[i])

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                               data['num_vehicles'], data['depot'])

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)


        # Create and register a transit callback.
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)


        # Add Capacity constraint.
        def demand_callback(from_index):
            """Returns the demand of the node."""
            # Convert from routing variable Index to demands NodeIndex.
            from_node = manager.IndexToNode(from_index)
            return data['demands'][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(
            demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data['vehicle_capacities'],  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity')

        penalty = 100000
        for node in range(1, len(data['distance_matrix'])):
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.FromSeconds(20)

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)



        # Print solution on console.
        if solution:
             solutions.append(print_solution(data, manager, routing, solution))

    data = {}
    data["costs"] = solutions
    # pickle.dump(data, open("vrp_20_OR_test_results_single.pkl", "wb"))

    solutions = np.array(solutions)


    print(solutions.mean())


if __name__ == '__main__':
    main()