#!/usr/bin/env python
# This Python file uses the following encoding: utf-8
# Copyright 2015 Tin Arm Engineering AB
# Copyright 2018 Google LLC
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
"""Capacitated Vehicle Routing Problem (CVRP).

   This is a sample using the routing library python wrapper to solve a CVRP
   problem while allowing multiple trips, i.e., vehicles can return to a depot
   to reset their load ("reload").

   A description of the CVRP problem can be found here:
   http://en.wikipedia.org/wiki/Vehicle_routing_problem.

   Distances are in meters.

   In order to implement multiple trips, new nodes are introduced at the same
   locations of the original depots. These additional nodes can be dropped
   from the schedule at 0 cost.

   The max_slack parameter associated to the capacity constraints of all nodes
   can be set to be the maximum of the vehicles' capacities, rather than 0 like
   in a traditional CVRP. Slack is required since before a solution is found,
   it is not known how much capacity will be transferred at the new nodes. For
   all the other (original) nodes, the slack is then re-set to 0.

   The above two considerations are implemented in `add_capacity_constraints()`.

   Last, it is useful to set a large distance between the initial depot and the
   new nodes introduced, to avoid schedules having spurious transits through
   those new nodes unless it's necessary to reload. This consideration is taken
   into account in `create_distance_evaluator()`.
"""


from functools import partial

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import numpy as np
import pickle
import csv


###########################
# Problem Data Definition #
###########################
def create_data_model(dt, j):
    """Stores the data for the problem"""
    data = {}
    _capacity = int(dt[3])
    _locations = []
    demands = []
    n_loc = len(dt[1])
    depot_mult_factor = [20,30]
    n_depot = int(np.array(dt[2]).sum()/_capacity)*depot_mult_factor[j] + 4
    for i in range(n_depot):
        _locations.append((dt[0][0]*100, dt[0][1]*100))
        demands.append(-dt[3])
    demands[0] = 0
    for i in range(n_loc):
        _locations.append((dt[1][i][0]*100,dt[1][i][1]*100))
        demands.append(dt[2][i])
    data['locations'] = _locations
    data['num_locations'] = len(data['locations'])
    data['demands'] = demands
    # data['demands']  = (-np.array(data['demands'])).tolist()
    data['num_vehicles'] = 1
    data['vehicle_capacity'] = _capacity
    data['vehicle_max_distance'] = 10_00000
    data[
        'vehicle_speed'] = 5 * 60 / 3.6  # Travel speed: 5km/h to convert in m/min
    data['depot'] = 0
    data['n_depot'] = n_depot
    return data


#######################
# Problem Constraints #
#######################
def euclidean_distance(position_1, position_2):
    """Computes the Manhattan distance between two points"""
    return ((position_1[0] - position_2[0])**2 +
     (position_1[1] - position_2[1])**2)**.5


def create_distance_evaluator(data):
    """Creates callback to return distance between points."""
    _distances = {}
    # precompute distance between location to have distance callback in O(1)
    for from_node in range(data['num_locations']):
        _distances[from_node] = {}
        for to_node in range(data['num_locations']):
            if from_node == to_node:
                _distances[from_node][to_node] = 0
            # Forbid start/end/reload node to be consecutive.
            elif from_node in range(data['n_depot']) and to_node in range(data['n_depot']):
                _distances[from_node][to_node] = data['vehicle_max_distance']
            else:
                _distances[from_node][to_node] = (euclidean_distance(
                    data['locations'][from_node], data['locations'][to_node]))

    def distance_evaluator(manager, from_node, to_node):
        """Returns the manhattan distance between the two nodes"""
        return _distances[manager.IndexToNode(from_node)][manager.IndexToNode(
            to_node)]

    return distance_evaluator


def create_demand_evaluator(data):
    """Creates callback to get demands at each location."""
    _demands = data['demands']

    def demand_evaluator(manager, from_node):
        """Returns the demand of the current node"""
        return _demands[manager.IndexToNode(from_node)]

    return demand_evaluator


def add_capacity_constraints(routing, manager, data, demand_evaluator_index):
    """Adds capacity constraint"""
    vehicle_capacity = data['vehicle_capacity']
    capacity = 'Capacity'
    routing.AddDimension(
        demand_evaluator_index,
        vehicle_capacity,
        vehicle_capacity,
        True,  # start cumul to zero
        capacity)

    # Add Slack for reseting to zero unload depot nodes.
    # e.g. vehicle with load 10/15 arrives at node 1 (depot unload)
    # so we have CumulVar = 10(current load) + -15(unload) + 5(slack) = 0.
    capacity_dimension = routing.GetDimensionOrDie(capacity)
    # Allow to drop reloading nodes with zero cost.
    for node in range(data['n_depot']):
        node_index = manager.NodeToIndex(node)
        routing.AddDisjunction([node_index], 0)

    # Allow to drop regular node with a cost.
    for node in range(data['n_depot'], len(data['demands'])):
        node_index = manager.NodeToIndex(node)
        capacity_dimension.SlackVar(node_index).SetValue(0)
        routing.AddDisjunction([node_index], 100_000)

###########
# Printer #
###########
def print_solution(data, manager, routing, assignment):  # pylint:disable=too-many-locals
    """Prints assignment on console"""
    print(f'Objective: {assignment.ObjectiveValue()}')
    total_distance = 0
    total_load = 0
    capacity_dimension = routing.GetDimensionOrDie('Capacity')
    dropped = []
    for order in range(6, routing.nodes()):
        index = manager.NodeToIndex(order)
        if assignment.Value(routing.NextVar(index)) == index:
            dropped.append(order)
    print(f'dropped orders: {dropped}')
    for reload in range(1, 6):
        index = manager.NodeToIndex(reload)
        if assignment.Value(routing.NextVar(index)) == index:
            dropped.append(reload)
    print(f'dropped reload stations: {dropped}')

    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = f'Route for vehicle {vehicle_id}:\n'
        distance = 0
        while not routing.IsEnd(index):
            load_var = capacity_dimension.CumulVar(index)
            plan_output += ' {0} Load({1})) ->'.format(
                manager.IndexToNode(index),
                assignment.Value(load_var))
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            distance += routing.GetArcCostForVehicle(previous_index, index,
                                                     vehicle_id)
        load_var = capacity_dimension.CumulVar(index)
        plan_output += ' {0} Load({1}))\n'.format(
            manager.IndexToNode(index),
            assignment.Value(load_var))
        plan_output += f'Distance of the route: {distance}m\n'
        plan_output += f'Load of the route: {assignment.Value(load_var)}\n'
        # plan_output += f'Time of the route: {assignment.Value(time_var)}min\n'
        print(plan_output)
        total_distance += distance
        total_load += assignment.Value(load_var)
    print('Total Distance of all routes: {}m'.format(total_distance))
    print('Total Load of all routes: {}'.format(total_load))
    if assignment.ObjectiveValue() > 100_000:
        return 0.0
    else:
        return total_distance


########
# Main #
########
def main():
    """Entry point of the program"""
    # n_task = 200

    tasks = [1000, 2000]
    times = [100 ,200]
    for j in range(len(tasks)):
        n_task = tasks[j]
        data_sets = pickle.load(open("data/vrp_" + str(n_task) + ".pkl", "rb"))
        solutions = []
        for i in range(100):
        # Instantiate the data problem.
            data = create_data_model(data_sets[i], j)

            # Create the routing index manager
            manager = pywrapcp.RoutingIndexManager(data['num_locations'],
                                                   data['num_vehicles'], data['depot'])

            # Create Routing Model
            routing = pywrapcp.RoutingModel(manager)

            # Define weight of each edge
            distance_evaluator_index = routing.RegisterTransitCallback(
                partial(create_distance_evaluator(data), manager))
            routing.SetArcCostEvaluatorOfAllVehicles(distance_evaluator_index)


            # Add Capacity constraint
            demand_evaluator_index = routing.RegisterUnaryTransitCallback(
                partial(create_demand_evaluator(data), manager))
            add_capacity_constraints(routing, manager, data, demand_evaluator_index)

            # Setting first solution heuristic (cheapest addition).
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)  # pylint: disable=no-member
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
            search_parameters.time_limit.FromSeconds(times[j])

            # Solve the problem.
            solution = routing.SolveWithParameters(search_parameters)
            if solution:
                solutions.append(print_solution(data, manager, routing, solution))
            else:
                print("No solution found !")
        file_name = "Results/PCA_GLS_"+ str(n_task)+".csv"
        with open(file_name, 'w') as f:
            write = csv.writer(f)
            write.writerows((np.array(solutions).T).reshape((100,1)).tolist())


if __name__ == '__main__':
    main()