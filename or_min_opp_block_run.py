import argparse
from gerrychain import Graph, GeographicPartition, Partition, Election, accept
from gerrychain.updaters import Tally, cut_edges
from gerrychain import MarkovChain
from gerrychain.proposals import recom
from gerrychain.accept import always_accept
from gerrychain import constraints
import geopandas as gpd
import numpy as np
from functools import partial
from gerrychain.tree import recursive_tree_part
import pickle
import random


## Set up argument parser

parser = argparse.ArgumentParser(description="Neutral ensemble for OR", 
                                 prog="or_blocks_neutral_chain.py")

parser.add_argument("map", metavar="map", type=str,
                    choices=["state_house"],
                    help="the map to redistrict")
parser.add_argument("n", metavar="iterations", type=int,
                    help="the number of plans to sample")
parser.add_argument("eps", metavar="epsilon", type=float,
                    help="population constraint")
args = parser.parse_args()

num_districts_in_map = {"congress" : 5,
                        "congress_2020" : 6,
                        "state_senate" : 30,
                        "state_house" : 60}

# epsilons = {"congress" : 0.01,
#             "congress_2020" : 0.01,
#             "state_senate" : 0.02,
#             "state_house" : 0.02} 

POP_COL = "TOTPOP"
NUM_DISTRICTS = num_districts_in_map[args.map]
ITERS = args.n
EPS = args.eps #epsilons[args.map]


## Pull in graph and set up updaters

print("Reading in Data/Graph")

df = gpd.read_file("data/OR_blocks/OR_blocks.shp")
with open("data/OR_blocks/OR_block_graph.p", "rb") as f_in:
    graph = pickle.load(f_in)


or_updaters = {"population" : Tally(POP_COL, alias="population"),
               "cut_edges": cut_edges,
               "VAP": Tally("VAP"),
               "WVAP": Tally("WVAP"),
               "HVAP": Tally("HVAP"),
               "ASIANVAP": Tally("ASIANVAP"),
               "HVAP_perc": lambda p: {k: (v / p["VAP"][k]) for k, v in p["HVAP"].items()},
               "WVAP_perc": lambda p: {k: (v / p["VAP"][k]) for k, v in p["WVAP"].items()},
               "ASIANVAP_perc": lambda p: {k: (v / p["VAP"][k]) for k, v in p["ASIANVAP"].items()},
               "HAVAP_perc": lambda p: {k: ((p["HVAP"][k] + p["ASIANVAP"][k]) / v) for k, v in p["VAP"].items()},}

# election_updaters = {election.name: election for election in elections}
# or_updaters.update(election_updaters)

## Create seed plans and Set up Markov chain

print("Creating seed plan")

total_pop = sum(df[POP_COL])
ideal_pop = total_pop / NUM_DISTRICTS

cddict = recursive_tree_part(graph=graph, parts=range(NUM_DISTRICTS), 
                             pop_target=ideal_pop, pop_col=POP_COL, epsilon=EPS)

init_partition = Partition(graph, assignment=cddict, updaters=or_updaters)


## Setup chain

proposal = partial(recom, pop_col=POP_COL, pop_target=ideal_pop, epsilon=EPS, 
                   node_repeats=1)

compactness_bound = constraints.UpperBound(lambda p: len(p["cut_edges"]), 
                                           2*len(init_partition["cut_edges"]))

chain = MarkovChain(
        proposal,
        constraints=[
            constraints.within_percent_of_ideal_population(init_partition, EPS),
            compactness_bound],
        accept=accept.always_accept,
        initial_state=init_partition,
        total_steps=ITERS)


## Run chain

print("Starting Markov Chain")

def init_min_results():
    data = {"cutedges": np.zeros(ITERS)}
    data["HVAP"] = np.zeros((ITERS, NUM_DISTRICTS))
    data["ASIANVAP"] = np.zeros((ITERS, NUM_DISTRICTS))
    data["HVAP_perc"] = np.zeros((ITERS, NUM_DISTRICTS))
    data["ASIANVAP_perc"] = np.zeros((ITERS, NUM_DISTRICTS))
    data["HAVAP_perc"] = np.zeros((ITERS, NUM_DISTRICTS))
    data["WVAP"] = np.zeros((ITERS, NUM_DISTRICTS))
    data["WVAP_perc"] = np.zeros((ITERS, NUM_DISTRICTS))

    return data #, parts

def tract_min_results(data, part, i):
    data["cutedges"][i] = len(part["cut_edges"])

    data["HVAP"][i] = sorted(part["HVAP"].values())
    data["ASIANVAP"][i] = sorted(part["ASIANVAP"].values())
    data["HVAP_perc"][i] = sorted(part["HVAP_perc"].values())
    data["ASIANVAP_perc"][i] = sorted(part["ASIANVAP_perc"].values())
    data["HAVAP_perc"][i] = sorted(part["HAVAP_perc"].values())
    data["WVAP"][i] = sorted(part["WVAP"].values())
    data["WVAP_perc"][i] = sorted(part["WVAP_perc"].values())

# def update_saved_parts(parts, part, elections, i):
#     if i % (ITERS / 10) == 99: parts["samples"].append(part)
    
#     if len(part["cut_edges"]) < 150: parts["compact"].append(part)


chain_results = init_min_results()

for i, part in enumerate(chain):
    tract_min_results(chain_results, part, i)
    # update_saved_parts(parts, part, ELECTS, i)

    if i % 100 == 0:
        print("*", end="", flush=True)
print()

## Save results

print("Saving results")

output = "data/neutral_min_opp_ensemble/OR_{}_{}_blocks_{}%.p".format(args.map, ITERS, args.eps)

with open(output, "wb") as f_out:
    pickle.dump(chain_results, f_out)

# for i, part in enumerate(parts["samples"]):
#     part.to_json("/cluster/tufts/mggg/jmatth03/sample_parts/sample_part_{}_{}_blocks_{}%.json".format(args.map, 
#                                                                                                       i, args.eps),
#                  save_assignment_as="District", include_geometries_as_geojson=True)

# comp_samp = parts["compact"] if len(parts["compact"]) <= 10 else random.sample(parts["compact"], 10)
# for i, part in enumerate(comp_samp):
#     if i < 10:
#         part.to_json("/cluster/tufts/mggg/jmatth03/sample_parts/compact_part_{}_{}_blocks_{}%.json".format(args.map, 
#                                                                                                            i, args.eps),
#                  save_assignment_as="District", include_geometries_as_geojson=True)
