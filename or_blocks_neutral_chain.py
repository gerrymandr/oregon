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
ELECTS = ["PRES16", "SEN16", "GOV16", "GOV18", "AG16", "SOS16", "USH18"]


## Pull in graph and set up updaters

print("Reading in Data/Graph")

df = gpd.read_file("data/OR_blocks/OR_blocks.shp")
with open("data/OR_blocks/OR_block_graph.p", "rb") as f_in:
    graph = pickle.load(f_in)

elections = [Election("USH18",{"Dem": "USH18D","Rep":"USH18R"}),
             Election("PRES16",{"Dem": "PRES16D","Rep":"PRES16R"}),
             Election("SEN16", {"Dem": "SEN16D", "Rep": "SEN16R"}),
             Election("GOV16", {"Dem": "GOV16D", "Rep": "GOV16R"}),
             Election("AG16", {"Dem": "AG16D", "Rep": "AG16R"}),
             Election("SOS16", {"Dem": "SOS16D", "Rep": "SOS16R"}),
             Election("GOV18", {"Dem": "GOV18D", "Rep": "GOV18R"})]


or_updaters = {"population" : Tally(POP_COL, alias="population"),
               "cut_edges": cut_edges}

election_updaters = {election.name: election for election in elections}
or_updaters.update(election_updaters)

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

def init_election_results(elections):
    data = {"cutedges": np.zeros(ITERS)}
    parts = {"samples": [], "compact": []}
    for election in elections:
        name = election.lower()
        data["seats_{}".format(name)] = np.zeros(ITERS)
        data["results_{}".format(name)] = np.zeros((ITERS, NUM_DISTRICTS))
        data["efficiency_gap_{}".format(name)] = np.zeros(ITERS)
        data["mean_median_{}".format(name)] = np.zeros(ITERS)
        data["partisan_gini_{}".format(name)] = np.zeros(ITERS)

    return data, parts

def tract_election_results(data, elections, part, i):
    for election in elections:
        name = election.lower()
        data["results_{}".format(name)][i] = sorted(part[election].percents("Dem"))
        data["seats_{}".format(name)][i] = part[election].seats("Dem")
        data["efficiency_gap_{}".format(name)][i] = part[election].efficiency_gap()
        data["mean_median_{}".format(name)][i] = part[election].mean_median()
        data["partisan_gini_{}".format(name)][i] = part[election].partisan_gini()


def update_saved_parts(parts, part, elections, i):
    if i % (ITERS / 10) == 99: parts["samples"].append(part)
    
    if len(part["cut_edges"]) < 150: parts["compact"].append(part)


chain_results, parts = init_election_results(ELECTS)

for i, part in enumerate(chain):
    chain_results["cutedges"][i] = len(part["cut_edges"])
    tract_election_results(chain_results, ELECTS, part, i)
    # update_saved_parts(parts, part, ELECTS, i)

    if i % 100 == 0:
        print("*", end="", flush=True)
print()

## Save results

print("Saving results")

output = "data/partisan_results_blocks/OR_{}_{}_blocks_{}%.p".format(args.map, ITERS, args.eps)

with open(output, "wb") as f_out:
    pickle.dump(chain_results, f_out)
