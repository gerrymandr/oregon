import geopandas as gpd
import pickle
from gerrychain import Graph, Partition

graph = Graph.from_file("data/OR_blocks/OR_blocks.shp")

with open("data/OR_blocks/OR_block_graph.p", "wb") as f_out:
    pickle.dump(graph, f_out)