import pickle
import numpy as np
import os
from run_benchmarks import SolutionTree

for file in os.listdir( ".checkpoints/2020-10-05" ):
    if file[-4:] != ".dat":
        continue

    print( "-" * 40 )
    print( file )
    file = os.path.join( ".checkpoints/2020-10-05", file )
    file = open( file, 'rb' )
    data = pickle.load( file )
    if len( data ) > 2:
        print( data[1] )
        print( data[2] )
        print( data[3] )

print( "-" * 40 )
