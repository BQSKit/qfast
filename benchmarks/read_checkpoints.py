import pickle
import numpy as np
import os

for file in os.listdir( ".checkpoints/2020-09-26" ):
    if file[-4:] != ".dat":
        continue

    print( "-" * 40 )
    print( file )
    file = os.path.join( ".checkpoints/2020-09-26", file )
    file = open( file, 'rb' )
    data = pickle.load( file )
    if len( data ) > 2:
        print( data[1] )
        print( data[2] )
        print( data[3] )

print( "-" * 40 )
