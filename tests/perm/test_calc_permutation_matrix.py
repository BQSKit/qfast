
"""
I = np.array( [ [ 1, 0 ], [ 0, 1 ] ], dtype = np.complex128 )
II = np.kron( I, I )
IIII = np.kron( II, II )
X = np.array( [ [ 0, 1 ], [ 1, 0 ] ], dtype = np.complex128 )
XX = np.kron( X, X )
XXI = np.kron( XX, I )
IXX = np.kron( I, XX )
IIXX = np.kron( I, IXX )
IX = np.kron( I, X )
IXIX = np.kron( IX, IX )
XXXX = np.kron( XX, XX )
IXIXIXIX = np.kron( IXIX, IXIX )

U = la.expm( -1j * IXIXIXIX )

P = calc_permutation_matrix( 8, [1, 3, 5, 7] )
A = la.expm( -1j * XXXX )

print( np.allclose( P @ np.kron( A, IIII ) @ P.T, U ) )
"""

print( calc_permutation_matrix( 2, [0, 1] ) )
print()
print()
print()
print( calc_permutation_matrix( 2, [1] ) )
print()
print()
print()
print()
print( calc_permutation_matrix( 2, [1, 0] ) )
