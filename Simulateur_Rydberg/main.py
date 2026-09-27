import numpy as np
import matplotlib.pyplot as plt
import sys

######################################################## FUNCTIONS ########################################################

# This function takes a number L of atoms and returns a list of binary vectors (i.e., a matrix)
# containing all POSSIBLE combinations of 0s and 1s for a given L.
# Note: There are fewer than 2^L total vectors, because if two "1"s appear consecutively,
# the corresponding ket vector is not allowed.
def all_possible_vectors_generation(L):
#### Creation of a matrix containing all possible configurations, including the forbidden ones ####
    all_vectors = np.zeros((2 ** L, L), dtype=int)

    for i in range(2 ** L):
        for j in range(L):
            # Extracting the j^th bit
            all_vectors[i, (L-1) - j] = (i >> j) & 1


#### Identifying the indices of the POSSIBLE configurations ####
    sorted_vectors = []  # list that will contain only the vectors of the POSSIBLE configurations
    sorted_indices = []
    index = 0

    for i in range(all_vectors.shape[0]):
        index = i
        for j in range(all_vectors.shape[1]):
            if j == 0 and all_vectors[i, 0] == 1 and all_vectors[i, 1] == 1:
                # if |1 1...>, we put index = -1
                index = -1

            elif j == (all_vectors.shape[1] - 1) and all_vectors[i, (all_vectors.shape[1] - 2)] == 1 and all_vectors[i, all_vectors.shape[1] - 1] == 1:
                # if |...1 1>, then we put index = -1
                index = -1

            elif j != 0 and j != (all_vectors.shape[1] - 1) and all_vectors[i, (j - 1)] == 1 and all_vectors[i, j] == 1  or  j != 0 and j != (all_vectors.shape[1] - 1) and all_vectors[i, j] == 1 and all_vectors[i, (j + 1)] == 1:
                # if j != 0 and j != (2^L - 1) and the state is of the form |...1 1...>, then we set index = -1
                index = -1

        if index != -1: # If index != -1, then the configuration is valid and we store the corresponding vector in the list sorted_vectors
            sorted_indices.append(index)
            sorted_vectors.append(all_vectors[index])

    return np.array(sorted_indices), np.array(sorted_vectors)

# Exemple i=10
'''
i >> j: décalage binaire vers la droite de j positions: 10 >> 1
i = 10 (0b001010)  (on met le préfixe "0b" pour préciser que le nombre suivant est en binaire car "b" comme " binaire")

10 >> 0 -> 0b001010 (reste identique)
10 >> 1 -> 0b000101
10 >> 2 -> 0b000010
10 >> 3 -> 0b000001
10 >> 4 -> 0b000000
10 >> 5 -> 0b000000


x & 1: garde le dernier bit de x

10 & 1 -> 0b001010 & 000001 = 0 (car le dernier bit (à droite) de 10 est 0)
21 & 1 -> 0b010101 & 000001 = 1 (car le dernier bit de 21 est 1)
etc...
'''

###########################################################
# This function takes a ket vector |Ψ⟩ of size L and returns a NumPy array of vectors (i.e., a matrix)
# that make up the terms of the linear combination of H|Ψ⟩, where H is the Hamiltonian matrix of the system
def CL_H_psi(vect_binaire):
    H_psi_mat = []  #matrice qui contient les vecteurs de la C.L
    vect_tampon = np.copy(vect_binaire) # ATTENTION: il faut utiliser copy car sinon on égale les références des deux tableaux et donc si on
                                        # modifiait vect_tampon, vect_binaire se modifiait aussi

#### Transformation of 0 in 1 if possible ####
    for j in range(vect_binaire.shape[0]):
        if j == 0 and vect_binaire[0] == 0 and vect_binaire[1] == 0:  # if |0 0 ...>, then we construct |1 0 ...>
            vect_tampon[0] = 1
            H_psi_mat.append(vect_tampon)
            vect_tampon = np.copy(vect_binaire) # The buffer vector is reset to the binary vector |Ψ⟩

        if j == (vect_binaire.shape[0] - 1) and vect_binaire[vect_binaire.shape[0] - 2] == 0 and vect_binaire[vect_binaire.shape[0] - 1] == 0:
            # if |...0 0>, then we construct |...0 1>
            vect_tampon[vect_binaire.shape[0] - 1] = 1
            H_psi_mat.append(vect_tampon)
            vect_tampon = np.copy(vect_binaire)  # Reset the buffer vector to the binary vector |Ψ⟩

        if j != 0 and j != (vect_binaire.shape[0] - 1) and vect_binaire[j-1] == 0 and vect_binaire[j] == 0  and vect_binaire[j+1] == 0:
            # If |...0 0 0...>, then we construct |...0 1 0...>
            vect_tampon[j] = 1
            H_psi_mat.append(vect_tampon)
            vect_tampon = np.copy(vect_binaire)  # Reinitialize the buffer vector to the binary vector |Ψ⟩


#### Transformation of the 1 in 0 ####
    for j in range(vect_binaire.shape[0]):
        if vect_binaire[j] == 1:
            vect_tampon[j] = 0
            H_psi_mat.append(vect_tampon)
            vect_tampon = np.copy(vect_binaire)

    return np.array(H_psi_mat)


###########################################################
# This function takes a matrix containing several binary vectors of size L, and returns a binary vector of size (2^L - number of
# forbidden configurations), containing "1"s at the positions corresponding to the decimal values of the binary vectors
# in the input matrix.
def H_psi(H_psi_mat):
    all_indices, all_vectors = all_possible_vectors_generation(H_psi_mat.shape[1])
    H_psi_binaire = np.zeros(all_indices.shape[0]) # Vector filled with 0s of size (2^L - number of forbidden configurations)


    for i in range(H_psi_mat.shape[0]):
        for k in range(all_vectors.shape[0]): # k ∈ {0,...,20}
            if np.array_equal(H_psi_mat[i] , all_vectors[k]):
                H_psi_binaire[k] = 1


    return H_psi_binaire


###########################################################
# This function takes a number L of atoms and returns the Hamiltonian that only includes the VALID configurations of the system
def get_hamiltonian(L):
    all_indices, all_vectors = all_possible_vectors_generation(L)
    dim_H = all_indices.shape[0]
    H = np.zeros((dim_H,dim_H))

    for i in range(dim_H):
        ket_i = all_vectors[i] # Retrieve the ket |i> from the matrix all_vectors, which contains all possible binary vectors.
                               # This vector is of size L.
        H_ket_i_mat = CL_H_psi(ket_i) # Matrix containing the vectors (of size L) obtained by considering all possible configurations
                                      # starting from the ket |i>.
        ket_H_psi = H_psi(H_ket_i_mat) # Convert to binary indices; a vector of size (2^L - number of forbidden configurations)
                                       # with ones and zeros.

        for j in range(ket_H_psi.shape[0]):
            if ket_H_psi[j] == 1:  # If the transition from ket |i> to ket |j> is possible
                H[i, j] = 1  # We can set the corresponding coefficient in the Hamiltonian

    return H

'''
Si nombres est une liste, par exemple nombres = [10, 20, 30, 40, 50], alors la fonction enumerate(nombres), transformer cette liste en une
autre liste de couple (indice , valeur)

--> enumerate(nombres) va produire [(0,10), (1, 20), (2, 30), (3, 40), (4,50)]
'''


###########################################################
# This function performs the product of matrix M1 with matrix M2 and returns the resulting matrix
def prod_mat(M1 , M2):
    if M1.shape[1] != M2.shape[0]:
        print("ERREUR : le nombre de colonnes de M1 n'est pas égal au nombre de lignes de M2.")
        sys.exit(1)  # Exit the program with an error code

    else:
        result = np.zeros((M1.shape[0] , M2.shape[1]))

        for i in range(M1.shape[0]):
            for j in range(M2.shape[1]):
                for k in range(M1.shape[1]):
                    if M1[i,k] != 0 and M2[k,j] != 0:
                        result[i,j] = result[i,j] + (M1[i,k] * M2[k,j]) # We fill result[i,j] row by row
    return result


###########################################################
# This function performs the product of a matrix with a vector and returns the resulting vector
def prod_mat_vect(M , v):
    if M.shape[1] != v.shape[0]:
        print("ERREUR : le produit matrice-vecteur n'est pas défini.")
        sys.exit(1)  # Exit the programm with an error code

    else:
        result = np.zeros(M.shape[0] , dtype=complex)

        for i in range(M.shape[0]):
            for j in range(v.shape[0]):
                if M[i,j] != 0:
                    result[i] = result[i] + (M[i,j] * v[j])

    return result


###########################################################
# This function computes the factorial of n
def factorielle(n):
    if n < 0:
        raise ValueError("La factorielle n'est pas définie pour les nombres négatifs.")
    elif n == 0 or n == 1:
        return 1
    else:
        return n* factorielle(n-1)


###########################################################
# This function takes a Hamiltonian, an initial psi function, a time step dt, and a natural number K, which is the term of the
# series to be truncated, and returns the wave function at a time t = t_0 + dt, which is "slightly further" than t_0
def compute_psi_t(H , psi_0 , dt , K):
    psi_t = np.copy(psi_0)
    H_k = np.zeros((H.shape[0] , H.shape[1]))

    for n in range(H_k.shape[0]): # We initialize H_k to the identity matrix
        H_k[n,n] = 1

#### Calculation of psi at time t_0 + dt ###
    for k in range(1 , K+1): # We start at k=1 and end at k=K because psi_t has already been initialized to psi_0
        H_k = prod_mat(H_k, H) # We compute H^k
        psi_t = np.copy(psi_t) + (((-1j * dt) ** k) / factorielle(k)) * prod_mat_vect(H_k , psi_0)

    return psi_t


###########################################################
# This function converts any decimal number n into a vector representing its binary form
def decimal_to_binary(n, L):
    #If n<0, We don't want it
    if n < 0:
        raise ValueError("Le nombre doit être positif ou nul.")

    binaire = bin(n)[2:]  # Converts to binary and removes the "0b" prefix from the bin(n) string
    liste_binaire = []  # Empty list to store the bits

    for bit in binaire:  # Loop over each binary character
        liste_binaire.append(int(bit))  # Convert to integer and append to the list

    while(len(liste_binaire) != L):
        liste_binaire.insert(0,0)

    return np.array(liste_binaire)


###########################################################
# This function takes a binary vector of dimension L and returns the corresponding decimal number
def binary_to_decimal(vect_binaire):
    decimal_number = 0
    for i in range(len(vect_binaire)):
        decimal_number = decimal_number + vect_binaire[i]*(2 ** ((len(vect_binaire)-1)-i))
    return decimal_number


###########################################################
# This function takes a dimension L vector and all_indices, and returns a dimension-n vector corresponding to the dimension L vector
def dim_L_to_dim_n(vect_binaire, all_indices):
    decimal_number = binary_to_decimal(vect_binaire)
    psi = np.zeros(all_indices.shape[0])
    for i in range(all_indices.shape[0]):
        if decimal_number == all_indices[i]:
            psi[i] = 1

    return psi


###########################################################
# This function takes a wave function of dimension 21, a number of atoms L, an atom index f, and a list all_indices.
# It returns the average of sigma_z for atom number f.
def sigma_z(psi, L, f, all_indices):

    n = 0
    mean_sigma_z_atom = 0

    for j in range(len(psi)):
        if psi[j] != 0:
            n = all_indices[j]
            vecteur = decimal_to_binary(n,L) # We convert the decimal number into a binary vector (the vector is a temporary buffer)

            if vecteur[f] == 0:
                mean_sigma_z_atom += (abs(psi[j])) ** 2

            else:
                mean_sigma_z_atom -= (abs(psi[j])) ** 2


    return mean_sigma_z_atom


###########################################################
def plot_and_save_1(x, y, titre, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, linestyle='-', color='b', label="Atom 2")

    # Adding titles and captions
    plt.xlabel("Time")
    plt.ylabel("Mean of " + r"$\sigma_{z}^2$")
    plt.title(titre)
    plt.legend()
    plt.grid(True)

    # Saving the plot
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()  # Ferme la figure pour éviter d'afficher dans certains environnements


###########################################################
def plot_and_save_2(x, y1, y2, f1, f2, titre, filename):
    # Creating the plot
    plt.figure(figsize=(8, 5))
    plt.plot(x, y1, linestyle='-', color='b', label="atom f = " + f1) # marker='o' fait que l'on plot les points avec ou non
    plt.plot(x, y2, linestyle='-', color='r', label="atom f = " + f2)


    # Adding titles and legends
    plt.xlabel("Time")
    plt.ylabel(r"Mean of $\sigma_{z}^f$")
    plt.title(titre)
    plt.legend()
    plt.grid(True)

    # Saving the graphic
    plt.savefig(filename, dpi=300)
    plt.close()  # Ferme la figure pour éviter d'afficher dans certains environnements


###########################################################
def plot_and_save_5(x, y1, y2, y3, y4, y5, title, filename):
    """
       This function generates a plot with 5 stacked curves, and limits the y-values to the range [-1, 1].

       Args:
       - x : List or array of data for the x-axis.
       - y1, y2, y3, y4, y5 : Lists or arrays containing the data for each curve.
       - title : Title of the plot.
       - filename : Name of the file to save the image (with .png extension).
       """
    # Create subplots
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(10, 12))

    # List of all curves (y1, y2, ..., y5)
    y_data = [y1, y2, y3, y4, y5]

    # Plot each curve and limit the y-values to [-1, 1]
    for i, ax in enumerate(axes):
        # Clip the y-values to be between -1 and 1
        y_clipped = np.clip(y_data[i], -1, 1)

        # Plot the curve
        ax.plot(x, y_clipped, label=f'Curve {i + 1}')

        # Add a label for each subplot
        ax.set_ylabel(f'L = {i + 4}')
        ax.grid(True)

    # Add an x-label only for the last subplot
    axes[-1].set_xlabel("Time")

    # Add a main title for the entire plot
    fig.suptitle(title)

    # Adjust spacing to avoid overlaps
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the plot
    plt.savefig(filename, dpi=300)

    # Display the plot
    plt.show()


###########################################################






######################################################## MAIN ########################################################


dt = 1  # Time step for the calculation of psi_t
K = 20  # Truncation term of the series
L = 6
f = 4
initial_config = [1,0,0,0,0,0]
number_of_points = 2000
vecteur_binaire_initial = []
all_indices, all_vectors = all_possible_vectors_generation(L)
H = get_hamiltonian(L)
vect2 = dim_L_to_dim_n(initial_config , all_indices)



# Plot the overlapping graphs of sigma_z for atoms 1 and 2
x = []  # Vector of x-values (abscissa points)
y = []
psi_0 = np.copy(vect2)

for i in range(number_of_points):
    x.append(i*dt)
    psi_t = compute_psi_t(H,psi_0,dt,K)
    y.append(sigma_z(psi_t, L, f, all_indices))
    psi_0 = psi_t


titre = "Configuration " + r"$[1,0,0,0,0,0]$"

plot_and_save_1(x, y, titre, filename="graph.png")





'''
x = []

### Fill the x-axis vector
for i in range(number_of_points):
    x.append(i * dt)

y = []
y4 = []
y5 = []
y6 = []
y7 = []
y8 = []
y_i = 0  # For calculating the current spin values
y_max = 0  # To normalize the data points

for L in range(4, 9):  # We make a plot for each number of atoms 4, 5, 6, 7, 8
    #### Fill the initial binary vector configuration as: [1,0,1,0,1,...] starting with an excited atom ####
    initial_binary_vector = []

    for i in range(L):
        if i % 2 == 0:
            initial_binary_vector.append(1)
        else:
            initial_binary_vector.append(0)

    #### Construct the system's Hamiltonian for a given number of atoms L AND translate the initial configuration of dimension L of the system into a vector of dimension n ####
    all_indices, all_vectors = all_possible_vectors_generation(L)
    H = get_hamiltonian(L)
    vect = dim_L_to_dim_n(initial_binary_vector, all_indices)  # Construct the vector of dim n instead of dim L
    psi_0 = compute_psi_t(H, vect, 0, 20)
    y_max = 0  # Reset y_max since it always holds the value from the L-1 iteration

    #### Compute the correlation points for each list yj ####
    for i in range(number_of_points):
        psi_t = compute_psi_t(H, psi_0, dt, K)
        psi_0 = psi_t  # Reset psi_0 to the current value of psi_t
        for f in range(L - 1):  # Loop goes up to 4 inclusive to avoid accessing vector[6] (which doesn't exist) in sigma_z when we do "f+1"
            y_i = y_i + (sigma_z(psi_t, L, f, all_indices) * sigma_z(psi_t, L, f + 1, all_indices))

    #### Store the results in the corresponding list based on the number of atoms ####
        if L == 4:
            y4.append(y_i)
        if L == 5:
            y5.append(y_i)
        if L == 6:
            y6.append(y_i)
        if L == 7:
            y7.append(y_i)
        if L == 8:
            y8.append(y_i)

    #### Find the largest element in the list in absolute value to normalize later ####
        if abs(y_i) > y_max:
            y_max = abs(y_i)
        y_i = 0

    #### Normalize the data based on the number of atoms L being considered ####
    for i in range(len(x)):
        if L == 4:
            y4[i] /= y_max
        if L == 5:
            y5[i] /= y_max
        if L == 6:
            y6[i] /= y_max
        if L == 7:
            y7[i] /= y_max
        if L == 8:
            y8[i] /= y_max

title = r"Correlation function for different values of L:  $\sum_{f=0}^{L-2} \sigma_{z}^{f} \sigma_{z}^{f+1}$ ; with the initial states [1,0,1,0,1,... ]"
filename = "Corr_atoms.png"

plot_and_save_5(x, y4, y5, y6, y7, y8, title, filename)

'''


