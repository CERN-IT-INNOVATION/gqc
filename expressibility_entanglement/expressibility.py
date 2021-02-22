# computes entanglement capability and expressbility of circuits

# based on: https://arxiv.org/abs/1905.10876

import os
import numpy as np
import pennylane as qml
from qiskit.quantum_info import partial_trace, DensityMatrix, random_statevector,state_fidelity

import scipy as sc
def sc_normalize_vector(vector):
    """
    Normalize the input state vector.
    """
    return vector / sc.linalg.norm(vector)

# number of threads to use
os.environ['OMP_NUM_THREADS'] = '2'

def compute_expressibility(fidelities, N_qubits):
    N_bins   = 75 
    N_sample = fidelities.shape[0]
    # Convert fidelities to a histogram
    binning = np.linspace(0,1,N_bins+1)
    bin_centers = np.array([(binning[i+1]+binning[i])/2 for i in range(N_bins)])
    fids, bins = np.histogram(fidelities,bins=binning)
    fids = (fids / N_sample) # normalize the histogram

    # Compute P_haar(F)
    P_haar = ((2**N_qubits) - 1) * (1 - bin_centers)**((2**N_qubits)-2)
    P_haar = P_haar / sum(P_haar) # normalize
    
    # Compute Kullback-Leibler (KL) Divergence
    TOLERANCE = 1e-18
    D_kl = 0
    for i in range(N_bins):
        value = fids[i] 
        if (value > TOLERANCE) and (P_haar[i] > TOLERANCE):
            D_kl += value * np.log(value/P_haar[i])

    return D_kl

def compute_Q_ptrace(density_matrix, N):
    TOLERANCE = 1e-6
    # Calculate Q (Meyer and Wallach entanglement measure)
    entanglement_sum = 0
    for k in range(N):
        rho_k = partial_trace(density_matrix,[k]).data
        rho_k_sq = np.conjugate(np.transpose(rho_k)) @ rho_k
        entanglement_sum += rho_k_sq.trace()  
    Q = 2*(1 - (entanglement_sum.real/N))
    if Q < -TOLERANCE or Q > (1. + TOLERANCE):
        print(Q)
    assert Q >= -TOLERANCE and Q <= (1. + TOLERANCE)
    return Q

def compute_Q(input_, params, qnode):
    rho = qnode(input_, params)
    return compute_Q_ptrace(DensityMatrix(rho), input_.shape[0]) 

def circuit(weights):
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i) 
    
    for idx in range(n_iter):
        for i in range(n_qubits):
            qml.CZ(wires=[n_qubits-1-i,(n_qubits-2-i)%n_qubits]) 

        for i in range(n_qubits):
            qml.RY(weights[i+n_qubits*(idx+1)],wires=i)

    return qml.density_matrix(np.arange(0, n_qubits))
'''
Section with circuit definitions:

'''
#FIXME: Works only for n_qubits=4, that's what we care for atm.
def my_circuit14(weights): 
    for rep in range(n_iter):
        for i in range(n_qubits):
            qml.RY(weights[i], wires=i) 
        qml.CRX(weights[4],wires=[3,0])
        qml.CRX(weights[5],wires=[2,3])
        qml.CRX(weights[6],wires=[1,2])
        qml.CRX(weights[7],wires=[0,1])
        for i,iq in zip(range(8,12),range(n_qubits)):
            qml.RY(weights[i],wires=iq)
        qml.CRX(weights[12],wires=[3,2])
        qml.CRX(weights[13],wires=[0,3])
        qml.CRX(weights[14],wires=[1,0])
        qml.CRX(weights[15],wires=[2,1])
    return qml.density_matrix(np.arange(0, n_qubits))

def amp_enc(weights):
    #FIXME: Need to normalize amp to 1. map data to data/norm
    for qubit in range(n_qubits):
        #qml.U3(weights[qubit],0,0,wires=qubit)#equivalent to RY
        qml.RY(weights[qubit],wires=qubit)
    iqubit = range(n_qubits)
    qml.CNOT(wires=[iqubit[n_qubits-1],iqubit[n_qubits-2]])
    qml.RY(weights[4],wires = iqubit[n_qubits-2])
    qml.CNOT(wires = [iqubit[n_qubits-1],iqubit[n_qubits-2]])
    qml.CNOT(wires = [n_qubits-2,n_qubits-3])
    qml.RY(weights[5],wires=iqubit[n_qubits-3])
    qml.CNOT(wires = [n_qubits-1,n_qubits-3])
    qml.RY(weights[6],wires=iqubit[n_qubits-3])
    qml.CNOT(wires = [n_qubits-2,n_qubits-3])
    qml.RY(weights[7],wires=iqubit[n_qubits-3])
    qml.CNOT(wires = [n_qubits-1,n_qubits-3])
    qml.CNOT(wires = [n_qubits-3,n_qubits-4])
    qml.RY(weights[8],wires=iqubit[n_qubits-4])
    qml.CNOT(wires = [n_qubits-2,n_qubits-4])
    qml.RY(weights[9],wires=iqubit[n_qubits-4])
    qml.CNOT(wires = [n_qubits-3,n_qubits-4])
    qml.RY(weights[10],wires=iqubit[n_qubits-4])
    qml.CNOT(wires = [n_qubits-1,n_qubits-4])
    qml.RY(weights[11],wires=iqubit[n_qubits-4])
    qml.CNOT(wires = [n_qubits-3,n_qubits-4])
    qml.RY(weights[12],wires=iqubit[n_qubits-4])
    qml.CNOT(wires = [n_qubits-2,n_qubits-4])
    qml.RY(weights[13],wires=iqubit[n_qubits-4])
    qml.CNOT(wires = [n_qubits-3,n_qubits-4])
    qml.RY(weights[14],wires=iqubit[n_qubits-4])
    qml.CNOT(wires = [n_qubits-1,n_qubits-4])
    #FIXME: check the weights enumeration...
    
    return qml.density_matrix(np.arange(0, n_qubits))

#8 qubits:
def u2_reuploading(weights,n_qubits = 8):
    for feature,qubit in zip(range(0,2*n_qubits,2),range(n_qubits
    )):
        #Pennylane decomposes to Ry,Rz or Rφ(phase) gates:
        qml.U3(np.pi/2,weights[feature],weights[feature+1],wires=qubit)
    for qubit in range(n_qubits):
        if qubit == n_qubits - 1 :
            break
        qml.CNOT(wires=[qubit,qubit+1])
    
    for feature,qubit in zip(range(0,2*n_qubits,2),range(n_qubits
    )):
        #Pennylane decomposes to Ry,Rz or Rφ(phase) gates:
        qml.U3(weights[feature],weights[feature+1],0,wires=qubit)
    
    return qml.density_matrix(np.arange(0,n_qubits))

if __name__ ==  '__main__':
    # needed to work with density matrices in pennylane
    qml.enable_tape()

    # setup circuit
    n_qubits = 8
    n_params = 16
    
    n_iter = 1 # number of layers of the circuit
    #n_params = n_qubits * (1+n_iter) # calculate number of paramters of the circuit
    
    dev = qml.device('default.qubit', wires=n_qubits)
    #qnode = qml.QNode(circuit, dev)
    qnode=qml.QNode(u2_reuploading,dev)
    n_trials = int(1e+4/2)
    #set it to 10 to check if everything works
    # set it to a large number to get accurate results
    #n_trials = int(1e+5)

    ent = np.zeros(n_trials)
    entanglement = np.zeros(n_trials*2)
    fidelities = np.zeros(n_trials)

    #Draw circuit:
    params = np.random.rand(n_params)*2*np.pi
    rho1 = qnode(params)
    print(qnode.draw())
    
    for n in range(n_trials):

        # run circuit with random parameters and get the state vector
        params = np.random.rand(n_params)*2*np.pi
        #params = np.random.rand(n_params)*np.pi
        rho1 = qnode(params)
        
        params = np.random.rand(n_params)*2*np.pi
        #params = np.random.rand(n_params)*np.pi
        rho2 = qnode(params)
        
        entanglement[n] = compute_Q_ptrace(DensityMatrix(rho1), n_qubits) 
        entanglement[n+n_trials] = compute_Q_ptrace(DensityMatrix(rho2), n_qubits) 

        fidelities[n] = state_fidelity(DensityMatrix(rho1),DensityMatrix(rho2))
       

    ent_avg = np.mean(entanglement)
    exp = compute_expressibility(fidelities, N_qubits=n_qubits)

    print('Entangling capability: {:.4f}, Expressibility: {:.4f}'.format(ent_avg, exp))
     

