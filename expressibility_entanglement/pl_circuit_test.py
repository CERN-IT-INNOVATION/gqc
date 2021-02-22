
import numpy as np
import pennylane as qml
from qiskit.quantum_info import partial_trace, DensityMatrix, random_statevector,state_fidelity

dev1 = qml.device('default.qubit', wires=2)
#Circuit definitions. Draw circuits for x-check:
@qml.qnode(dev1)
def circuit(x):#test
    qml.RZ(x, wires=0)
    qml.CNOT(wires=[0,1])
    qml.RY(x, wires=1)
    return qml.expval(qml.PauliZ(0))

##############################################
# 4 QUBIT CIRCUITS: weight = features here!...

#FIXME: Works only for n_qubits=4, that's what we care for atm.
#Test feature map with good expressibility and entanglement capability from the work: https://zenodo.org/record/4298781

dev_4qubit = qml.device('default.qubit',wires=4)
@qml.qnode(dev_4qubit)
def circuit14(weights,n_qubits = 4,n_iter=1): 
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
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev_4qubit)
#Copying from qiskit implementation of StateVectorCircuit.
def amp_enc(weights,n_qubits = 4):
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

    return qml.expval(qml.PauliZ(0))

#8 QUBIT CIRCUITS:
dev_8qubit = qml.device('default.qubit',wires = 8)
@qml.qnode(dev_8qubit)
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
    
    return qml.expval(qml.PauliZ(0))
##############################################

circuit_dictionary = {'circuit14':circuit14,'u2_reuploading':u2_reuploading,
'amp_enc':amp_enc}
circuit_names = list(circuit_dictionary.keys())
circuit_funcs = list(circuit_dictionary.values())

if __name__ == '__main__':

    n_features = 16
    np.random.seed(0)
    feature_vector = np.random.rand(n_features)
    for i,icircuit in enumerate(circuit_funcs):
        print(f'Circuit: {circuit_names[i]} \n')
        icircuit(feature_vector)
        #print(icircuit(feature_vector))
        print(icircuit.draw())