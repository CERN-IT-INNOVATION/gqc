from qiskit.circuit.library import PauliFeatureMap


class ZZFeatureMap(PauliFeatureMap):
    def __init__(
        self,
        feature_dimension,
        reps=2,
        entanglement="linear",
        data_map_func=None,
        insert_barriers=False,
        name="ZZFeatureMap",
        parameter_prefix="x",
    ):
        """
        Create a new second-order Pauli-Z expansion.
        @feature_dimension :: Number of features.
        @reps              :: The number of repeated circuits, has a min. 1.
        @entanglement      :: Specifies the entanglement structure. Refer to
        @data_map_func     :: A mapping function for data x.
        @insert_barriers   :: If True, barriers are inserted in between the
                              evolution instructions  and hadamard layers.
        """
        if feature_dimension < 2:
            raise ValueError(
                "The ZZFeatureMap contains 2-local interactions"
                "and cannot be defined for less than 2 qubits."
                f"You provided {feature_dimension}."
            )

        super().__init__(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            paulis=["Z", "ZZ"],
            data_map_func=data_map_func,
            insert_barriers=insert_barriers,
            name=name,
            parameter_prefix=parameter_prefix,
        )
