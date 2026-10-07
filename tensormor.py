"""Last updated: Oct 6, 2026

Tensor products for filtered bordered-Floer multimodules.

This is a simple Python 3 adaptation of the tensor-product part of
Hanselman's graph_manifolds_HFhat.py.

It is written for the filtered Multimodule convention

    [start, W_monomial, boundary_1, ..., boundary_n, end]

where the W coefficient may be "", "W_1", "W_2", "W_1^2W_2", etc.

Put this file in the same folder as filtered_multimodules.py.
"""

from filtered_multimodules import Multimodule


def make_new_operation(module1, A_op, boundary1, module2, list_of_D_ops, boundary2):
    """Make a tensor operation from one A operation and a matching D path."""
    initial_generator = (A_op[0], list_of_D_ops[0][0])
    final_generator = (A_op[-1], list_of_D_ops[-1][-1])
    W_terms = [A_op[1]] + [op[1] for op in list_of_D_ops]
    W_coefficient = module1.multiply_W_monomials(*W_terms)
    new_operation = [initial_generator, W_coefficient]
    for i in range(len(module1.boundary_types)):
        if i != boundary1:
            new_operation.append(A_op[i + 2])
    for i in range(len(module2.boundary_types)):
        if i == boundary2:
            continue
        new_sequence = [op[i + 2] for op in list_of_D_ops]
        if module2.boundary_types[i] == "A":
            result = []
            for term in new_sequence:
                result += term
        else:
            result = module2.torus_product_list(new_sequence)
            if result is None:
                return None
        new_operation.append(result)
    new_operation.append(final_generator)
    return new_operation


def make_new_operation_from_Adiff(module1, A_op, boundary1, module2, D_gen, boundary2):
    """Make a tensor operation from an A operation with no glued A inputs."""
    initial_generator = (A_op[0], D_gen)
    final_generator = (A_op[-1], D_gen)
    new_op = [initial_generator, A_op[1]]
    for i in range(len(module1.boundary_types)):
        if i != boundary1:
            new_op.append(A_op[i + 2])
    for i in range(len(module2.boundary_types)):
        if i == boundary2:
            continue
        if module2.boundary_types[i] == "A":
            new_op.append([])
        else:
            new_op.append("")
    new_op.append(final_generator)
    return new_op


def make_new_operation_from_Ddiff(module1, A_gen, boundary1, module2, D_op, boundary2):
    """Make a tensor operation from a D operation with no glued D output."""
    initial_generator = (A_gen, D_op[0])
    final_generator = (A_gen, D_op[-1])
    new_op = [initial_generator, D_op[1]]
    for i in range(len(module1.boundary_types)):
        if i == boundary1:
            continue
        if module1.boundary_types[i] == "A":
            new_op.append([])
        else:
            new_op.append("")
    for i in range(len(module2.boundary_types)):
        if i != boundary2:
            new_op.append(D_op[i + 2])
    new_op.append(final_generator)
    return new_op


def tensor(module1, boundary1, module2, boundary2, simplify=False):
    """
    Compute the box tensor product along one A boundary and one D boundary.

    module1.boundary_types[boundary1] must be "A".
    module2.boundary_types[boundary2] must be "D".

    The formal W coefficients multiply along matched paths.

    By default simplify=False, so differential cancellation is NOT automatic.
    If simplify=True, remove_differentials() is called at the end.
    """
    if module1.boundary_types[boundary1] != "A":
        raise ValueError("The selected boundary of module1 must be type A.")
    if module2.boundary_types[boundary2] != "D":
        raise ValueError("The selected boundary of module2 must be type D.")

    # Determine the tensor generators.
    new_gens = []
    for gen1 in module1.generators:
        for gen2 in module2.generators:
            if module1.idempotence[gen1][boundary1] == module2.idempotence[gen2][boundary2]:
                new_gens.append((gen1, gen2))

    # Remove the two glued boundaries.
    new_boundary_types = (
        module1.boundary_types[:boundary1]
        + module1.boundary_types[boundary1 + 1:]
        + module2.boundary_types[:boundary2]
        + module2.boundary_types[boundary2 + 1:]
    )

    # Idempotents of the tensor generators on the remaining boundaries.
    new_idempotence = {}
    for gen1, gen2 in new_gens:
        new_idempotence[(gen1, gen2)] = (
            module1.idempotence[gen1][:boundary1]
            + module1.idempotence[gen1][boundary1 + 1:]
            + module2.idempotence[gen2][:boundary2]
            + module2.idempotence[gen2][boundary2 + 1:]
        )

    new_operations = []

    # Reeb_chains stores all D paths which match partial A-input sequences.
    Reeb_chains = {}
    for op in module1.operations:
        Reeb_sequence = op[boundary1 + 2]
        initial_gen = op[0]
        initial_idem = module1.idempotence[initial_gen][boundary1]
        for i in range(1, len(Reeb_sequence) + 1):
            key = tuple([initial_idem] + Reeb_sequence[:i])
            Reeb_chains[key] = []

    # Start the D paths with one matching D operation.
    for op in module2.operations:
        initial_gen = op[0]
        initial_idem = module2.idempotence[initial_gen][boundary2]
        Reeb_chord = op[boundary2 + 2]
        key = (initial_idem, Reeb_chord)
        if key in Reeb_chains:
            Reeb_chains[key].append([op])

    # Extend the D paths until they match all partial A-input sequences.
    partial_chains = sorted(Reeb_chains.keys(), key=len)
    for pc in partial_chains:
        pclist = list(pc)
        for path in list(Reeb_chains[pc]):
            for op in module2.operations_starting_with(path[-1][-1]):
                next_key = tuple(pclist + [op[boundary2 + 2]])
                if next_key in Reeb_chains:
                    if module2.is_valid_D_sequence(path + [op], boundary2):
                        Reeb_chains[next_key].append(path + [op])

    # Operations coming from A operations.
    for op in module1.operations:
        initial_Agen = op[0]
        initial_idem = module1.idempotence[initial_Agen][boundary1]
        Reeb_sequence = op[boundary1 + 2]

        # A operation with no inputs on the glued boundary.
        if Reeb_sequence == []:
            for Dgen in module2.generators:
                if initial_idem == module2.idempotence[Dgen][boundary2]:
                    new_operations.append(
                        make_new_operation_from_Adiff(
                            module1, op, boundary1, module2, Dgen, boundary2
                        )
                    )

        # A operation with a nonempty Reeb-input sequence.
        else:
            key = tuple([initial_idem] + Reeb_sequence)
            for path in Reeb_chains.get(key, []):
                new_op = make_new_operation(
                    module1, op, boundary1, module2, path, boundary2
                )
                if new_op is not None:
                    new_operations.append(new_op)

    # Operations coming from D operations with empty output on the glued boundary.
    for D_op in module2.operations:
        if D_op[boundary2 + 2] != "":
            continue
        initial_D_gen = D_op[0]
        initial_D_idem = module2.idempotence[initial_D_gen][boundary2]
        for A_gen in module1.generators:
            if module1.idempotence[A_gen][boundary1] == initial_D_idem:
                new_operations.append(
                    make_new_operation_from_Ddiff(
                        module1, A_gen, boundary1, module2, D_op, boundary2
                    )
                )

    name = str(module1.name) + "_box_" + str(module2.name)
    new_module = Multimodule(
        name,
        new_boundary_types,
        new_gens,
        new_idempotence,
        new_operations,
    )

    # Mod-2 reduction is algebraic simplification, not differential cancellation.
    new_module.reduce_mod_2()

    # Differential cancellation happens only if explicitly requested.
    if simplify:
        new_module.remove_differentials()

    return new_module


box_tensor_product = tensor

# ================================================================
# Convert a filtered multimodule to the unfiltered one by setting
# every W_i = 1, then reduce mod 2 and cancel removable differentials.
# ================================================================

def unfiltered_reduced(module):
    new_operations = []

    for op in module.operations:
        new_op = list(op)
        new_op[1] = ""
        new_operations.append(new_op)

    new_idempotence = {}
    for g in module.generators:
        new_idempotence[g] = list(module.idempotence[g])

    unfiltered = Multimodule(
        module.name + "_unfiltered",
        list(module.boundary_types),
        list(module.generators),
        new_idempotence,
        new_operations,
    )

    unfiltered.reduce_mod_2()
    unfiltered.remove_differentials()

    return unfiltered

# ================================================================
# Convert a filtered multimodule to the associated graded object
# by setting every formal variable W_i = 0.
#
# Thus:
#     coefficient ""      survives
#     coefficient W_1     becomes 0
#     coefficient W_1W_2  becomes 0
#     coefficient W_i^k   becomes 0
#
# By default we do NOT cancel differentials afterward, so the
# returned object can be used in further tensor/morphism calculations.
# ================================================================

def gradedW0(module, simplify=False):
    """
    Return the specialization of a filtered Multimodule at

        W_1 = W_2 = W_3 = ... = 0.

    Any operation containing a positive power of a formal variable W_i
    is deleted. Operations with W coefficient 1, stored as "", survive.

    Parameters
    ----------
    module : Multimodule
        The filtered multimodule.

    simplify : bool, default False
        If True, cancel removable coefficient-1 differentials after
        taking W=0. By default no differential cancellation is done.

    Returns
    -------
    Multimodule
        The associated graded / W=0 specialization.
    """

    new_operations = []

    for op in module.operations:

        # Normalize first, in case the W coefficient was entered
        # in a non-canonical form.
        W_coefficient = module.normalize_W_monomial(op[1])

        # At W_i = 0, only coefficient 1 survives.
        if W_coefficient != "":
            continue

        new_op = []

        for entry in op:
            if isinstance(entry, list):
                new_op.append(list(entry))
            else:
                new_op.append(entry)

        # Explicitly store surviving coefficient as 1.
        new_op[1] = ""

        new_operations.append(new_op)

    # Copy idempotence data.
    new_idempotence = {
        g: list(module.idempotence[g])
        for g in module.generators
    }

    graded = Multimodule(
        module.name + "_gradedW0",
        list(module.boundary_types),
        list(module.generators),
        new_idempotence,
        new_operations,
    )

    # Equal surviving operations cancel over F_2.
    graded.reduce_mod_2()

    # Optional homotopy reduction.
    if simplify:
        graded.remove_differentials()

    return graded
