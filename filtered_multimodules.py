"""Last updated: Oct 6, 2026

Callable multimodules over the torus algebra, with coefficients in F_2
and monomial coefficients in formal variables W_1, W_2, ...

This is a simple Python 3 adaptation of the tensor-product part of
Hanselman's graph_manifolds_HFhat.py.

Example:
    from multimodules import (
        fCFDsolidtorus_1,
        fCFDsolidtorus_2,
        tube_cutting_piece,
    )

    torus_1 = fCFDsolidtorus_1()
    torus_2 = fCFDsolidtorus_2()
    tube = tube_cutting_piece()

    torus_1.print_ops()
    torus_2.print_ops()
    tube.print_ops()


Every function creates a new module with its own lists and dictionary.


OPERATIONS
----------

Every operation is stored in the form

    [start, W_monomial, boundary_1_entry, ..., boundary_n_entry, end].

The second entry always records a monomial in the formal variables

    W_1, W_2, W_3, ...

The convention is

    ""              = 1
    "W_1"           = W_1
    "W_2"           = W_2
    "W_1^2"         = W_1^2
    "W_1W_2"        = W_1 * W_2
    "W_1^2W_2^2"    = W_1^2 * W_2^2
    ...

Thus an empty string in the W_monomial position means coefficient 1.

For example,

    ["a", "", "1", "b"]

represents an operation from a to b with coefficient rho_1.

Similarly,

    ["a", "W_1", "12", "b"]

represents an operation with coefficient W_1*rho_12,

and

    ["a", "W_1^2W_2^2", "23", "b"]

represents an operation with coefficient W_1^2*W_2^2*rho_23.


MULTIPLE W VARIABLES
--------------------

A multimodule may use one or several W variables.

For example, a module with two filtration variables may contain

    ["a", "W_1", "1", "b"]
    ["b", "W_2", "3", "c"]
    ["c", "W_1W_2", "12", "d"]
    ["d", "W_1^2W_2^3", "23", "a"]

The variables are stored directly in the operation string.

The Multimodule class below handles the formal-variable monomials directly.
All helper routines used by the class are defined inside the class itself.
Differential cancellation, Hochschild homology, and self-gluing are never
performed unless their corresponding methods are explicitly called.


BOUNDARY ENTRIES
----------------

For a type D boundary, the boundary entry is an output string such as

    "1", "2", "3", "12", "23", "123".

For a type A boundary, the boundary entry is a list of input strings,
for example

    ["1", "2"].

The strings

    "1", "2", "3", "12", "23", "123"

represent the corresponding Reeb chords.

At a type D boundary,

    ""

means the compatible idempotent contribution; it does not mean zero.

At a type A boundary,

    []

means that there are no algebra inputs.


IMPORTANT EMPTY-STRING CONVENTION
---------------------------------

The meaning of "" depends on its position.

For example,

    ["a", "", "", "b"]

contains two empty strings.

The first empty string is the W_monomial entry and means coefficient 1.

The second empty string is the type D boundary entry and denotes the
compatible idempotent.


IDEMPOTENTS
-----------

There are exactly two idempotent labels:

    0 = iota_0 = bullet
    1 = iota_1 = circle

Idempotence has one label per boundary, in the same order as
boundary_types.

For example,

    idempotence = {
        "a": [0],
        "b": [1],
    }

means that a lies in iota_0 and b lies in iota_1.

For a bimodule,

    "a": [0, 1]

means that a lies in iota_0 on the first boundary and iota_1 on the
second boundary."""


class Multimodule:
    """
    Filtered bordered-Floer multimodule over the torus algebra with coefficients
    in F_2[W_1, W_2, ...].

    Every operation is stored as

        [start, W_monomial, boundary_1_entry, ..., boundary_n_entry, end].

    Differential cancellation, Hochschild homology, and self-gluing are only
    performed when their corresponding methods are explicitly called.
    """

    def __init__(self, name, boundary_types, generators, idempotence, operations):
        self.name = name
        self.boundary_types = list(boundary_types)
        self.generators = list(generators)
        self.idempotence = {gen: list(idempotence[gen]) for gen in idempotence}
        self.operations = []
        for op in operations:
            new_op = []
            for entry in op:
                new_op.append(list(entry) if isinstance(entry, list) else entry)
            self.operations.append(new_op)
        self._validate_and_normalize()

    def _freeze(self, value):
        """Convert nested lists to tuples so an operation can be used as a key."""
        if isinstance(value, list):
            return tuple(self._freeze(x) for x in value)
        if isinstance(value, tuple):
            return tuple(self._freeze(x) for x in value)
        return value

    def _parse_W_monomial(self, monomial):
        """Parse a monomial such as W_1^2W_3 into an exponent dictionary."""
        if not isinstance(monomial, str):
            raise TypeError("The W coefficient must be stored as a string.")
        s = monomial.replace(" ", "").replace("*", "")
        if s == "" or s == "1":
            return {}
        result = {}
        i = 0
        while i < len(s):
            if not s.startswith("W_", i):
                raise ValueError("Invalid W-monomial: " + repr(monomial))
            i += 2
            start = i
            while i < len(s) and s[i].isdigit():
                i += 1
            if start == i:
                raise ValueError("Missing variable index in " + repr(monomial))
            variable = int(s[start:i])
            exponent = 1
            if i < len(s) and s[i] == "^":
                i += 1
                start = i
                while i < len(s) and s[i].isdigit():
                    i += 1
                if start == i:
                    raise ValueError("Missing exponent in " + repr(monomial))
                exponent = int(s[start:i])
            result[variable] = result.get(variable, 0) + exponent
        return result

    def _format_W_monomial(self, exponents):
        """Write an exponent dictionary in canonical W_1,W_2,... order."""
        pieces = []
        for variable in sorted(exponents):
            exponent = exponents[variable]
            if exponent < 0:
                raise ValueError("Negative W-exponents are not supported.")
            if exponent == 0:
                continue
            pieces.append("W_" + str(variable) if exponent == 1 else "W_" + str(variable) + "^" + str(exponent))
        return "".join(pieces)

    def normalize_W_monomial(self, monomial):
        """Canonicalize a formal-variable monomial."""
        return self._format_W_monomial(self._parse_W_monomial(monomial))

    def multiply_W_monomials(self, *monomials):
        """Multiply formal monomials by adding exponents."""
        total = {}
        for monomial in monomials:
            exponents = self._parse_W_monomial(monomial)
            for variable in exponents:
                total[variable] = total.get(variable, 0) + exponents[variable]
        return self._format_W_monomial(total)

    def torus_product(self, left, right):
        """Multiply two torus-algebra basis elements; return None if the product is zero."""
        if left == "":
            return right
        if right == "":
            return left
        if (left, right) in {("1", "2"), ("1", "23"), ("2", "3"), ("12", "3")}:
            return left + right
        return None

    def torus_product_list(self, elements):
        """Multiply a sequence of torus-algebra basis elements."""
        result = ""
        for element in elements:
            result = self.torus_product(result, element)
            if result is None:
                return None
        return result

    def list_sum(self, list_of_lists):
        """Concatenate a sequence of lists."""
        result = []
        for input_list in list_of_lists:
            result += input_list
        return result

    def connected_components(self, list_of_ops):
        """Group operations by connected components of the underlying unoriented graph."""
        adjacency = {}
        for op in list_of_ops:
            start, end = op[0], op[-1]
            adjacency.setdefault(start, set()).add(end)
            adjacency.setdefault(end, set()).add(start)
        visited = set()
        generator_components = []
        for generator in adjacency:
            if generator in visited:
                continue
            stack = [generator]
            component = set()
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                for neighbour in adjacency.get(current, set()):
                    if neighbour not in visited:
                        stack.append(neighbour)
            generator_components.append(component)
        operation_components = []
        for component in generator_components:
            operation_components.append([op for op in list_of_ops if op[0] in component])
        return operation_components

    def _entry_is_empty(self, entry):
        """True for an empty type-D output or empty type-A input list."""
        return entry == "" or entry == []

    def _reduce_operation_list_mod_2(self, operations):
        """Reduce a list of operations over F_2."""
        parity = {}
        representatives = {}
        order = []
        for op in operations:
            normalized = []
            for index, entry in enumerate(op):
                if index == 1:
                    normalized.append(self.normalize_W_monomial(entry))
                elif isinstance(entry, list):
                    normalized.append(list(entry))
                else:
                    normalized.append(entry)
            key = self._freeze(normalized)
            if key not in parity:
                parity[key] = 0
                representatives[key] = normalized
                order.append(key)
            parity[key] = (parity[key] + 1) % 2
        return [representatives[key] for key in order if parity[key] == 1]

    def _operation_boundary_index(self, boundary):
        """Translate a boundary index to the corresponding operation-list index."""
        if boundary < 0 or boundary >= len(self.boundary_types):
            raise IndexError("Boundary index out of range.")
        return boundary + 2

    def _validate_and_normalize(self):
        """Check operation lengths and normalize W coefficients."""
        expected_length = len(self.boundary_types) + 3
        for op in self.operations:
            if len(op) != expected_length:
                raise ValueError("Operation " + repr(op) + " has the wrong length.")
            op[1] = self.normalize_W_monomial(op[1])
        for gen in self.generators:
            if gen not in self.idempotence:
                raise ValueError("Missing idempotence data for generator " + repr(gen))
            if len(self.idempotence[gen]) != len(self.boundary_types):
                raise ValueError("Generator " + repr(gen) + " does not have one idempotent entry per boundary.")

    def print_ops(self):
        """Print every operation."""
        for op in self.operations:
            print(op)

    def print_ops2(self):
        """Print operations grouped by connected component."""
        for component in self.connected_components(self.operations):
            for op in component:
                print(op)
            print()

    def operations_starting_with(self, generator):
        """Return all operations whose initial generator is generator."""
        return [op for op in self.operations if op[0] == generator]

    def operations_ending_with(self, generator):
        """Return all operations whose final generator is generator."""
        return [op for op in self.operations if op[-1] == generator]

    def reduce_mod_2(self):
        """Cancel identical operations in pairs over F_2."""
        self.operations = self._reduce_operation_list_mod_2(self.operations)
        return self

    def relabel_generators(self):
        """Relabel generators a,b,c,... or g0,g1,... ."""
        available_names = list("abcdefghijklmnopqrstuvwxyz") if len(self.generators) < 26 else ["g" + str(i) for i in range(len(self.generators))]
        translation = {}
        new_generators = []
        new_idempotence = {}
        for i, old_name in enumerate(self.generators):
            new_name = available_names[i]
            translation[old_name] = new_name
            new_generators.append(new_name)
            new_idempotence[new_name] = list(self.idempotence[old_name])
        self.generators = new_generators
        self.idempotence = new_idempotence
        for op in self.operations:
            op[0] = translation[op[0]]
            op[-1] = translation[op[-1]]
        return translation

    def find_differentials(self):
        """Return all operations with no Reeb chords or A-inputs."""
        result = []
        for op in self.operations:
            if all(self._entry_is_empty(entry) for entry in op[2:-1]):
                result.append(op)
        return result

    def find_removable_differential(self):
        """Return one coefficient-1 cancellable differential, or None."""
        for differential in self.find_differentials():
            if self.normalize_W_monomial(differential[1]) != "":
                continue
            initial_generator, final_generator = differential[0], differential[-1]
            other_ops = [op for op in self.operations_starting_with(initial_generator) if op is not differential]
            if final_generator not in [op[-1] for op in other_ops]:
                return differential
        return None

    def remove_differentials(self):
        """Explicitly cancel all removable unit differentials."""
        differential = self.find_removable_differential()
        while differential is not None:
            initial_generator, final_generator = differential[0], differential[-1]
            incoming_ops = [op for op in self.operations_ending_with(final_generator) if op is not differential]
            outgoing_ops = [op for op in self.operations_starting_with(initial_generator) if op is not differential]
            new_operations = []
            for op1 in incoming_ops:
                for op2 in outgoing_ops:
                    new_op = [op1[0], self.multiply_W_monomials(op1[1], op2[1])]
                    valid = True
                    for boundary in range(len(self.boundary_types)):
                        op_index = self._operation_boundary_index(boundary)
                        if self.boundary_types[boundary] == "D":
                            product = self.torus_product(op1[op_index], op2[op_index])
                            if product is None:
                                valid = False
                                break
                            new_op.append(product)
                        elif self.boundary_types[boundary] == "A":
                            new_op.append(list(op1[op_index]) + list(op2[op_index]))
                        else:
                            raise ValueError("Unknown boundary type " + repr(self.boundary_types[boundary]))
                    if valid:
                        new_op.append(op2[-1])
                        if new_op[0] not in [initial_generator, final_generator] and new_op[-1] not in [initial_generator, final_generator]:
                            new_operations.append(new_op)
            for op in self.operations:
                if op[0] not in [initial_generator, final_generator] and op[-1] not in [initial_generator, final_generator]:
                    new_operations.append(op)
            self.operations = new_operations
            if initial_generator in self.generators:
                self.generators.remove(initial_generator)
            if final_generator in self.generators:
                self.generators.remove(final_generator)
            self.idempotence.pop(initial_generator, None)
            self.idempotence.pop(final_generator, None)
            self.reduce_mod_2()
            differential = self.find_removable_differential()
        return self

    def is_valid_D_sequence(self, list_of_ops, boundary):
        """Check nonzero torus products on every other type-D boundary."""
        if len(list_of_ops) == 0:
            raise ValueError("list_of_ops must be nonempty.")
        for b in range(len(self.boundary_types)):
            if b == boundary or self.boundary_types[b] != "D":
                continue
            op_index = self._operation_boundary_index(b)
            if self.torus_product_list([op[op_index] for op in list_of_ops]) is None:
                return False
        return True

    def d_squared(self, print_result=True):
        """Compute filtered d^2 for an all-type-D multimodule."""
        if any(boundary_type != "D" for boundary_type in self.boundary_types):
            raise ValueError("d_squared() is intended for all-type-D multimodules.")
        double_operations = []
        for op1 in self.operations:
            for op2 in self.operations:
                if op1[-1] != op2[0]:
                    continue
                outputs = []
                valid = True
                for boundary in range(len(self.boundary_types)):
                    op_index = self._operation_boundary_index(boundary)
                    product = self.torus_product(op1[op_index], op2[op_index])
                    if product is None:
                        valid = False
                        break
                    outputs.append(product)
                if valid:
                    double_operations.append([op1[0], self.multiply_W_monomials(op1[1], op2[1])] + outputs + [op2[-1]])
        result = self._reduce_operation_list_mod_2(double_operations)
        if print_result:
            print(result)
        return result

    def rank(self, cancel_differentials=False):
        """Return the number of generators of a chain complex with no boundaries."""
        if len(self.boundary_types) != 0:
            raise ValueError("rank() is only defined here for a chain complex with no boundaries.")
        if cancel_differentials:
            self.remove_differentials()
        return len(self.generators)


    def is_there_a_cycle(self, gen, ancestors):
        """Recursive helper for is_bounded()."""
        if gen in ancestors:
            return True
        next_ancestors = ancestors + [gen]
        for op in self.operations_starting_with(gen):
            if self.is_there_a_cycle(op[-1], next_ancestors):
                return True
        return False

    def is_bounded(self):
        """Return True iff the underlying directed operation graph has no cycle."""
        for gen in self.generators:
            if self.is_there_a_cycle(gen, []):
                return False
        return True


# ================================================================
# Solid torus:
# p lies in idempotent 0 = iota_0 = bullet
# differential: W_1*rho_12
def fCFDsolidtorus_1():
    generators = ["p"]
    boundary_types = ["D"]
    idempotence = {
        "p": [0],
    }
    operations = [
        ["p", "W_1", "12", "p"],
    ]
    return Multimodule(
        "fCFDsolidtorus_1",
        boundary_types,
        generators,
        idempotence,
        operations,
    )

# ================================================================
# Solid torus:
# p lies in idempotent 1 = iota_1 = circle
# differential: W_1*rho_23
def fCFDsolidtorus_2():
    generators = ["p"]
    boundary_types = ["D"]
    idempotence = {
        "p": [1],
    }
    operations = [
        ["p", "W_1", "23", "p"],
    ]
    return Multimodule(
        "fCFDsolidtorus_2",
        boundary_types,
        generators,
        idempotence,
        operations,
    )

# ================================================================
# Identity type AA bimodule: CFAA(Id)
def CFAA_id():
    generators = [
        "w_1",
        "w_2",
        "x",
        "y",
        "z_1",
        "z_2",
    ]
    boundary_types = ["A", "A"]
    idempotence = {
        "w_1": [1, 0],
        "w_2": [1, 0],
        "x":   [0, 0],
        "y":   [1, 1],
        "z_1": [0, 1],
        "z_2": [0, 1],
    }
    operations = [
        # Operations starting at w_1
        ["w_1", "", [], ["1"], "y"],
        ["w_1", "", ["2"], ["12"], "x"],
        ["w_1", "", [], [], "w_2"],
        ["w_1", "", ["23"], ["12"], "w_2"],
        ["w_1", "", ["2"], ["123"], "z_2"],
        ["w_1", "", ["2"], ["3", "2", "1"], "z_2"],
        # Operations starting at z_1
        ["z_1", "", ["1"], [], "y"],
        ["z_1", "", ["12"], ["2"], "x"],
        ["z_1", "", [], [], "z_2"],
        ["z_1", "", ["12"], ["23"], "z_2"],
        ["z_1", "", ["123"], ["2"], "w_2"],
        # Operations starting at y
        ["y", "", ["2"], ["2"], "x"],
        ["y", "", ["23"], ["2"], "w_2"],
        ["y", "", ["2"], ["23"], "z_2"],
        # Operations starting at x
        ["x", "", ["3"], [], "w_2"],
        ["x", "", [], ["3"], "z_2"],
    ]
    return Multimodule(
        "CFAA_id",
        boundary_types,
        generators,
        idempotence,
        operations,
    )

# ================================================================
# Identity type DD bimodule: CFDD_id
def CFDD_id():
    generators = [
        "a",
        "b",
    ]
    boundary_types = ["D", "D"]
    idempotence = {
        "a": [1, 1],
        "b": [0, 0],
    }
    operations = [
        ["a", "", "2", "2", "b"],

        ["b", "", "1", "3", "a"],

        ["b", "", "123", "123", "a"],

        ["b", "", "3", "1", "a"],
    ]
    return Multimodule(
        "CFDD_id",
        boundary_types,
        generators,
        idempotence,
        operations,
    )

# ================================================================
# Tube-cutting piece / self-gluing DD bimodule
def tube_cutting_piece():
    generators = [
        "afi", "afj", "afk", "afl", "ahi",
        "ahj", "ahk", "ahl", "ang", "amg",
        "ebi", "ebj", "ebk", "ebl", "edi",
        "edj", "edk", "edl", "enc", "emc",
    ]
    boundary_types = ["D", "D"]
    idempotence = {
        "afi": [1, 1],
        "afj": [1, 0],
        "afk": [1, 1],
        "afl": [1, 0],
        "ahi": [1, 1],
        "ahj": [1, 0],
        "ahk": [1, 1],
        "ahl": [1, 0],
        "ang": [1, 1],
        "amg": [1, 1],

        "ebi": [0, 1],
        "ebj": [0, 0],
        "ebk": [0, 1],
        "ebl": [0, 0],
        "edi": [0, 1],
        "edj": [0, 0],
        "edk": [0, 1],
        "edl": [0, 0],
        "enc": [0, 0],
        "emc": [0, 0],
    }
    operations = [

        ["afi", "", "2", "", "edi"],

        ["afj", "", "", "1", "afi"],
        ["afj", "", "2", "", "edj"],

        ["afk", "", "", "2", "afj"],
        ["afk", "", "2", "", "edk"],

        ["afl", "", "", "123", "afi"],
        ["afl", "", "", "3", "afk"],
        ["afl", "", "", "1", "ang"],
        ["afl", "", "2", "", "edl"],

        ["ahj", "", "", "1", "ahi"],

        ["ahk", "", "", "2", "ahj"],

        ["ahl", "", "", "123", "ahi"],
        ["ahl", "", "", "3", "ahk"],

        ["ang", "", "2", "2", "enc"],

        ["amg", "", "", "23", "afi"],
        ["amg", "", "", "", "ahi"],
        ["amg", "", "23", "", "ahk"],
        ["amg", "", "2", "2", "emc"],

        ["ebi", "", "3", "", "afi"],
        ["ebi", "", "123", "", "ang"],

        ["ebj", "", "3", "", "afj"],
        ["ebj", "", "", "1", "ebi"],
        ["ebj", "", "12", "", "enc"],

        ["ebk", "", "3", "", "afk"],
        ["ebk", "", "1", "", "ang"],
        ["ebk", "", "", "2", "ebj"],

        ["ebl", "", "3", "", "afl"],
        ["ebl", "", "", "123", "ebi"],
        ["ebl", "", "", "3", "ebk"],
        ["ebl", "", "", "", "enc"],

        ["edi", "", "1", "", "ahi"],

        ["edj", "", "1", "", "ahj"],
        ["edj", "", "", "1", "edi"],

        ["edk", "", "1", "", "ahk"],
        ["edk", "", "", "2", "edj"],

        ["edl", "", "1", "", "ahl"],
        ["edl", "", "", "123", "edi"],
        ["edl", "", "", "3", "edk"],
        ["edl", "", "", "12", "enc"],

        ["enc", "", "1", "3", "ang"],
        ["enc", "", "3", "1", "ang"],
        ["enc", "", "123", "123", "ang"],

        ["emc", "", "1", "3", "amg"],
        ["emc", "", "3", "1", "amg"],
        ["emc", "", "123", "123", "amg"],
        ["emc", "", "3", "", "ahj"],
        ["emc", "", "", "3", "edi"],
        ["emc", "", "123", "", "ahl"],
        ["emc", "", "", "123", "ebi"],

        ["ahl", "", "", "123", "ang"],
    ]
    return Multimodule(
        "tube_cutting_piece",
        boundary_types,
        generators,
        idempotence,
        operations,
    )

# ================================================================
# Type DA bimodule:
# Positive Dehn twist about alpha_1 on the type A side
# ================================================================
def CFDA_twist1():
    generators = ["x", "y", "z"]
    boundary_types = ["D", "A"]
    idempotence = {
        "x": [1, 1],
        "y": [0, 1],
        "z": [0, 0],
    }
    operations = [
        ["x", "", "2", ["2", "1"], "y"],
        ["x", "", "2", ["2", "12"], "z"],
        ["x", "", "23", ["2", "123"], "x"],
        ["y", "", "1", [], "x"],
        ["y", "", "", ["2"], "z"],
        ["y", "", "3", ["23"], "x"],
        ["z", "", "12", ["1"], "y"],
        ["z", "", "12", ["12"], "z"],
        ["z", "", "123", ["123"], "x"],
        ["z", "", "3", ["3"], "x"],
    ]
    return Multimodule(
        "CFDA_twist1",
        boundary_types,
        generators,
        idempotence,
        operations,
    )


# ================================================================
# Type DA bimodule:
# Negative Dehn twist about alpha_1 on the type A side
# ================================================================
def CFDA_twist1_inv():
    generators = ["x", "y", "z"]
    boundary_types = ["D", "A"]
    idempotence = {
        "x": [0, 0],
        "y": [0, 1],
        "z": [1, 1],
    }
    operations = [
        ["x", "", "3", ["3"], "z"],
        ["x", "", "", ["1"], "y"],
        ["x", "", "12", ["12"], "x"],
        ["x", "", "123", ["123"], "z"],
        ["x", "", "1", ["12", "1"], "z"],
        ["y", "", "12", ["2"], "x"],
        ["y", "", "123", ["23"], "z"],
        ["y", "", "1", ["2", "1"], "z"],
        ["z", "", "2", [], "y"],
    ]
    return Multimodule(
        "CFDA_twist1_inv",
        boundary_types,
        generators,
        idempotence,
        operations,
    )


# ================================================================
# Type DA bimodule:
# Positive Dehn twist about alpha_2 on the type A side
# ================================================================
def CFDA_twist2():
    generators = ["x", "y", "z"]
    boundary_types = ["D", "A"]
    idempotence = {
        "x": [0, 0],
        "y": [1, 0],
        "z": [1, 1],
    }
    operations = [
        ["x", "", "1", ["1"], "z"],
        ["x", "", "123", ["12"], "y"],
        ["x", "", "123", ["123"], "z"],
        ["x", "", "3", ["3", "2"], "y"],
        ["x", "", "3", ["3", "23"], "z"],
        ["y", "", "", ["3"], "z"],
        ["y", "", "2", [], "x"],
        ["z", "", "23", ["2"], "y"],
        ["z", "", "23", ["23"], "z"],
    ]
    return Multimodule(
        "CFDA_twist2",
        boundary_types,
        generators,
        idempotence,
        operations,
    )


# ================================================================
# Type DA bimodule:
# Negative Dehn twist about alpha_2 on the type A side
# ================================================================
def CFDA_twist2_inv():
    generators = ["x", "y", "z"]
    boundary_types = ["D", "A"]
    idempotence = {
        "x": [0, 0],
        "y": [1, 0],
        "z": [1, 1],
    }
    operations = [
        ["x", "", "1", ["1"], "z"],
        ["x", "", "1", ["12"], "y"],
        ["x", "", "123", ["123"], "z"],
        ["x", "", "12", ["123", "2"], "x"],
        ["x", "", "3", [], "y"],
        ["y", "", "23", ["3"], "z"],
        ["y", "", "2", ["3", "2"], "x"],
        ["z", "", "", ["2"], "y"],
        ["z", "", "2", ["23", "2"], "x"],
        ["z", "", "23", ["23"], "z"],
    ]
    return Multimodule(
        "CFDA_twist2_inv",
        boundary_types,
        generators,
        idempotence,
        operations,
    )

# ================================================================
# Type DDD trimodule: CFDDD
# Pair-of-pants x S^1
def CFDDD():
    generators = [
        "v",
        "w",
        "x",
        "y",
        "z",
    ]
    boundary_types = ["D", "D", "D"]
    idempotence = {
        "v": [0, 0, 0],
        "w": [0, 0, 0],
        "x": [1, 0, 0],
        "y": [1, 1, 1],
        "z": [0, 1, 0],
    }
    operations = [
        ["v", "", "1", "123", "3", "y"],
        ["v", "", "123", "123", "123", "y"],
        ["v", "", "", "3", "", "z"],
        ["v", "", "3", "", "", "x"],

        ["w", "", "1", "1", "3", "y"],
        ["w", "", "3", "", "12", "x"],
        ["w", "", "123", "1", "123", "y"],

        ["x", "", "2", "", "12", "v"],
        ["x", "", "", "3", "1", "y"],
        ["x", "", "2", "", "", "w"],

        ["y", "", "", "2", "2", "x"],
        ["y", "", "2", "", "2", "z"],

        ["z", "", "", "2", "", "w"],
        ["z", "", "3", "", "1", "y"],
    ]

    return Multimodule(
        "CFDDD",
        boundary_types,
        generators,
        idempotence,
        operations,
    )

# ================================================================
# Mirror type DDD trimodule: CFDDD_mirror
def CFDDD_mirror():
    generators = [
        "v",
        "w",
        "x",
        "y",
        "z",
    ]
    boundary_types = ["D", "D", "D"]
    idempotence = {
        "v": [1, 1, 1],
        "w": [1, 1, 1],
        "x": [0, 1, 1],
        "y": [0, 0, 0],
        "z": [1, 0, 1],
    }
    operations = [

        ["y", "", "3", "123", "1", "v"],
        ["y", "", "123", "123", "123", "v"],
        ["z", "", "", "1", "", "v"],
        ["x", "", "1", "", "", "v"],

        ["y", "", "3", "3", "1", "w"],
        ["x", "", "1", "", "23", "w"],
        ["y", "", "123", "3", "123", "w"],

        ["v", "", "2", "", "23", "x"],
        ["y", "", "", "1", "3", "x"],
        ["w", "", "2", "", "", "x"],

        ["x", "", "", "2", "2", "y"],
        ["z", "", "2", "", "2", "y"],

        ["w", "", "", "2", "", "z"],
        ["y", "", "1", "", "3", "z"],
    ]

    return Multimodule(
        "CFDDD_mirror",
        boundary_types,
        generators,
        idempotence,
        operations,
    )