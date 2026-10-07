"""Last updated: Oct 6, 2026

LaTeX display helpers for all-type-D bordered Floer multimodules.

Works for:
    D, DD, DDD, DDDD, ...

Idempotent convention:
    0 = iota_0 = bullet
    1 = iota_1 = circle

Display conventions:
    - If every D-output is empty, use a long double arrow.
    - If W is present on an empty algebra arrow, put W over the double arrow.
    - If a Reeb chord is present, put the full coefficient over the arrow.
    - For a single D boundary, write W_1 rho_12 without extra parentheses.
    - For DD, DDD, ... group the tensor coefficient when W is present.
"""


def generator_tex(g):
    """
    Convert a generator name to LaTeX.

    Tuple generators are displayed as tensor products.
    """
    if isinstance(g, tuple):
        return (
            '('
            + r'\otimes '.join(generator_tex(x) for x in g)
            + ')'
        )

    return str(g).replace('_', r'\_')


def W_tex(W):
    """
    Convert a W-monomial to LaTeX.

    Examples:
        ""          -> ""
        "W_1"       -> W_{1}
        "W_1^2"     -> W_{1}^{2}
        "W_1W_3"    -> W_{1}W_{3}
        "W_1^2W_3"  -> W_{1}^{2}W_{3}
    """
    if W == '':
        return ''

    result = ''
    i = 0

    while i < len(W):
        if W[i:i+2] != 'W_':
            raise ValueError('Invalid W-monomial: ' + W)

        i += 2

        # Read the variable index.
        start = i
        while i < len(W) and W[i].isdigit():
            i += 1

        index = W[start:i]

        if index == '':
            raise ValueError('Invalid W-monomial: ' + W)

        result += r'W_{' + index + '}'

        # Read an optional exponent.
        if i < len(W) and W[i] == '^':
            i += 1
            start = i

            while i < len(W) and W[i].isdigit():
                i += 1

            exponent = W[start:i]

            if exponent == '':
                raise ValueError('Invalid W-monomial: ' + W)

            result += '^{' + exponent + '}'

    return result


def idempotent_tex(i):
    """
    Generator idempotent notation.

        0 = iota_0 = bullet
        1 = iota_1 = circle
    """
    if i == 0:
        return r'\bullet'

    if i == 1:
        return r'\circ'

    raise ValueError('Idempotent must be 0 or 1.')


def algebra_idempotent_tex(i):
    """
    Algebra idempotent notation.
    """
    if i == 0:
        return r'\iota_{0}'

    if i == 1:
        return r'\iota_{1}'

    raise ValueError('Idempotent must be 0 or 1.')


def reeb_symbol(boundary):
    """
    Choose a Reeb-chord symbol for each D boundary.

        boundary 0 -> rho
        boundary 1 -> sigma
        boundary 2 -> tau

    Further boundaries use indexed rho notation.
    """
    if boundary == 0:
        return r'\rho'

    if boundary == 1:
        return r'\sigma'

    if boundary == 2:
        return r'\tau'

    return r'\rho^{(' + str(boundary + 1) + ')}'


def display_type_D_differential(module, show=True):
    """
    Display any multimodule whose remaining boundaries are all type D.

    Supported:
        D, DD, DDD, DDDD, ...

    Operation convention:
        [start, W, D_1, D_2, ..., D_n, end]
    """
    if len(module.boundary_types) == 0:
        raise ValueError('Module has no remaining D boundary.')

    if not all(bt == 'D' for bt in module.boundary_types):
        raise ValueError(
            'display_type_D_differential requires all boundaries to be type D.'
        )

    number_of_boundaries = len(module.boundary_types)

    # ============================================================
    # Generators
    # ============================================================

    generator_formulas = []

    for g in module.generators:
        idempotents = [
            idempotent_tex(i)
            for i in module.idempotence[g]
        ]

        if number_of_boundaries == 1:
            idem_text = idempotents[0]
        else:
            idem_text = (
                r'\left('
                + ','.join(idempotents)
                + r'\right)'
            )

        generator_formulas.append(
            idem_text
            + r'\qquad '
            + generator_tex(g)
        )

    # ============================================================
    # Differential
    # ============================================================

    operation_formulas = []

    for g in module.generators:
        outgoing = [
            op
            for op in module.operations
            if op[0] == g
        ]

        if outgoing == []:
            operation_formulas.append(
                r'\delta('
                + generator_tex(g)
                + r')&=0'
            )
            continue

        for j, op in enumerate(outgoing):
            W = W_tex(op[1])

            # D outputs are op[2], op[3], ..., op[n+1].
            D_outputs = [
                op[2 + boundary]
                for boundary in range(number_of_boundaries)
            ]

            all_empty = all(
                entry == ''
                for entry in D_outputs
            )

            # ----------------------------------------------------
            # Build the algebra coefficient.
            # ----------------------------------------------------

            algebra_factors = []

            for boundary in range(number_of_boundaries):
                entry = D_outputs[boundary]

                if entry == '':
                    # Empty D output means the compatible algebra idempotent.
                    source_i = module.idempotence[op[0]][boundary]
                    factor = algebra_idempotent_tex(source_i)
                else:
                    symbol = reeb_symbol(boundary)
                    factor = (
                        symbol
                        + r'_{'
                        + entry
                        + '}'
                    )

                algebra_factors.append(factor)

            if all_empty:
                algebra_tex = ''
            else:
                algebra_tex = r'\otimes '.join(algebra_factors)

            # ----------------------------------------------------
            # Build the full coefficient.
            #
            # Single D boundary:
            #     W_1 rho_12
            #
            # DD, DDD, ...:
            #     W_1 (rho_1 tensor sigma_3 ...)
            # ----------------------------------------------------

            if W != '' and algebra_tex != '':
                if number_of_boundaries == 1:
                    coefficient = W + algebra_tex
                else:
                    coefficient = (
                        W
                        + r'\left('
                        + algebra_tex
                        + r'\right)'
                    )

            elif W != '':
                coefficient = W

            else:
                coefficient = algebra_tex

            # ----------------------------------------------------
            # Differential term.
            # ----------------------------------------------------

            if coefficient == '':
                term = generator_tex(op[-1])
            else:
                term = (
                    coefficient
                    + r'\otimes '
                    + generator_tex(op[-1])
                )

            # ----------------------------------------------------
            # Source and target idempotents.
            # ----------------------------------------------------

            source_idempotents = [
                idempotent_tex(i)
                for i in module.idempotence[op[0]]
            ]

            target_idempotents = [
                idempotent_tex(i)
                for i in module.idempotence[op[-1]]
            ]

            if number_of_boundaries == 1:
                source_text = source_idempotents[0]
                target_text = target_idempotents[0]
            else:
                source_text = (
                    r'\left('
                    + ','.join(source_idempotents)
                    + r'\right)'
                )

                target_text = (
                    r'\left('
                    + ','.join(target_idempotents)
                    + r'\right)'
                )

            # ----------------------------------------------------
            # Arrow between idempotents.
            # ----------------------------------------------------

            if all_empty:
                if W == '':
                    idem_arrow = (
                        source_text
                        + r'\Longrightarrow '
                        + target_text
                    )
                else:
                    idem_arrow = (
                        source_text
                        + r'\overset{'
                        + W
                        + r'}{\Longrightarrow}'
                        + target_text
                    )
            else:
                idem_arrow = (
                    source_text
                    + r'\xrightarrow{'
                    + coefficient
                    + r'}'
                    + target_text
                )

            idem_map = (
                r'\qquad\left['
                + idem_arrow
                + r'\right]'
            )

            # ----------------------------------------------------
            # Assemble this line of the differential.
            # ----------------------------------------------------

            if j == 0:
                operation_formulas.append(
                    r'\delta('
                    + generator_tex(g)
                    + r')&='
                    + term
                    + idem_map
                )
            else:
                operation_formulas.append(
                    r'&\quad +'
                    + term
                    + idem_map
                )

    # ============================================================
    # Assemble LaTeX.
    # ============================================================

    generators_latex = (
        r'\begin{aligned}'
        + r'\\'.join(generator_formulas)
        + r'\end{aligned}'
    )

    operations_latex = (
        r'\begin{aligned}'
        + r'\\'.join(operation_formulas)
        + r'\end{aligned}'
    )

    # ============================================================
    # Display in Jupyter.
    # ============================================================

    if show:
        from IPython.display import display, Math, Markdown

        display(Markdown('### Generators'))
        display(Math(generators_latex))

        display(Markdown('### Type D differential'))
        display(Math(operations_latex))

    return generators_latex, operations_latex
