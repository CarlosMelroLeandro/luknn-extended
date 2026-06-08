"""
Translate a crystallized ŁNN into a symbolic Łukasiewicz formula.

Process (§4 of Leandro ALT 2009):
  1.  For each neuron in each layer, classify the configuration (Prop. 3).
  2.  If representable → write the corresponding formula.
  3.  If un-representable → use best λ-approximation from rule R.
  4.  Substitute layer-by-layer from inputs to output.
"""

from dataclasses import dataclass

import torch

from ..network.crystallization import crisp_crystallize_weights
from ..network.luknn import LukNN
from .classifier import NeuronKind, classify_neuron
from .approximation import best_approximation


@dataclass
class ExtractionResult:
    formula: str
    representable: bool          # True if every neuron was representable
    layer_formulas: list[list[str]]


def extract_formula(model: LukNN, input_names: list[str] | None = None,
                    n_values: int = 4) -> ExtractionResult:
    """
    Extract a symbolic formula from a trained (and crystallized) ŁNN.

    input_names : variable names like ['x1','x2',…].  Defaults to x1, x2, …
    n_values    : truth-table resolution used for λ-similarity (default 4).
    """
    layers = model.weight_matrix_repr()   # [(W, b), …]
    n_inputs = layers[0][0].shape[1]

    if input_names is None:
        input_names = [f"x{i+1}" for i in range(n_inputs)]

    # Symbols available at each stage: start with input variable names
    symbols: list[str] = list(input_names)
    all_layer_formulas: list[list[str]] = []
    fully_representable = True

    for W, b in layers:
        n_out, _ = W.shape
        W_crisp = crisp_crystallize_weights(W)
        b_crisp = b.round().int()
        layer_syms: list[str] = []

        for j in range(n_out):
            row_w = W_crisp[j]
            row_b = b_crisp[j]

            # Build the formula for neuron j using current symbols
            cfg = classify_neuron(row_w, row_b)
            nonzero = [(symbols[i], int(row_w[i].item()))
                       for i in range(len(symbols)) if row_w[i] != 0]

            if cfg.kind == NeuronKind.CONJUNCTION:
                terms = [f"¬{s}" if wi < 0 else s for s, wi in nonzero]
                sym = " ⊗ ".join(terms) if terms else "1"

            elif cfg.kind == NeuronKind.DISJUNCTION:
                terms = [f"¬{s}" if wi < 0 else s for s, wi in nonzero]
                sym = " ⊕ ".join(terms) if terms else "0"

            elif cfg.kind in (NeuronKind.CONSTANT_ZERO, NeuronKind.CONSTANT_ONE):
                sym = "1" if cfg.kind == NeuronKind.CONSTANT_ONE else "0"

            else:
                # Un-representable: use best approximation
                fully_representable = False
                w_list = row_w.int().tolist()
                b_val = int(row_b.item())
                approx = best_approximation(w_list, b_val, n_values)
                sym = approx.formula if approx else f"ψ_{b_val}({w_list})"

            layer_syms.append(sym)

        all_layer_formulas.append(layer_syms)
        symbols = layer_syms   # outputs become inputs for next layer

    # The final layer should have exactly one neuron
    final_formula = symbols[0] if symbols else "?"

    return ExtractionResult(
        formula=final_formula,
        representable=fully_representable,
        layer_formulas=all_layer_formulas,
    )
