"""
Łukasiewicz formula extraction for LukResidualNet.

Strategy: symbolic walk layer by layer.
  - Linear layers: same as the standard extractor.
  - Residual blocks: save the input symbols, process the inner layers,
    then merge with the skip connection using the crystallized bias.

Merge after crystallization:
  b_j = 0  → sym_j = F(x)_j ⊕ x_j
  b_j = -1 → sym_j = F(x)_j ⊗ x_j
  other    → sym_j = ψ_{b_j}(F(x)_j, x_j)  [non-representable]
"""

from dataclasses import dataclass

from ..extraction.classifier import NeuronKind, classify_neuron
from ..extraction.approximation import best_approximation
from ..network.crystallization import crisp_crystallize_weights
from ..layers.lukasiewicz_linear import LukasiewiczLinear
from ..layers.residual import LukResidualBlock


@dataclass
class ResidualExtractionResult:
    formula: str
    representable: bool
    layer_formulas: list[list[str]]   # per layer, including merge


# ── Internal helpers ──────────────────────────────────────────────────────────

def _walk_linear(
    layer: LukasiewiczLinear,
    symbols: list[str],
    n_values: int = 4,
) -> tuple[list[str], bool]:
    """Return (new_symbols, fully_representable)."""
    W = crisp_crystallize_weights(layer.weight.data)
    b = layer.bias.data.round().int()
    n_out, n_in = W.shape
    out_syms: list[str] = []
    all_rep = True

    for j in range(n_out):
        row_w = W[j]
        row_b = b[j]
        cfg = classify_neuron(row_w, row_b)
        nonzero = [
            (symbols[i], int(row_w[i].item()))
            for i in range(min(n_in, len(symbols)))
            if row_w[i] != 0
        ]

        if cfg.kind == NeuronKind.CONJUNCTION:
            terms = [f"¬{s}" if wi < 0 else s for s, wi in nonzero]
            sym = " ⊗ ".join(terms) if terms else "1"
        elif cfg.kind == NeuronKind.DISJUNCTION:
            terms = [f"¬{s}" if wi < 0 else s for s, wi in nonzero]
            sym = " ⊕ ".join(terms) if terms else "0"
        elif cfg.kind == NeuronKind.CONSTANT_ONE:
            sym = "1"
        elif cfg.kind == NeuronKind.CONSTANT_ZERO:
            sym = "0"
        else:
            all_rep = False
            w_list = row_w.int().tolist()
            b_val = int(row_b.item())
            approx = best_approximation(w_list, b_val, n_values)
            sym = approx.formula if approx else f"ψ_{b_val}({w_list})"

        out_syms.append(sym)

    return out_syms, all_rep


def _walk_block(
    block: LukResidualBlock,
    symbols: list[str],
    n_values: int = 4,
) -> tuple[list[str], bool, list[str]]:
    """
    Process a LukResidualBlock.
    Return (merged_symbols, all_representable, merge_symbols_for_log).
    """
    skip_symbols = list(symbols)
    cur_syms = list(symbols)
    all_rep = True

    for layer in block.inner_layers:
        cur_syms, rep = _walk_linear(layer, cur_syms, n_values)
        all_rep = all_rep and rep

    merge_syms: list[str] = []
    for j in range(block.width):
        b_j = int(block.merge_bias[j].round().item())
        fx = cur_syms[j]
        xj = skip_symbols[j]

        if b_j == 0:
            sym = f"({fx} ⊕ {xj})"
        elif b_j == -1:
            sym = f"({fx} ⊗ {xj})"
        else:
            all_rep = False
            sym = f"ψ_{b_j}({fx}, {xj})"

        merge_syms.append(sym)

    return merge_syms, all_rep, merge_syms


# ── Public API ───────────────────────────────────────────────────────────────

def extract_formula_residual(
    model,
    input_names: list[str] | None = None,
    n_values: int = 4,
) -> ResidualExtractionResult:
    """
    Extract the symbolic formula from a crystallized LukResidualNet.

    Parameters
    ----------
    model       : LukResidualNet
    input_names : names of the input variables (e.g. ['x1','x2',…])
    n_values    : truth-table resolution for λ-similarity
    """
    n_inputs = model.n_inputs
    if input_names is None:
        input_names = [f"x{i+1}" for i in range(n_inputs)]

    symbols: list[str] = list(input_names)
    fully_rep = True
    all_layer_formulas: list[list[str]] = []

    # Projection layer (optional)
    if model.proj is not None and isinstance(model.proj, LukasiewiczLinear):
        symbols, rep = _walk_linear(model.proj, symbols, n_values)
        all_layer_formulas.append(list(symbols))
        fully_rep = fully_rep and rep

    # Residual blocks
    for block in model.blocks:
        symbols, rep, merge_log = _walk_block(block, symbols, n_values)
        all_layer_formulas.append(list(symbols))
        fully_rep = fully_rep and rep

    # Output layer
    symbols, rep = _walk_linear(model.output_layer, symbols, n_values)
    all_layer_formulas.append(list(symbols))
    fully_rep = fully_rep and rep

    return ResidualExtractionResult(
        formula=symbols[0] if symbols else "?",
        representable=fully_rep,
        layer_formulas=all_layer_formulas,
    )
