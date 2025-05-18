import json
import itertools
import sys

def reconstruct_all_formulas(data, latex: bool = True, threshold: float = 0.9):
    # 1) Build candidate lists and record x‐positions
    sym_cands = {}
    x_pos     = {}
    for s in data["symbols"]:
        idx = s["index"]
        # sort candidates by confidence descending
        sorted_syms = sorted(
            s["possible_symbols_confidence"].items(),
            key=lambda kv: kv[1],
            reverse=True
        )
        best_sym, best_conf = sorted_syms[0]
        if best_conf > threshold:
            # high‐confidence: only keep the top
            sym_cands[idx] = [best_sym]
        else:
            # low‐confidence: keep _all_ possibilities
            sym_cands[idx] = [sym for sym,_ in sorted_syms]
        x_pos[idx] = s["bounding_box"]["cx"]

    # 2) Build exponent & fraction maps
    exp_map  = {}
    for e in data.get("exponents", []):
        exp_map.setdefault(e["base_idx"], []).append(e["exp_idx"])

    frac_map = {}
    for f in data.get("fractions", []):
        frac_map[f["frac_index"]] = (f["numerator"], f["denominator"])

    # 3) Decide which indices to skip when emitting 'plain' symbols
    skip = set()
    # skip exponent symbols themselves
    for exps in exp_map.values():
        skip.update(exps)
    # skip fraction numerator & denominator symbols, but NOT the fraction key
    for nums, dens in frac_map.values():
        skip.update(nums)
        skip.update(dens)

    # 4) Prepare for Cartesian product over ambiguous symbols
    all_indices = sorted(sym_cands.keys(), key=lambda i: x_pos[i])
    all_lists   = [sym_cands[idx] for idx in all_indices]
    formulas    = []

    # 5) Generate every assignment
    for assignment in itertools.product(*all_lists):
        chosen = dict(zip(all_indices, assignment))
        tokens = []

        for idx in all_indices:
            # FRACTION? (do this first so we don't skip the fraction root)
            if idx in frac_map:
                nums, dens = frac_map[idx]
                num_str = "".join(chosen[i] for i in sorted(nums, key=lambda i: x_pos[i]))
                den_str = "".join(chosen[i] for i in sorted(dens, key=lambda i: x_pos[i]))
                if latex:
                    tokens.append(f"\\frac{{{num_str}}}{{{den_str}}}")
                else:
                    tokens.append(f"({num_str})/({den_str})")

            # EXPONENT?
            elif idx in exp_map:
                base = chosen[idx]
                exps = sorted(exp_map[idx], key=lambda i: x_pos[i])
                exp_str = "".join(chosen[i] for i in exps)
                if latex:
                    tokens.append(f"{base}^{{{exp_str}}}")
                else:
                    tokens.append(f"{base}^{exp_str}")

            # SKIP sub‐symbol?
            elif idx in skip:
                continue

            # PLAIN SYMBOL
            else:
                tokens.append(chosen[idx])

        formulas.append(" ".join(tokens))

    # remove duplicates and return
    return sorted(set(formulas))


if __name__ == "__main__":
    with open("output.json") as f:
        data = json.load(f)

    all_exprs = reconstruct_all_formulas(data, latex=False, threshold=0.9)
    for expr in all_exprs:
        print(expr)