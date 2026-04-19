import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .resources import data_path
from .results import ContactAnalysis, ProdigyResults
from .utils.residue_classification import ResidueClassification

LOGGER = logging.getLogger(__name__)
_RELATIVE_SASA = {residue: asa.total for residue, asa in ResidueClassification().rel_asa.items()}


def _load_prodigy_modules():
    try:
        from prodigy_prot.modules.models import IC_NIS
        from prodigy_prot.modules.parsers import parse_structure
        from prodigy_prot.modules.prodigy import Prodigy, analyse_contacts, analyse_nis, calculate_ic
    except ImportError as exc:  # pragma: no cover - exercised by integration environments
        raise ImportError("CPU prediction requires the optional 'prodigy-prot' dependency.") from exc

    return Prodigy, analyse_contacts, analyse_nis, calculate_ic, IC_NIS, parse_structure


def execute_freesasa(structure, sphere_points: int | None = None):
    """Compute SASA using freesasa and return atom- and residue-level values."""
    try:
        import freesasa
        from freesasa import Classifier, calc, structureFromBioPDB
    except ImportError as exc:  # pragma: no cover - exercised by integration environments
        raise ImportError("CPU prediction requires the optional 'freesasa' dependency.") from exc

    if sphere_points:
        parameters = freesasa.Parameters({"algorithm": freesasa.ShrakeRupley, "n-points": sphere_points})
    else:
        parameters = freesasa.Parameters({"algorithm": freesasa.LeeRichards, "n-slices": 20})

    with data_path("naccess.config") as config_path:
        classifier = Classifier(str(config_path))
        struct = structureFromBioPDB(structure, classifier)
        result = calc(struct, parameters)

    atom_sasa, residue_sasa, absolute_difference = {}, {}, {}
    for atom_index in range(struct.nAtoms()):
        atom_name = struct.atomName(atom_index)
        residue_name = struct.residueName(atom_index)
        residue_number = struct.residueNumber(atom_index)
        chain_label = struct.chainLabel(atom_index)
        atom_key = (chain_label, residue_name, residue_number, atom_name)
        residue_key = (chain_label, residue_name, residue_number)

        area = result.atomArea(atom_index)
        atom_sasa[atom_key] = area
        residue_sasa[residue_key] = residue_sasa.get(residue_key, 0.0) + area
        absolute_difference[residue_key] = absolute_difference.get(residue_key, 0.0) + area

    residue_sasa.update(
        (residue_key, sasa / _RELATIVE_SASA[residue_key[1]]) for residue_key, sasa in residue_sasa.items()
    )
    absolute_difference.update(
        (residue_key, abs(sasa - _RELATIVE_SASA[residue_key[1]]))
        for residue_key, sasa in absolute_difference.items()
    )
    return atom_sasa, residue_sasa, absolute_difference


def _run_prodigy_prediction(
    structure,
    selection: str,
    temperature: float,
    distance_cutoff: float,
    acc_threshold: float,
    sphere_points: int,
):
    Prodigy, analyse_contacts, analyse_nis, calculate_ic, IC_NIS, _ = _load_prodigy_modules()
    selection_groups = [chain.strip() for chain in selection.split(",") if chain.strip()]
    predictor = Prodigy(structure, name=getattr(structure, "id", ""), selection=selection_groups, temp=temperature)
    freesasa_structure = structure.get_parent() if getattr(structure, "get_parent", None) else structure

    chains = selection_groups
    if len(chains) != 2:
        raise ValueError("Selection must contain exactly two chains.")

    selection_dict = {chain: idx for idx, chain in enumerate(chains)}
    predictor.temperature = temperature
    predictor.distance_cutoff = distance_cutoff
    predictor.acc_threshold = acc_threshold
    predictor.sphere_points = sphere_points
    predictor.ic_network = calculate_ic(structure, d_cutoff=distance_cutoff, selection=selection_dict)
    predictor.bins = analyse_contacts(predictor.ic_network)
    predictor.asa_data, predictor.rsa_data, predictor.abs_diff_data = execute_freesasa(
        freesasa_structure,
        sphere_points=sphere_points,
    )
    predictor.nis_a, predictor.nis_c, predictor.nis_p = analyse_nis(
        predictor.rsa_data,
        acc_threshold=acc_threshold,
    )
    predictor.ba_val = IC_NIS(
        predictor.bins["CC"],
        predictor.bins["AC"],
        predictor.bins["PP"],
        predictor.bins["AP"],
        predictor.nis_a,
        predictor.nis_c,
    )
    predictor.kd_val = np.exp(predictor.ba_val / (0.0019858775 * (temperature + 273.15)))
    return predictor


def predict_binding_affinity(
    struct_path: str | Path,
    selection: Optional[str] = None,
    temperature: float = 25.0,
    distance_cutoff: float = 5.5,
    acc_threshold: float = 0.05,
    sphere_points: int = 100,
    save_results: bool = False,
    output_dir: str | Path = ".",
    quiet: bool = True,
) -> ProdigyResults:
    """Predict binding affinity using the CPU PRODIGY path."""
    _, _, _, _, _, parse_structure = _load_prodigy_modules()
    models, n_chains, n_residues = parse_structure(str(struct_path))
    if not models:
        raise ValueError(f"No models parsed from structure: {struct_path}")

    model = models[0]
    LOGGER.info("Parsed %s with %s chains and %s residues", model.id, n_chains, n_residues)

    prediction = _run_prodigy_prediction(
        structure=model,
        selection=selection or "A,B",
        temperature=temperature,
        distance_cutoff=distance_cutoff,
        acc_threshold=acc_threshold,
        sphere_points=sphere_points,
    )

    results = ProdigyResults(
        structure_id=Path(struct_path).stem,
        contact_types=ContactAnalysis(
            [
                prediction.bins["AA"],
                prediction.bins["CC"],
                prediction.bins["PP"],
                prediction.bins["AC"],
                prediction.bins["AP"],
                prediction.bins["CP"],
            ]
        ),
        binding_affinity=np.float32(prediction.ba_val),
        dissociation_constant=np.float32(prediction.kd_val),
        nis_aliphatic=np.float32(prediction.nis_a),
        nis_charged=np.float32(prediction.nis_c),
        nis_polar=np.float32(prediction.nis_p),
        sasa_data=np.array(
            [
                {
                    "chain": chain,
                    "resname": residue_name,
                    "resindex": int(residue_number),
                    "atomname": atom_name,
                    "atom_sasa": area,
                    "relative_sasa": prediction.rsa_data.get((chain, residue_name, residue_number), 0.0),
                }
                for (chain, residue_name, residue_number, atom_name), area in prediction.asa_data.items()
            ]
        ),
    )

    if save_results:
        results.save_results(output_dir)
    if not quiet:
        LOGGER.info("CPU prediction complete for %s", results.structure_id)
    return results
