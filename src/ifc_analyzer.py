from __future__ import annotations

import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import ifcopenshell
except ImportError:  # pragma: no cover - handled at runtime in the Streamlit app
    ifcopenshell = None


MAIN_TYPES = ["IfcWall", "IfcSlab", "IfcDoor", "IfcWindow", "IfcBeam", "IfcColumn"]


@dataclass
class IFCAnalysis:
    project_name: str | None
    total_entities: int
    class_counts: pd.DataFrame
    storeys: pd.DataFrame
    missing_name: pd.DataFrame
    missing_material: pd.DataFrame
    missing_psets: pd.DataFrame
    missing_quantities: pd.DataFrame
    main_types: pd.DataFrame
    completeness_score: float
    completeness_label: str
    preliminary_conclusion: str

    def to_serializable(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "total_entities": self.total_entities,
            "class_counts": self.class_counts.to_dict(orient="records"),
            "storeys": self.storeys.to_dict(orient="records"),
            "missing_name": self.missing_name.to_dict(orient="records"),
            "missing_material": self.missing_material.to_dict(orient="records"),
            "missing_psets": self.missing_psets.to_dict(orient="records"),
            "missing_quantities": self.missing_quantities.to_dict(orient="records"),
            "main_types": self.main_types.to_dict(orient="records"),
            "completeness_score": self.completeness_score,
            "completeness_label": self.completeness_label,
            "preliminary_conclusion": self.preliminary_conclusion,
        }


def _require_ifcopenshell() -> None:
    if ifcopenshell is None:
        raise ImportError(
            "ifcopenshell nao esta instalado. Instale as dependencias antes de rodar a aplicacao."
        )


def _open_ifc_from_bytes(content: bytes, file_name: str | None = None):
    _require_ifcopenshell()
    suffix = Path(file_name or "model.ifc").suffix or ".ifc"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        return ifcopenshell.open(temp_path)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _entity_name(entity: Any) -> str:
    return _safe_text(getattr(entity, "Name", ""))


def _entity_global_id(entity: Any) -> str:
    return _safe_text(getattr(entity, "GlobalId", "")) or str(getattr(entity, "id", lambda: "")())


def _element_label(entity: Any) -> str:
    return _entity_name(entity) or _entity_global_id(entity) or "Sem identificacao"


def _has_material(entity: Any) -> bool:
    for rel in getattr(entity, "HasAssociations", []) or []:
        if rel.is_a("IfcRelAssociatesMaterial"):
            return True
    return False


def _has_property_set(entity: Any) -> bool:
    for rel in getattr(entity, "IsDefinedBy", []) or []:
        if rel.is_a("IfcRelDefinesByProperties"):
            definition = getattr(rel, "RelatingPropertyDefinition", None)
            if definition is not None and definition.is_a("IfcPropertySet"):
                return True
    return False


def _has_quantity_set(entity: Any) -> bool:
    for rel in getattr(entity, "IsDefinedBy", []) or []:
        if rel.is_a("IfcRelDefinesByProperties"):
            definition = getattr(rel, "RelatingPropertyDefinition", None)
            if definition is not None and definition.is_a("IfcElementQuantity"):
                return True
    return False


def _storey_element_count(storey: Any) -> int:
    total = 0
    for rel in getattr(storey, "ContainsElements", []) or []:
        if rel.is_a("IfcRelContainedInSpatialStructure"):
            total += len(getattr(rel, "RelatedElements", []) or [])
    return total


def _as_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _top_class_frame(class_counts: Counter[str]) -> pd.DataFrame:
    rows = [
        {"Classe IFC": class_name, "Quantidade": quantity}
        for class_name, quantity in class_counts.most_common()
    ]
    return _as_dataframe(rows)


def _missing_entities_frame(entities: list[Any]) -> pd.DataFrame:
    rows = []
    for entity in entities:
        rows.append(
            {
                "Expressao": _element_label(entity),
                "Classe IFC": entity.is_a(),
                "GlobalId": _entity_global_id(entity),
            }
        )
    return _as_dataframe(rows)


def _main_types_frame(class_counts: Counter[str]) -> pd.DataFrame:
    rows = [{"Classe IFC": class_name, "Quantidade": int(class_counts.get(class_name, 0))} for class_name in MAIN_TYPES]
    return _as_dataframe(rows)


def _completeness_score(
    total_entities: int,
    missing_name: int,
    missing_material: int,
    missing_psets: int,
    missing_quantities: int,
) -> tuple[float, str, str]:
    if total_entities <= 0:
        return 0.0, "Indeterminado", "Modelo IFC sem entidades para auditoria."

    penalties = (
        (missing_name / total_entities) * 25.0
        + (missing_material / total_entities) * 25.0
        + (missing_psets / total_entities) * 25.0
        + (missing_quantities / total_entities) * 25.0
    )
    score = max(0.0, 100.0 - penalties)

    if score >= 90:
        label = "Muito alta"
        conclusion = "O modelo apresenta boa completude informacional e poucos vazios estruturais."
    elif score >= 75:
        label = "Boa"
        conclusion = "O modelo tem base consistente, mas ainda mostra lacunas informacionais relevantes."
    elif score >= 50:
        label = "Moderada"
        conclusion = "O modelo possui inconsistencias importantes e merece revisao antes de uso analitico."
    else:
        label = "Baixa"
        conclusion = "O modelo apresenta alta incompletude informacional e risco para uso em analises downstream."

    return round(score, 1), label, conclusion


def analyze_ifc_file(content: bytes, file_name: str | None = None) -> IFCAnalysis:
    model = _open_ifc_from_bytes(content, file_name=file_name)

    entities = list(model)
    total_entities = len(entities)
    class_counts = Counter(entity.is_a() for entity in entities)

    project = next(iter(model.by_type("IfcProject")), None)
    project_name = None
    if project is not None:
        project_name = _safe_text(getattr(project, "Name", "")) or _safe_text(
            getattr(project, "LongName", "")
        )

    elements = list(model.by_type("IfcElement"))

    missing_name_entities = [entity for entity in elements if not _entity_name(entity)]
    missing_material_entities = [entity for entity in elements if not _has_material(entity)]
    missing_pset_entities = [entity for entity in elements if not _has_property_set(entity)]
    missing_quantity_entities = [entity for entity in elements if not _has_quantity_set(entity)]

    storeys_rows = []
    for storey in model.by_type("IfcBuildingStorey"):
        storeys_rows.append(
            {
                "Nome": _safe_text(getattr(storey, "Name", "")) or "Sem nome",
                "Elevation": getattr(storey, "Elevation", None),
                "Elementos contidos": _storey_element_count(storey),
            }
        )

    storeys_df = _as_dataframe(storeys_rows)
    if not storeys_df.empty and "Elevation" in storeys_df.columns:
        storeys_df = storeys_df.sort_values(
            by="Elevation",
            na_position="last",
            ignore_index=True,
        )

    missing_name_df = _missing_entities_frame(missing_name_entities)
    missing_material_df = _missing_entities_frame(missing_material_entities)
    missing_psets_df = _missing_entities_frame(missing_pset_entities)
    missing_quantities_df = _missing_entities_frame(missing_quantity_entities)

    class_counts_df = _top_class_frame(class_counts)
    main_types_df = _main_types_frame(class_counts)

    completeness_score, completeness_label, preliminary_conclusion = _completeness_score(
        total_entities=len(elements),
        missing_name=len(missing_name_entities),
        missing_material=len(missing_material_entities),
        missing_psets=len(missing_pset_entities),
        missing_quantities=len(missing_quantity_entities),
    )

    return IFCAnalysis(
        project_name=project_name,
        total_entities=total_entities,
        class_counts=class_counts_df,
        storeys=storeys_df,
        missing_name=missing_name_df,
        missing_material=missing_material_df,
        missing_psets=missing_psets_df,
        missing_quantities=missing_quantities_df,
        main_types=main_types_df,
        completeness_score=completeness_score,
        completeness_label=completeness_label,
        preliminary_conclusion=preliminary_conclusion,
    )


def format_analysis_for_display(analysis: IFCAnalysis) -> dict[str, Any]:
    return {
        "project_name": analysis.project_name,
        "total_entities": analysis.total_entities,
        "class_counts": analysis.class_counts,
        "storeys": analysis.storeys,
        "missing_name": analysis.missing_name,
        "missing_material": analysis.missing_material,
        "missing_psets": analysis.missing_psets,
        "missing_quantities": analysis.missing_quantities,
        "main_types": analysis.main_types,
        "missing_name_count": len(analysis.missing_name),
        "missing_material_count": len(analysis.missing_material),
        "missing_pset_count": len(analysis.missing_psets),
        "missing_quantity_count": len(analysis.missing_quantities),
        "completeness_score": analysis.completeness_score,
        "completeness_label": analysis.completeness_label,
        "preliminary_conclusion": analysis.preliminary_conclusion,
    }
