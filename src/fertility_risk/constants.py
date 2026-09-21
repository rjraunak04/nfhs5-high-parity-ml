"""Shared feature contracts and human-readable category mappings."""

MODEL_FEATURES = [
    "current_age",
    "residence",
    "education",
    "wealth",
    "in_union",
]

PRIMARY_INDIA_FEATURES = [
    "current_age",
    "residence",
    "education",
    "wealth",
    "religion",
    "caste",
    "state",
]

RAW_TO_CLEAN = {
    "v012": "current_age",
    "v025": "residence",
    "v106": "education",
    "v130": "religion",
    "v190": "wealth",
    "v024": "state",
    "s116": "caste",
    "v501": "marital_status",
    "v201": "children_ever_born",
    "v005": "sample_weight_raw",
    "v021": "psu",
    "v022": "stratum",
}

BLOCKED_OUTCOME_PROXIMAL = {
    "v201",
    "children_ever_born",
    "fertility_class",
    "v212",
    "v511",
    "v509",
    "v312",
    "v313",
    "v602",
    "v613",
    "v621",
}

RESIDENCE_LABELS = {1: "Urban", 2: "Rural"}
EDUCATION_LABELS = {0: "No education", 1: "Primary", 2: "Secondary", 3: "Higher"}
WEALTH_LABELS = {1: "Poorest", 2: "Poorer", 3: "Middle", 4: "Richer", 5: "Richest"}
UNION_LABELS = {0: "Not currently in union", 1: "Currently in union"}
