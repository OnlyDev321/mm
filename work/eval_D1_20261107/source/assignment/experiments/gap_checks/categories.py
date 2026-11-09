"""MMMU discipline (category) -> subject mapping used by the gap_checks scripts."""
CATEGORIES = {
    "Art & Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": ["Biology", "Chemistry", "Geography", "Math", "Physics"],
    "Health & Medicine": ["Basic_Medical_Science", "Clinical_Medicine",
                          "Diagnostics_and_Laboratory_Medicine", "Pharmacy", "Public_Health"],
    "Humanities & Social Science": ["History", "Literature", "Psychology", "Sociology"],
    "Tech & Engineering": ["Agriculture", "Architecture_and_Engineering", "Computer_Science",
                           "Electronics", "Energy_and_Power", "Materials", "Mechanical_Engineering"],
}
assert sum(len(v) for v in CATEGORIES.values()) == 30
SUBJECT_TO_CATEGORY = {s: c for c, subs in CATEGORIES.items() for s in subs}
