"""Shared paths and the source-filename → slug map for the dataset pipeline.

Every stage imports its directories from here so the on-disk layout is defined
in exactly one place.
"""
from __future__ import annotations

from pathlib import Path

# config.py lives at src/build_dataset/config.py, so ROOT is three levels up.
ROOT = Path(__file__).resolve().parent.parent.parent
RAW_HOSPITAL = ROOT / "data" / "raw" / "hospital"
CLEAN_DIR = ROOT / "data" / "clean"
INTERIM_DIR = ROOT / "data" / "interim"
QA_DIR = ROOT / "data" / "qa"

# Explicit source-filename -> clean-file slug map (stable, human-readable).
SLUG_MAP = {
    "FAQ- Appointments and channelling.pdf": "faq-appointments-and-channelling",
    "FAQ - Inpatient Admission.pdf": "faq-inpatient-admission",
    "FAQ - Insurance and Payments.pdf": "faq-insurance-and-payments",
    "FAQ - Laboratory Tests and Results.pdf": "faq-laboratory-tests-and-results",
    "FAQ - Surgery and Pre Operative Care.pdf": "faq-surgery-and-preoperative-care",
    "Accident, Emergency and Casualty Services - Protocol Reference.pdf": "emergency-casualty-protocol",
    "Inpatient Admission - Patient and Family Handbook.pdf": "inpatient-admission-handbook",
    "Insurance, Payments and Billing - Policy Document.pdf": "insurance-payments-billing-policy",
    "Laboratory and Diagnostic Services - Service Catalog.pdf": "laboratory-diagnostic-services-catalog",
    "Outpatient Department (OPD) - Services Directory.pdf": "opd-services-directory",
    "Pharmacy Services - Information Sheet.pdf": "pharmacy-services-info",
    "Serendib General Hospital - Document Library.pdf": "document-library",
    "Surgical Procedures and Estimated Pricing.pdf": "surgical-procedures-pricing",
}
