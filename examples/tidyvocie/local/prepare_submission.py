#!/usr/bin/env python3
"""
Submission File Preparation Tool for TidyVoice 2026 challenge

This script processes score files and generates submission files in the correct format.
It supports multiple input formats and validates the data before creating the submission.

--------------------------------------------------------------------------------
EXPECTED FORMAT OF EACH INPUT FILE
--------------------------------------------------------------------------------

1) REFERENCE FILES (trial pair lists - provided by the challenge):
   - TV26_EVAL_A_REF_FILE (e.g. tv26_eval-A.txt)
   - TV26_EVAL_U_REF_FILE (e.g. tv26_eval-U.txt)

   Format: 2 columns per line, space- or tab-separated, NO header.
   Column 1: enrollment filename
   Column 2: test filename

   Example:
     j8pvmcu6iuh v7l9r79188y
     o0wrjr1rqur vgbprjwts7v
     nsv25uueewd axs1fn4f1jr
     drmmy1q8bp1 il2xf2tvhql

   - tv26_eval-A.txt must have exactly 4,000,000 lines (trials).
   - tv26_eval-U.txt must have exactly 1,280,000 lines (trials).

--------------------------------------------------------------------------------

2) SCORE FILES (your system outputs - you provide these):
   - TV26_EVAL_A_SCORE_FILE (e.g. tv26_eval-A_score.txt)
   - TV26_EVAL_U_SCORE_FILE (e.g. tv26_eval-U_score.txt)

   Format: 3 columns per line, space- or tab-separated, NO header.
   Either:
     Column 1: enrollment filename   Column 2: test filename   Column 3: score (float)
     
   or (also accepted):
     Column 1: score (float)   Column 2: enrollment filename   Column 3: test filename

   Example (enrollment, test, score order):
     j8pvmcu6iuh v7l9r79188y 0.0123
     o0wrjr1rqur vgbprjwts7v 0.2345
     nsv25uueewd axs1fn4f1jr 0.3456
     drmmy1q8bp1 il2xf2tvhql 0.4567


   - Same trial pairs as in the corresponding reference file, one line per trial.
   - Score: floating-point number (higher = more likely same speaker).
   - Filenames must match the reference file (with or without .wav); order can differ,
     the script matches by (enroll, test) and writes scores in the reference order.
--------------------------------------------------------------------------------
"""

import sys
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List, Optional

# ============================================================================
# FILE PATHS CONFIGURATION - Update these paths as needed
# ============================================================================

# tv26_eval_A files
TV26_EVAL_A_SCORE_FILE = Path("/media/rf/T9/Tidyvoice2/final_evalSet/tv26_eval-A_score.txt")
TV26_EVAL_A_REF_FILE = Path("/media/rf/T9/Tidyvoice2/final_evalSet/tv26_eval-A.txt")

# tv26_eval_U files
TV26_EVAL_U_SCORE_FILE = Path("/media/rf/T9/Tidyvoice2/final_evalSet/tv26_eval-U_score.txt")
TV26_EVAL_U_REF_FILE = Path("/media/rf/T9/Tidyvoice2/final_evalSet/tv26_eval-U.txt")

# Output directory (where submission files will be created)
OUTPUT_DIR = Path.cwd()  # Change this to a specific directory if needed, e.g., Path("/path/to/output")

# ============================================================================
# ============================================================================

# Expected row counts (without headers)
TV26_EVAL_U_EXPECTED_ROWS = 1280000
TV26_EVAL_A_EXPECTED_ROWS = 4000000


def detect_separator(line: str) -> str:
    """Detect if the line uses space or tab separator."""
    if '\t' in line:
        return '\t'
    elif ' ' in line:
        # Count spaces - if there are multiple spaces, it's likely space-separated
        parts = line.strip().split()
        if len(parts) >= 2:
            return ' '  # Space-separated (will split on any whitespace)
    return None


def parse_line(line: str, separator: Optional[str] = None) -> List[str]:
    """Parse a line with the given separator."""
    line = line.strip()
    if not line:
        return []
    
    if separator == '\t':
        return line.split('\t')
    elif separator == ' ':
        # Split on any whitespace
        return line.split()
    else:
        # Try to detect
        if '\t' in line:
            return line.split('\t')
        else:
            return line.split()


def detect_format(parts: List[str]) -> str:
    """
    Detect the format of the score file.
    Returns: 'enroll_test_score' or 'score_enroll_test'
    """
    if len(parts) < 3:
        raise ValueError(f"Expected 3 columns, got {len(parts)}")
    
    # Check if first column is numeric (score)
    try:
        float(parts[0])
        return 'score_enroll_test'
    except ValueError:
        # Check if last column is numeric (score)
        try:
            float(parts[-1])
            return 'enroll_test_score'
        except ValueError:
            raise ValueError("Cannot detect format: score column not found")


def is_numeric(value: str) -> bool:
    """Check if a string represents a valid number."""
    try:
        float(value)
        return True
    except ValueError:
        return False


def normalize_filename(filename: str) -> str:
    """
    Normalize filename by removing .wav extension if present.
    This allows matching to work whether filenames have .wav extension or not.
    """
    if filename.lower().endswith('.wav'):
        return filename[:-4]  # Remove .wav extension
    return filename


def is_header_line(parts: List[str]) -> bool:
    """
    Detect if a line is a header by checking for common header patterns.
    Returns True if the line appears to be a header.
    Uses conservative detection - only flags clear header patterns.
    """
    if len(parts) < 2:
        return False
    
    # Common header keywords (case-insensitive) - must match exactly or be part of common header phrases
    header_keywords = [
        'enroll', 'enrollment', 'test', 'utterance', 'pair', 'id',
        'filename', 'file', 'speaker', 'segment', 'trial', 'score',
        'similarity', 'distance', 'label'
    ]
    
    parts_lower = [p.lower().strip() for p in parts]
    
    # Check for exact matches or common header patterns
    for part in parts_lower:
        # Check if part is exactly a header keyword
        if part in header_keywords:
            return True
        
        # Check if part starts with header keywords followed by common separators
        for keyword in header_keywords:
            if part.startswith(keyword + ' ') or part.startswith(keyword + '\t') or part == keyword:
                return True
    

    if len(parts) >= 2:
        part0_lower = parts_lower[0]
        part1_lower = parts_lower[1]
        
        # Only flag if both parts have multiple words AND contain header-like terms
        has_multiple_words_0 = ' ' in part0_lower or len(part0_lower.split()) > 1
        has_multiple_words_1 = ' ' in part1_lower or len(part1_lower.split()) > 1
        
        if has_multiple_words_0 and has_multiple_words_1:
            # Both have multiple words - check if they contain header keywords
            for keyword in header_keywords:
                if keyword in part0_lower or keyword in part1_lower:
                    return True
    
    return False


def load_reference_file(ref_path: Path, expected_rows: int = None) -> List[Tuple[str, str]]:
    """
    Load reference file and return list of (enroll, test) tuples.
    Filenames are normalized (removes .wav extension if present).
    Automatically detects and skips headers (only checks first line).
    
    Args:
        ref_path: Path to reference file
        expected_rows: Expected number of data rows (excluding headers). If provided, validates row count.
    
    Returns:
        List of (enroll, test) tuples
    """
    pairs = []
    header_skipped = False
    first_line_checked = False
    
    with open(ref_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = parse_line(line)
            if len(parts) < 2:
                raise ValueError(f"Reference file {ref_path} line {line_num}: expected 2 columns, got {len(parts)}")
            
            if not first_line_checked:
                first_line_checked = True
                if is_header_line(parts):
                    print(f"  Detected header at line {line_num}, skipping: {line[:80]}")
                    header_skipped = True
                    continue
            
            enroll = normalize_filename(parts[0])
            test = normalize_filename(parts[1])
            pairs.append((enroll, test))
    
    if expected_rows is not None:
        if len(pairs) != expected_rows:
            raise ValueError(
                f"Reference file {ref_path}: expected {expected_rows} data rows "
                f"(excluding headers), but found {len(pairs)} rows"
            )
    
    return pairs


def load_score_file(score_path: Path) -> Dict[Tuple[str, str], str]:
    """
    Load score file and return a dictionary mapping (enroll, test) -> score.
    Automatically detects format and separator. Detects and skips headers (only checks first line).
    """
    scores = {}
    separator = None
    format_type = None
    header_skipped = False
    first_line_processed = False
    
    with open(score_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # Detect separator on first non-empty line
            if separator is None:
                separator = detect_separator(line)
                if separator is None:
                    raise ValueError(f"Cannot detect separator in {score_path} at line {line_num}")
            
            parts = parse_line(line, separator)
            
            if len(parts) < 3:
                raise ValueError(f"Score file {score_path} line {line_num}: expected 3 columns, got {len(parts)}")
            
            # Process first line - detect format and check for header
            if not first_line_processed:
                first_line_processed = True
                
                # Try to detect format
                try:
                    format_type = detect_format(parts)
                except ValueError:
                    # If format detection fails, check if it's a header
                    if is_header_line(parts[:2]) or (not is_numeric(parts[0]) and not is_numeric(parts[-1])):
                        print(f"  Detected header at line {line_num}, skipping: {line[:80]}")
                        header_skipped = True
                        continue
                    else:
                        raise
                
                # Check if score column is numeric (if not, it's likely a header)
                if format_type == 'enroll_test_score':
                    score_col = parts[2]
                else:  # score_enroll_test
                    score_col = parts[0]
                
                # If score column is not numeric, check if it's a header
                if not is_numeric(score_col):
                    if is_header_line(parts[:2]) or score_col.lower() in ['score', 'scores', 'similarity', 'distance', 'label']:
                        print(f"  Detected header at line {line_num}, skipping: {line[:80]}")
                        header_skipped = True
                        continue
                    else:
                        raise ValueError(f"Score file {score_path} line {line_num}: score column '{score_col}' is not numeric and doesn't appear to be a header")
            
            # Extract columns based on format
            if format_type == 'enroll_test_score':
                enroll, test, score = parts[0], parts[1], parts[2]
            else:  # score_enroll_test
                score, enroll, test = parts[0], parts[1], parts[2]
            
            # Normalize filenames (remove .wav extension if present)
            enroll = normalize_filename(enroll)
            test = normalize_filename(test)
            
            # Validate score is numeric
            if not is_numeric(score):
                raise ValueError(f"Score file {score_path} line {line_num}: score '{score}' is not numeric")
            
            key = (enroll, test)
            if key in scores:
                print(f"Warning: Duplicate pair ({enroll}, {test}) found at line {line_num}, using last occurrence")
            
            scores[key] = score
    
    return scores


def process_submission(
    exclusive_score_file: Path,
    exclusive_ref_file: Path,
    semi_score_file: Path,
    semi_ref_file: Path,
    output_dir: Path = None
) -> Path:
    """
    Process score files and create submission zip.
    Creates a unique folder for each run to avoid overwriting existing files.
    
    Returns:
        Path to the created zip file
    """
    if output_dir is None:
        base_output_dir = Path.cwd()
    else:
        base_output_dir = Path(output_dir)
        base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a unique folder for this run (using timestamp)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_folder_name = f"submission_{timestamp}"
    output_dir = base_output_dir / unique_folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating unique output directory: {output_dir}")
    
    print("Loading reference files...")
    exclusive_pairs = load_reference_file(exclusive_ref_file, expected_rows=TV26_EVAL_U_EXPECTED_ROWS)
    semi_pairs = load_reference_file(semi_ref_file, expected_rows=TV26_EVAL_A_EXPECTED_ROWS)
    
    print(f"  tv26_eval-U pairs: {len(exclusive_pairs)} (expected: {TV26_EVAL_U_EXPECTED_ROWS})")
    print(f"  tv26_eval-A pairs: {len(semi_pairs)} (expected: {TV26_EVAL_A_EXPECTED_ROWS})")
    
    # Double-check row counts
    if len(exclusive_pairs) != TV26_EVAL_U_EXPECTED_ROWS:
        raise ValueError(
            f"tv26_eval-U reference file must have exactly {TV26_EVAL_U_EXPECTED_ROWS} rows "
            f"(excluding headers), but found {len(exclusive_pairs)}"
        )
    
    if len(semi_pairs) != TV26_EVAL_A_EXPECTED_ROWS:
        raise ValueError(
            f"tv26_eval-A reference file must have exactly {TV26_EVAL_A_EXPECTED_ROWS} rows "
            f"(excluding headers), but found {len(semi_pairs)}"
        )
    
    print("\nLoading score files...")
    exclusive_scores = load_score_file(exclusive_score_file)
    semi_scores = load_score_file(semi_score_file)
    
    print(f"  Exclusive scores: {len(exclusive_scores)}")
    print(f"  Semi scores: {len(semi_scores)}")
    
    # Validate row counts
    if len(exclusive_scores) != len(exclusive_pairs):
        raise ValueError(
            f"Row count mismatch for exclusive: reference has {len(exclusive_pairs)} rows, "
            f"score file has {len(exclusive_scores)} rows"
        )
    
    if len(semi_scores) != len(semi_pairs):
        raise ValueError(
            f"Row count mismatch for semi: reference has {len(semi_pairs)} rows, "
            f"score file has {len(semi_scores)} rows"
        )
    
    # Generate output files
    print("\nGenerating output files...")
    
    # Process exclusive (U)
    output_u = output_dir / "tv26_eval-U.txt"
    missing_u = []
    with open(output_u, 'w', encoding='utf-8') as f:
        for enroll, test in exclusive_pairs:
            key = (enroll, test)
            if key not in exclusive_scores:
                missing_u.append((enroll, test))
                print(f"Warning: Missing score for pair ({enroll}, {test})")
                # Use 0.0 as default or raise error - let's raise error for safety
                raise ValueError(f"Missing score for exclusive pair ({enroll}, {test})")
            f.write(f"{exclusive_scores[key]}\n")
    
    print(f"  Created: {output_u} ({len(exclusive_pairs)} scores)")
    
    # Process semi (A)
    output_a = output_dir / "tv26_eval-A.txt"
    missing_a = []
    with open(output_a, 'w', encoding='utf-8') as f:
        for enroll, test in semi_pairs:
            key = (enroll, test)
            if key not in semi_scores:
                missing_a.append((enroll, test))
                print(f"Warning: Missing score for pair ({enroll}, {test})")
                raise ValueError(f"Missing score for semi pair ({enroll}, {test})")
            f.write(f"{semi_scores[key]}\n")
    
    print(f"  Created: {output_a} ({len(semi_pairs)} scores)")
    
    # Validate all scores are numeric in output files
    print("\nValidating output files...")
    with open(output_u, 'r') as f:
        for line_num, line in enumerate(f, 1):
            score = line.strip()
            if not is_numeric(score):
                raise ValueError(f"Output file {output_u} line {line_num}: non-numeric score '{score}'")
    
    with open(output_a, 'r') as f:
        for line_num, line in enumerate(f, 1):
            score = line.strip()
            if not is_numeric(score):
                raise ValueError(f"Output file {output_a} line {line_num}: non-numeric score '{score}'")
    
    print("  All scores validated as numeric")
    
    # Create zip file with unique name (reuse timestamp from folder)
    zip_name = f"submission_{timestamp}.zip"
    zip_path = output_dir / zip_name
    
    print(f"\nCreating zip file: {zip_path}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_u, arcname="tv26_eval-U.txt")
        zf.write(output_a, arcname="tv26_eval-A.txt")
    
    print(f"  Created: {zip_path}")
    print(f"\nSubmission package created successfully!")
    print(f"  Zip file: {zip_path}")
    print(f"  Contains: tv26_eval-U.txt ({len(exclusive_pairs)} scores)")
    print(f"            tv26_eval-A.txt ({len(semi_pairs)} scores)")
    
    return zip_path


def main():
    """Main function to process submission files."""
    
    # Validate input files exist
    print("Checking input files...")
    for file_path, name in [
        (TV26_EVAL_U_SCORE_FILE, "tv26_eval_U-scores"),
        (TV26_EVAL_U_REF_FILE, "tv26_eval_U-ref"),
        (TV26_EVAL_A_SCORE_FILE, "tv26_eval_A-scores"),
        (TV26_EVAL_A_REF_FILE, "tv26_eval_A-ref"),
    ]:
        if not file_path.exists():
            print(f"Error: {name} file not found: {file_path}", file=sys.stderr)
            sys.exit(1)
        print(f"   {name}: {file_path}")
    
    print()
    
    try:
        zip_path = process_submission(
            exclusive_score_file=TV26_EVAL_U_SCORE_FILE,
            exclusive_ref_file=TV26_EVAL_U_REF_FILE,
            semi_score_file=TV26_EVAL_A_SCORE_FILE,
            semi_ref_file=TV26_EVAL_A_REF_FILE,
            output_dir=OUTPUT_DIR
        )
        print(f"\nSuccess! Submission file: {zip_path}")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()