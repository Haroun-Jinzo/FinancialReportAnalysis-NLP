import pandas as pd
import json
import re
from typing import List, Dict, Tuple
import ast

"""
Convert XBRL-tagged financial data to NER format

This converter extracts financial entities from SEC filings with XBRL tags
and converts them to standard NER format with entities marked.
"""

def extract_entities_from_xbrl(text: str, xbrl_data: str) -> List[Dict]:
    """
    Extract entities from text using XBRL tag data
    
    Args:
        text: The financial text
        xbrl_data: JSON string with XBRL tags and values
    
    Returns:
        List of entities with text, label, start, end
    """
    entities = []
    
    # Parse XBRL data
    if xbrl_data == "No XBRL associated data.":
        # No entities - still use for training (negative examples)
        return entities
    
    try:
        # Parse the JSON-like string
        xbrl_dict = ast.literal_eval(xbrl_data)
    except:
        return entities
    
    # For each XBRL tag, find the values in the text
    for tag, values in xbrl_dict.items():
        if not isinstance(values, list):
            values = [values]
        
        for value in values:
            # Convert value to string and handle different formats
            value_str = str(value)
            
            # Try to find this value in the text
            # Look for patterns like: $X.X million, X million, $X.X, etc.
            patterns = [
                rf'\$\s*{re.escape(value_str)}\s*(?:million|billion|thousand)?',
                rf'{re.escape(value_str)}\s*(?:million|billion|thousand)',
                rf'\$\s*{re.escape(value_str)}',
                rf'{re.escape(value_str)}'
            ]
            
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entity_text = match.group().strip()
                    
                    # Determine entity type based on XBRL tag
                    entity_type = map_xbrl_to_entity_type(tag)
                    
                    entities.append({
                        'text': entity_text,
                        'label': entity_type,
                        'start': match.start(),
                        'end': match.end(),
                        'xbrl_tag': tag
                    })
    
    # Also extract common financial entities using regex
    entities.extend(extract_money_entities(text))
    entities.extend(extract_date_entities(text))
    entities.extend(extract_percent_entities(text))
    entities.extend(extract_org_entities(text))
    
    # Remove duplicates and overlaps
    entities = remove_overlapping_entities(entities)
    
    return entities


def map_xbrl_to_entity_type(xbrl_tag: str) -> str:
    """Map XBRL tag to entity type"""
    
    # Map common XBRL tags to entity types
    tag_mapping = {
        'EquityMethodInvestments': 'MONEY',
        'IncomeLossFromEquityMethodInvestments': 'MONEY',
        'UnrecognizedTaxBenefits': 'MONEY',
        'Revenue': 'MONEY',
        'NetIncome': 'MONEY',
        'Assets': 'MONEY',
        'Liabilities': 'MONEY',
        'Cash': 'MONEY',
        'Debt': 'MONEY',
        'EPS': 'METRIC',
        'EBITDA': 'METRIC',
        'SharesOutstanding': 'METRIC',
    }
    
    # Check if tag contains certain keywords
    if any(word in xbrl_tag.lower() for word in ['income', 'revenue', 'asset', 'liability', 'cash', 'debt', 'equity']):
        return 'MONEY'
    elif any(word in xbrl_tag.lower() for word in ['eps', 'ratio', 'rate', 'percent']):
        return 'METRIC'
    else:
        return 'MONEY'  # Default to MONEY for financial values


def extract_money_entities(text: str) -> List[Dict]:
    """Extract money amounts"""
    entities = []
    
    # Pattern: $X million/billion/thousand
    pattern = r'\$\s*[\d,]+\.?\d*\s*(?:million|billion|thousand|[BMK])'
    for match in re.finditer(pattern, text, re.IGNORECASE):
        entities.append({
            'text': match.group().strip(),
            'label': 'MONEY',
            'start': match.start(),
            'end': match.end()
        })
    
    return entities


def extract_date_entities(text: str) -> List[Dict]:
    """Extract dates"""
    entities = []
    
    # Pattern: Month Day, Year
    pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\s*,\s*\d{4}'
    for match in re.finditer(pattern, text, re.IGNORECASE):
        entities.append({
            'text': match.group().strip(),
            'label': 'DATE',
            'start': match.start(),
            'end': match.end()
        })
    
    # Pattern: Quarters
    pattern = r'\b(?:three|six)\s+months\s+ended\s+\w+\s+\d{1,2}\s*,\s*\d{4}'
    for match in re.finditer(pattern, text, re.IGNORECASE):
        entities.append({
            'text': match.group().strip(),
            'label': 'DATE',
            'start': match.start(),
            'end': match.end()
        })
    
    return entities


def extract_percent_entities(text: str) -> List[Dict]:
    """Extract percentages"""
    entities = []
    
    pattern = r'\d+\.?\d*\s*%'
    for match in re.finditer(pattern, text):
        entities.append({
            'text': match.group().strip(),
            'label': 'PERCENT',
            'start': match.start(),
            'end': match.end()
        })
    
    return entities


def extract_org_entities(text: str) -> List[Dict]:
    """Extract organization names"""
    entities = []
    
    # Common company indicators
    org_patterns = [
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Corporation|Corp|Company|Co|Inc|Incorporated|LLC|LLP|Ltd|Limited)\b',
    ]
    
    for pattern in org_patterns:
        for match in re.finditer(pattern, text):
            entities.append({
                'text': match.group().strip(),
                'label': 'ORG',
                'start': match.start(),
                'end': match.end()
            })
    
    return entities


def remove_overlapping_entities(entities: List[Dict]) -> List[Dict]:
    """Remove overlapping entities, keeping longer ones"""
    if not entities:
        return entities
    
    # Sort by start position, then by length (longest first)
    sorted_entities = sorted(entities, key=lambda x: (x['start'], -(x['end'] - x['start'])))
    
    filtered = []
    for entity in sorted_entities:
        # Check if this entity overlaps with any already added
        overlaps = False
        for existing in filtered:
            if not (entity['end'] <= existing['start'] or entity['start'] >= existing['end']):
                overlaps = True
                break
        
        if not overlaps:
            filtered.append(entity)
    
    return sorted(filtered, key=lambda x: x['start'])


def convert_xbrl_csv_to_ner(input_csv: str, output_json: str, max_examples: int = None):
    """
    Convert XBRL CSV to NER JSON format
    
    Args:
        input_csv: Path to CSV file with columns: [text_column, xbrl_column]
        output_json: Path to output JSON file
        max_examples: Maximum number of examples to process
    """
    print(f"📖 Reading from: {input_csv}")
    
    # Read CSV
    df = pd.read_csv(input_csv)
    
    print(f"   Columns: {list(df.columns)}")
    print(f"   Total rows: {len(df)}")
    
    # Detect columns (first column is usually the prompt, second is text, third is XBRL)
    if len(df.columns) >= 3:
        text_col = df.columns[1]
        xbrl_col = df.columns[2]
    else:
        print("❌ Error: Expected at least 3 columns (prompt, text, xbrl_data)")
        return
    
    ner_data = []
    processed = 0
    skipped = 0
    
    for idx, row in df.iterrows():
        if max_examples and processed >= max_examples:
            break
        
        text = str(row[text_col])
        xbrl_data = str(row[xbrl_col])
        
        # Skip very short texts
        if len(text) < 20:
            skipped += 1
            continue
        
        # Extract entities
        entities = extract_entities_from_xbrl(text, xbrl_data)
        
        # Include examples even without entities (for negative examples)
        ner_data.append({
            'text': text,
            'entities': entities
        })
        processed += 1
        
        if (idx + 1) % 100 == 0:
            print(f"   Processed {idx + 1} rows, kept {processed} examples")
    
    # Save
    print(f"\n💾 Saving to: {output_json}")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(ner_data, f, indent=2, ensure_ascii=False)
    
    # Statistics
    with_entities = sum(1 for item in ner_data if item['entities'])
    without_entities = len(ner_data) - with_entities
    
    print(f"\n✅ Conversion complete!")
    print(f"   📊 Total examples: {processed}")
    print(f"   ✨ With entities: {with_entities} ({with_entities/processed*100:.1f}%)")
    print(f"   ⚪ Without entities: {without_entities} ({without_entities/processed*100:.1f}%)")
    print(f"   ⏭️  Skipped: {skipped}")
    
    # Show samples
    print(f"\n📋 Sample outputs:")
    samples_shown = 0
    for item in ner_data[:50]:
        if item['entities'] and samples_shown < 3:
            print(f"\n{samples_shown + 1}.")
            print(f"Text: {item['text'][:100]}...")
            print(f"Entities: {item['entities'][:3]}")
            samples_shown += 1
    
    return ner_data


def main():
    """Main conversion function"""
    print("\n" + "="*70)
    print("XBRL FINANCIAL DATA TO NER CONVERTER")
    print("="*70)
    
    # Configure paths
    input_csv = 'data/training/Finance_Entities.csv'  # Change this to your CSV file path
    output_json = 'data/training/xbrl_financial_ner.json'
    
    # Check if file exists
    import os
    if not os.path.exists(input_csv):
        print(f"\n❌ Error: Input file not found: {input_csv}")
        print("\nPlease update the input_csv path in the script to point to your CSV file")
        return
    
    # Convert
    ner_data = convert_xbrl_csv_to_ner(
        input_csv=input_csv,
        output_json=output_json,
        max_examples=5000  # Process up to 5000 examples (remove limit by setting to None)
    )
    
    if ner_data and len(ner_data) > 100:
        print("\n" + "="*70)
        print("✅ READY FOR NER TRAINING!")
        print("="*70)
        print(f"\nThis dataset has {len(ner_data)} examples - much better!")
        print(f"\nNow run:")
        print(f"  python Ner_FineTuning.py {output_json}")
        print("="*70)
    else:
        print("\n⚠️  WARNING: Very few examples generated!")


if __name__ == "__main__":
    main()