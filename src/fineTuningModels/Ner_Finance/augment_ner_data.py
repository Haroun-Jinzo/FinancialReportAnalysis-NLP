import json
import re
import random
from typing import List, Dict
from copy import deepcopy

def augment_ner_dataset(input_file: str, output_file: str):
    
    print(f"📖 Loading data from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_count = len(data)
    augmented_data = list(data)  # Start with original data
    
    print(f"   Original examples: {original_count}")
    print("\n🔄 Augmenting dataset...")
    
    for item in data:
        text = item['text']
        entities = item['entities']
        
        # Augmentation 1: Replace "million" with "billion", "M", "B"
        if 'million' in text.lower():
            variations = create_money_variations(text, entities)
            augmented_data.extend(variations)
        
        # Augmentation 2: Simplify company names
        if any(e['label'] == 'ORG' for e in entities):
            variations = simplify_company_names(text, entities)
            augmented_data.extend(variations)
        
        # Augmentation 3: Shorten formal phrases
        if 'months ended' in text.lower() or 'year ended' in text.lower():
            variations = shorten_dates(text, entities)
            augmented_data.extend(variations)
    
    # Augmentation 4: Add synthetic conversational examples
    conversational = create_conversational_examples()
    augmented_data.extend(conversational)
    
    print(f"\n✅ Augmentation complete!")
    print(f"   Original: {original_count}")
    print(f"   Augmented: {len(augmented_data)}")
    print(f"   Added: {len(augmented_data) - original_count}")
    
    # Save
    print(f"\n💾 Saving to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, indent=2, ensure_ascii=False)
    
    return augmented_data


def create_money_variations(text: str, entities: List[Dict]) -> List[Dict]:
    """Create variations with different money formats"""
    variations = []
    
    # Variation 1: million → billion
    new_text = text.replace('million', 'billion')
    new_entities = update_entity_positions(text, new_text, entities)
    if new_entities:
        variations.append({'text': new_text, 'entities': new_entities})
    
    # Variation 2: million → M
    new_text = re.sub(r'\bmillion\b', 'M', text, flags=re.IGNORECASE)
    new_entities = update_entity_positions(text, new_text, entities)
    if new_entities:
        variations.append({'text': new_text, 'entities': new_entities})
    
    # Variation 3: billion → B
    if 'billion' in text.lower():
        new_text = re.sub(r'\bbillion\b', 'B', text, flags=re.IGNORECASE)
        new_entities = update_entity_positions(text, new_text, entities)
        if new_entities:
            variations.append({'text': new_text, 'entities': new_entities})
    
    return variations


def simplify_company_names(text: str, entities: List[Dict]) -> List[Dict]:
    """Simplify company names (e.g., remove 'Corporation')"""
    variations = []
    
    for entity in entities:
        if entity['label'] != 'ORG':
            continue
        
        org_name = entity['text']
        
        # Simplification options
        simplified_names = []
        if 'Corporation' in org_name:
            simplified_names.append(org_name.replace(' Corporation', ''))
        if 'Company' in org_name:
            simplified_names.append(org_name.replace(' Company', ''))
        if 'Group' in org_name:
            simplified_names.append(org_name.replace(' Group', ''))
        
        for simple_name in simplified_names:
            new_text = text.replace(org_name, simple_name)
            new_entities = update_entity_positions(text, new_text, entities)
            if new_entities:
                variations.append({'text': new_text, 'entities': new_entities})
    
    return variations


def shorten_dates(text: str, entities: List[Dict]) -> List[Dict]:
    """Shorten date phrases"""
    variations = []
    
    # Pattern: "three months ended March 31, 2016" → "Q1 2016"
    quarter_map = {
        ('january', 'february', 'march'): 'Q1',
        ('april', 'may', 'june'): 'Q2',
        ('july', 'august', 'september'): 'Q3',
        ('october', 'november', 'december'): 'Q4'
    }
    
    # Find month in text
    for months, quarter in quarter_map.items():
        for month in months:
            if month in text.lower():
                # Try to extract year
                year_match = re.search(r'\b(20\d{2})\b', text)
                if year_match:
                    year = year_match.group(1)
                    
                    # Replace long phrase with quarter
                    patterns = [
                        r'(?:three|six) months ended \w+ \d{1,2}, ' + year,
                        r'quarter ended \w+ \d{1,2}, ' + year
                    ]
                    
                    for pattern in patterns:
                        new_text = re.sub(pattern, f'{quarter} {year}', text, flags=re.IGNORECASE)
                        if new_text != text:
                            new_entities = update_entity_positions(text, new_text, entities)
                            if new_entities:
                                variations.append({'text': new_text, 'entities': new_entities})
    
    return variations


def update_entity_positions(old_text: str, new_text: str, entities: List[Dict]) -> List[Dict]:
    """Update entity positions after text modification"""
    if old_text == new_text:
        return None
    
    new_entities = []
    
    for entity in entities:
        entity_text = entity['text']
        old_start = entity['start']
        
        # Find entity in new text
        new_start = new_text.find(entity_text)
        
        if new_start == -1:
            # Entity text changed, try to find similar text
            # For now, skip this entity
            continue
        
        new_entity = {
            'text': entity_text,
            'label': entity['label'],
            'start': new_start,
            'end': new_start + len(entity_text)
        }
        new_entities.append(new_entity)
    
    return new_entities


def create_conversational_examples() -> List[Dict]:
    """Create synthetic conversational financial examples"""
    
    examples = [
        {
            "text": "Apple Inc. reported Q3 revenue of $90.1 billion, up 5% from last year",
            "entities": [
                {"text": "Apple Inc.", "label": "ORG", "start": 0, "end": 10},
                {"text": "Q3", "label": "DATE", "start": 20, "end": 22},
                {"text": "$90.1 billion", "label": "MONEY", "start": 34, "end": 47},
                {"text": "5%", "label": "PERCENT", "start": 52, "end": 54}
            ]
        },
        {
            "text": "Tesla's stock (TSLA) rose 5.2% to $245.30 after earnings",
            "entities": [
                {"text": "Tesla", "label": "ORG", "start": 0, "end": 5},
                {"text": "TSLA", "label": "TICKER", "start": 15, "end": 19},
                {"text": "5.2%", "label": "PERCENT", "start": 26, "end": 30},
                {"text": "$245.30", "label": "MONEY", "start": 34, "end": 41}
            ]
        },
        {
            "text": "Microsoft acquired Activision Blizzard for $68.7 billion",
            "entities": [
                {"text": "Microsoft", "label": "ORG", "start": 0, "end": 9},
                {"text": "Activision Blizzard", "label": "ORG", "start": 19, "end": 38},
                {"text": "$68.7 billion", "label": "MONEY", "start": 43, "end": 56}
            ]
        },
        {
            "text": "Amazon's EPS increased to $1.29, beating expectations",
            "entities": [
                {"text": "Amazon", "label": "ORG", "start": 0, "end": 6},
                {"text": "EPS", "label": "METRIC", "start": 9, "end": 12},
                {"text": "$1.29", "label": "MONEY", "start": 26, "end": 31}
            ]
        },
        {
            "text": "Google reported $76.7B in Q4 2023 revenue",
            "entities": [
                {"text": "Google", "label": "ORG", "start": 0, "end": 6},
                {"text": "$76.7B", "label": "MONEY", "start": 16, "end": 22},
                {"text": "Q4 2023", "label": "DATE", "start": 26, "end": 33}
            ]
        },
        {
            "text": "Goldman Sachs upgraded Netflix with price target of $650",
            "entities": [
                {"text": "Goldman Sachs", "label": "ORG", "start": 0, "end": 13},
                {"text": "Netflix", "label": "ORG", "start": 23, "end": 30},
                {"text": "$650", "label": "MONEY", "start": 52, "end": 56}
            ]
        }
    ]
    
    return examples


def main():
    print("\n" + "="*70)
    print("NER DATA AUGMENTATION")
    print("="*70)
    
    input_file = 'data/training/xbrl_financial_ner.json'
    output_file = 'data/training/xbrl_financial_ner_augmented.json'
    
    import os
    if not os.path.exists(input_file):
        print(f"\n❌ Error: Input file not found: {input_file}")
        return
    
    augmented_data = augment_ner_dataset(input_file, output_file)
    
    print("\n" + "="*70)
    print("✅ READY FOR IMPROVED TRAINING!")
    print("="*70)
    print(f"\nNow retrain with augmented data:")
    print(f"  python Ner_FineTuning.py {output_file}")
    print("="*70)


if __name__ == "__main__":
    main()