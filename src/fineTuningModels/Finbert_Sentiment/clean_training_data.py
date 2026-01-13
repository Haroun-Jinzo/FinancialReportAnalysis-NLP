import pandas as pd
import sys
from pathlib import Path


def clean_csv_data(input_file, output_file=None, text_column='text', label_column='label'):
    print("\n" + "="*70)
    print("CLEANING TRAINING DATA")
    print("="*70)
    
    # Load data
    print(f"\n📂 Loading: {input_file}")
    df = pd.read_csv(input_file)
    
    initial_count = len(df)
    print(f"✅ Loaded {initial_count} rows")
    
    # Show columns
    print(f"\n📋 Columns: {list(df.columns)}")
    
    # Check if columns exist
    if text_column not in df.columns:
        print(f"\n❌ ERROR: Column '{text_column}' not found!")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    if label_column not in df.columns:
        print(f"\n❌ ERROR: Column '{label_column}' not found!")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    # Start cleaning
    print(f"\n🧹 Starting data cleaning...")
    print("="*70)
    
    # 1. Show initial state
    print(f"\n1️⃣ Initial Data:")
    print(f"   Total rows: {len(df)}")
    print(f"   Text nulls: {df[text_column].isna().sum()}")
    print(f"   Label nulls: {df[label_column].isna().sum()}")
    
    # 2. Remove rows with null values
    before_null = len(df)
    df = df.dropna(subset=[text_column, label_column])
    after_null = len(df)
    removed_null = before_null - after_null
    
    if removed_null > 0:
        print(f"\n   ✓ Removed {removed_null} rows with null values")
    
    # 3. Remove empty strings
    before_empty = len(df)
    df[text_column] = df[text_column].astype(str).str.strip()
    df[label_column] = df[label_column].astype(str).str.strip()
    df = df[df[text_column] != '']
    df = df[df[label_column] != '']
    after_empty = len(df)
    removed_empty = before_empty - after_empty
    
    if removed_empty > 0:
        print(f"   ✓ Removed {removed_empty} rows with empty strings")
    
    # 4. Remove 'nan' strings
    before_nan = len(df)
    df = df[~df[text_column].str.lower().isin(['nan', 'none', 'null'])]
    df = df[~df[label_column].str.lower().isin(['nan', 'none', 'null'])]
    after_nan = len(df)
    removed_nan = before_nan - after_nan
    
    if removed_nan > 0:
        print(f"   ✓ Removed {removed_nan} rows with 'nan' strings")
    
    # 5. Standardize labels to lowercase
    df[label_column] = df[label_column].str.lower()
    
    # 6. Validate labels
    print(f"\n2️⃣ Validating Labels:")
    valid_labels = {'positive', 'negative', 'neutral'}
    unique_labels = set(df[label_column].unique())
    
    print(f"   Found labels: {unique_labels}")
    
    invalid_labels = unique_labels - valid_labels
    
    if invalid_labels:
        print(f"\n   ⚠️  WARNING: Invalid labels found: {invalid_labels}")
        print(f"   Valid labels must be: {valid_labels}")
        
        # Try to map common variations
        label_mapping = {
            'pos': 'positive',
            'neg': 'negative',
            'neu': 'neutral',
            'good': 'positive',
            'bad': 'negative',
            'ok': 'neutral',
            '1': 'positive',
            '0': 'negative',
            '-1': 'negative',
            'true': 'positive',
            'false': 'negative'
        }
        
        df[label_column] = df[label_column].replace(label_mapping)
        
        # Check again
        unique_labels_after = set(df[label_column].unique())
        still_invalid = unique_labels_after - valid_labels
        
        if still_invalid:
            print(f"\n   ❌ Cannot auto-fix labels: {still_invalid}")
            print(f"   Removing rows with invalid labels...")
            before_invalid = len(df)
            df = df[df[label_column].isin(valid_labels)]
            after_invalid = len(df)
            removed_invalid = before_invalid - after_invalid
            print(f"   ✓ Removed {removed_invalid} rows with invalid labels")
        else:
            print(f"   ✓ Auto-mapped labels successfully")
    else:
        print(f"   ✅ All labels are valid!")
    
    # 7. Remove duplicates
    before_dup = len(df)
    df = df.drop_duplicates(subset=[text_column])
    after_dup = len(df)
    removed_dup = before_dup - after_dup
    
    if removed_dup > 0:
        print(f"\n3️⃣ Duplicates:")
        print(f"   ✓ Removed {removed_dup} duplicate rows")
    
    # 8. Final statistics
    print(f"\n" + "="*70)
    print(f"CLEANING SUMMARY")
    print(f"="*70)
    
    total_removed = initial_count - len(df)
    
    print(f"\n📊 Results:")
    print(f"   Initial rows: {initial_count}")
    print(f"   Final rows: {len(df)}")
    print(f"   Removed: {total_removed} ({total_removed/initial_count*100:.1f}%)")
    
    if total_removed > 0:
        print(f"\n   Breakdown:")
        if removed_null > 0:
            print(f"     - Null values: {removed_null}")
        if removed_empty > 0:
            print(f"     - Empty strings: {removed_empty}")
        if removed_nan > 0:
            print(f"     - 'nan' strings: {removed_nan}")
        if removed_dup > 0:
            print(f"     - Duplicates: {removed_dup}")
    
    print(f"\n📈 Label Distribution:")
    label_counts = df[label_column].value_counts()
    for label, count in label_counts.items():
        pct = count / len(df) * 100
        print(f"   {label:10s}: {count:6d} ({pct:5.1f}%)")
    
    # Check balance
    if len(label_counts) > 0:
        max_count = label_counts.max()
        min_count = label_counts.min()
        ratio = max_count / min_count
        
        print(f"\n⚖️  Balance Ratio: {ratio:.2f}:1")
        
        if ratio > 3:
            print(f"   ⚠️  Dataset is imbalanced!")
            print(f"   Consider using balance_classes=True when training")
        else:
            print(f"   ✅ Dataset is reasonably balanced")
    
    # 9. Save cleaned data
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"\n💾 Saved cleaned data to: {output_file}")
    
    print(f"\n" + "="*70)
    
    return df


def interactive_clean():
    """Interactive cleaning"""
    print("\n" + "="*70)
    print("INTERACTIVE DATA CLEANER")
    print("="*70)
    
    # Get input file
    input_file = input("\n📁 Enter CSV file path: ").strip().strip('"').strip("'")
    
    if not Path(input_file).exists():
        print(f"\n❌ File not found: {input_file}")
        return
    
    # Get column names
    df = pd.read_csv(input_file)
    
    print(f"\n📋 Available columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i}. {col}")
    
    text_col = input(f"\nEnter text column name: ").strip()
    label_col = input(f"Enter label column name: ").strip()
    
    # Output file
    output_file = input(f"\nSave cleaned data to (press Enter for '_cleaned.csv'): ").strip()
    
    if not output_file:
        base = Path(input_file).stem
        output_file = f"data/training/{base}_cleaned.csv"
    
    # Clean
    clean_df = clean_csv_data(input_file, output_file, text_col, label_col)
    
    if clean_df is not None:
        print(f"\n✅ SUCCESS!")
        print(f"\nNow use this file for training:")
        print(f"   {output_file}")


def main():
    """Main function"""
    if len(sys.argv) > 1:
        # Command line mode
        input_file = sys.argv[1]
        text_col = sys.argv[2] if len(sys.argv) > 2 else 'text'
        label_col = sys.argv[3] if len(sys.argv) > 3 else 'label'
        
        base = Path(input_file).stem
        output_file = f"{base}_cleaned.csv"
        
        clean_csv_data(input_file, output_file, text_col, label_col)
    else:
        # Interactive mode
        interactive_clean()


if __name__ == "__main__":
    main()