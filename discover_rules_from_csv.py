import pandas as pd

def discover_rules_stage1(df, target_col, condition_cols, source_cols, confidence_threshold=1.0):
    """
    Stage 1: Discovers 'If-Else' rules where the target column takes its value from a source column
    based on a condition in a condition column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the target column (e.g., 'flag').
        condition_cols (list): A list of column names to be used for forming conditions.
        source_cols (list): A list of column names that could be the source for the target column's value.
        confidence_threshold (float): The minimum proportion of matches required to consider a rule valid (0.0 to 1.0).

    Returns:
        list: A list of strings, where each string describes a discovered rule.
    """
    discovered_rules = []
    print(f"Starting Stage 1: Discovering rules with confidence threshold {confidence_threshold:.2f}\n")

    for cond_c in condition_cols:
        if cond_c not in df.columns:
            print(f"Warning: Condition column '{cond_c}' not found in DataFrame. Skipping.")
            continue
        
        # Ensure condition values are treated appropriately, especially if they are strings already
        # For numeric columns, unique values are fine. For object/string, they are also fine.
        unique_conditions = df[cond_c].unique()
        print(f"Analyzing condition column: '{cond_c}' with unique values: {unique_conditions}")

        for cond_v in unique_conditions:
            # Create a subset of the DataFrame based on the condition
            # Handle NaN correctly in conditions if necessary, though unique() usually excludes it unless present
            if pd.isna(cond_v):
                df_subset = df[df[cond_c].isna()]
            else:
                df_subset = df[df[cond_c] == cond_v]

            if df_subset.empty:
                continue

            # print(f"  Condition: {cond_c} == {cond_v} (Subset size: {len(df_subset)})")

            for src_c in source_cols:
                if src_c not in df.columns:
                    print(f"Warning: Source column '{src_c}' not found in DataFrame. Skipping for this source.")
                    continue
                
                if src_c == target_col: # Source column cannot be the target column itself in this logic
                    continue

                # Check if the target column consistently matches the source column in this subset
                # Ensure consistent data types for comparison if necessary, though pandas usually handles it.
                # However, explicit conversion might be needed if 'flag' is int and source is str '1'.
                # For this dataset, 'flag' and source_cols like 'cflag' appear to be strings.
                matches = (df_subset[target_col].astype(str) == df_subset[src_c].astype(str))
                match_count = matches.sum()
                
                if len(df_subset) > 0: # Avoid division by zero
                    confidence = match_count / len(df_subset)
                else:
                    confidence = 0

                if confidence >= confidence_threshold:
                    rule_description = f"IF '{cond_c}' == '{cond_v}' THEN '{target_col}' = value from '{src_c}' (Confidence: {confidence:.2%})"
                    # Further check: ensure all values in target_col are indeed from src_col
                    # This is already covered by confidence == 1.0
                    # Example of specific values if needed for clarity:
                    # if confidence == 1.0:
                    #    example_src_val = df_subset[src_c].iloc[0]
                    #    example_target_val = df_subset[target_col].iloc[0]
                    #    if example_src_val == example_target_val:
                    #        rule_description += f" (e.g., {target_col} is '{example_target_val}')"
                    discovered_rules.append(rule_description)
                    # print(f"    Found Rule: {rule_description}")
    
    return discovered_rules

def main():
    csv_file_path = 'test_data.csv'
    try:
        # nrows can be adjusted for very large files during development
        df = pd.read_csv(csv_file_path) #, nrows=1000)
        print(f"Successfully loaded {csv_file_path}. Shape: {df.shape}\n")
        print("First 5 rows of the dataset:")
        print(df.head())
        print("\nData types of columns:")
        print(df.dtypes)
        print("\n")
    except FileNotFoundError:
        print(f"Error: The file {csv_file_path} was not found.")
        return
    except Exception as e:
        print(f"Error loading or processing CSV: {e}")
        return

    target_column = 'flag'
    # Columns that might define the condition (e.g., 'eflag')
    potential_condition_columns = ['eflag', 'cflag', 'dflag', 'bflag']
    # Columns that might be the source of the 'flag' value (e.g., 'cflag', 'dflag', 'bflag')
    potential_source_columns = ['eflag', 'cflag', 'dflag', 'bflag'] 
    # Exclude '操作名称' as it's constant and 'flag' itself from sources

    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in the CSV.")
        return

    # --- Stage 1: Discovering If-Else rules ---
    # We are looking for 100% confidence rules initially
    rules_stage1 = discover_rules_stage1(df, target_column, potential_condition_columns, potential_source_columns, confidence_threshold=1.0)

    print("\n--- Stage 1 Results: Discovered Rules ---")
    if rules_stage1:
        for rule in rules_stage1:
            print(rule)
    else:
        print("No strong rules (100% confidence) discovered in Stage 1.")
        print("Consider lowering the confidence_threshold or checking data patterns manually.")

    # --- Stage 2: Modeling and Inverse Engineering ---
    print("\n--- Stage 2: Interpretation ---")
    print("The problem states: '目标列 flag 的取值可能是其他特征列中某一列的值'.")
    print("Stage 1 aimed to identify these direct assignment rules.")
    if rules_stage1:
        print("The rules found in Stage 1 represent this direct relationship.")
        print("For example, a rule like 'IF eflag == 1 THEN flag = value from cflag' means that when eflag is 1, the 'flag' column's value is taken directly from the 'cflag' column for that row.")
        print("In this context, Stage 2 involves confirming and documenting these identified direct assignments rather than complex mathematical modeling.")
    else:
        print("Since no strong direct assignment rules were found in Stage 1 with 100% confidence, Stage 2 would involve more exploratory analysis.")
        print("This could include:")
        print("  - Checking for rules with lower confidence.")
        print("  - Investigating if 'flag' is derived from a combination of columns or a transformation.")
        print("  - Using more advanced techniques like decision trees or symbolic regression if numerical relationships are suspected (though current data seems categorical).")
    
    print("\nAnalysis complete. You can run this script to analyze your data.")

if __name__ == '__main__':
    main()