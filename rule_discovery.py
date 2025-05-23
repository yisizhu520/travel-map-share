import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error # Added accuracy_score
import numpy as np
from gplearn.genetic import SymbolicRegressor
from sklearn.utils.validation import check_X_y, check_array # For gplearn or general use

# 定义CSV文件路径，实际使用时可以修改或作为参数传入
CSV_FILE_PATH = 'c:\\workspace\\private\\travel-map-share\\test_data.csv'

# 定义用于规则发现的候选列
CANDIDATE_RULE_COLUMNS = ['eflag'] 
# 定义用于构建布尔特征的基础比较列和源比较列
BASE_COMPARISON_COL = 'flag'
SOURCE_COMPARISON_COLS = ['cflag', 'dflag', 'bflag']
# 定义额外的列，用于创建基于其值的布尔特征 (例如: aflag='a', eflag=1)
# 这些列可以帮助发现更复杂的组合规则
# 注意：如果这里的列也是 CANDIDATE_RULE_COLUMNS 中的目标列，它将不会被用来为自己生成特征。
ADDITIONAL_CONDITION_COLS = ['aflag', 'eflag'] # 示例，根据数据集进行配置

def load_and_preprocess_data(file_path, nrows=None):
    """加载并预处理数据"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, nrows=nrows)
    print(f"Successfully loaded data from: {file_path}")
    print(f"Loaded {len(df)} rows.")

    # 为基于值的特征创建列的字符串副本
    for col in ADDITIONAL_CONDITION_COLS:
        if col in df.columns:
            df[f"{col}_for_features"] = df[col].astype(str)
        else:
            print(f"Warning: Additional condition column '{col}' not found. Cannot create '{col}_for_features'.")

    all_relevant_cols = list(set(SOURCE_COMPARISON_COLS + [BASE_COMPARISON_COL] + CANDIDATE_RULE_COLUMNS + ADDITIONAL_CONDITION_COLS))

    for col in all_relevant_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in CSV. Skipping preprocessing for it.")
            continue
        
        if col in SOURCE_COMPARISON_COLS + [BASE_COMPARISON_COL]:
            df[f"{col}_numeric"] = pd.to_numeric(df[col], errors='coerce')
            if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].astype(str)
            else:
                df[col] = df[col].astype(str)
        
        if col in CANDIDATE_RULE_COLUMNS:
            try:
                df[col] = df[col].astype(int)
            except ValueError:
                print(f"Warning: Could not convert candidate rule column '{col}' to int. Using LabelEncoder.")
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
    return df


def _calculate_segment_consistency_score(df_data_for_scoring, predicted_segment_values, base_comparison_col, source_comparison_cols):
    """
    Calculates a score based on how consistently the base_comparison_col matches one of the
    source_comparison_cols within each predicted segment.
    """
    if base_comparison_col not in df_data_for_scoring.columns:
        print(f"  Error in _calculate_segment_consistency_score: base_comparison_col '{base_comparison_col}' not found.")
        return 0.0
    
    all_src_cols_present = True
    for sc_col in source_comparison_cols:
        if sc_col not in df_data_for_scoring.columns:
            print(f"  Error in _calculate_segment_consistency_score: source_comparison_col '{sc_col}' not found.")
            all_src_cols_present = False
    if not all_src_cols_present:
        return 0.0

    temp_df = df_data_for_scoring.copy()
    temp_df['_predicted_segment_'] = predicted_segment_values

    total_weighted_consistency = 0
    total_rows_in_segments = 0

    unique_predicted_segments = pd.Series(predicted_segment_values).unique()

    for seg_val in unique_predicted_segments:
        segment_df = temp_df[temp_df['_predicted_segment_'] == seg_val]
        
        if segment_df.empty:
            continue

        num_rows_in_segment = len(segment_df)
        best_consistency_for_this_segment = 0.0
        
        for src_col in source_comparison_cols:
            matches = (segment_df[base_comparison_col].astype(str) == segment_df[src_col].astype(str))
            consistency = matches.mean() if num_rows_in_segment > 0 else 0.0
            if consistency > best_consistency_for_this_segment:
                best_consistency_for_this_segment = consistency
        
        total_weighted_consistency += best_consistency_for_this_segment * num_rows_in_segment
        total_rows_in_segments += num_rows_in_segment

    if total_rows_in_segments == 0:
        # This case might occur if y_pred is empty or all segments are empty, though unlikely with typical data.
        print("  Warning in _calculate_segment_consistency_score: No rows in any segment after prediction.")
        return 0.0
    
    final_score = total_weighted_consistency / total_rows_in_segments
    return final_score


def _generate_rules_for_target(df_original, target_col_name, base_comparison_col, source_comparison_cols, additional_condition_cols):
    """为单个目标列生成决策规则（内部函数）"""
    print(f"\nAttempting to generate rules for target column: '{target_col_name}'")
    df = df_original.copy()

    boolean_feature_names = []
    for src_col in source_comparison_cols:
        if src_col not in df.columns or base_comparison_col not in df.columns:
            print(f"  Skipping boolean feature for '{src_col}' or '{base_comparison_col}' as one/both are missing.")
            continue
        feature_name = f'{base_comparison_col}_eq_{src_col}'
        df[feature_name] = (df[base_comparison_col].astype(str) == df[src_col].astype(str)).astype(int)
        boolean_feature_names.append(feature_name)

    # 基于 ADDITIONAL_CONDITION_COLS 创建布尔特征
    for mc_col in additional_condition_cols:
        if mc_col == target_col_name: # 避免使用目标列为自身创建特征
            print(f"  Skipping '{mc_col}' for feature generation as it is the target column.")
            continue

        feature_source_col = f"{mc_col}_for_features"
        if feature_source_col not in df_original.columns:
            if mc_col in df_original.columns:
                print(f"  Warning: Using raw column '{mc_col}' for features as '{feature_source_col}' not found. Ensure it's string type.")
                feature_source_col = mc_col 
            else:
                print(f"  Feature source column '{feature_source_col}' (and raw '{mc_col}') for multi-condition rule not found in DataFrame. Skipping '{mc_col}'.")
                continue
        
        try:
            unique_values = df_original[feature_source_col].unique()
            MAX_UNIQUE_VALUES_FOR_FEATURE = 20 
            if len(unique_values) > MAX_UNIQUE_VALUES_FOR_FEATURE:
                print(f"  Column '{mc_col}' (from '{feature_source_col}') has {len(unique_values)} unique values, exceeding limit of {MAX_UNIQUE_VALUES_FOR_FEATURE}. Skipping feature generation for this column to avoid excessive features.")
                continue

            for val in unique_values:
                clean_val = str(val).replace(" ", "_").replace(".", "_dot_") 
                feature_name = f'{mc_col}_is_{clean_val}'[:250] 
                df[feature_name] = (df_original[feature_source_col].astype(str) == str(val)).astype(int)
                boolean_feature_names.append(feature_name)
        except KeyError:
            print(f"  KeyError accessing '{feature_source_col}' for multi-condition features. Skipping {mc_col}.")
            continue
        except Exception as e:
            print(f"  Error generating features for multi-condition column '{mc_col}' (from '{feature_source_col}'): {e}. Skipping.")
            continue
    
    if not boolean_feature_names:
        print(f"  No boolean features generated for target '{target_col_name}'. Skipping.")
        return None, [], "", 0.0, 0.0

    X_bool = df[boolean_feature_names]
    
    if target_col_name not in df.columns:
        print(f"  Target column '{target_col_name}' not found in DataFrame. Skipping.")
        return None, [], "", 0.0, 0.0
        
    y_bool = df[target_col_name]

    if y_bool.ndim > 1:
        y_bool = y_bool.squeeze()

    model = DecisionTreeClassifier(random_state=42, max_depth=max(3, len(source_comparison_cols)), min_samples_leaf=10) 
    try:
        model.fit(X_bool, y_bool)
    except Exception as e:
        print(f"  Error training Decision Tree for '{target_col_name}': {e}")
        return None, [], "", 0.0, 0.0

    rules_text = ""
    try:
        rules_text = export_text(model, feature_names=boolean_feature_names)
        print(f"\n  Decision Tree Rules (for {target_col_name} using boolean features):")
        print(rules_text)
    except Exception as e:
        print(f"  Error exporting rules for '{target_col_name}': {e}")

    y_pred = model.predict(X_bool)
    acc = accuracy_score(y_bool, y_pred) # Accuracy of predicting the target_col_name itself
    print(f"  Accuracy for predicting '{target_col_name}' (raw): {acc:.4f}")

    # Calculate the new segment consistency score
    # df is the local copy within _generate_rules_for_target, which has all necessary columns
    segment_consistency_score = _calculate_segment_consistency_score(
        df, # Pass the DataFrame used for generating X_bool and y_pred
        y_pred,      # These are the predicted values for target_col_name (e.g., eflag)
        base_comparison_col, 
        source_comparison_cols
    )
    
    return model, boolean_feature_names, rules_text, acc, segment_consistency_score

def discover_segmentation_rules(df, candidate_target_cols, base_comparison_col, source_comparison_cols, additional_condition_cols):
    """第一阶段：发现主要的分割规则列及其规则"""
    print("\n--- Stage 1: Discovering Segmentation Rules ---")
    
    best_rule_info = {
        'target_col_name': None,
        'model': None,
        'feature_names': [],
        'rules_text': "",
        'segment_consistency_score': -1.0, # This is the new primary score
        'target_prediction_accuracy': -1.0, # This is the old 'acc'
        'depth': float('inf')
    }

    for target_col in candidate_target_cols:
        # Unpack 5 values now
        model, features, rules, target_pred_acc, seg_consistency_score = _generate_rules_for_target(
            df, target_col, base_comparison_col, source_comparison_cols, additional_condition_cols
        )
        
        if model is not None:
            depth = model.get_depth()
            
            is_better = False
            # Primary criterion: higher segment_consistency_score
            if seg_consistency_score > best_rule_info['segment_consistency_score']:
                is_better = True
            elif seg_consistency_score == best_rule_info['segment_consistency_score']:
                # Secondary criterion: higher target_prediction_accuracy
                if target_pred_acc > best_rule_info['target_prediction_accuracy']:
                    is_better = True
                elif target_pred_acc == best_rule_info['target_prediction_accuracy']:
                    # Tertiary criterion: lower depth
                    if depth < best_rule_info['depth']:
                        is_better = True

            if is_better:
                best_rule_info['target_col_name'] = target_col
                best_rule_info['model'] = model
                best_rule_info['feature_names'] = features
                best_rule_info['rules_text'] = rules
                best_rule_info['segment_consistency_score'] = seg_consistency_score
                best_rule_info['target_prediction_accuracy'] = target_pred_acc
                best_rule_info['depth'] = depth
                print(f"  New best rule set found for target '{target_col}' "
                      f"(Seg. Consistency: {seg_consistency_score:.4f}, "
                      f"Target Acc: {target_pred_acc:.4f}, Depth: {depth})")

    if best_rule_info['model'] is None:
        print("\nCould not find any suitable segmentation rules.")
        return None, None, [], ""
    
    print(f"\n--- Best Segmentation Column Identified: '{best_rule_info['target_col_name']}' ---")
    print(f"  Segment Consistency Score: {best_rule_info['segment_consistency_score']:.4f}")
    print(f"  Target Prediction Accuracy: {best_rule_info['target_prediction_accuracy']:.4f}")
    print(f"  Tree Depth: {best_rule_info['depth']}")
    print("  Rules:")
    print(best_rule_info['rules_text'])
    
    return best_rule_info['target_col_name'], best_rule_info['model'], best_rule_info['feature_names'], best_rule_info['rules_text']

def stage2_discover_mathematical_relationships(df, segmentation_col_name, rules_text_for_segmentation):
    """第二阶段：使用符号回归发现潜在的数学关系"""
    print("\n--- Stage 2: Discovering Mathematical Relationships (Symbolic Regression) ---")

    if segmentation_col_name is None or segmentation_col_name not in df.columns:
        print("  Error: Segmentation column name is invalid or column not found. Skipping Stage 2.")
        return

    base_feature_cols_for_sr = SOURCE_COMPARISON_COLS 
    source_col_map = {}
    if segmentation_col_name == 'eflag': 
        source_col_map = {1: 'cflag', 2: 'dflag', 3: 'bflag'}
        print(f"  Using default source_col_map for eflag: {source_col_map}")
    else:
        print(f"  Warning: Dynamic source_col_map generation for '{segmentation_col_name}' is basic and may need refinement based on rules_text_for_segmentation.")
        # Placeholder for more advanced rule parsing to build source_col_map
        # For now, if not eflag, this map will be empty, limiting fallback rule descriptions.

    for seg_value in sorted(df[segmentation_col_name].unique()):
        print(f"\nProcessing segment where {segmentation_col_name} = {seg_value}:")
        segment_df = df[df[segmentation_col_name] == seg_value].copy()

        if segment_df.empty:
            print("  No data in this segment.")
            continue

        primary_source_col_name = source_col_map.get(seg_value)
        target_col_numeric_name = f"{BASE_COMPARISON_COL}_numeric"
        
        if target_col_numeric_name not in segment_df.columns or segment_df[target_col_numeric_name].isnull().all():
            print(f"  Target column '{target_col_numeric_name}' is missing or all NaN for {segmentation_col_name} = {seg_value}.")
            if primary_source_col_name and primary_source_col_name in segment_df.columns and BASE_COMPARISON_COL in segment_df.columns:
                if (segment_df[BASE_COMPARISON_COL].astype(str) == segment_df[primary_source_col_name].astype(str)).all():
                    print(f"  Fallback Rule: {BASE_COMPARISON_COL} = {primary_source_col_name} (categorical selection confirmed).")
                else:
                    mismatches = segment_df[segment_df[BASE_COMPARISON_COL].astype(str) != segment_df[primary_source_col_name].astype(str)]
                    print(f"  Fallback Rule: {BASE_COMPARISON_COL} is related to {primary_source_col_name} (categorical), but not always equal. Found {len(mismatches)} mismatches out of {len(segment_df)}.")
            elif not primary_source_col_name and source_col_map:
                 print(f"  Segment value {seg_value} not found in source_col_map. Cannot determine primary source for fallback.")
            else:
                print("  Cannot determine fallback rule due to missing columns or unmapped segment value.")
            continue
        
        y_data = segment_df[target_col_numeric_name].dropna()
        if y_data.empty:
            print(f"  Target column '{target_col_numeric_name}' has no non-NaN values after dropping NaNs for {segmentation_col_name} = {seg_value}.")
            continue

        X_data_for_sr_list = []
        valid_feature_names_for_sr = []
        for base_col_name_sr in base_feature_cols_for_sr:
            numeric_feature_col_name = f"{base_col_name_sr}_numeric"
            if numeric_feature_col_name in segment_df.columns and not segment_df[numeric_feature_col_name].isnull().all():
                feature_series = segment_df[numeric_feature_col_name].dropna()
                if not feature_series.empty:
                    X_data_for_sr_list.append(feature_series)
                    valid_feature_names_for_sr.append(base_col_name_sr) 
            else:
                print(f"  Numeric feature '{numeric_feature_col_name}' is missing or all NaN. Excluding from SR.")

        if not X_data_for_sr_list:
            print(f"  No valid numeric feature columns available for {segmentation_col_name} = {seg_value}. Cannot perform symbolic regression.")
            if primary_source_col_name and (segment_df[BASE_COMPARISON_COL].astype(str) == segment_df[primary_source_col_name].astype(str)).all():
                 print(f"  Fallback Rule: {BASE_COMPARISON_COL} = {primary_source_col_name} (categorical selection confirmed).")
            continue

        common_index = y_data.index
        for x_series in X_data_for_sr_list:
            common_index = common_index.intersection(x_series.index)

        if common_index.empty or len(common_index) < 2:
            print(f"  Not enough common non-NaN data points ({len(common_index)}) between target and features for {segmentation_col_name} = {seg_value}.")
            continue
            
        y_aligned = y_data.loc[common_index]
        aligned_data_dict = {}
        for i, name in enumerate(valid_feature_names_for_sr):
            aligned_data_dict[name] = X_data_for_sr_list[i].loc[common_index]
        X_aligned_df = pd.DataFrame(aligned_data_dict)
        
        if X_aligned_df.empty or y_aligned.empty:
            print(f"  Aligned data is empty for {segmentation_col_name} = {seg_value}.")
            continue

        print(f"  Attempting symbolic regression for '{target_col_numeric_name}' using features: {list(X_aligned_df.columns)}")
        print(f"  Number of data points for SR: {len(y_aligned)}")

        sr = SymbolicRegressor(population_size=500, generations=10, 
                               stopping_criteria=0.01, p_crossover=0.7, 
                               p_subtree_mutation=0.1, p_hoist_mutation=0.05, 
                               p_point_mutation=0.1, max_samples=0.9, 
                               verbose=0, 
                               function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv'),
                               metric='mse', parsimony_coefficient=0.005,
                               random_state=42, feature_names=list(X_aligned_df.columns))
        
        try:
            X_fit = X_aligned_df.values
            y_fit = y_aligned.values
            sr.fit(X_fit, y_fit)
            discovered_formula = str(sr._program)
            print(f"  Discovered formula for {target_col_numeric_name}: {discovered_formula}")
        except Exception as e:
            print(f"  Symbolic regression failed for {segmentation_col_name} = {seg_value}: {e}")
            if primary_source_col_name and (segment_df[BASE_COMPARISON_COL].astype(str) == segment_df[primary_source_col_name].astype(str)).all():
                print(f"  Fallback Rule: {BASE_COMPARISON_COL} = {primary_source_col_name} (categorical selection confirmed).")

def verify_flag_selection(df, segmentation_col_name, rules_text_for_segmentation):
    """验证基于发现的分割规则的选择逻辑"""
    print(f"\n--- Verifying selection logic based on discovered segmentation column '{segmentation_col_name}' ---")

    if segmentation_col_name is None or segmentation_col_name not in df.columns:
        print("  Error: Segmentation column name is invalid or column not found. Skipping verification.")
        return

    if segmentation_col_name == 'eflag': # Default verification for eflag
        source_col_map = {1: 'cflag', 2: 'dflag', 3: 'bflag'}
        for seg_value, src_col in source_col_map.items():
            segment_data = df[df[segmentation_col_name] == seg_value]
            if not segment_data.empty:
                if src_col not in df.columns or BASE_COMPARISON_COL not in df.columns:
                    print(f"  Cannot verify for {segmentation_col_name}={seg_value} as '{src_col}' or '{BASE_COMPARISON_COL}' is missing.")
                    continue
                accuracy = (segment_data[BASE_COMPARISON_COL].astype(str) == segment_data[src_col].astype(str)).mean()
                print(f"For {segmentation_col_name} = {seg_value}: '{BASE_COMPARISON_COL}' == '{src_col}' holds for {accuracy*100:.2f}% of cases.")
                if accuracy < 1.0:
                    mismatches = segment_data[segment_data[BASE_COMPARISON_COL].astype(str) != segment_data[src_col].astype(str)]
                    print(f"  Mismatches: {len(mismatches)} rows")
            else:
                print(f"For {segmentation_col_name} = {seg_value}: No data in this segment to verify.")
    else:
        # For other discovered rule columns, a more general verification based on 'rules_text_for_segmentation' would be needed.
        # This is a complex task involving parsing the decision tree rules.
        # For now, we'll just state that specific verification for non-eflag columns needs to be manually checked or this function extended.
        print(f"  Verification logic for segmentation column '{segmentation_col_name}' (if not 'eflag') is not fully generalized here.")
        print(f"  The discovered rules were: \n{rules_text_for_segmentation}")
        print(f"  You may need to manually inspect data or extend this verification function based on these rules.")

import argparse # Added based on user's intended replacement structure

def main():
    # --- 参数配置区 --- (comment from user's intended replace block)
    parser = argparse.ArgumentParser(description='Discover segmentation rules from a CSV file.')
    parser.add_argument('csv_file', type=str, help='Path to the input CSV file.')
    parser.add_argument('--nrows', type=int, default=20000, help='Number of rows to read from CSV.') # Adapted from original usage
    args = parser.parse_args()

    # Global CSV_FILE_PATH is no longer used by this main logic.
    # Global CANDIDATE_RULE_COLUMNS, BASE_COMPARISON_COL, SOURCE_COMPARISON_COLS are still used.
    df_data = load_and_preprocess_data(args.csv_file, nrows=args.nrows) 
    
    if not df_data.empty:
        discovered_seg_col, seg_model, seg_features, seg_rules_text = discover_segmentation_rules(
            df_data.copy(), 
            CANDIDATE_RULE_COLUMNS, 
            BASE_COMPARISON_COL, 
            SOURCE_COMPARISON_COLS,
            ADDITIONAL_CONDITION_COLS
        )
        
        if discovered_seg_col:
            stage2_discover_mathematical_relationships(df_data.copy(), discovered_seg_col, seg_rules_text)
            verify_flag_selection(df_data.copy(), discovered_seg_col, seg_rules_text)
        else:
            print("\nSkipping Stage 2 and Verification as no segmentation rules were discovered.")
        
        print("\n--- Analysis Complete ---")
        print("This script implements a two-stage approach:")
        print("1. Stage 1: Discovers a primary segmentation column and its decision rules based on boolean features (e.g., flag_eq_cflag).")
        if discovered_seg_col:
            print(f"   - Discovered Segmentation Column: '{discovered_seg_col}'")
        print("2. Stage 2: For each segment, attempts symbolic regression for numeric relationships or describes categorical rules.")
        print("3. Verification: Checks consistency of data with discovered/assumed selection logic.")
    else:
        print("No data loaded, skipping analysis.")

if __name__ == "__main__":
    main()