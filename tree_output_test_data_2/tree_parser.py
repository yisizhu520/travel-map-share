import re

def parse_tree_rules(input_file_path, output_file_path):
    """
    Parses decision tree rules from an input file and writes human-readable
    versions to an output file.
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                stripped_line = line.strip()
                indentation = line[:len(line) - len(line.lstrip())]
                
                # Regex to match lines like: |--- dflag_2 <= 0.50 [最接近值: 2]
                match_rule = re.match(r'\|--- (\w+)_(\w+) ([<>=]+) (\S+) \[最接近值: (\S+)\]',
                                      stripped_line)
                # Regex to match lines like: |--- dflag_2 >  0.50
                match_rule_no_closest = re.match(r'\|--- (\w+)_(\w+) ([<>=]+) (\S+)', stripped_line)

                # Regex to match lines like: |   |--- class: 4
                match_class = re.match(r'\|--- class: (\S+)', stripped_line)
                
                # Regex to match lines like: |   |   |--- truncated branch of depth 16
                match_truncated = re.match(r'\|--- truncated branch of depth (\d+)', stripped_line)

                if match_rule:
                    feature_prefix, feature_val, operator, _, closest_val = match_rule.groups()
                    feature_name = f"{feature_prefix}_{feature_val}"
                    if operator == '<=':
                        # Assuming <= 0.50 means it's NOT the closest value
                        outfile.write(f"{indentation}|--- 如果 {feature_name} 不是 {closest_val}\n")
                    elif operator == '>':
                        # Assuming > 0.50 means it IS the closest value
                        outfile.write(f"{indentation}|--- 如果 {feature_name} 是 {closest_val}\n")
                    else:
                        outfile.write(line) # Fallback for unhandled operators
                elif match_rule_no_closest:
                    # This case might occur if [最接近值: X] is missing for some reason
                    # For now, let's try to interpret it based on the operator and common patterns
                    feature_prefix, feature_val, operator, _ = match_rule_no_closest.groups()
                    feature_name = f"{feature_prefix}_{feature_val}"
                    # This is a guess; the logic might need refinement based on actual data meaning
                    if operator == '<=':
                        outfile.write(f"{indentation}|--- 如果 {feature_name} 的值较小或等于某个阈值\n")
                    elif operator == '>':
                        outfile.write(f"{indentation}|--- 如果 {feature_name} 的值较大\n")
                    else:
                        outfile.write(line) # Fallback
                elif match_class:
                    class_val = match_class.groups()[0]
                    outfile.write(f"{indentation}|--- 则分类为 {class_val}\n")
                elif match_truncated:
                    depth = match_truncated.groups()[0]
                    outfile.write(f"{indentation}|--- (省略了深度为 {depth} 的分支)\n")
                else:
                    # For lines that don't match known patterns, write them as is
                    # This could be the root or other structural lines
                    outfile.write(line)
        print(f"Successfully converted rules to {output_file_path}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # Get absolute paths based on the script's location or a fixed base path
    # Assuming the script is in tree_output_test_data_2 and the data file is also there.
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'tree_closest.txt')
    output_file = os.path.join(script_dir, 'tree_closest_human_readable.txt')
    
    # If your script is elsewhere, you might need to adjust paths:
    # base_path = r'c:\workspace\private\travel-map-share\tree_output_test_data_2'
    # input_file = os.path.join(base_path, 'tree_closest.txt')
    # output_file = os.path.join(base_path, 'tree_closest_human_readable.txt')

    parse_tree_rules(input_file, output_file)