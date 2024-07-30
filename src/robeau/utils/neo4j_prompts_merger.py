import json


# Function to read JSON content from a file
def read_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


# Function to write JSON content to a file
def write_json(file_path, content):
    with open(file_path, "w") as file:
        json.dump(content, file, indent=4)


# Function to merge the two JSON files with logging
def merge_json_with_synonyms(old, new):
    preserved_keys = ["StopCommand", "StopCommandRude", "StopCommandPolite"]
    protected_subkeys = ["synonyms"]
    merged = {}
    additions = []
    updates = []
    deletions = []
    log_entries = []

    for key in new.keys():
        if key in old:
            # Create a dictionary for easy lookup of nodes in old by id
            original_nodes = {entry["id"]: entry for entry in old[key]}
            # Create a dictionary for easy lookup of nodes in new by id
            new_nodes = {entry["id"]: entry for entry in new[key]}

            # Create the merged list for the current key
            merged_list = []

            for node_id, new_entry in new_nodes.items():
                if node_id in original_nodes:
                    original_entry = original_nodes[node_id]
                    merged_entry = (
                        new_entry.copy()
                    )  # Start with new_entry and add missing old properties

                    changes = []

                    for k, v in original_entry.items():
                        if k not in new_entry and k not in protected_subkeys:
                            changes.append(f"{k} removed")
                        elif k in protected_subkeys:
                            merged_entry[k] = original_entry[k]
                        elif k != "id" and original_entry.get(k) != v:
                            merged_entry[k] = v
                            changes.append(f"{k} changed to {v}")

                    if changes:
                        updates.append(merged_entry)
                        log_entries.append(
                            f"Updated entry for key '{key}', id '{node_id}': {', '.join(changes)}"
                        )

                    merged_list.append(merged_entry)
                else:
                    merged_list.append(new_entry)
                    additions.append(new_entry)
                    log_entries.append(
                        f"Added new entry for key '{key}', id '{node_id}': {new_entry}"
                    )

            merged[key] = merged_list
        else:
            merged[key] = new[key]
            additions.extend(new[key])
            log_entries.append(f"Added all new entries for new key '{key}'")

    # Detect deletions and preserve specific keys
    for key in old.keys():
        if key not in new:
            if key in preserved_keys:
                merged[key] = old[key]
                log_entries.append(f"Preserved key '{key}'")
            else:
                deletions.extend(old[key])
                log_entries.append(f"Key '{key}' removed")

    return merged, additions, updates, deletions, log_entries


# Input file paths
old_file_path = "src/robeau/jsons/processed_for_robeau/robeau_prompts.json"
new_file_path = "src/robeau/jsons/raw_from_neo4j/neo4j_prompts.json"
log_file_path = "src/robeau/jsons/temp/outputs_from_prompts_merge/last_merge_log.txt"

# Result file paths
additions_file_path = (
    "src/robeau/jsons/temp/outputs_from_prompts_merge/last_additions.json"
)
updates_file_path = "src/robeau/jsons/temp/outputs_from_prompts_merge/last_updates.json"
deletions_file_path = (
    "src/robeau/jsons/temp/outputs_from_prompts_merge/last_deletions.json"
)
backup_file_path = (
    "src/robeau/jsons/temp/outputs_from_prompts_merge/OLD_robeau_prompts.json"
)

# Read JSON content from files
original_json = read_json(old_file_path)
new_json = read_json(new_file_path)

# Merging the JSON files
merged_json, additions, updates, deletions, log_entries = merge_json_with_synonyms(
    original_json, new_json
)

# Save the original old data to a new file with a .old.json extension
write_json(backup_file_path, original_json)

# Write the merged JSON to the original file path
write_json(old_file_path, merged_json)
write_json(additions_file_path, {"additions": additions})
write_json(updates_file_path, {"updates": updates})
write_json(deletions_file_path, {"deletions": deletions})

# Save the log entries
with open(log_file_path, "w") as log_file:
    log_file.write("\n".join(log_entries))

# Print a confirmation message
print(f"Old file backed up as {backup_file_path}")
print(f"Merged file saved to {old_file_path}")
print(f"Additions saved to {additions_file_path}")
print(f"Updates saved to {updates_file_path}")
print(f"Deletions saved to {deletions_file_path}")
print(f"Log saved to {log_file_path}")