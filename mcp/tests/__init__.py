# Tests for Personal Fitness Assistant

# Define file paths for saving user data
set_base_directory()
define_user_data_directory()
ensure_directory_exists()

# Save user query and assistant response
def save_user_query(user_id, query, response):
    path = build_user_history_path(user_id)
    history = load_existing_history(path)
    history.append({
        "timestamp": current_time(),
        "query": query,
        "response": response
    })
    write_json(path, history)

# Retrieve previous queries
def get_user_query_history(user_id, limit=None):
    path = get_history_path(user_id)
    if not file_exists(path):
        return []
    history = read_json(path)
    return sort_by_timestamp(history)[:limit]

# Format structured fitness data for readable output
def format_fitness_data(data):
    for key, value in data.items():
        if value is dict:
            format_nested_dict(key, value)
        else:
            format_key_value(key, value)

# Extract basic fitness metrics from user text
def parse_fitness_metrics(text):
    metrics = {}
    if "bmi" in text: metrics["bmi"] = "Mentioned"
    if "weight" in text: metrics["weight"] = "Mentioned"
    if "calories" in text: metrics["calories"] = "Mentioned"
    return metrics
