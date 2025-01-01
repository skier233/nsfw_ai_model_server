import os
import yaml
import curses

# Static file paths
active_ai_yaml_path = "./config/active_ai.yaml"
ai_models_directory = "./config/models"

def load_active_ai_models():
    with open(active_ai_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('active_ai_models', []) or []

def save_active_ai_models(active_ai_models):
    with open(active_ai_yaml_path, 'w') as f:
        yaml.dump({'active_ai_models': active_ai_models}, f, default_flow_style=False, sort_keys=False)

def load_available_ai_models():
    models = []
    for file in os.listdir(ai_models_directory):
        if file.endswith('.yaml'):
            yaml_path = os.path.join(ai_models_directory, file)
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            if data.get('type') == 'model' and data.get('model_category', None) is not None:
                data['yaml_file_name'] = file.replace(".yaml", "")  # Use the file name itself
                models.append(data)
    return models

def load_model_data(model_file_name):
    yaml_path = os.path.join(ai_models_directory, model_file_name + '.yaml')
    with open(yaml_path, 'r') as f:
        data =  yaml.safe_load(f)
        data['yaml_file_name'] = model_file_name.replace(".yaml", "")
        return data

def display_model(stdscr, y, x, model, highlight=False, incompatible=False, reason=""):
    categories = ", ".join(model['model_category'])
    display_text = f"{model['yaml_file_name']:<20} | {categories:<15} | {model['model_version']:<5} | {model['model_image_size']:<5} | {model['model_info']:<30}"
    if highlight:
        stdscr.addstr(y, x, display_text, curses.A_REVERSE)
    elif incompatible:
        stdscr.addstr(y, x, display_text + f" (Incompatible: {reason})", curses.color_pair(1))
    else:
        stdscr.addstr(y, x, display_text)

def choose_active_models():
    curses.wrapper(open_ui)

def open_ui(stdscr):
    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_RED)
    stdscr.clear()

    active_ai_model_names = load_active_ai_models()
    available_ai_models = load_available_ai_models()

    # Load the actual YAML data for active AI models
    active_ai_models = [load_model_data(model_name) for model_name in active_ai_model_names]

    # Remove active models from available models list
    available_ai_models = [model for model in available_ai_models if model['yaml_file_name'] not in active_ai_model_names]

    # Group and sort models by category and identifier
    available_ai_models.sort(key=lambda x: (x['model_category'][0], int(x['model_identifier'])))
    active_ai_models.sort(key=lambda x: (x['model_category'][0], int(x['model_identifier'])))

    # Determine the image size and categories of active models
    active_image_sizes = {model['model_image_size'] for model in active_ai_models}
    active_categories = {category for model in active_ai_models for category in model['model_category']}

    current_list = 'available'
    current_index = 0

    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, "Available AI Models:")
        stdscr.addstr(1, 0, f"{'File Name':<20} | {'Category':<15} | {'Ver':<5} | {'Size':<5} | {'Info':<30}")
        stdscr.addstr(2, 0, "-" * 80)
        y_offset = 3
        for idx, model in enumerate(available_ai_models):
            highlight = current_list == 'available' and idx == current_index
            incompatible = any(size != model['model_image_size'] for size in active_image_sizes) or any(category in active_categories for category in model['model_category'])
            reason = "Different image size" if any(size != model['model_image_size'] for size in active_image_sizes) else "Category already active" if any(category in active_categories for category in model['model_category']) else ""
            display_model(stdscr, y_offset, 0, model, highlight, incompatible, reason)
            y_offset += 1

        stdscr.addstr(y_offset + 1, 0, "Active AI Models:")
        stdscr.addstr(y_offset + 2, 0, f"{'File Name':<20} | {'Category':<15} | {'Ver':<5} | {'Size':<5} | {'Info':<30}")
        stdscr.addstr(y_offset + 3, 0, "-" * 80)
        y_offset += 4
        for idx, model in enumerate(active_ai_models):
            highlight = current_list == 'active' and idx == current_index
            display_model(stdscr, y_offset, 0, model, highlight)
            y_offset += 1

        # Display key bindings
        stdscr.addstr(y_offset + 2, 0, "Keys: UP/DOWN - Navigate | ENTER/RIGHT - Move | TAB - Switch List | q - Quit")

        key = stdscr.getch()

        if key == curses.KEY_UP:
            if current_index > 0:
                current_index -= 1
        elif key == curses.KEY_DOWN:
            if current_list == 'available' and current_index < len(available_ai_models) - 1:
                current_index += 1
            elif current_list == 'active' and current_index < len(active_ai_models) - 1:
                current_index += 1
        elif key == curses.KEY_RIGHT or key in [curses.KEY_ENTER, 10, 13]:
            if current_list == 'available' and available_ai_models:
                model = available_ai_models[current_index]
                incompatible = any(size != model['model_image_size'] for size in active_image_sizes) or any(category in active_categories for category in model['model_category'])
                if not incompatible:
                    available_ai_models.pop(current_index)
                    active_ai_models.append(model)
                    active_image_sizes.add(model['model_image_size'])
                    active_categories.update(model['model_category'])
                    if current_index >= len(available_ai_models):
                        current_index = len(available_ai_models) - 1
                    # Re-sort the lists
                    available_ai_models.sort(key=lambda x: (x['model_category'][0], int(x['model_identifier'])))
                    active_ai_models.sort(key=lambda x: (x['model_category'][0], int(x['model_identifier'])))
            elif current_list == 'active' and active_ai_models:
                model = active_ai_models.pop(current_index)
                available_ai_models.append(load_model_data(model['yaml_file_name']))
                active_image_sizes = {model['model_image_size'] for model in active_ai_models}
                active_categories = {category for model in active_ai_models for category in model['model_category']}
                if current_index >= len(active_ai_models):
                    current_index = len(active_ai_models) - 1
                # Re-sort the lists
                available_ai_models.sort(key=lambda x: (x['model_category'][0], int(x['model_identifier'])))
                active_ai_models.sort(key=lambda x: (x['model_category'][0], int(x['model_identifier'])))
        elif key == curses.KEY_LEFT:
            if current_list == 'active' and active_ai_models:
                model = active_ai_models.pop(current_index)
                available_ai_models.append(load_model_data(model['yaml_file_name']))
                active_image_sizes = {model['model_image_size'] for model in active_ai_models}
                active_categories = {category for model in active_ai_models for category in model['model_category']}
                if current_index >= len(active_ai_models):
                    current_index = len(active_ai_models) - 1
                # Re-sort the lists
                available_ai_models.sort(key=lambda x: (x['model_category'][0], int(x['model_identifier'])))
                active_ai_models.sort(key=lambda x: (x['model_category'][0], int(x['model_identifier'])))
            elif current_list == 'available' and available_ai_models:
                model = available_ai_models[current_index]
                incompatible = any(size != model['model_image_size'] for size in active_image_sizes) or any(category in active_categories for category in model['model_category'])
                if not incompatible:
                    available_ai_models.pop(current_index)
                    active_ai_models.append(model)
                    active_image_sizes.add(model['model_image_size'])
                    active_categories.update(model['model_category'])
                    if current_index >= len(available_ai_models):
                        current_index = len(available_ai_models) - 1
                    # Re-sort the lists
                    available_ai_models.sort(key=lambda x: (x['model_category'][0], int(x['model_identifier'])))
                    active_ai_models.sort(key=lambda x: (x['model_category'][0], int(x['model_identifier'])))
        elif key == ord('q'):
            break
        elif key == ord('\t'):
            if current_list == 'available':
                current_list = 'active'
            else:
                current_list = 'available'
            current_index = 0

        stdscr.refresh()

    save_active_ai_models([model['yaml_file_name'] for model in active_ai_models])

if __name__ == "__main__":
    choose_active_models()