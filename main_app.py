# main_app.py
import pluggy
from insilicoICH import hooks  # Your hooks.py


def main():
    pm = pluggy.PluginManager(hooks.PROJECT_NAME)
    pm.add_hookspecs(hooks.PhantomSpecs)
    num_loaded = pm.load_setuptools_entrypoints(group=hooks.PROJECT_NAME)
    print(f"Loaded {num_loaded} plugins via entry points.")

    # --- Call the hook to get all registered phantom types ---
    # The hook returns a list of lists (one list per plugin implementation that returned something)
    list_of_results = pm.hook.register_phantom_types()

    # Flatten the list of lists and filter out None or empty lists from plugins
    discovered_phantom_classes = []
    for result_list in list_of_results:
        if result_list:  # Check if the plugin returned a non-empty list
            for phantom_cls in result_list:
                if phantom_cls not in discovered_phantom_classes: # Avoid duplicates if somehow registered twice
                     discovered_phantom_classes.append(phantom_cls)

    print("Discovered Phantom Types (Classes):")
    for cls in discovered_phantom_classes:
        print(f"- {cls.__name__} (from module: {cls.__module__})")

    # Get just the names for your list
    phantom_type_names = [cls.__name__ for cls in discovered_phantom_classes]
    print("\nList of Discovered Phantom Names:")
    print(phantom_type_names)


if __name__ == "__main__":
    # To run this, ensure your PYTHONPATH is set up correctly if phantom_project is not installed,
    # or run as a module from the parent directory: python -m phantom_project.main_app
    main()