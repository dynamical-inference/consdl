class DotDict(dict):
    """
    A dictionary that allows accessing values using dot notation.
    Example:
        d = DotDict({'foo': {'bar': 42}})
        d.foo.bar  # returns 42
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Convert nested dictionaries to DotDict
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DotDict(value)
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")

    def to_dict(self):
        """Convert DotDict back to regular dictionary"""
        return {k: v.to_dict() if isinstance(v, DotDict) else v for k, v in self.items()}

    def __str__(self):
        """String representation of the DotDict"""
        return f"DotDict({super().__str__()})"

    def __repr__(self):
        """Detailed string representation of the DotDict"""
        return f"DotDict({super().__repr__()})"

    def print_all(self, indent=0):
        """
        Print all keys and values in a nicely formatted tree structure.

        Args:
            indent (int): Number of spaces to indent nested items
        """
        for key, value in self.items():
            if isinstance(value, DotDict):
                print(" " * indent + f"{key}:")
                value.print_all(indent + 4)
            else:
                print(" " * indent + f"{key}: {value}")

    def get_all_keys(self):
        """
        Get all keys in the DotDict, including nested keys.
        Returns a list of dot-notation keys.
        """
        keys = []
        for key, value in self.items():
            keys.append(key)
            if isinstance(value, DotDict):
                nested_keys = value.get_all_keys()
                keys.extend([f"{key}.{k}" for k in nested_keys])
        return keys

    def get_all_values(self):
        """
        Get all values in the DotDict, including nested values.
        Returns a list of all values.
        """
        values = []
        for value in self.values():
            if isinstance(value, DotDict):
                values.extend(value.get_all_values())
            else:
                values.append(value)
        return values
