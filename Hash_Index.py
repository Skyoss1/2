class HashIndex:
    def __init__(self):
        self.index = {}

    def insert(self, key, value):
        hash_key = self._hash_func(key)
        if hash_key not in self.index:
            self.index[hash_key] = []
        self.index[hash_key].append(value)

    def search(self, key):
        # Tìm kiếm tất cả các documents có review_id = key
        hash_key = self._hash_func(key)
        if hash_key in self.index:
            results = [item for item in self.index[hash_key] if item.get("review_id") == key]
            return results
        return []

    def _hash_func(self, key):
        return hash(key)
