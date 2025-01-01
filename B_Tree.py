import bisect

class BTreeNode:
    def __init__(self, t, leaf=False):
        self.t = t                # Bậc của B-Tree
        self.keys = []            # Danh sách các cặp (key, list_of_values)
        self.children = []        # Danh sách con
        self.leaf = leaf          # True nếu là nút lá

    def traverse(self):
        """In các khóa trong cây"""
        for i in range(len(self.keys)):
            if not self.leaf:
                self.children[i].traverse()
            # keys[i][0] là key, keys[i][1] là list_of_values
            print(self.keys[i][0], end=" ") 
        if not self.leaf:
            self.children[-1].traverse()

    def search(self, key):
        """Tìm kiếm khóa trong cây, trả về danh sách tất cả các giá trị"""
        i = 0
        while i < len(self.keys) and key > self.keys[i][0]:
            i += 1

        # Nếu tìm thấy khóa
        if i < len(self.keys) and self.keys[i][0] == key:
            return self.keys[i][1]  # Trả về list các giá trị tương ứng

        # Nếu là nút lá, không tìm thấy
        if self.leaf:
            return []

        # Tìm kiếm trong con phù hợp
        return self.children[i].search(key)


class BTree:
    def __init__(self, t):
        self.t = t                # Bậc của B-Tree
        self.root = BTreeNode(t, True)

    def insert(self, key, value):
        """Hàm chèn cặp (key, value). Nếu key đã tồn tại, thêm value vào danh sách."""
        root = self.root
        # Nếu gốc đầy, cần chia gốc
        if len(root.keys) == 2 * self.t - 1:
            new_root = BTreeNode(self.t)
            new_root.children.append(self.root)
            new_root.leaf = False
            self.split_child(new_root, 0)
            self.root = new_root

        self.insert_non_full(self.root, key, value)

    def split_child(self, parent, index):
        """Tách nút con đầy thành hai nút"""
        t = self.t
        child = parent.children[index]
        new_child = BTreeNode(t, child.leaf)

        # Đưa khóa ở giữa lên nút cha
        parent.keys.insert(index, child.keys[t - 1])
        parent.children.insert(index + 1, new_child)

        # Phân chia khóa và con giữa hai nút
        new_child.keys = child.keys[t:]
        child.keys = child.keys[:t - 1]

        if not child.leaf:
            new_child.children = child.children[t:]
            child.children = child.children[:t]

    def insert_non_full(self, node, key, value):
        """Chèn (key, value) vào nút chưa đầy.
           Nếu key tồn tại, append value vào list_of_values.
           Nếu không, chèn key mới.
        """
        i = len(node.keys) - 1

        if node.leaf:
            # Chèn cặp (key, value) trực tiếp vào nút lá
            # Kiểm tra xem key đã tồn tại hay chưa
            while i >= 0 and key < node.keys[i][0]:
                i -= 1
            i += 1

            # Nếu key đã tồn tại
            if i < len(node.keys) and node.keys[i][0] == key:
                node.keys[i][1].append(value)
            else:
                # Chèn mới
                node.keys.insert(i, (key, [value]))
        else:
            # Tìm con phù hợp để chèn
            while i >= 0 and key < node.keys[i][0]:
                i -= 1
            i += 1

            # Nếu con đầy, cần chia nút con
            if len(node.children[i].keys) == 2 * self.t - 1:
                self.split_child(node, i)
                if key > node.keys[i][0]:
                    i += 1

            self.insert_non_full(node.children[i], key, value)

    def traverse(self):
        """In toàn bộ cây"""
        if self.root:
            self.root.traverse()

    def search(self, key):
        """Tìm kiếm khóa trong cây, trả về danh sách các value cho key"""
        return self.root.search(key)
