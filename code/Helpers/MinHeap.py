import heapq


class MinHeap:
    """
    Min heap where each node is tuple of the form (vertex index, value)
    """

    def __init__(self, max_size=0):
        """
        Constructor for Min heap.
        Heap's elements mapped to a list such that, with 0-based indexing, if a node is stored at index k then its
        LEFT CHILD is stored at index 2k + 1 and its RIGHT CHILD is stored at index 2k + 2
        """
        self.nodes_lst = []
        self.__size = 0
        self.nodes_positions = []

    def add_node(self, vertex, dist):
        """
        Adds node to the heap
        :param vertex:
        :param dist:
        :return:
        """
        self.nodes_lst.append([vertex, dist])
        self.nodes_positions.append(vertex)
        self.__size += 1

    def swap(self, first_node, second_node):
        """
        Swaps two nodes at the heap - Needed for "heapify_down" function
        """
        temp_node = self.nodes_lst[first_node]
        self.nodes_lst[first_node] = self.nodes_lst[second_node]
        self.nodes_lst[second_node] = temp_node

    def heapify_down(self, node_to_heapify):
        """
        Apply heapify at a given idx. This function also updates position of nodes for update_vertex_val function
        """
        smallest = node_to_heapify
        left = 2 * node_to_heapify + 1
        right = 2 * node_to_heapify + 2

        # Finds the smallest node between node_to_heapify, its left child and its right child

        # Check if the left child exists and its value is smaller than node's values
        if left < self.__size and self.nodes_lst[left][1] < self.nodes_lst[smallest][1]:
            smallest = left

        # Check if the right child exists and its value is smaller than node's values
        if right < self.__size and self.nodes_lst[right][1] < self.nodes_lst[smallest][1]:
            smallest = right

        # Check if 'smallest' had changed
        if smallest != node_to_heapify:
            # Swap positions
            self.nodes_positions[self.nodes_lst[smallest][0]] = node_to_heapify
            self.nodes_positions[self.nodes_lst[node_to_heapify][0]] = smallest

            # Swap nodes
            self.swap(smallest, node_to_heapify)

            self.heapify_down(smallest)

    def extract_min(self):
        """
        Extract node with the minimal value at the heap
        """
        # Return None if heap is empty
        if self.is_empty():
            return

        # Save the root node
        root = self.nodes_lst[0]

        # Replace root node with last node
        last_node = self.nodes_lst[self.__size - 1]
        self.nodes_lst[0] = last_node

        # Update position of last node
        self.nodes_positions[last_node[0]] = 0
        self.nodes_positions[root[0]] = self.__size - 1

        # Reduce heap size and heapify root
        self.__size -= 1
        self.heapify_down(0)

        return root

    def is_empty(self):
        """
        Check if the heap empty
        """
        return self.__size == 0

    def update_vertex_val(self, vertex, val):
        """
        Update vertex's value and place it at the right place in the Min heap
        """
        vertex_heap_ind = self.nodes_positions[vertex]

        # Update node's value
        self.nodes_lst[vertex_heap_ind][1] = val

        vertex_parent_ind = (vertex_heap_ind - 1) // 2

        # Travel up while the complete tree is not heapified.
        while vertex_heap_ind > 0 and self.nodes_lst[vertex_heap_ind][1] < self.nodes_lst[vertex_parent_ind][1]:
            # Swap this node with its parents
            self.nodes_positions[self.nodes_lst[vertex_heap_ind][0]] = vertex_parent_ind
            self.nodes_positions[self.nodes_lst[vertex_parent_ind][0]] = vertex_heap_ind
            self.swap(vertex_heap_ind, vertex_parent_ind)

            # Move to parent's index
            vertex_heap_ind = vertex_parent_ind
            # Update parent to parent
            vertex_parent_ind = (vertex_heap_ind - 1) // 2

    def in_heap(self, vertex):
        """
        Check if a given vertex is in min heap or not
        """
        return self.nodes_positions[vertex] < self.__size

    def set_heap_size(self, size):
        self.__size = size