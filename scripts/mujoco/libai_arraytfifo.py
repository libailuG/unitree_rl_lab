import numpy as np

class ArrayFIFO:
    """
    固定大小的数组FIFO缓存，支持任意数量的数据流
    
    数据排列顺序: 所有组的数组1, 所有组的数组2, ...
    即: A1...An, B1...Bn, C1...Cn, D1...Dn, ...
    
    buffer布局:
    [A0, A1, ..., A9, B0, B1, ..., B9, C0, C1, ..., C9, ...]
    """
    def __init__(self, num_groups, num_arrays=3):
        """
        初始化FIFO
        
        Parameters:
        -----------
        num_groups : int
            存储的组数（FIFO深度）
        num_arrays : int
            每组中数组的个数
        """
        self.num_groups = num_groups
        self.num_arrays = num_arrays
        
        # 存储整个FIFO的固定大小数组
        self.buffer = None
        self.total_size = 0
        
        # 记录每个数组的长度（第一次push时确定）
        self.array_lengths = None
        # 记录每个数组在buffer中的起始位置
        self.array_offsets = None
        
        # 当前存储的组数
        self.current_groups = 0
        
    def push(self, *arrays):
        """
        向FIFO中添加一组新数据
        
        Returns:
        --------
        fifo_data : ndarray
            整个FIFO的一维数组
        """
        # 检查数组数量是否正确
        if len(arrays) != self.num_arrays:
            raise ValueError(f"需要 {self.num_arrays} 个数组，但传入了 {len(arrays)} 个")
        
        # 确保所有输入都是一维数组
        arrays = [np.array(arr).flatten() for arr in arrays]
        
        # 如果是第一次push，初始化buffer
        if self.buffer is None:
            # 记录每个数组的长度
            self.array_lengths = [len(arr) for arr in arrays]
            
            # 计算每个数组在buffer中的起始位置
            # 每个数组占据: num_groups * array_length 个位置
            self.array_offsets = []
            offset = 0
            for length in self.array_lengths:
                self.array_offsets.append(offset)
                offset += self.num_groups * length
            self.total_size = offset
            
            # 初始化buffer，用NaN填充
            self.buffer = np.full(self.total_size, np.nan)
            self.current_groups = 0
        
        # 检查数组长度是否一致
        for i, arr in enumerate(arrays):
            if len(arr) != self.array_lengths[i]:
                raise ValueError(
                    f"数组{i}长度 {len(arr)} 与首次记录的长度 {self.array_lengths[i]} 不一致"
                )
        
        # 如果已满，移除最旧的一组
        if self.current_groups == self.num_groups:
            # 对每个数组，移除最旧的数据（每组移除第一个元素）
            for i in range(self.num_arrays):
                arr_len = self.array_lengths[i]
                start_pos = self.array_offsets[i]
                # 左移：移除第一个元素
                self.buffer[start_pos:start_pos + (self.num_groups - 1) * arr_len] = \
                    self.buffer[start_pos + arr_len:start_pos + self.num_groups * arr_len]
                # 清空最后一组的位置
                self.buffer[start_pos + (self.num_groups - 1) * arr_len:start_pos + self.num_groups * arr_len] = np.nan
            self.current_groups -= 1
        
        # 在末尾添加新数据
        for i, arr in enumerate(arrays):
            arr_len = self.array_lengths[i]
            start_pos = self.array_offsets[i]
            # 计算插入位置（该数组的当前末尾）
            insert_pos = start_pos + self.current_groups * arr_len
            self.buffer[insert_pos:insert_pos + arr_len] = arr
        
        # 更新组数
        self.current_groups += 1
        
        return self.get_fifo()
    
    def get_fifo(self):
        """获取整个FIFO的一维数组"""
        if self.buffer is None:
            return np.array([])
        return self.buffer.copy()
    
    def get_fifo_valid(self):
        """获取当前有效数据（去除NaN）"""
        if self.buffer is None or self.current_groups == 0:
            return np.array([])
        
        result = []
        for i in range(self.num_arrays):
            arr_len = self.array_lengths[i]
            start_pos = self.array_offsets[i]
            total_len = self.current_groups * arr_len
            result.append(self.buffer[start_pos:start_pos + total_len].copy())
        
        return np.concatenate(result)
    
    def get_fifo_by_array(self):
        """按数组类型获取数据"""
        if self.buffer is None or self.current_groups == 0:
            return [np.array([]) for _ in range(self.num_arrays)]
        
        result = []
        for i in range(self.num_arrays):
            arr_len = self.array_lengths[i]
            start_pos = self.array_offsets[i]
            total_len = self.current_groups * arr_len
            result.append(self.buffer[start_pos:start_pos + total_len].copy())
        return result
    
    def get_fifo_by_group(self):
        """按组获取数据"""
        if self.buffer is None or self.current_groups == 0:
            return []
        
        result = []
        for g in range(self.current_groups):
            group_data = []
            for i in range(self.num_arrays):
                arr_len = self.array_lengths[i]
                start_pos = self.array_offsets[i] + g * arr_len
                group_data.append(self.buffer[start_pos:start_pos + arr_len].copy())
            result.append(tuple(group_data))
        return result
    
    def get_group(self, index):
        """获取指定组的数据"""
        if self.buffer is None or self.current_groups == 0:
            return None
        
        if index < 0:
            index = self.current_groups + index
        
        if index < 0 or index >= self.current_groups:
            raise IndexError(f"组索引 {index} 超出范围")
        
        group_data = []
        for i in range(self.num_arrays):
            arr_len = self.array_lengths[i]
            start_pos = self.array_offsets[i] + index * arr_len
            group_data.append(self.buffer[start_pos:start_pos + arr_len].copy())
        return tuple(group_data)
    
    def get_array_lengths(self):
        return self.array_lengths.copy() if self.array_lengths else None
    
    def get_total_size(self):
        return self.total_size
    
    def clear(self):
        if self.buffer is not None:
            self.buffer.fill(np.nan)
            self.current_groups = 0
    
    def reset(self):
        self.buffer = None
        self.total_size = 0
        self.array_lengths = None
        self.array_offsets = None
        self.current_groups = 0
    
    def is_full(self):
        return self.current_groups == self.num_groups
    
    def __len__(self):
        return self.current_groups
    
    def __repr__(self):
        return (f"ArrayFIFO(num_groups={self.num_groups}, num_arrays={self.num_arrays}, "
                f"current_groups={self.current_groups})\n"
                f"整个FIFO: {self.get_fifo()}")
