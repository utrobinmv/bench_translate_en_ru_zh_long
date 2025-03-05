class BenchDataset:
    def __init__(self, list_items: list[dict], batch_size: int = 1):
        """
        list dict {'input': text,'target': text}
        """
        self.batch_size = batch_size

        self.list_items = list_items

    def __iter__(self):
        for i in range(0, len(self.list_items), self.batch_size):
            batch = self.list_items[i:i + self.batch_size]
            yield batch

    def __len__(self):
        return (len(self.list_items) + self.batch_size - 1) // self.batch_size
