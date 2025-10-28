class DataHandler: 
    def __init__(self, df):
        self.data = df.sort_index();
        self.pointer = 0;
        self.max_index =len(df)

    def reset(self):
        self.pointer = 0;

    def next(self):
        if self.pointer < self.max_index:
            row = self.data.iloc[self.pointer];
            self.pointer += 1;
            return row.name, row.to_dict();
        return None, None;

