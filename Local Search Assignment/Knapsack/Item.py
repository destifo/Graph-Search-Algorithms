class Item:

    name: str
    weight: float
    value: int
    value_per_weight: float

    def __init__(self, name, weight, value) -> None:
        self.name = name
        self.weight = weight
        self.value = value
        self.value_per_weight = value/weight


    def __str__(self) -> str:
        print(f'''
            item name: {self.name},
            item wieght: {self.weight},
            item value: {self.value} 
        ''')


    def __repr__(self) -> str:
        return (f'''
            item name: {self.name},
            item weight: {self.weight},
            item value: {self.value} 
        ''')
