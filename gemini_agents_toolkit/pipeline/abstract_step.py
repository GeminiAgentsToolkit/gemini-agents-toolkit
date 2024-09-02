class AbstractStep(object):

    def __init__(self, name: str = None, debug: bool = False):
        self._next_step = None
        self.previous_step = None
        self.executed = False
        self.debug = debug
        self.name = name

    def execute(self, data: str = None):
        if self.debug:
            print(f"Running step: {self.name}")
            print(f"Data: {data}")
        if self.previous_step and not self.previous_step.executed:
            if self.debug:
                print(f"Chaining from previous step: {self.previous_step.name}")
            return self.previous_step.execute()
        new_data = self.run(data)
        if self.debug:
            print(f"New data: {new_data}")
        self.executed = True
        if self._next_step:
            if self.debug:
                print(f"Chaining to next step: {self._next_step.name}")
            return self._next_step.execute(new_data)
        else:
            if self.debug:
                print("No more steps to chain to")
            return new_data

    def run(self, data):
        pass

    def set_next_step(self, step):
        while step.previous_step:
            step = step.previous_step
        if not self._next_step:
            self._next_step = step
            step.previous_step = self
            return step
        else:
            raise ValueError("Next step already exists")

    def __or__(self, other):
        if isinstance(other, AbstractStep):
            return self.set_next_step(other)
        else:
            raise ValueError("Can only chain steps with other steps")
