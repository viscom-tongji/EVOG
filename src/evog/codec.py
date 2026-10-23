import torch

from evog.geometry import canonical_boxes, straight_through_values


class SpatialCodec:
    def __init__(self, tokenizer, frames, points, include_boxes=True):
        self.tokenizer = tokenizer
        self.frames = frames
        self.points = points
        self.include_boxes = include_boxes
        self.numeric_ids = []
        for i in range(10):
            encoded = tokenizer.encode(str(i), add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"Tokenizer must encode decimal digits as single tokens; failed at {i}")
            self.numeric_ids.append(encoded[0])
        if len(set(self.numeric_ids)) != 10:
            raise ValueError("Numeric tokens must have distinct IDs")
        self.id_to_value = {token: i / 9 for i, token in enumerate(self.numeric_ids)}
        self.layout = []
        self.numeric_positions = []
        for frame in range(frames):
            if frame:
                self.literal("\n")
            if include_boxes:
                self.literal("<BOX>")
                for i in range(4):
                    if i:
                        self.literal(" ")
                    self.number()
                self.literal("</BOX>; ")
            self.literal("<POINTS>")
            for i in range(points):
                if i:
                    self.literal("; ")
                for j in range(3):
                    if j:
                        self.literal(" ")
                    self.number()
            self.literal("</POINTS>")

    def literal(self, value):
        self.layout.extend(self.tokenizer.encode(value, add_special_tokens=False))

    def number(self):
        for _ in range(3):
            self.numeric_positions.append(len(self.layout))
            self.layout.append(None)

    def allowed(self, prefix_length):
        def choose(batch_id, input_ids):
            index = len(input_ids) - prefix_length
            if index >= len(self.layout):
                return [self.tokenizer.eos_token_id]
            token = self.layout[index]
            return self.numeric_ids if token is None else [token]

        return choose

    def values(self, generated, logits=None, temperature=1.0):
        tokens = generated[0].tolist()
        if len(tokens) != len(self.layout):
            raise ValueError(f"Truncated spatial output: {len(tokens)} of {len(self.layout)} tokens")
        for i, expected in enumerate(self.layout):
            if expected is not None and tokens[i] != expected:
                raise ValueError(f"Invalid spatial syntax at token {i}")
        values = torch.tensor(
            [self.id_to_value[tokens[i]] for i in self.numeric_positions], device=generated.device
        )
        if logits is not None:
            values = straight_through_values(logits, values, temperature)
        weights = values.new_tensor([100, 10, 1])
        values = (values.reshape(-1, 3) * weights).sum(-1) / 111
        width = (4 if self.include_boxes else 0) + 3 * self.points
        values = values.view(self.frames, width)
        boxes = canonical_boxes(values[:, :4]) if self.include_boxes else None
        points = values[:, 4 if self.include_boxes else 0 :].view(self.frames, self.points, 3)
        return boxes, points
